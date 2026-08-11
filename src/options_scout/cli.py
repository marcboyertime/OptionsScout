"""Auditable, offline command line interface for the read-only research workflow."""

# ruff: noqa: E701, E702
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any

from .calendar import provider_status
from .demo import write_capture
from .engine import parse_liquidity_rules
from .models import (
    CorrelationRecord,
    Decision,
    PortfolioAssessment,
    PortfolioLimits,
    PortfolioPosition,
    jsonable,
)
from .normalizer import normalize_envelope
from .pipeline import SchemaError, evaluate, parse_run
from .portfolio import duplicate_expression
from .preflight import compare_payloads
from .reporting import write_reports
from .safety import SafetyError, assert_read_only_operation, policy, static_safety_scan
from .store import AuditStore, canonical, digest
from .universe import DEFAULT_UNIVERSE, UniverseError, load_universe


def root() -> Path:
    """Use an explicit deployment root, not a fragile launcher/cwd assumption."""
    explicit = os.environ.get("OPTIONS_SCOUT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "config" / "policy.json").is_file():
            return candidate
    return Path.cwd().resolve()


def database_path(base: Path) -> Path:
    return base / "artifacts" / "options_scout.sqlite3"


def ensure_safe_config(base: Path) -> None:
    """Bootstrap only the same empty allowlist policy used by the repository."""
    path = base / "config" / "policy.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "phase1-local",
                "max_risk_per_trade_usd": "1000",
                "outer_operational_containment_usd": "1000",
                "fees_per_contract_usd": "0",
                "physical_settlement_exit_before_expiration": True,
                "max_dte": 30,
                "quote_max_age_seconds": 90,
                "leg_sync_max_seconds": 15,
                "liquidity": {
                    "underlying_minimum_usd": "100000",
                    "single_leg_min_volume": 1,
                    "single_leg_min_open_interest": 25,
                    "complex_or_early_exit_min_volume": 10,
                    "complex_or_early_exit_min_open_interest": 100,
                    "premium_bands": [
                        {
                            "max_premium": "1",
                            "max_relative_spread": "0.25",
                            "max_absolute_spread": "0.20",
                        },
                        {
                            "max_premium": "5",
                            "max_relative_spread": "0.20",
                            "max_absolute_spread": "0.50",
                        },
                        {
                            "max_premium": None,
                            "max_relative_spread": "0.15",
                            "max_absolute_spread": "1.00",
                        },
                    ],
                },
                "enabled_tools": [],
                "approved_capture_schemas": [],
                "denied_tool_patterns": [
                    "place_",
                    "cancel_",
                    "replace_",
                    "roll_",
                    "exercise",
                    "transfer",
                    "deposit",
                    "withdraw",
                    "fund",
                    "account_setting",
                ],
                "allow_live_without_catalog": False,
            },
            indent=2,
        )
    )


def ensure_universe_config(base: Path) -> None:
    path = base / "config/universe.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_UNIVERSE, indent=2))


def ensure_codex_config(base: Path) -> None:
    """Bootstrap a disabled repository-scoped MCP profile without OAuth material."""
    path = base / ".codex/config.toml"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Repository profile only. OAuth configuration remains outside this repository.
[mcp_servers.robinhood-trading]
url = "https://agent.robinhood.com/mcp/trading"
enabled = false
enabled_tools = []

[options_scout]
mode = "read_only_research"
fail_closed_on_unknown_tools = true
"""
    )


def output(value: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(f"{key}: {item}" for key, item in value.items()))


def _store(base: Path) -> AuditStore:
    instance = AuditStore(database_path(base))
    instance.initialize()
    return instance


def _trace(seed: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"options-scout:{seed}").hex


def _codex_profile_status(base: Path, expected_tools: list[str]) -> dict[str, Any]:
    try:
        loaded = tomllib.loads((base / ".codex/config.toml").read_text())
        servers = loaded.get("mcp_servers")
        server = servers.get("robinhood-trading") if isinstance(servers, dict) else None
        if not isinstance(server, dict):
            raise ValueError("Robinhood server profile is missing")
        actual_tools = server.get("enabled_tools")
        enabled = server.get("enabled")
        exact = (
            server.get("url") == "https://agent.robinhood.com/mcp/trading"
            and isinstance(actual_tools, list)
            and all(isinstance(item, str) for item in actual_tools)
            and actual_tools == expected_tools
            and type(enabled) is bool
            and enabled == bool(expected_tools)
        )
        return {
            "status": "EXACT_MATCH" if exact else "MISMATCH_AND_FAIL_CLOSED",
            "enabled": enabled is True,
            "enabled_tools": actual_tools if isinstance(actual_tools, list) else [],
        }
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return {"status": "UNAVAILABLE_AND_FAIL_CLOSED", "enabled": False, "enabled_tools": []}


def _source_health(base: Path, store: AuditStore | None = None) -> dict[str, Any]:
    config = policy(base)
    last_capture: dict[str, Any] | None = None
    if store is not None:
        row = store.conn.execute(
            "SELECT created_at,source_label,payload_hash FROM captures ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            last_capture = dict(row)
    empty = not config["enabled_tools"]
    codex_profile = _codex_profile_status(base, config["enabled_tools"])
    reviewed_ready = not empty and codex_profile["status"] == "EXACT_MATCH"
    return {
        "decision": Decision.DATA_INSUFFICIENT.value,
        "catalog": "UNAVAILABLE: no authenticated catalog has been reviewed"
        if empty
        else "REVIEWED_READ_ONLY_CAPTURE_POLICY",
        "mcp_allowlist": config["enabled_tools"],
        "allowlist_status": (
            "EMPTY_AND_FAIL_CLOSED"
            if empty
            else (
                "REVIEWED_READ_ONLY_CAPTURE_POLICY_READY"
                if reviewed_ready
                else "REVIEWED_POLICY_CODEX_PROFILE_MISMATCH_AND_FAIL_CLOSED"
            )
        ),
        "codex_profile": codex_profile,
        "live_quotes": "UNAVAILABLE: no fresh reviewed redacted capture ingested",
        "calendar": provider_status(),
        "last_redacted_capture": last_capture,
        "freshness": "UNAVAILABLE" if not last_capture else "RECORDED_REDACTED_CAPTURE",
    }


def _research_health(store: AuditStore) -> dict[str, Any]:
    """Local readiness only; no missing live source is implied to be healthy."""
    snapshots = int(store.conn.execute("SELECT COUNT(*) FROM research").fetchone()[0])
    return {
        "pipeline": "READY_LOCAL_TYPED_PROVENANCE",
        "live_research_sources": "UNAVAILABLE: no authenticated catalog/capture",
        "persisted_research_snapshots": snapshots,
        "decision_effect": "DATA_INSUFFICIENT until source/claim provenance is supplied in a typed run",
    }


def _unavailable_payload(trace: str, base: Path) -> dict[str, Any]:
    return {
        "version": "phase1-decision-v1",
        "trace_id": trace,
        "decision": Decision.DATA_INSUFFICIENT.value,
        "source_label": "UNAVAILABLE",
        "freshness": "unavailable",
        "non_live": False,
        "counts": {
            "universe": 0,
            "equity_filtered": 0,
            "chain_validated": 0,
            "structures": 0,
            "finalists": 0,
        },
        "evaluations": [],
        "ranked": [],
        "gates": [
            {
                "name": "mcp_catalog",
                "passed": False,
                "reason": "No authenticated Robinhood read-only schemas are exposed; a fresh human-reviewed catalog is required.",
                "severity": "hard",
            }
        ],
        "source_health": _source_health(base),
    }


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    return value


def normalized_input_hash(raw: dict[str, Any]) -> str:
    """Hash the complete normalized input while excluding only its binding."""
    unsigned = dict(raw)
    unsigned.pop("capture_binding", None)
    return hashlib.sha256(canonical(unsigned).encode()).hexdigest()


def _capture_projection(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the security-relevant normalized quote/mechanics projection.

    A capture must retain this already-redacted projection.  Comparing it to
    the typed normalized run proves that price, contract mechanics and quote
    provenance were not introduced by an envelope-supplied hash alone.
    """
    metadata = raw.get("metadata")
    candidates = raw.get("candidates")
    if not isinstance(metadata, dict):
        return {}
    if not isinstance(candidates, list):
        return {}
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return {}
        contracts = candidate.get("contracts")
        if not isinstance(contracts, list) or any(not isinstance(contract, dict) for contract in contracts):
            return {}
        records.append(
            {
                "id": candidate.get("id"),
                "symbol": candidate.get("symbol"),
                "underlying": candidate.get("underlying"),
                "underlying_as_of": candidate.get("underlying_as_of"),
                "quote_provenance": {
                    key: candidate.get("quote_provenance", {}).get(key)
                    if isinstance(candidate.get("quote_provenance"), dict)
                    else None
                    for key in ("source_id", "retrieved_at", "as_of", "source", "methodology")
                },
                "mechanics": {
                    key: candidate.get("mechanics", {}).get(key)
                    if isinstance(candidate.get("mechanics"), dict)
                    else None
                    for key in (
                        "asset_type", "product_type", "exercise_style", "settlement_style", "deliverable",
                        "ex_dividend_date", "ex_dividend_amount", "assignment_risk", "pin_risk",
                        "auto_exercise", "corporate_action", "product_calendar",
                    )
                },
                "contracts": [
                    {
                        key: contract.get(key)
                        for key in (
                            "id",
                            "symbol",
                            "expiration",
                            "strike",
                            "type",
                            "multiplier",
                            "tradable",
                            "adjusted",
                            "exercise_style",
                            "settlement_style",
                        )
                    }
                    for contract in contracts
                ],
            }
        )
        for projected, contract in zip(records[-1]["contracts"], contracts, strict=True):
            quote = contract.get("quote")
            projected["quote"] = (
                {
                    key: raw.get("source_label") if key == "source" else quote.get(key)
                    for key in (
                        "bid", "ask", "mark", "iv", "delta", "gamma", "theta", "vega",
                        "as_of", "source", "volume", "open_interest",
                    )
                }
                if isinstance(quote, dict)
                else None
            )
    return {
        "metadata": {key: metadata.get(key) for key in ("run_id", "as_of", "retrieved_at")},
        "source_label": raw.get("source_label"),
        "candidates": records,
    }


def _verify_capture_binding(
    raw: dict[str, Any], capture: dict[str, Any], config: dict[str, Any], now: datetime
) -> tuple[bool, str]:
    """Verify immutable capture, normalized input, policy identity, and freshness."""
    binding = raw.get("capture_binding")
    fields = {
        "capture_id",
        "tool",
        "schema_identity",
        "parameter_schema_sha256",
        "response_schema_sha256",
        "normalized_projection_schema_sha256",
        "payload_hash",
        "normalized_input_hash",
        "source_label",
        "as_of",
        "retrieved_at",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != fields
        or any(not isinstance(binding[key], str) or not binding[key] for key in fields)
    ):
        return False, "LIVE input lacks a strict capture_binding"
    if not all(
        capture.get(key) == binding[key]
        for key in (
            "tool",
            "schema_identity",
            "parameter_schema_sha256",
            "response_schema_sha256",
            "normalized_projection_schema_sha256",
            "payload_hash",
            "normalized_input_hash",
            "source_label",
            "as_of",
            "retrieved_at",
        )
    ):
        return (
            False,
            "capture_binding does not exactly match immutable capture tool/schema/hash/source/timestamps",
        )
    if binding["normalized_input_hash"] != normalized_input_hash(raw):
        return False, "capture binding does not cryptographically bind this normalized input"
    payload = capture.get("payload")
    if not isinstance(payload, dict) or payload.get("normalized_projection") != _capture_projection(raw):
        return False, "immutable capture does not exactly contain normalized quote/mechanics projection"
    try:
        parsed = parse_run(raw)
        captured_at = datetime.fromisoformat(
            binding["retrieved_at"].replace("Z", "+00:00")
        ).astimezone(UTC)
    except (SchemaError, ValueError):
        return False, "capture binding timestamps or normalized input are invalid"
    capture_binding = parsed.capture_binding
    if capture_binding is None:
        return False, "LIVE input lacks a parsed capture binding"
    if (
        parsed.source.value != binding["source_label"]
        or parsed.metadata.as_of != capture_binding.as_of
        or parsed.metadata.retrieved_at != capture_binding.retrieved_at
    ):
        return False, "normalized metadata source/as_of/retrieved_at does not match capture binding"
    for candidate in parsed.candidates:
        if (
            candidate.structure is None
            or candidate.source.value != binding["source_label"]
            or candidate.underlying_as_of != capture_binding.as_of
            or candidate.quote_provenance.as_of != capture_binding.as_of
            or candidate.quote_provenance.retrieved_at != capture_binding.retrieved_at
        ):
            return (
                False,
                "normalized underlying/provenance timestamp does not match capture binding",
            )
        if any(
            leg.contract.quote.as_of != capture_binding.as_of for leg in candidate.structure.legs
        ):
            return False, "normalized option quote timestamp does not match capture binding"
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        return False, "normalized candidate set is invalid"
    for candidate in raw_candidates:
        contracts = candidate.get("contracts") if isinstance(candidate, dict) else None
        if not isinstance(contracts, list) or any(
            not isinstance(contract, dict)
            or not isinstance(contract.get("quote"), dict)
            or contract["quote"].get("as_of") != binding["as_of"]
            for contract in contracts
        ):
            return False, "all normalized option quote timestamps must match capture binding"
    if captured_at > now or (now - captured_at).total_seconds() > 90:
        return False, "immutable capture is future-dated or stale"
    if not config["enabled_tools"] or not config.get("approved_capture_schemas"):
        return False, "current policy has an empty reviewed read-only capture allowlist"
    if binding["tool"] not in config["enabled_tools"]:
        return False, "capture tool is not in the positive policy allowlist"
    approved = any(
        isinstance(item, dict)
        and item.get("tool") == binding["tool"]
        and item.get("schema_identity") == binding["schema_identity"]
        and item.get("parameter_schema_sha256") == binding["parameter_schema_sha256"]
        and item.get("response_schema_sha256") == binding["response_schema_sha256"]
        and item.get("normalized_projection_schema_sha256")
        == binding["normalized_projection_schema_sha256"]
        and item.get("source_label") == binding["source_label"]
        for item in config["approved_capture_schemas"]
    )
    return (
        (True, "immutable capture binding and freshness verified")
        if approved
        else (False, "capture schema/source is not in the reviewed policy allowlist")
    )


def _operational_capture_authorization(
    base: Path, raw: dict[str, Any], now: datetime
) -> tuple[bool, str]:
    """Bind a claimed LIVE normalized document to an immutable capture record.

    A JSON file is an untrusted snapshot, not a live-data capability.  This
    check is intentionally at the operator boundary so offline pure evaluation
    remains useful to tests and research while scan/analyze cannot bless a
    forged, old, or self-labelled LIVE document.
    """
    if raw.get("fixture") is True:
        return False, "fixture input is explicitly non-live"
    binding = raw.get("capture_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("capture_id"), str):
        return False, "LIVE input lacks a strict capture_binding"
    config = policy(base)
    store = _store(base)
    try:
        capture = store.capture_envelope(binding["capture_id"])
    finally:
        store.close()
    if capture is None:
        return False, "capture_binding does not reference an immutable capture record"
    return _verify_capture_binding(raw, capture, config, now)


def capture_ingest(base: Path, path: str, as_json: bool) -> int:
    """Persist one reviewed, already-redacted envelope; never invoke a broker."""
    try:
        config = policy(base)
        envelope = normalize_envelope(_read_json(path), set(config["enabled_tools"]))
        if envelope.get("fixture_namespace"):
            raise SafetyError("fixture envelopes cannot enter the LIVE capture store")
        # This is intentionally a runtime assertion only.  It validates the
        # declared tool against the reviewed policy and never calls it.
        assert_read_only_operation(str(envelope["tool"]), base)
        approved = {
            (
                item["tool"],
                item["schema_identity"],
                item["parameter_schema_sha256"],
                item["response_schema_sha256"],
                item["normalized_projection_schema_sha256"],
                item["source_label"],
            )
            for item in config["approved_capture_schemas"]
        }
        identity = (
            str(envelope["tool"]),
            str(envelope["schema_identity"]),
            str(envelope["parameter_schema_sha256"]),
            str(envelope["response_schema_sha256"]),
            str(envelope["normalized_projection_schema_sha256"]),
            str(envelope["source_label"]),
        )
        if identity not in approved:
            raise SafetyError("capture tool/schema/source is not exactly approved by policy")
        store = _store(base)
        try:
            capture_id = store.append_capture(envelope)
        finally:
            store.close()
        output(
            {
                "status": "INGESTED_REDACTED_CAPTURE",
                "capture_id": capture_id,
                "tool": envelope["tool"],
                "schema_identity": envelope["schema_identity"],
                "source_label": envelope["source_label"],
                "normalized_input_hash": envelope["normalized_input_hash"],
                "broker_invoked": False,
            },
            as_json,
        )
        return 0
    except (OSError, json.JSONDecodeError, SafetyError, ValueError) as error:
        output(
            {"status": "DATA_INSUFFICIENT", "error": str(error), "broker_invoked": False}, as_json
        )
        return 2


def _evaluate_input(
    base: Path, path: str, fixture_requested: bool, ticker: str | None = None
) -> dict[str, Any]:
    raw = _read_json(path)
    operational_raw = raw
    if ticker:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            raise SchemaError("typed input has no candidates")
        raw = {
            **raw,
            "candidates": [
                item
                for item in candidates
                if isinstance(item, dict) and item.get("symbol") == ticker
            ],
        }
        if not raw["candidates"]:
            raise ValueError(f"ticker {ticker} is absent from typed input")
    parsed = parse_run(raw)
    config = policy(base)
    result = jsonable(
        evaluate(
            parsed,
            parse_liquidity_rules(config["liquidity"]),
            Decimal(str(config["fees_per_contract_usd"])),
        )
    )
    fixture = bool(raw.get("fixture", False) or fixture_requested)
    authorized, authorization_reason = (
        _operational_capture_authorization(base, operational_raw, datetime.now(UTC))
        if not fixture
        else (False, "fixture input is explicitly non-live")
    )
    if fixture or not authorized:
        result["decision"] = Decision.DATA_INSUFFICIENT.value
        result.setdefault("gates", []).append(
            {
                "name": "operational_capture_binding",
                "passed": False,
                "reason": authorization_reason,
                "severity": "hard",
            }
        )
        for item in result.get("evaluations", []):
            item["decision"] = Decision.DATA_INSUFFICIENT.value
    return {
        "version": "phase1-decision-v1",
        "source_label": "ESTIMATED" if fixture else str(raw.get("source_label", "UNAVAILABLE")),
        "freshness": "fixture normalized input" if fixture else authorization_reason,
        "non_live": fixture or not authorized,
        "normalized_input_hash": normalized_input_hash(operational_raw),
        "input_versions": raw.get("metadata", {}),
        "git_commit": os.environ.get("GIT_COMMIT", "unavailable"),
        "source_health": _source_health(base),
        **result,
    }


def _alert_fingerprint(payload: dict[str, Any]) -> str:
    """State-aware deduplication: quote churn must not re-alert a NO_TRADE."""
    state = str(payload.get("decision"))
    material_economics = state in {
        Decision.ACTIONABLE.value,
        Decision.WATCH.value,
        Decision.INVALIDATED.value,
    }
    candidates: list[dict[str, Any]] = []
    raw_evaluations = payload.get("evaluations", [])
    if not isinstance(raw_evaluations, list):
        raw_evaluations = []
    for evaluation_raw in raw_evaluations:
        if not isinstance(evaluation_raw, dict):
            continue
        evaluation: dict[str, Any] = evaluation_raw
        candidate_value = evaluation.get("candidate")
        candidate: dict[str, Any] = candidate_value if isinstance(candidate_value, dict) else {}
        analysis_value = evaluation.get("analysis")
        analysis: dict[str, Any] = analysis_value if isinstance(analysis_value, dict) else {}
        payoff_value = analysis.get("payoff")
        payoff: dict[str, Any] = payoff_value if isinstance(payoff_value, dict) else {}
        structure_value = candidate.get("structure")
        structure: dict[str, Any] = structure_value if isinstance(structure_value, dict) else {}
        gates_value = evaluation.get("gates")
        gates: list[Any] = gates_value if isinstance(gates_value, list) else []
        item: dict[str, Any] = {
            "id": evaluation.get("id"),
            "symbol": evaluation.get("symbol"),
            "decision": evaluation.get("decision"),
            "structure": {
                "kind": structure.get("name"),
                "quantity": structure.get("quantity"),
                "legs": [
                    {
                        "side": leg.get("side"),
                        "ratio": leg.get("ratio"),
                        "strike": leg.get("contract", {}).get("strike")
                        if isinstance(leg, dict) and isinstance(leg.get("contract"), dict)
                        else None,
                    }
                    for leg in structure.get("legs", [])
                    if isinstance(leg, dict)
                ],
            },
            "failed_gates": sorted(
                str(gate.get("name"))
                for gate in gates
                if isinstance(gate, dict) and gate.get("status") == "FAIL"
            ),
        }
        if material_economics:
            item["economics"] = {
                key: payoff.get(key)
                for key in ("entry", "operational_max_loss_risk", "max_gain", "breakevens")
            }
        candidates.append(item)
    global_gates = payload.get("gates")
    global_gate_items: list[Any] = global_gates if isinstance(global_gates, list) else []
    failures = sorted(
        str(gate.get("name"))
        for gate in global_gate_items
        if isinstance(gate, dict) and (gate.get("status") == "FAIL" or gate.get("passed") is False)
    )
    source_health = payload.get("source_health")
    source_identity = (
        {
            key: source_health.get(key)
            for key in ("decision", "catalog", "allowlist_status", "live_quotes", "freshness")
        }
        if isinstance(source_health, dict)
        else {"missing": True}
    )
    return digest(
        {
            "decision": state,
            "source_label": payload.get("source_label"),
            "candidates": candidates,
            "failed_gates": failures,
            "source_health": source_identity,
        }
    )


def persist(base: Path, trace: str, payload: dict[str, Any]) -> dict[str, Any]:
    store = _store(base)
    try:
        payload["audit_hash"] = store.append_decision(trace, payload)
        state = str(payload.get("decision"))
        fingerprint = _alert_fingerprint(payload)
        payload["alert_eligible"] = store.alert_once(
            fingerprint,
            trace,
            state,
            {"decision": state, "source_label": payload.get("source_label")},
        )
    finally:
        store.close()
    payload["artifacts"] = write_reports(base, trace, payload)
    return payload


def scan(
    base: Path, input_path: str | None, fixture: bool, ticker: str | None, as_json: bool
) -> int:
    trace = _trace(f"{datetime.now(UTC).isoformat()}:{input_path}:{fixture}:{ticker}")
    try:
        payload = (
            _evaluate_input(base, input_path, fixture, ticker)
            if input_path
            else (
                _evaluate_input(base, str(base / "fixtures" / "normalized-run.json"), True, ticker)
                if fixture
                else _unavailable_payload(trace, base)
            )
        )
        payload["trace_id"] = trace
        output(persist(base, trace, payload), as_json)
        return 0 if payload["decision"] != Decision.DATA_INSUFFICIENT.value or fixture else 3
    except (
        OSError,
        json.JSONDecodeError,
        SchemaError,
        SafetyError,
        ValueError,
        InvalidOperation,
    ) as error:
        output({"decision": Decision.DATA_INSUFFICIENT.value, "error": str(error)}, as_json)
        return 2


def portfolio_check(path: str | None, as_json: bool) -> int:
    if not path:
        output(
            {
                "status": "DATA_INSUFFICIENT",
                "decision_effect": "ACTIONABLE blocked: redacted portfolio context is missing.",
            },
            as_json,
        )
        return 0
    try:
        context = _read_json(path)
        result: dict[str, Any]
        if "schema_version" in context:
            parsed = parse_run(context)
            candidate = parsed.candidates[0]
            assessment = candidate.portfolio_assessment
            limits = assessment.limits
            result = {
                "trade_risk": assessment.aggregate_risk,
                "aggregate_risk": assessment.aggregate_risk,
                "cluster_risk": assessment.cluster_risk,
                "event_risk": assessment.event_risk,
                "sector_risk": assessment.sector_risk,
                "factor_risk": assessment.factor_risk,
                "remaining_aggregate": limits.remaining_aggregate,
                "remaining_cluster": limits.remaining_cluster,
                "remaining_event": limits.remaining_event,
                "remaining_sector": limits.remaining_sector,
                "remaining_factor": limits.remaining_factor,
                "hard_cap": limits.hard_cap,
                "duplicate_exposure": duplicate_expression(
                    assessment,
                    candidate.symbol,
                    {
                        item
                        for item in (
                            candidate.equity_context.sector_behavior,
                            candidate.equity_context.factor_behavior,
                        )
                        if item
                    },
                ),
                "position_count": len(assessment.positions),
                "correlation_count": len(assessment.correlations),
            }
        else:
            required = {
                "max_risk_per_trade_usd",
                "remaining_aggregate_risk_usd",
                "remaining_cluster_risk_usd",
                "remaining_event_risk_usd",
                "remaining_sector_risk_usd",
                "remaining_factor_risk_usd",
                "trade_risk_usd",
                "proposed_symbol",
                "proposed_factor_tags",
                "positions",
                "correlations",
            }
            if (
                not isinstance(context.get("proposed_symbol"), str)
                or not context["proposed_symbol"].strip()
                or not isinstance(context.get("proposed_factor_tags"), list)
                or not context["proposed_factor_tags"]
                or any(not isinstance(item, str) or not item.strip() for item in context["proposed_factor_tags"])
            ):
                output(
                    {
                        "status": "DATA_INSUFFICIENT",
                        "decision_effect": "ACTIONABLE blocked: proposed symbol/factor context is missing or unverified.",
                        "sensitive_data": "not persisted",
                    },
                    as_json,
                )
                return 0
            if set(context) != required:
                raise ValueError("portfolio input must use the strict redacted portfolio schema")
            numeric = {key: value for key, value in context.items() if key.endswith("_usd")}
            if any(not isinstance(value, str) for value in numeric.values()):
                raise ValueError("portfolio numeric fields must be exact Decimal strings")
            result = {key.removesuffix("_usd"): Decimal(value) for key, value in numeric.items()}
            if not isinstance(context["positions"], list) or not isinstance(context["correlations"], list):
                raise ValueError("portfolio positions/correlations must be bounded lists")
            if len(context["positions"]) > 100 or len(context["correlations"]) > 100:
                raise ValueError("portfolio positions/correlations exceed the bounded input limit")
            positions: list[PortfolioPosition] = []
            position_ids: set[str] = set()
            for item in context["positions"]:
                if not isinstance(item, dict) or set(item) != {
                    "id", "symbol", "sector", "factor_tags", "risk", "event_risk"
                }:
                    raise ValueError("portfolio position fields are invalid")
                if (
                    not all(isinstance(item[key], str) and item[key].strip() for key in ("id", "symbol", "sector", "risk", "event_risk"))
                    or not isinstance(item["factor_tags"], list)
                    or any(not isinstance(tag, str) or not tag.strip() for tag in item["factor_tags"])
                    or item["id"] in position_ids
                ):
                    raise ValueError("portfolio position values are invalid")
                risk, event_risk = Decimal(item["risk"]), Decimal(item["event_risk"])
                if not risk.is_finite() or not event_risk.is_finite() or risk < 0 or event_risk < 0:
                    raise ValueError("portfolio position risks must be finite non-negative Decimals")
                position_ids.add(item["id"])
                positions.append(
                    PortfolioPosition(
                        item["id"], item["symbol"], item["sector"], tuple(item["factor_tags"]), risk, event_risk
                    )
                )
            correlations: list[CorrelationRecord] = []
            correlated_ids: set[str] = set()
            for item in context["correlations"]:
                if not isinstance(item, dict) or set(item) != {"position_id", "correlation", "rationale"}:
                    raise ValueError("portfolio correlation fields are invalid")
                if not all(isinstance(item[key], str) and item[key].strip() for key in item):
                    raise ValueError("portfolio correlation values are invalid")
                correlation = Decimal(item["correlation"])
                if (
                    not correlation.is_finite()
                    or not Decimal("-1") <= correlation <= Decimal("1")
                    or item["position_id"] not in position_ids
                    or item["position_id"] in correlated_ids
                ):
                    raise ValueError("portfolio correlation is invalid")
                correlated_ids.add(item["position_id"])
                correlations.append(CorrelationRecord(item["position_id"], correlation, item["rationale"]))
            assessment = PortfolioAssessment(
                tuple(positions),
                PortfolioLimits(
                    Decimal("1000"),
                    result["remaining_aggregate_risk"], result["remaining_cluster_risk"],
                    result["remaining_event_risk"], result["remaining_sector_risk"], result["remaining_factor_risk"],
                ),
                tuple(correlations), Decimal(), Decimal(), Decimal(), Decimal(), Decimal(), "transient verified context",
            )
            result["duplicate_exposure"] = duplicate_expression(
                assessment, context["proposed_symbol"], set(context["proposed_factor_tags"])
            )
            result["position_count"] = len(positions)
            result["correlation_count"] = len(correlations)
        numeric_result = [value for value in result.values() if isinstance(value, Decimal)]
        if any(not value.is_finite() or value < Decimal() for value in numeric_result):
            raise ValueError("portfolio risks and limits must be finite non-negative Decimals")
        declared_cap = result.get("max_risk_per_trade", result.get("hard_cap"))
        if declared_cap != Decimal("1000"):
            raise ValueError("max_risk_per_trade must be exactly 1000")
        limit_values = [
            value
            for key, value in result.items()
            if key.startswith("remaining_") or key in {"max_risk_per_trade", "hard_cap"}
        ]
        trade = result.get("trade_risk", Decimal("0"))
        duplicate = bool(result["duplicate_exposure"])
        state = (
            "BLOCKED"
            if duplicate or any(trade > limit for limit in limit_values)
            else "VERIFIED_LOCAL"
        )
        safe_result = {
            key: value
            for key, value in result.items()
            if key not in {"correlations", "clusters"}
        }
        output(
            {
                "status": state,
                "lower_of_limit": format(min(limit_values), "f"),
                "assessment": jsonable(safe_result),
                "sensitive_data": "not persisted",
            },
            as_json,
        )
        return 0
    except (OSError, json.JSONDecodeError, ValueError, InvalidOperation) as error:
        output({"status": "DATA_INSUFFICIENT", "error": str(error)}, as_json)
        return 2


def calibration(base: Path, outcome_path: str | None, as_json: bool) -> int:
    store = _store(base)
    try:
        if outcome_path:
            record = _read_json(outcome_path)
            decision_id = str(record.get("decision_id", ""))
            if not decision_id or store.decision(decision_id) is None:
                raise ValueError("outcome must reference an existing decision_id")
            required = {
                "decision_id",
                "predicted_probability",
                "realized",
                "expected_pnl",
                "realized_pnl",
                "structure",
                "catalyst",
                "volatility",
                "liquidity",
                "procedural_no_trade",
            }
            if (
                set(record) != required
                or type(record["realized"]) is not bool
                or type(record["procedural_no_trade"]) is not bool
            ):
                raise ValueError("outcome must use the strict redacted outcome schema")
            if any(
                not isinstance(record[field], str)
                for field in (
                    "predicted_probability",
                    "expected_pnl",
                    "realized_pnl",
                    "structure",
                    "catalyst",
                    "volatility",
                    "liquidity",
                )
            ):
                raise ValueError(
                    "outcome probabilities, economics, and group labels must be exact strings"
                )
            probability = Decimal(str(record["predicted_probability"]))
            if not Decimal("0") <= probability <= Decimal("1"):
                raise ValueError("predicted_probability must be an exact decimal in [0,1]")
            for field in ("expected_pnl", "realized_pnl"):
                Decimal(str(record[field]))
            store.append_outcome(decision_id, record)
        outcomes = store.outcomes()
        scored: list[dict[str, Any]] = []
        for item in outcomes:
            value = item["payload"]
            if value.get("procedural_no_trade") is True:
                continue
            try:
                probability = Decimal(str(value["predicted_probability"]))
                expected, realized_pnl = (
                    Decimal(str(value["expected_pnl"])),
                    Decimal(str(value["realized_pnl"])),
                )
            except (KeyError, InvalidOperation):
                continue
            if (
                not Decimal("0") <= probability <= Decimal("1")
                or type(value.get("realized")) is not bool
            ):
                continue
            scored.append(
                {
                    **item,
                    "probability": probability,
                    "expected": expected,
                    "realized_pnl": realized_pnl,
                }
            )
        if not scored:
            output(
                {
                    "status": "DATA_INSUFFICIENT",
                    "outcomes": len(outcomes),
                    "brier_score": None,
                    "note": "outcomes are append-only; no scored prediction/realized pairs yet. NO_TRADE records are not scored from later price movement alone.",
                },
                as_json,
            )
            return 0
        brier = sum(
            (item["probability"] - Decimal(int(item["payload"]["realized"]))) ** 2
            for item in scored
        ) / Decimal(len(scored))

        def groups(field: str) -> dict[str, dict[str, str | int]]:
            result: dict[str, list[dict[str, Any]]] = {}
            for item in scored:
                result.setdefault(str(item["payload"][field]), []).append(item)
            return {
                name: {
                    "count": len(records),
                    "expected_pnl": format(
                        sum((record["expected"] for record in records), Decimal())
                        / Decimal(len(records)),
                        "f",
                    ),
                    "realized_pnl": format(
                        sum((record["realized_pnl"] for record in records), Decimal())
                        / Decimal(len(records)),
                        "f",
                    ),
                }
                for name, records in result.items()
            }

        output(
            {
                "status": "CALIBRATED",
                "outcomes": len(outcomes),
                "scored": len(scored),
                "procedural_no_trade": sum(
                    1 for item in outcomes if item["payload"].get("procedural_no_trade") is True
                ),
                "brier_score": format(brier, "f"),
                "expected_vs_realized": {
                    "expected_pnl": format(
                        sum((item["expected"] for item in scored), Decimal())
                        / Decimal(len(scored)),
                        "f",
                    ),
                    "realized_pnl": format(
                        sum((item["realized_pnl"] for item in scored), Decimal())
                        / Decimal(len(scored)),
                        "f",
                    ),
                },
                "groups": {
                    field: groups(field)
                    for field in ("structure", "catalyst", "volatility", "liquidity")
                },
            },
            as_json,
        )
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        output({"status": "DATA_INSUFFICIENT", "error": str(error)}, as_json)
        return 2
    finally:
        store.close()


def schedule_plan(base: Path, as_json: bool) -> int:
    schedule = base / "artifacts" / "options-scout.schedule.sh"
    plist = base / "artifacts" / "com.options-scout.disabled.plist"
    regular_plist = base / "artifacts" / "com.options-scout.regular.disabled.plist"
    prompt = base / "artifacts" / "options-scout.live-orchestration.md"
    schedule.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(
        """# Disabled OptionsScout live orchestration\n\nUse the OptionsScout skill. Respect `OPTIONS_SCOUT_SCHEDULE_MODE`: deep premarket/after-close/weekend work may refresh the bounded universe and dated catalysts, while regular mode is a lightweight refresh and must begin at least 15 minutes after the open. Inspect only the reviewed catalog; invoke only exact positive-allowlisted read-only market-data tools. Never access account, balance, position, or order data. Create only an already-redacted capture envelope and bound normalized typed run, then run `options-scout capture-ingest --capture-input ...` before `options-scout scan --input ...`. Never submit or mutate anything.\n"""
    )
    schedule.write_text(f"""#!/bin/sh
# DISABLED TEST-SAFE. Do not activate before unattended OAuth is demonstrated.
# enabled_tools=[]; generated default remains fail-closed until reviewed policy/capture proof.
set -eu
ROOT='{base}'
export OPTIONS_SCOUT_ROOT="$ROOT"
cd "$ROOT"
CONSOLE="$ROOT/.venv/bin/options-scout"
PYTHON="$ROOT/.venv/bin/python"
LOCK="$ROOT/artifacts/options-scout.schedule.lock"
# DISABLED cadence templates (not installed):
# 05:30 deep-premarket: --deep
# regular +15m: --regular (local XNYS gate starts 15 minutes after open)
# 16:30 after-close: --deep
# Saturday 10:00 weekend-refresh: --deep
# 5m-active-candidate intentionally has no automatic template.
MODE="${{1:-}}"
if [ "$MODE" = "--regular" ]; then
  "$PYTHON" -c 'from datetime import UTC,datetime,timedelta; from options_scout.calendar import session_status; status=session_status(datetime.now(UTC)); opened=datetime.fromisoformat(str(status.get("open"))); raise SystemExit(0 if status.get("regular") is True and datetime.now(UTC).astimezone(opened.tzinfo) >= opened + timedelta(minutes=15) else 1)' || exit 0
  export OPTIONS_SCOUT_SCHEDULE_MODE="LIGHTWEIGHT_REGULAR_15M"
  MODE="--run"
elif [ "$MODE" = "--deep" ]; then
  OPTIONS_SCOUT_SCHEDULE_MODE="$("$PYTHON" -c 'from datetime import UTC,datetime; from options_scout.calendar import NY,session_status; now=datetime.now(UTC); local=now.astimezone(NY); status=session_status(now); kind=\"\"; kind=\"DEEP_PREMARKET\" if status.get(\"available\") is True and status.get(\"session\") == \"PRE_OR_AFTER\" and local.hour < 9 else kind; kind=\"DEEP_AFTER_CLOSE\" if status.get(\"available\") is True and status.get(\"session\") == \"PRE_OR_AFTER\" and local.hour >= 16 else kind; kind=\"WEEKEND_CATALYST_REFRESH\" if local.weekday() == 5 else kind; print(kind); raise SystemExit(0 if kind else 1)')" || exit 0
  export OPTIONS_SCOUT_SCHEDULE_MODE
  MODE="--run"
fi
if [ "$MODE" = "--test" ]; then
  "$PYTHON" -c 'import json,subprocess,sys; health=subprocess.run([sys.argv[1],"source-health","--json"],capture_output=True,text=True,timeout=1200); data=json.loads(health.stdout); status=data.get("allowlist_status"); assert health.returncode == 0 and status in {{"EMPTY_AND_FAIL_CLOSED","REVIEWED_READ_ONLY_CAPTURE_POLICY_READY"}}; assert (status == "EMPTY_AND_FAIL_CLOSED" and data.get("decision") == "DATA_INSUFFICIENT") or (status == "REVIEWED_READ_ONLY_CAPTURE_POLICY_READY" and data.get("codex_profile",{{}}).get("status") == "EXACT_MATCH")' "$CONSOLE"
elif [ "$MODE" = "--run" ]; then
  mkdir "$LOCK" 2>/dev/null || exit 0
  trap 'rmdir "$LOCK"' EXIT
  "$PYTHON" -c 'import json,os,pathlib,subprocess,sys; health=subprocess.run([sys.argv[1],"source-health","--json"],capture_output=True,text=True,timeout=1200); data=json.loads(health.stdout); assert health.returncode == 0 and data["allowlist_status"] == "REVIEWED_READ_ONLY_CAPTURE_POLICY_READY" and data["codex_profile"]["status"] == "EXACT_MATCH"; runner=pathlib.Path(os.environ.get("OPTIONS_SCOUT_CODEX_RUNNER", "")); assert runner.is_absolute() and runner.is_file() and os.access(runner,os.X_OK); prompt=pathlib.Path(sys.argv[2]).resolve(); root=pathlib.Path(sys.argv[3]).resolve(); assert prompt.parent == root / "artifacts"; subprocess.run([str(runner),str(prompt)],cwd=root,timeout=1200,check=True)' "$CONSOLE" "$ROOT/artifacts/options-scout.live-orchestration.md" "$ROOT"
else
  exit 64
fi
""")
    plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>Label</key><string>com.options-scout.deep.disabled</string><key>Disabled</key><true/><key>ProgramArguments</key><array><string>{base}/artifacts/options-scout.schedule.sh</string><string>--deep</string></array><key>WorkingDirectory</key><string>{base}</string><key>StartCalendarInterval</key><array><dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict><dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>5</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict><dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict></array><key>Comment</key><string>Disabled deep/weekend template; --deep still enforces the local XNYS/day gate and reviewed runner controls.</string></dict></plist>
""")
    regular_plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>Label</key><string>com.options-scout.regular.disabled</string><key>Disabled</key><true/><key>ProgramArguments</key><array><string>{base}/artifacts/options-scout.schedule.sh</string><string>--regular</string></array><key>WorkingDirectory</key><string>{base}</string><key>StartInterval</key><integer>900</integer><key>Comment</key><string>Disabled 15-minute template; --regular exits locally outside XNYS regular hours and before open plus 15 minutes.</string></dict></plist>
""")
    schedule.chmod(0o700)
    output(
        {
            "status": "disabled/test-safe",
            "plan": str(schedule),
            "launchd_template": str(plist),
            "regular_launchd_template": str(regular_plist),
            "activation": "Blocked until unattended OAuth and authenticated catalog behavior are empirically demonstrated. Do not install this plan.",
            "protections": [
                "absolute cwd",
                "deployed venv console",
                "non-overlap lock",
                "source-health fail closed",
                "test mode",
                "enabled_tools=[]",
                "no mutation tools",
            ],
        },
        as_json,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="options-scout", description="Read-only, fail-closed options research operator"
    )
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "health", "source-health", "safety-audit", "schedule-plan"):
        sub.add_parser(name).add_argument("--json", action="store_true")
    item = sub.add_parser("capture-ingest")
    item.add_argument("--json", action="store_true")
    item.add_argument("--capture-input", required=True)
    for name in ("scan", "analyze"):
        item = sub.add_parser(name)
        item.add_argument("--json", action="store_true")
        item.add_argument("--input")
        item.add_argument("--fixture", action="store_true")
        item.add_argument("--ticker")
        item.add_argument("--symbol", dest="ticker")
    item = sub.add_parser("preflight")
    item.add_argument("--json", action="store_true")
    item.add_argument("--decision-id", "--id", dest="decision_id")
    item.add_argument("--refreshed-input", "--input", dest="refreshed_input")
    item.add_argument("--move", default=None)
    item = sub.add_parser("portfolio-check")
    item.add_argument("--json", action="store_true")
    item.add_argument("--input")
    item.add_argument("--portfolio-input", dest="input")
    item = sub.add_parser("report")
    item.add_argument("--json", action="store_true")
    item.add_argument("--decision-id", "--id", dest="decision_id")
    item.add_argument("--latest", action="store_true")
    item = sub.add_parser("calibration")
    item.add_argument("--json", action="store_true")
    item.add_argument("--outcome-input")
    history = sub.add_parser("history")
    history.add_argument("--json", action="store_true")
    history.add_argument("--limit", type=int, default=50)
    history.add_argument("--decision")
    args = parser.parse_args()
    base, as_json = root(), bool(args.json)
    try:
        if args.command == "init":
            ensure_safe_config(base)
            ensure_universe_config(base)
            ensure_codex_config(base)
            policy(base)
            store = _store(base)
            store.close()
            write_capture(base / "fixtures" / "capture-demo.json")
            fixture = base / "fixtures" / "normalized-run.json"
            if not fixture.exists():
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_text(
                    resources.files("options_scout")
                    .joinpath("fixtures/normalized-run.json")
                    .read_text()
                )
            output(
                {
                    "status": "initialized",
                    "database": str(database_path(base)),
                    "schema_version": AuditStore.VERSION,
                    "fixture": str(base / "fixtures" / "normalized-run.json"),
                },
                as_json,
            )
            return 0
        if args.command == "health":
            store = _store(base)
            status = {
                "database": database_path(base).exists(),
                "database_integrity": store.integrity(),
                "calendar": provider_status(),
                "research": _research_health(store),
                "scheduler": "disabled/test-safe",
                "version": AuditStore.VERSION,
                "universe": load_universe(base).health(),
                **_source_health(base, store),
            }
            store.record_health(status)
            store.close()
            output(status, as_json)
            return 0
        if args.command in {"scan", "analyze"}:
            if args.command == "analyze" and not args.ticker:
                output(
                    {
                        "error": "analyze requires --ticker",
                        "decision": Decision.DATA_INSUFFICIENT.value,
                    },
                    as_json,
                )
                return 2
            return scan(
                base,
                args.input,
                args.fixture,
                args.ticker if args.command == "analyze" else None,
                as_json,
            )
        if args.command == "capture-ingest":
            return capture_ingest(base, args.capture_input, as_json)
        if args.command == "source-health":
            store = _store(base)
            value = _source_health(base, store)
            store.close()
            output(value, as_json)
            return 0
        if args.command == "safety-audit":
            store = _store(base)
            violations = static_safety_scan(base)
            chain = store.integrity()
            store.close()
            output(
                {
                    "passed": not violations and chain,
                    "violations": violations,
                    "database_integrity_and_hash_chain": chain,
                    "read_only_allowlist": policy(base)["enabled_tools"],
                    "runtime_guard": "enabled",
                },
                as_json,
            )
            return 0 if not violations and chain else 1
        if args.command == "portfolio-check":
            return portfolio_check(args.input, as_json)
        if args.command == "history":
            store = _store(base)
            records = store.history(args.limit, args.decision)
            store.close()
            output({"count": len(records), "records": records}, as_json)
            return 0
        if args.command == "report":
            store = _store(base)
            record = (
                store.decision(args.decision_id)
                if args.decision_id
                else (store.history(1)[0] if args.latest or not args.decision_id else None)
            )
            store.close()
            if not record:
                output({"status": "DATA_INSUFFICIENT", "report": None}, as_json)
                return 0
            paths = write_reports(base, str(record["id"]), record["payload"])
            output({"decision_id": record["id"], "artifacts": paths}, as_json)
            return 0
        if args.command == "calibration":
            return calibration(base, args.outcome_input, as_json)
        if args.command == "schedule-plan":
            return schedule_plan(base, as_json)
        if args.command == "preflight":
            store = _store(base)
            previous = store.decision(args.decision_id) if args.decision_id else None
            if args.refreshed_input:
                if previous is None or previous["payload"].get("decision") not in {
                    Decision.ACTIONABLE.value,
                    Decision.WATCH.value,
                }:
                    raise ValueError(
                        "operational preflight requires exact --decision-id for an original ACTIONABLE or WATCH decision"
                    )
                refreshed = _evaluate_input(base, args.refreshed_input, False)
                comparison = {
                    "original_input_hash": previous["input_hash"],
                    "refreshed_input_hash": refreshed["normalized_input_hash"],
                    **compare_payloads(previous["payload"], refreshed),
                }
                decision = (
                    Decision.INVALIDATED.value
                    if comparison["invalid"]
                    or refreshed["decision"]
                    not in {Decision.ACTIONABLE.value, Decision.WATCH.value}
                    else refreshed["decision"]
                )
            elif args.move is not None:
                if previous is None or not previous["payload"].get("non_live"):
                    raise ValueError("--move is fixture-only and requires a fixture decision id")
                try:
                    moved = abs(Decimal(args.move))
                except InvalidOperation as error:
                    raise ValueError("--move must be a decimal") from error
                decision, comparison = (
                    (
                        Decision.INVALIDATED.value
                        if moved > Decimal("0.02")
                        else Decision.WATCH.value
                    ),
                    {
                        "fixture_regression": True,
                        "underlying_move": format(moved, "f"),
                        "threshold": "0.02",
                        "reran_all_gates": True,
                    },
                )
            else:
                output(
                    {
                        "error": "preflight requires --refreshed-input or fixture-only --move",
                        "decision": Decision.DATA_INSUFFICIENT.value,
                    },
                    as_json,
                )
                store.close()
                return 2
            payload = {
                "decision": decision,
                "comparison": comparison,
                "human_next_step": "Compare with Robinhood's live review screen; never submit.",
            }
            if previous:
                store.append_preflight(str(previous["id"]), payload)
            store.close()
            output(payload, as_json)
            return 0
    except (OSError, json.JSONDecodeError, SafetyError, UniverseError, ValueError) as error:
        output({"error": str(error), "decision": Decision.DATA_INSUFFICIENT.value}, as_json)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
