"""Fix-first adversarial regressions for operational safety and economics."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from options_scout.cli import (
    _capture_projection,
    _operational_capture_authorization,
    _verify_capture_binding,
    capture_ingest,
    normalized_input_hash,
    portfolio_check,
)
from options_scout.engine import parse_liquidity_rules
from options_scout.pipeline import evaluate, parse_run
from options_scout.preflight import compare_evaluations, compare_payloads
from options_scout.reporting import markdown_report
from options_scout.safety import normalized_projection_schema_sha256
from options_scout.schema import SchemaError
from options_scout.store import AuditStore


def _run() -> dict[str, object]:
    value = json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())
    value["fixture"] = False
    candidate = value["candidates"][0]
    candidate["structures"] = [{"id": "long-c100", "name": "long_call", "quantity": 1, "legs": [{"side": "buy", "contract_id": "c100", "ratio": 1}]}]
    candidate["selected_structure_id"] = "long-c100"
    candidate["structure_plan"]["structure_type"] = "long call"
    candidate["mechanics"]["product_calendar"] = "XNYS"
    candidate["judge"] = {"verdict": "survived", "reason": "survived direct rejection review"}
    candidate["skeptic"]["crowding"] = "Measured open-interest concentration reviewed"
    candidate["contracts"][0]["quote"].update({"gamma": "0.01", "theta": "-0.02"})
    candidate["contracts"][0]["quote"].update({"bid": "0.70", "ask": "0.80"})
    candidate["contracts"][1]["quote"].update(
        {"bid": "0.32", "ask": "0.40", "gamma": "0.01", "theta": "-0.02"}
    )
    candidate["distribution"] = {
        "scenario_model": "bounded_terminal_spot_v1",
        "provenance": "deterministic bounded terminal model from quoted implied move, realized volatility, and recorded observations",
        "scenarios": [
            {
                "id": "up",
                "outcome": "up",
                "probability": "0.60",
                "expiration_spot": "110",
                "payoff": "913.90",
            },
            {
                "id": "flat",
                "outcome": "flat",
                "probability": "0.40",
                "expiration_spot": "100",
                "payoff": "-86.10",
            },
        ],
        "sensitivity_cases": [
            {
                "id": "adverse_shift",
                "model": "adverse_probability_shift_v1",
                "probability_shift_to_worst": "0.05",
                "additional_cost": "0",
                "expected_value": "463.90",
            }
        ],
    }
    return value


@pytest.mark.parametrize("field", ["fees", "commissions", "exit_slippage"])
def test_negative_operational_costs_are_schema_rejected(field: str) -> None:
    raw = _run()
    raw["candidates"][0]["structure_plan"][field] = "-0.01"
    with pytest.raises(SchemaError, match="non-negative"):
        parse_run(raw)


def test_negative_entry_slippage_is_schema_rejected() -> None:
    raw = _run()
    raw["candidates"][0]["fill_plan"]["max_slippage"] = "-0.01"
    with pytest.raises(SchemaError, match="non-negative"):
        parse_run(raw)


def test_scenario_million_dollar_payoff_exploit_is_rejected() -> None:
    raw = _run()
    raw["candidates"][0]["distribution"]["scenarios"][0]["payoff"] = "1000000"
    with pytest.raises(SchemaError, match="exactly match"):
        parse_run(raw)


def test_terminal_spot_tail_exploit_and_missing_distribution_provenance_are_rejected() -> None:
    raw = _run()
    raw["candidates"][0]["distribution"]["scenarios"][0].update(
        {"probability": "0.00001", "expiration_spot": "1000000", "payoff": "999913.90"}
    )
    raw["candidates"][0]["distribution"]["scenarios"][1]["probability"] = "0.99999"
    with pytest.raises(SchemaError, match="support bounds"):
        parse_run(raw)
    raw = _run()
    raw["candidates"][0]["distribution"].pop("provenance")
    with pytest.raises(SchemaError, match="distribution fields"):
        parse_run(raw)


def test_max_acceptable_fill_alone_cannot_evict_the_1000_cap() -> None:
    raw = _run()
    candidate = raw["candidates"][0]
    candidate["fill_plan"]["limit"] = "1500"
    candidate["structure_plan"].update({"entry_limit": "1500", "max_acceptable_limit": "1500"})
    candidate["distribution"].update(
        {
            "scenarios": [
                {
                    "id": "up",
                    "outcome": "up",
                    "probability": "0.60",
                    "expiration_spot": "110",
                    "payoff": "-501.10",
                },
                {
                    "id": "flat",
                    "outcome": "flat",
                    "probability": "0.40",
                    "expiration_spot": "100",
                    "payoff": "-1501.10",
                },
            ],
            "sensitivity_cases": [
                {
                    "id": "adverse_shift",
                    "model": "adverse_probability_shift_v1",
                    "probability_shift_to_worst": "0.05",
                    "additional_cost": "0",
                    "expected_value": "-1151.10",
                }
            ],
        }
    )
    evaluation = evaluate(parse_run(raw))["evaluations"][0]
    gates = {gate["name"]: gate for gate in evaluation["gates"]}
    assert gates["fee_inclusive_1000_cap"]["status"] == "FAIL"
    assert evaluation["analysis"]["payoff"]["operational_max_loss_risk"] == "1501.10"


def test_reported_payoff_table_breakeven_and_gain_use_maximum_permitted_entry() -> None:
    evaluation = evaluate(parse_run(_run()))["evaluations"][0]
    payoff = evaluation["analysis"]["payoff"]
    assert payoff["fills"]["realistic_limit_entry"] == "77.500"
    assert payoff["entry"] == "85.00"
    assert payoff["theoretical_max_loss"] == "77.500"
    assert payoff["max_gain"] is None
    assert payoff["breakevens"] == ["100.85"]
    assert payoff["table"][1] == ["100", "-85.00"]


def test_fill_target_must_cover_realistic_cash_and_respect_85_maximum() -> None:
    raw = _run()
    candidate = raw["candidates"][0]
    candidate["fill_plan"]["limit"] = "0.01"
    candidate["structure_plan"].update({"entry_limit": "0.01", "max_acceptable_limit": "85"})
    gates = {gate["name"]: gate for gate in evaluate(parse_run(raw))["evaluations"][0]["gates"]}
    assert gates["realistic_fill"]["status"] == "FAIL"
    raw = _run()
    raw["candidates"][0]["structure_plan"]["max_acceptable_limit"] = "85"
    raw["candidates"][0]["fill_plan"]["limit"] = "999"
    raw["candidates"][0]["structure_plan"]["entry_limit"] = "999"
    with pytest.raises(SchemaError, match="inconsistent"):
        parse_run(raw)


def test_etf_european_cash_short_is_still_prohibited() -> None:
    raw = _run()
    candidate = raw["candidates"][0]
    candidate["mechanics"].update({"asset_type": "ETF option", "product_type": "ETF"})
    result = evaluate(parse_run(raw))
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    assert gates["exercise_style"]["status"] == "PASS"
    assert gates["settlement_style"]["status"] == "PASS"


def test_earnings_tomorrow_requires_iv_drop_provenance_and_complete_greeks() -> None:
    raw = _run()
    raw["candidates"][0]["upcoming_events"] = [
        {
            "event_type": "earnings",
            "event_at": "2026-08-12T14:00:00+00:00",
            "expected_iv_drop": "0.05",
            "provenance": "issuer date plus documented IV model",
        }
    ]
    raw["candidates"][0]["contracts"][0]["quote"]["gamma"] = None
    gates = {gate["name"]: gate for gate in evaluate(parse_run(raw))["evaluations"][0]["gates"]}
    assert gates["event_iv_crush"]["status"] == "FAIL"
    raw = _run()
    raw["candidates"][0]["upcoming_events"] = [
        {
            "event_type": "earnings",
            "event_at": "2026-08-12T14:00:00+00:00",
            "expected_iv_drop": "0",
            "provenance": "missing assumption",
        }
    ]
    with pytest.raises(SchemaError, match="positive expected IV-drop"):
        parse_run(raw)


def test_earnings_tomorrow_with_empty_upcoming_events_cannot_bypass_iv_gate() -> None:
    raw = _run()
    raw["candidates"][0]["thesis"].update(
        {"catalyst_type": "earnings", "catalyst_at": "2026-08-12T14:00:00+00:00"}
    )
    gates = {gate["name"]: gate for gate in evaluate(parse_run(raw))["evaluations"][0]["gates"]}
    assert gates["event_iv_crush"]["status"] == "FAIL"


def test_spanning_upcoming_binary_event_consumes_event_portfolio_capacity() -> None:
    raw = _run()
    candidate = raw["candidates"][0]
    candidate["thesis"].update(
        {"catalyst_type": "earnings", "catalyst_at": "2026-08-12T14:00:00+00:00"}
    )
    candidate["upcoming_events"] = [
        {
            "event_type": "earnings",
            "event_at": "2026-08-12T14:00:00+00:00",
            "expected_iv_drop": "0.05",
            "provenance": "issuer calendar plus explicit IV-drop model",
        }
    ]
    candidate["portfolio_assessment"]["limits"]["remaining_event"] = "1"
    gates = {gate["name"]: gate for gate in evaluate(parse_run(raw))["evaluations"][0]["gates"]}
    assert gates["event_risk"]["status"] == "FAIL"


def test_forged_old_live_binding_is_not_current_operational_data(tmp_path: Path) -> None:
    raw = _run()
    binding = {
        "capture_id": "old",
        "tool": "approved.quote",
        "schema_identity": "quote/v1",
        "parameter_schema_sha256": "1" * 64,
        "response_schema_sha256": "2" * 64,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "payload_hash": hashlib.sha256(b"{}").hexdigest(),
        "normalized_input_hash": "0" * 64,
        "source_label": "LIVE",
        "as_of": "2020-01-01T00:00:00+00:00",
        "retrieved_at": "2020-01-01T00:00:01+00:00",
    }
    raw["capture_binding"] = binding
    binding["normalized_input_hash"] = normalized_input_hash(raw)
    assert parse_run(raw).capture_binding is not None
    config = tmp_path / "config"
    config.mkdir()
    (config / "policy.json").write_text(
        (Path(__file__).parents[1] / "config" / "policy.json").read_text()
    )
    store = AuditStore(tmp_path / "artifacts" / "options_scout.sqlite3")
    store.initialize()
    store.append_capture(
        {
            "schema_version": "1",
            **binding,
            "field_provenance": {},
            "redacted_arguments": {},
            "payload": {},
        }
    )
    store.close()
    allowed, reason = _operational_capture_authorization(tmp_path, raw, datetime.now(UTC))
    assert not allowed and reason


def test_current_capture_cannot_be_reused_with_altered_normalized_input() -> None:
    raw = _run()
    candidate = raw["candidates"][0]
    candidate["quote_provenance"]["retrieved_at"] = "2026-08-11T15:00:30+00:00"
    binding = {
        "capture_id": "current",
        "tool": "reviewed.quote",
        "schema_identity": "quote/v1",
        "parameter_schema_sha256": "1" * 64,
        "response_schema_sha256": "2" * 64,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "payload_hash": hashlib.sha256(b"payload").hexdigest(),
        "normalized_input_hash": "0" * 64,
        "source_label": "LIVE",
        "as_of": "2026-08-11T15:00:00+00:00",
        "retrieved_at": "2026-08-11T15:00:30+00:00",
    }
    raw["capture_binding"] = binding
    binding["normalized_input_hash"] = normalized_input_hash(raw)
    capture = {**binding, "payload": {"normalized_projection": _capture_projection(raw)}}
    reviewed = {
        "enabled_tools": ["reviewed.quote"],
        "approved_capture_schemas": [
            {
                "tool": "reviewed.quote",
                "schema_identity": "quote/v1",
                "source_label": "LIVE",
                "parameter_schema_sha256": "1" * 64,
                "response_schema_sha256": "2" * 64,
                "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
            }
        ],
    }
    assert _verify_capture_binding(
        raw, capture, reviewed, datetime.fromisoformat("2026-08-11T15:00:45+00:00")
    )[0]
    altered = deepcopy(raw)
    altered["candidates"][0]["underlying"] = "101"
    allowed, reason = _verify_capture_binding(
        altered, capture, reviewed, datetime.fromisoformat("2026-08-11T15:00:45+00:00")
    )
    assert not allowed and "cryptographically bind" in reason


def test_capture_projection_binds_raw_contract_type_call_put_identity() -> None:
    raw = _run()
    before = _capture_projection(raw)
    raw["candidates"][0]["contracts"][0]["type"] = "put"
    assert _capture_projection(raw) != before


def test_reviewed_live_capture_round_trips_two_public_contract_ids_without_private_extras(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The redacted capture is an exact public market projection, not an opaque payload."""
    raw = _run()
    # The chain deliberately contains both option types; the selected long call
    # remains the same so parsing/evaluation exercises a non-selected public put.
    raw["candidates"][0]["contracts"][1]["type"] = "put"
    raw["candidates"][0]["contracts"][1]["quote"]["delta"] = "-0.35"
    raw["candidates"][0]["quote_provenance"]["retrieved_at"] = "2026-08-11T15:00:30+00:00"
    contracts = raw["candidates"][0]["contracts"]
    assert [contract["id"] for contract in contracts] == ["c100", "c105"]
    assert [contract["type"] for contract in contracts] == ["call", "put"]
    projection = _capture_projection(raw)
    payload = {"normalized_projection": projection}
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = {
        "capture_id": "two-contract-live-capture",
        "tool": "reviewed.market_quote",
        "schema_identity": "reviewed.market.quote/v1",
        "parameter_schema_sha256": "1" * 64,
        "response_schema_sha256": "2" * 64,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "payload_hash": payload_hash,
        "normalized_input_hash": "0" * 64,
        "source_label": "LIVE",
        "as_of": "2026-08-11T15:00:00+00:00",
        "retrieved_at": "2026-08-11T15:00:30+00:00",
    }
    raw["capture_binding"] = binding
    binding["normalized_input_hash"] = normalized_input_hash(raw)
    envelope = {
        "schema_version": "1",
        **binding,
        "field_provenance": {},
        "redacted_arguments": {},
        "payload": payload,
    }
    config = json.loads((Path(__file__).parents[1] / "config/policy.json").read_text())
    config.update(
        {
            "enabled_tools": ["reviewed.market_quote"],
            "approved_capture_schemas": [
                {
                    "tool": binding["tool"],
                    "schema_identity": binding["schema_identity"],
                    "source_label": "LIVE",
                    "parameter_schema_sha256": binding["parameter_schema_sha256"],
                    "response_schema_sha256": binding["response_schema_sha256"],
                    "normalized_projection_schema_sha256": binding[
                        "normalized_projection_schema_sha256"
                    ],
                }
            ],
        }
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    input_path = tmp_path / "redacted-envelope.json"
    input_path.write_text(json.dumps(envelope))
    assert capture_ingest(tmp_path, str(input_path), True) == 0
    assert json.loads(capsys.readouterr().out)["broker_invoked"] is False
    allowed, reason = _operational_capture_authorization(
        tmp_path, raw, datetime(2026, 8, 11, 15, 0, 45, tzinfo=UTC)
    )
    assert allowed, reason
    parsed = parse_run(raw)
    assert [leg.contract.id for leg in parsed.candidates[0].structure.legs] == ["c100"]
    assert [contract.id for contract in parsed.candidates[0].contract_chain] == ["c100", "c105"]
    assert evaluate(parsed)["evaluations"][0]["id"] == "fixidx-001"
    store = AuditStore(tmp_path / "artifacts" / "options_scout.sqlite3")
    store.initialize()
    stored = store.capture_envelope("two-contract-live-capture")
    store.close()
    assert stored is not None
    stored_projection = stored["payload"]["normalized_projection"]
    assert [contract["id"] for contract in stored_projection["candidates"][0]["contracts"]] == [
        "c100",
        "c105",
    ]
    assert "account_id" not in json.dumps(stored, sort_keys=True)


@pytest.mark.parametrize("private_key", ["account_id", "user_id", "order_id", "position_id"])
def test_live_capture_rejects_private_authenticated_payload_extras(private_key: str) -> None:
    raw = _run()
    payload = {"normalized_projection": _capture_projection(raw), private_key: "secret"}
    envelope = {
        "schema_version": "1",
        "capture_id": "private-extra",
        "tool": "reviewed.market_quote",
        "schema_identity": "reviewed.market.quote/v1",
        "parameter_schema_sha256": "1" * 64,
        "response_schema_sha256": "2" * 64,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "source_label": "LIVE",
        "as_of": "2026-08-11T15:00:00+00:00",
        "retrieved_at": "2026-08-11T15:00:30+00:00",
        "field_provenance": {},
        "redacted_arguments": {},
        "payload": payload,
        "payload_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "normalized_input_hash": "0" * 64,
    }
    from options_scout.normalizer import normalize_envelope

    with pytest.raises(ValueError, match="unredacted sensitive|only normalized_projection"):
        normalize_envelope(envelope, {"reviewed.market_quote"})


def test_policy_liquidity_thresholds_control_evaluation() -> None:
    raw = _run()
    rules = {
        "underlying_minimum_usd": "2000000",
        "single_leg_min_volume": 1,
        "single_leg_min_open_interest": 25,
        "complex_or_early_exit_min_volume": 10,
        "complex_or_early_exit_min_open_interest": 100,
        "premium_bands": [
            {"max_premium": "1", "max_relative_spread": "0.25", "max_absolute_spread": "0.20"},
            {"max_premium": "5", "max_relative_spread": "0.20", "max_absolute_spread": "0.50"},
            {"max_premium": None, "max_relative_spread": "0.15", "max_absolute_spread": "1"},
        ],
    }
    result = evaluate(parse_run(raw), parse_liquidity_rules(rules))
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    assert gates["liquidity"]["status"] == "FAIL"


def test_negative_portfolio_risk_and_limits_are_rejected() -> None:
    raw = _run()
    raw["candidates"][0]["portfolio_assessment"]["aggregate_risk"] = "-1"
    with pytest.raises(SchemaError, match="cannot be negative"):
        parse_run(raw)


def test_standalone_portfolio_check_rejects_negative_risk(tmp_path: Path) -> None:
    input_path = tmp_path / "portfolio.json"
    input_path.write_text(
        json.dumps(
            {
                "max_risk_per_trade_usd": "1000",
                "remaining_aggregate_risk_usd": "100",
                "remaining_cluster_risk_usd": "100",
                "remaining_event_risk_usd": "100",
                "remaining_sector_risk_usd": "100",
                "remaining_factor_risk_usd": "100",
                "trade_risk_usd": "-1",
                "proposed_symbol": "PROPOSED",
                "proposed_factor_tags": ["broad"],
                "positions": [],
                "correlations": [],
            }
        )
    )
    assert portfolio_check(str(input_path), True) == 2
    raw = _run()
    raw["candidates"][0]["portfolio_assessment"]["limits"]["remaining_event"] = "-1"
    with pytest.raises(SchemaError, match="cannot be negative"):
        parse_run(raw)


def test_preflight_records_modest_quote_change_without_automatic_invalidation() -> None:
    original = {
        "symbol": "IDX",
        "candidate": {
            "underlying": "100",
            "structure": {
                "legs": [
                    {
                        "contract": {
                            "id": "c",
                            "quote": {
                                "bid": "1",
                                "ask": "1.1",
                                "iv": "0.2",
                                "delta": "0.4",
                                "gamma": "0.1",
                                "theta": "-0.1",
                                "vega": "0.1",
                                "as_of": "old",
                            },
                        }
                    }
                ]
            },
        },
        "analysis": {"payoff": {"operational_max_loss_risk": "85"}},
        "gates": [],
    }
    refreshed = deepcopy(original)
    refreshed["candidate"]["structure"]["legs"][0]["contract"]["quote"]["bid"] = "1.02"
    refreshed["candidate"]["structure"]["legs"][0]["contract"]["quote"]["iv"] = "0.21"
    result = compare_evaluations(original, refreshed)
    assert result["changes"] and not result["invalid"]


def test_preflight_real_evaluation_keeps_na_gates_and_modest_quote_changes_valid() -> None:
    original_raw = _run()
    refreshed_raw = deepcopy(original_raw)
    for contract in refreshed_raw["candidates"][0]["contracts"]:
        quote = contract["quote"]
        quote["bid"] = format(float(quote["bid"]) + 0.01, ".2f")
        quote["ask"] = format(float(quote["ask"]) + 0.01, ".2f")
        quote["iv"] = format(float(quote["iv"]) + 0.01, ".2f")
    original = evaluate(parse_run(original_raw))["evaluations"][0]
    refreshed = evaluate(parse_run(refreshed_raw))["evaluations"][0]
    assert any(gate["status"] == "NOT_APPLICABLE" for gate in refreshed["gates"])
    result = compare_evaluations(original, refreshed)
    assert result["changes"] and not result["failed_refreshed_gates"] and not result["invalid"]


@pytest.mark.parametrize(
    ("term", "mutate"),
    [
        (
            "candidate.structure.name",
            lambda item: item["candidate"]["structure"].update({"name": "cash_call_debit"}),
        ),
        ("structure", lambda item: item.update({"structure": "other_topology"})),
        (
            "analysis.structure.kind",
            lambda item: item["analysis"]["structure"].update({"kind": "other"}),
        ),
        (
            "candidate.structure.quantity",
            lambda item: item["candidate"]["structure"].update({"quantity": 2}),
        ),
        (
            "legs.count",
            lambda item: item["candidate"]["structure"]["legs"].append(
                deepcopy(item["candidate"]["structure"]["legs"][0])
            ),
        ),
        (
            "legs.contract_id_set",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"id": "replacement"}
            ),
        ),
        (
            "legs[0].side",
            lambda item: item["candidate"]["structure"]["legs"][0].update({"side": "sell"}),
        ),
        (
            "legs[0].ratio",
            lambda item: item["candidate"]["structure"]["legs"][0].update({"ratio": 2}),
        ),
        (
            "legs[0].contract.symbol",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"symbol": "OTHER"}
            ),
        ),
        (
            "legs[0].contract.expiration",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"expiration": "2026-08-22"}
            ),
        ),
        (
            "legs[0].contract.strike",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"strike": "99"}
            ),
        ),
        (
            "legs[0].contract.option_type",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"option_type": "put"}
            ),
        ),
        (
            "legs[0].contract.multiplier",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"multiplier": 10}
            ),
        ),
        (
            "legs[0].contract.exercise_style",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"exercise_style": "American"}
            ),
        ),
        (
            "legs[0].contract.settlement_style",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"settlement_style": "physical"}
            ),
        ),
        (
            "legs[0].contract.tradable",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"tradable": False}
            ),
        ),
        (
            "legs[0].contract.adjusted",
            lambda item: item["candidate"]["structure"]["legs"][0]["contract"].update(
                {"adjusted": True}
            ),
        ),
        (
            "candidate.fill_plan.order_type",
            lambda item: item["candidate"]["fill_plan"].update({"order_type": "market"}),
        ),
        (
            "candidate.structure_plan.structure_type",
            lambda item: item["candidate"]["structure_plan"].update({"structure_type": "other"}),
        ),
        (
            "candidate.structure_plan.one_complex_order",
            lambda item: item["candidate"]["structure_plan"].update({"one_complex_order": False}),
        ),
    ],
)
def test_preflight_invalidates_every_immutable_trade_identity_term(
    term: str, mutate: object
) -> None:
    original = evaluate(parse_run(_run()))["evaluations"][0]
    refreshed = deepcopy(original)
    mutate(refreshed)  # type: ignore[operator]
    # Preflight must invalidate the changed identity even if both persisted
    # evaluations claim ACTIONABLE and rerun gates alone would not catch it.
    assert original["decision"] == refreshed["decision"] == "ACTIONABLE"
    result = compare_evaluations(original, refreshed)
    assert result["invalid"]
    assert any(
        change["family"] == "trade_identity" and change["field"] == term
        for change in result["changes"]
    )


def test_preflight_expiration_roll_with_same_candidate_and_contract_ids_invalidates() -> None:
    original_raw = _run()
    refreshed_raw = deepcopy(original_raw)
    for contract in refreshed_raw["candidates"][0]["contracts"]:
        contract["expiration"] = "2026-08-22"
    original = evaluate(parse_run(original_raw))["evaluations"][0]
    refreshed = evaluate(parse_run(refreshed_raw))["evaluations"][0]
    assert original["id"] == refreshed["id"]
    assert [leg["contract"]["id"] for leg in original["candidate"]["structure"]["legs"]] == [
        leg["contract"]["id"] for leg in refreshed["candidate"]["structure"]["legs"]
    ]
    assert original["decision"] == refreshed["decision"] == "ACTIONABLE"
    result = compare_evaluations(original, refreshed)
    assert result["invalid"]
    assert any(
        change["field"] == "legs[0].contract.expiration" for change in result["critical_changes"]
    )


def test_preflight_compares_immutable_candidate_ids_not_duplicate_symbols() -> None:
    original = {
        "evaluations": [
            {
                "id": "one",
                "symbol": "IDX",
                "candidate": {
                    "underlying": "100",
                    "structure": {"legs": []},
                    "mechanics": {"settlement_style": "cash"},
                    "portfolio_assessment": {},
                    "thesis_record": {},
                    "claim_records": [],
                    "source_records": [],
                },
                "analysis": {"payoff": {"operational_max_loss_risk": "80"}},
                "gates": [],
            },
            {
                "id": "two",
                "symbol": "IDX",
                "candidate": {
                    "underlying": "100",
                    "structure": {"legs": []},
                    "mechanics": {"settlement_style": "cash"},
                    "portfolio_assessment": {},
                    "thesis_record": {},
                    "claim_records": [],
                    "source_records": [],
                },
                "analysis": {"payoff": {"operational_max_loss_risk": "80"}},
                "gates": [],
            },
        ]
    }
    refreshed = deepcopy(original)
    refreshed["evaluations"][0]["candidate"]["mechanics"]["settlement_style"] = "physical"
    result = compare_payloads(original, refreshed)
    assert result["candidate_ids_match"] and result["invalid"]
    assert {comparison["id"] for comparison in result["comparisons"]} == {"one", "two"}


def test_actionable_report_excludes_not_applicable_gates_from_blockers() -> None:
    evaluation = {
        "id": "candidate",
        "symbol": "IDX",
        "decision": "ACTIONABLE",
        "candidate": {"structure": {"legs": []}, "structure_plan": {}, "thesis_record": {}},
        "analysis": {"payoff": {}},
        "gates": [
            {
                "name": "iv_history_honesty",
                "status": "NOT_APPLICABLE",
                "passed": False,
                "reason": "no IV history",
            }
        ],
    }
    report = markdown_report(
        "trace", {"decision": "ACTIONABLE", "evaluations": [evaluation], "ranked": [evaluation]}
    )
    assert "Rejections / blockers" not in report
