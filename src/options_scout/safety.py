"""Fail-closed capability, redaction, and repository-audit helpers."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .engine import parse_liquidity_rules

FORBIDDEN = (
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
)
MAX_PAYLOAD_BYTES = 512_000
_SENSITIVE = re.compile(
    r"(?:token|secret|password|credential|account|user|order|ssn|balance|position|email|phone|authorization|api[_-]?key|^id$)",
    re.I,
)
_POLICY_FIELDS = {
    "version",
    "max_risk_per_trade_usd",
    "outer_operational_containment_usd",
    "fees_per_contract_usd",
    "physical_settlement_exit_before_expiration",
    "max_dte",
    "quote_max_age_seconds",
    "leg_sync_max_seconds",
    "liquidity",
    "enabled_tools",
    "approved_capture_schemas",
    "denied_tool_patterns",
    "allow_live_without_catalog",
}
_CAPTURE_SCHEMA_FIELDS = {
    "tool",
    "schema_identity",
    "source_label",
    "parameter_schema_sha256",
    "response_schema_sha256",
    "normalized_projection_schema_sha256",
}
_NORMALIZED_PROJECTION_SCHEMA = (
    "metadata.run_id",
    "metadata.as_of",
    "metadata.retrieved_at",
    "source_label",
    "candidates[].id",
    "candidates[].symbol",
    "candidates[].underlying",
    "candidates[].underlying_as_of",
    "candidates[].quote_provenance",
    "candidates[].mechanics",
    "candidates[].contracts[].id",
    "candidates[].contracts[].symbol",
    "candidates[].contracts[].expiration",
    "candidates[].contracts[].strike",
    "candidates[].contracts[].type",
    "candidates[].contracts[].multiplier",
    "candidates[].contracts[].tradable",
    "candidates[].contracts[].adjusted",
    "candidates[].contracts[].exercise_style",
    "candidates[].contracts[].settlement_style",
    "candidates[].contracts[].quote",
)

_PROJECTION_METADATA_FIELDS = {"run_id", "as_of", "retrieved_at"}
_PROJECTION_CANDIDATE_FIELDS = {
    "id", "symbol", "underlying", "underlying_as_of", "quote_provenance", "mechanics", "contracts"
}
_PROJECTION_PROVENANCE_FIELDS = {"source_id", "retrieved_at", "as_of", "source", "methodology"}
_PROJECTION_MECHANICS_FIELDS = {
    "asset_type", "product_type", "exercise_style", "settlement_style", "deliverable",
    "ex_dividend_date", "ex_dividend_amount", "assignment_risk", "pin_risk",
    "auto_exercise", "corporate_action", "product_calendar",
}
_PROJECTION_CONTRACT_FIELDS = {
    "id", "symbol", "expiration", "strike", "type", "multiplier", "tradable", "adjusted",
    "exercise_style", "settlement_style", "quote",
}
_PROJECTION_QUOTE_FIELDS = {
    "bid", "ask", "mark", "iv", "delta", "gamma", "theta", "vega", "as_of", "source",
    "volume", "open_interest",
}


def normalized_projection_schema_sha256() -> str:
    """Fixed complete projection contract for capture-to-typed binding."""
    return hashlib.sha256(json.dumps(_NORMALIZED_PROJECTION_SCHEMA, separators=(",", ":")).encode()).hexdigest()


def validate_normalized_projection(value: object) -> None:
    """Accept only the fixed public market-data projection shape.

    The projection schema hash is meaningful only if ingest also rejects extra
    nested objects.  In particular this prevents an authenticated account-like
    object from inheriting a harmless-looking public candidate ``id`` path.
    Values remain untrusted data; typed schema parsing validates their precise
    market semantics after capture binding.
    """
    if not isinstance(value, dict) or set(value) != {"metadata", "source_label", "candidates"}:
        raise SafetyError("normalized projection has unknown or missing fields")
    metadata, candidates = value["metadata"], value["candidates"]
    if (
        not isinstance(metadata, dict)
        or set(metadata) != _PROJECTION_METADATA_FIELDS
        or not all(isinstance(item, str) and item for item in metadata.values())
        or not isinstance(value["source_label"], str)
        or value["source_label"] != "LIVE"
        or not isinstance(candidates, list)
    ):
        raise SafetyError("normalized projection metadata is invalid")
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != _PROJECTION_CANDIDATE_FIELDS:
            raise SafetyError("normalized projection candidate fields are invalid")
        if not all(isinstance(candidate[key], str) and candidate[key] for key in ("id", "symbol", "underlying", "underlying_as_of")):
            raise SafetyError("normalized projection candidate identity is invalid")
        provenance, mechanics, contracts = (
            candidate["quote_provenance"], candidate["mechanics"], candidate["contracts"]
        )
        if (
            not isinstance(provenance, dict)
            or set(provenance) != _PROJECTION_PROVENANCE_FIELDS
            or not all(isinstance(item, str) and item for item in provenance.values())
            or not isinstance(mechanics, dict)
            or set(mechanics) != _PROJECTION_MECHANICS_FIELDS
            or not all(item is None or isinstance(item, str) for item in mechanics.values())
            or not isinstance(contracts, list)
        ):
            raise SafetyError("normalized projection provenance or mechanics is invalid")
        for contract in contracts:
            if not isinstance(contract, dict) or set(contract) != _PROJECTION_CONTRACT_FIELDS:
                raise SafetyError("normalized projection contract fields are invalid")
            if (
                not all(isinstance(contract[key], str) and contract[key] for key in ("id", "symbol", "expiration", "strike", "type"))
                or type(contract["multiplier"]) is not int
                or type(contract["tradable"]) is not bool
                or type(contract["adjusted"]) is not bool
                or not all(contract[key] is None or isinstance(contract[key], str) for key in ("exercise_style", "settlement_style"))
                or not isinstance(contract["quote"], dict)
                or set(contract["quote"]) != _PROJECTION_QUOTE_FIELDS
            ):
                raise SafetyError("normalized projection contract values are invalid")
            quote = contract["quote"]
            if (
                not isinstance(quote["as_of"], str)
                or not quote["as_of"]
                or quote["source"] != "LIVE"
                or any(quote[key] is not None and not isinstance(quote[key], str) for key in ("bid", "ask", "mark", "iv", "delta", "gamma", "theta", "vega"))
                or any(quote[key] is not None and (type(quote[key]) is not int or quote[key] < 0) for key in ("volume", "open_interest"))
            ):
                raise SafetyError("normalized projection quote values are invalid")


class SafetyError(ValueError):
    pass


def policy(root: Path) -> dict[str, Any]:
    try:
        loaded: object = json.loads((root / "config/policy.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SafetyError(f"invalid policy: {error}") from error
    if not isinstance(loaded, dict) or set(loaded) != _POLICY_FIELDS:
        raise SafetyError("policy fields are unknown or incomplete")
    if (
        not isinstance(loaded["version"], str)
        or not loaded["version"].strip()
        or type(loaded["physical_settlement_exit_before_expiration"]) is not bool
        or loaded["physical_settlement_exit_before_expiration"] is not True
        or type(loaded["max_dte"]) is not int
        or loaded["max_dte"] != 30
        or type(loaded["quote_max_age_seconds"]) is not int
        or loaded["quote_max_age_seconds"] != 90
        or type(loaded["leg_sync_max_seconds"]) is not int
        or loaded["leg_sync_max_seconds"] != 15
        or type(loaded["allow_live_without_catalog"]) is not bool
        or loaded["allow_live_without_catalog"] is not False
        or not isinstance(loaded["enabled_tools"], list)
        or not isinstance(loaded["approved_capture_schemas"], list)
        or not isinstance(loaded["denied_tool_patterns"], list)
        or loaded["denied_tool_patterns"] != list(FORBIDDEN)
    ):
        raise SafetyError("policy has invalid fixed safety controls")
    for field in (
        "max_risk_per_trade_usd",
        "outer_operational_containment_usd",
        "fees_per_contract_usd",
    ):
        value = loaded[field]
        if not isinstance(value, str):
            raise SafetyError(f"policy {field} must be an exact Decimal string")
        try:
            decimal = Decimal(value)
        except InvalidOperation as error:
            raise SafetyError(f"policy {field} is not a Decimal") from error
        if not decimal.is_finite() or decimal < 0:
            raise SafetyError(f"policy {field} must be finite and non-negative")
        if field != "fees_per_contract_usd" and decimal != Decimal("1000"):
            raise SafetyError("universal all-in cap must be exactly $1,000")
    try:
        parse_liquidity_rules(loaded.get("liquidity"))
    except (ValueError, ArithmeticError) as error:
        raise SafetyError(f"invalid liquidity policy: {error}") from error
    tools = loaded["enabled_tools"]
    if (
        any(
            not isinstance(tool, str) or not tool.strip() or tool != tool.strip() or len(tool) > 200
            for tool in tools
        )
        or len(set(tools)) != len(tools)
        or any(any(token in tool.casefold() for token in FORBIDDEN) for tool in tools)
    ):
        raise SafetyError(
            "enabled_tools must be a unique human-reviewed read-only positive allowlist"
        )
    schemas = loaded["approved_capture_schemas"]
    approved: set[tuple[str, str, str]] = set()
    for item in schemas:
        if not isinstance(item, dict) or set(item) != _CAPTURE_SCHEMA_FIELDS:
            raise SafetyError("approved capture schema fields are invalid")
        tool, identity, source = item["tool"], item["schema_identity"], item["source_label"]
        parameter_hash = item["parameter_schema_sha256"]
        response_hash = item["response_schema_sha256"]
        projection_hash = item["normalized_projection_schema_sha256"]
        if (
            not isinstance(tool, str)
            or not isinstance(identity, str)
            or not isinstance(source, str)
            or not tool.strip()
            or not identity.strip()
            or tool != tool.strip()
            or identity != identity.strip()
            or len(tool) > 200
            or len(identity) > 200
            or source != "LIVE"
            or not all(
                isinstance(item_hash, str)
                and re.fullmatch(r"[0-9a-f]{64}", item_hash) is not None
                for item_hash in (parameter_hash, response_hash)
            )
            or projection_hash != normalized_projection_schema_sha256()
            or any(token in tool.casefold() or token in identity.casefold() for token in FORBIDDEN)
        ):
            raise SafetyError("approved capture schema must be a reviewed LIVE read-only record")
        approved.add((tool, identity, source))
    if (
        len(approved) != len(schemas)
        or {item[0] for item in approved} != set(tools)
        or len(approved) != len(tools)
    ):
        raise SafetyError("every enabled tool must have exactly one unique approved LIVE schema")
    return loaded


def assert_read_only_operation(tool_name: str, root: Path) -> None:
    allowed = set(policy(root)["enabled_tools"])
    if tool_name not in allowed or any(token in tool_name.casefold() for token in FORBIDDEN):
        raise SafetyError("broker operation rejected: no approved read-only market-data schema")


def redact(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Redact private identifiers while retaining schema-bounded market IDs.

    Generic ``id`` is sensitive everywhere except normalized candidate and
    option-contract records, whose unique public IDs are part of the fixed
    capture projection and needed to bind legs without ambiguity.
    """
    if isinstance(value, dict):
        return {
            str(key): (
                redact(item, (*path, str(key)))
                if str(key) == "id"
                and len(path) >= 2
                and path[-2] in {"candidates", "contracts"}
                and path[-1].isdigit()
                else "[REDACTED]"
                if _SENSITIVE.search(str(key))
                else redact(item, (*path, str(key)))
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, (*path, str(index))) for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [redact(item, (*path, str(index))) for index, item in enumerate(value)]
    return value


def _call_target(node: ast.expr) -> str:
    """Return an executable name/attribute chain without inspecting string prose."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_target(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def static_safety_scan(root: Path) -> list[str]:
    violations: list[str] = []
    paths = [*(root / "src").rglob("*.py"), *(root / "bin").glob("*.py")]
    for path in sorted(paths):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as error:
            violations.append(f"{path.relative_to(root)}:{error.lineno}: syntax")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = _call_target(node.func)
                literal = (
                    node.args[0].value
                    if target.rsplit(".", 1)[-1]
                    in {"invoke_tool", "call_tool", "run_tool", "broker_capture"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    else ""
                )
                if any(
                    token in target.casefold() or token in literal.casefold() for token in FORBIDDEN
                ):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: forbidden executable broker invocation"
                    )
    schedule = root / "artifacts/options-scout.schedule.sh"
    if schedule.exists():
        text = schedule.read_text()
        if "DISABLED" not in text or "enabled_tools=[]" not in text or "source-health" not in text:
            violations.append(
                "artifacts/options-scout.schedule.sh: schedule lacks fail-closed controls"
            )
        for number, line in enumerate(text.splitlines(), 1):
            executable = line.strip()
            if (
                executable
                and not executable.startswith("#")
                and any(token in executable.casefold() for token in FORBIDDEN)
            ):
                violations.append(
                    f"artifacts/options-scout.schedule.sh:{number}: forbidden executable mutation token"
                )
    for forbidden in (".env", ".env.local", "credentials.json", "token.json"):
        if (root / forbidden).exists():
            violations.append(f"{forbidden}: secret-like file present")
    return violations
