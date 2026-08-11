from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .models import Contract, OptionType, Quote, SourceLabel, money
from .safety import MAX_PAYLOAD_BYTES, SafetyError, redact, validate_normalized_projection

REQUIRED = {
    "schema_version",
    "tool",
    "capture_id",
    "as_of",
    "retrieved_at",
    "field_provenance",
    "source_label",
    "payload",
    "payload_hash",
    "normalized_input_hash",
    "schema_identity",
    "parameter_schema_sha256",
    "response_schema_sha256",
    "normalized_projection_schema_sha256",
    "redacted_arguments",
}


def parse_time(raw: str) -> datetime:
    point = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if point.tzinfo is None:
        raise ValueError("timestamp must have timezone")
    return point.astimezone(UTC)


def normalize_envelope(raw: dict[str, Any], allowed_tools: set[str]) -> dict[str, Any]:
    """Validate a broker envelope without ever accepting a schema by implication.

    Fixture envelopes are deliberately separate and never use a broker tool identity.
    """
    if set(raw) - (REQUIRED | {"fixture_namespace"}):
        raise ValueError("unknown envelope fields: schema drift")
    if not set(raw) >= REQUIRED or raw["schema_version"] != "1":
        raise ValueError("unsupported or incomplete capture envelope")
    fixture = raw.get("fixture_namespace") == "options-scout.fixture.v1"
    if fixture and raw["tool"] != "fixture.normalized-run":
        raise SafetyError("fixture namespace must use fixture.normalized-run")
    if not fixture and raw["tool"] not in allowed_tools:
        raise SafetyError("unapproved capture tool")
    if not isinstance(raw["schema_identity"], str) or not raw["schema_identity"].strip():
        raise ValueError("capture schema identity is required")
    if not all(
        isinstance(raw[field], str)
        and len(raw[field]) == 64
        and all(character in "0123456789abcdef" for character in raw[field])
        for field in (
            "parameter_schema_sha256",
            "response_schema_sha256",
            "normalized_projection_schema_sha256",
        )
    ):
        raise ValueError("capture must pin canonical parameter and response schema hashes")
    if not isinstance(raw["normalized_input_hash"], str) or len(raw["normalized_input_hash"]) != 64:
        raise ValueError("normalized input hash is required")
    if not isinstance(raw["redacted_arguments"], dict | list):
        raise ValueError("redacted arguments must be an object or list")
    # Redaction is a persistence boundary, not a lossy cleanup step.  An
    # operator must provide an already-redacted envelope whose hash therefore
    # binds exactly the stored content; silently hashing raw secret material
    # and then storing a different redacted payload would break that evidence.
    if (
        redact(raw["payload"]) != raw["payload"]
        or redact(raw["redacted_arguments"]) != raw["redacted_arguments"]
        or redact(raw["field_provenance"]) != raw["field_provenance"]
    ):
        raise SafetyError("capture envelope contains unredacted sensitive material")
    if not fixture and (
        not isinstance(raw["payload"], dict)
        or set(raw["payload"]) != {"normalized_projection"}
        or not isinstance(raw["payload"]["normalized_projection"], dict)
    ):
        raise SafetyError("LIVE capture payload must contain only normalized_projection")
    if not fixture:
        validate_normalized_projection(raw["payload"]["normalized_projection"])
    encoded = json.dumps(
        raw["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError("capture payload exceeds bound")
    if hashlib.sha256(encoded).hexdigest() != raw["payload_hash"]:
        raise ValueError("payload hash mismatch")
    as_of, retrieved = parse_time(raw["as_of"]), parse_time(raw["retrieved_at"])
    if retrieved < as_of:
        raise ValueError("retrieval precedes as_of")
    result = dict(raw)
    result["payload"] = redact(raw["payload"])
    result["redacted_arguments"] = redact(raw["redacted_arguments"])
    return result


def decimal_field(record: dict[str, Any], key: str, optional: bool = False) -> Decimal | None:
    value = record.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an exact decimal string")
    return money(value)


def normalize_contract(record: dict[str, Any], source: SourceLabel) -> Contract:
    required = {"id", "symbol", "expiration", "strike", "type", "bid", "ask", "as_of"}
    if not required <= set(record):
        raise ValueError("contract missing required fields")
    quote = Quote(
        bid=decimal_field(record, "bid", True),
        ask=decimal_field(record, "ask", True),
        mark=decimal_field(record, "mark", True),
        iv=decimal_field(record, "iv", True),
        delta=decimal_field(record, "delta", True),
        gamma=decimal_field(record, "gamma", True),
        theta=decimal_field(record, "theta", True),
        vega=decimal_field(record, "vega", True),
        as_of=parse_time(record["as_of"]),
        source=source,
        volume=record.get("volume"),
        open_interest=record.get("open_interest"),
    )
    return Contract(
        record["id"],
        record["symbol"],
        record["expiration"],
        decimal_field(record, "strike") or Decimal(),
        OptionType(record["type"]),
        quote,
        bool(record.get("tradable", True)),
        int(record.get("multiplier", 100)),
        bool(record.get("adjusted", False)),
        record.get("exercise_style"),
        record.get("settlement_style"),
    )
