from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from .models import (
    Candidate,
    ClaimRecord,
    Contract,
    Leg,
    OptionType,
    Quote,
    Side,
    SourceLabel,
    Structure,
    ThesisRecord,
)


def fixture_candidate(now: datetime | None = None, kind: str = "no_trade") -> Candidate:
    now = now or datetime.now(UTC)
    exp = (now + timedelta(days=14)).date().isoformat()

    def contract(identifier: str, strike: str, bid: str, ask: str, kind_: OptionType) -> Contract:
        return Contract(
            identifier,
            "DEMO",
            exp,
            Decimal(strike),
            kind_,
            Quote(
                Decimal(bid),
                Decimal(ask),
                now,
                SourceLabel.ESTIMATED,
                Decimal("1.00"),
                Decimal("0.45"),
                Decimal("0.50"),
                Decimal("0.01"),
                Decimal("-0.02"),
                Decimal("0.10"),
                100,
                1000,
            ),
            exercise_style="American",
            settlement_style="physical",
        )

    long = contract("demo-call-100", "100", "2.00", "2.20", OptionType.CALL)
    short = contract("demo-call-105", "105", "0.70", "0.85", OptionType.CALL)
    structure = Structure("call_debit_spread", (Leg(Side.BUY, long), Leg(Side.SELL, short)))
    thesis = ThesisRecord(
        Decimal("0.5"),
        Decimal("0.5"),
        "fixture outcome",
        "fixture evidence",
        "fixture",
        "fixture falsifier",
        (),
        "fixture catalyst",
        "fixture timing",
        (),
    )
    claim = ClaimRecord("fixture-claim", "fixture", (), "fixture", Decimal("0"))
    return Candidate(
        "DEMO",
        Decimal("100"),
        now,
        SourceLabel.ESTIMATED,
        structure,
        thesis,
        (claim,),
        False,
        ("fixture",) if kind == "actionable" else (),
        ("fixture",) if kind == "actionable" else (),
        True,
        (),
    )


def write_capture(path: Path, now: datetime | None = None) -> Path:
    now = now or datetime.now(UTC)
    payload = {"note": "demo only", "contracts": []}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    envelope = {
        "schema_version": "1",
        "tool": "fixture.normalized-run",
        "schema_identity": "options-scout.fixture.normalized-run/v1",
        "redacted_arguments": {},
        "fixture_namespace": "options-scout.fixture.v1",
        "capture_id": "demo-capture",
        "as_of": now.isoformat(),
        "retrieved_at": now.isoformat(),
        "field_provenance": {"note": "fixture"},
        "source_label": "ESTIMATED",
        "payload": payload,
        "payload_hash": hashlib.sha256(encoded).hexdigest(),
        "normalized_input_hash": hashlib.sha256(b"fixture-normalized-input").hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2))
    return path
