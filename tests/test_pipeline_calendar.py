import json
from datetime import datetime
from pathlib import Path

import pytest

from options_scout.calendar import provider_status, session_status
from options_scout.pipeline import SchemaError, evaluate, parse_run


def test_normalized_fixture_runs_staged_funnel() -> None:
    raw = json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())
    result = evaluate(parse_run(raw))
    assert result["counts"]["universe"] == 1
    assert result["counts"]["chain_validated"] == 1
    assert result["evaluations"][0]["decision"] != "ACTIONABLE"


def test_schema_rejects_unknown_bool_int_and_duplicate_contract() -> None:
    raw = json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())
    raw["unexpected"] = True
    with pytest.raises(SchemaError):
        parse_run(raw)
    raw.pop("unexpected")
    raw["candidates"][0]["structures"][0]["quantity"] = True
    with pytest.raises(SchemaError):
        parse_run(raw)
    raw["candidates"][0]["structures"][0]["quantity"] = 1
    raw["candidates"][0]["contracts"].append(raw["candidates"][0]["contracts"][0])
    with pytest.raises(SchemaError):
        parse_run(raw)


def test_verified_calendar_regular_and_closed_session() -> None:
    assert provider_status()["status"] == "AVAILABLE"
    assert session_status(datetime.fromisoformat("2026-08-11T15:00:00+00:00"))["regular"] is True
    assert session_status(datetime.fromisoformat("2026-08-09T15:00:00+00:00"))["regular"] is False
