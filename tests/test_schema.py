import copy
import json
from pathlib import Path

import pytest

from options_scout.models import Decimal, SourceLabel
from options_scout.pipeline import evaluate
from options_scout.schema import SchemaError, parse_run


def fixture() -> dict[str, object]:
    return json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())


def candidate(raw: dict[str, object]) -> dict[str, object]:
    return raw["candidates"][0]  # type: ignore[index, no-any-return]


def test_complete_fixture_builds_typed_provider_neutral_records() -> None:
    parsed = parse_run(fixture())
    item = parsed.candidates[0]
    assert item.source is SourceLabel.LIVE
    assert item.underlying == Decimal("100.00")
    assert item.mechanics.exercise_style == "European"
    assert item.mechanics.settlement_style == "cash"
    assert item.fixture is True


def test_structure_plan_must_name_the_selected_canonical_topology() -> None:
    raw = fixture()
    candidate(raw)["structure_plan"]["structure_type"] = "iron condor"
    with pytest.raises(SchemaError, match="selected canonical topology"):
        parse_run(raw)


def test_live_quote_source_cannot_be_stale_background_research() -> None:
    raw = fixture()
    raw["fixture"] = False
    candidate(raw)["sources"][0]["published_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(SchemaError, match="fresh linked source"):
        parse_run(raw)


def test_required_preexpiry_exit_uses_typed_post_event_valuation_not_expiration_payoff() -> None:
    raw = fixture()
    item = candidate(raw)
    item["structure_plan"]["exit_plan"]["required_before_expiry"] = True
    item["planned_exit_valuation"] = {
        "method": "intrinsic_close_v1",
        "valuation_at": item["structure_plan"]["exit_plan"]["time_exit_at"],
        "source_id": "source-fixture",
        "scenarios": [
            {"id": "up", "outcome": "post-event up", "probability": "0.60", "underlying_spot": "110"},
            {"id": "down", "outcome": "post-event down", "probability": "0.40", "underlying_spot": "90"},
        ],
        "sensitivity_cases": [
            {
                "id": "adverse", "model": "adverse_probability_shift_v1",
                "probability_shift_to_worst": "0.05", "additional_cost": "0", "expected_value": "188.9",
            }
        ],
    }
    parsed = parse_run(raw)
    evaluation = evaluate(parsed)["evaluations"][0]
    gates = {gate["name"]: gate for gate in evaluation["gates"]}
    assert gates["robust_positive_ev"]["status"] == "FAIL"
    assert "short-leg extrinsic" in gates["robust_positive_ev"]["reason"]
    assert evaluation["analysis"]["distribution"]["valuation_basis"] == "planned_exit_post_event"


def test_all_typed_alternatives_render_economics_and_dominated_selection_rejects() -> None:
    raw = fixture()
    item = candidate(raw)
    item["structures"].append(
        {
            "id": "unsupported-calendar",
            "name": "calendar",
            "quantity": 1,
            "legs": item["structures"][0]["legs"],
        }
    )
    evaluation = evaluate(parse_run(raw))["evaluations"][0]
    comparison = evaluation["analysis"]["structure_comparison"]
    selected = next(row for row in comparison if row["selected"])
    rejected = next(row for row in comparison if row["name"] == "calendar")
    assert selected["status"] == "VALUED"
    assert {"max_fill_payoff", "expected_value", "breakevens", "operational_max_loss_risk"} <= set(selected)
    assert rejected["status"] == "REJECTED"

    dominated = fixture()
    dominated_item = candidate(dominated)
    dominated_item["structures"].append(
        {
            "id": "z-tied-supported-alternative",
            "name": "call_debit",
            "quantity": 1,
            "legs": dominated_item["structures"][0]["legs"],
        }
    )
    dominated_evaluation = evaluate(parse_run(dominated))["evaluations"][0]
    assert not dominated_evaluation["analysis"]["selected_structure_best"]["passed"]


def test_alternative_cannot_inject_a_separate_probability_distribution() -> None:
    raw = fixture()
    item = candidate(raw)
    item["structures"].append(
        {
            "id": "second", "name": "call_debit", "quantity": 1,
            "legs": item["structures"][0]["legs"],
        }
    )
    item["alternative_economics"] = [
        {
            "structure_id": "second", "fees": "0", "commissions": "0",
            "exit_slippage": "0", "entry_slippage": "0",
            "distribution": {"scenarios": [{"id": "forged", "outcome": "forged", "probability": "1", "expiration_spot": "10000", "payoff": "999999"}], "sensitivity_cases": [], "scenario_model": "bounded_terminal_spot_v1", "provenance": "forged"},
        }
    ]
    with pytest.raises(SchemaError, match="unknown or missing alternative_economics"):
        parse_run(raw)


def test_alternative_costs_change_only_its_independent_economics() -> None:
    raw = fixture()
    item = candidate(raw)
    item["structures"].append({"id": "second", "name": "call_debit", "quantity": 1, "legs": item["structures"][0]["legs"]})
    item["alternative_economics"] = [{"structure_id": "second", "fees": "7", "commissions": "3", "exit_slippage": "2", "entry_slippage": "1"}]
    rows = evaluate(parse_run(raw))["evaluations"][0]["analysis"]["structure_comparison"]
    selected = next(row for row in rows if row["selected"])
    alternative = next(row for row in rows if row["id"] == "alternative-1")
    assert alternative["valuation_basis"] == "alternative_reviewed_costs_natural_max_fill"
    assert alternative["costs"]["fees"] == Decimal("7")
    assert selected["costs"]["fees"] != alternative["costs"]["fees"]


def test_unused_safe_chain_contract_can_dominate_and_hard_fail_structure_selection() -> None:
    raw = fixture()
    raw["fixture"] = False
    item = candidate(raw)
    unused = copy.deepcopy(item["contracts"][0])
    unused.update({"id": "unused-c95", "strike": "95"})
    unused["quote"].update({"bid": "0.05", "ask": "0.06", "mark": "0.055"})
    item["contracts"].append(unused)
    evaluation = evaluate(parse_run(raw))["evaluations"][0]
    gate = next(gate for gate in evaluation["gates"] if gate["name"] == "structure_selection")
    assert gate["status"] == "FAIL"
    assert evaluation["decision"] != "ACTIONABLE"


def test_generated_physical_long_and_non_european_cash_short_are_unavailable() -> None:
    raw = fixture()
    item = candidate(raw)
    physical = copy.deepcopy(item["contracts"][0])
    physical.update({"id": "physical-c95", "strike": "95", "exercise_style": "American", "settlement_style": "physical"})
    item["contracts"].append(physical)
    rows = evaluate(parse_run(raw))["evaluations"][0]["analysis"]["structure_comparison"]
    generated = [row for row in rows if row.get("generated")]
    assert any("physical long" in str(row.get("reason", "")) for row in generated)
    assert any("European cash" in str(row.get("reason", "")) for row in generated)


def test_planned_exit_rejects_arbitrary_plus_2m_minus_1_5m_payoff_exploit() -> None:
    raw = fixture()
    item = candidate(raw)
    item["structure_plan"]["exit_plan"]["required_before_expiry"] = True
    item["planned_exit_valuation"] = {
        "method": "intrinsic_close_v1",
        "valuation_at": item["structure_plan"]["exit_plan"]["time_exit_at"],
        "source_id": "source-fixture",
        "scenarios": [
            {"id": "gain", "outcome": "forged", "probability": "0.50", "payoff": "2000000"},
            {"id": "loss", "outcome": "forged", "probability": "0.50", "payoff": "-1500000"},
        ],
        "sensitivity_cases": [],
    }
    with pytest.raises(SchemaError):
        parse_run(raw)


@pytest.mark.parametrize(
    ("location", "field", "variant"),
    [
        ("contract", "settlement_style", "physically settled"),
        ("contract", "settlement_style", "cash settled"),
        ("contract", "settlement_style", "Physical"),
        ("contract", "settlement_style", "CASH"),
        ("contract", "settlement_style", "physical "),
        ("mechanics", "settlement_style", "physically settled"),
        ("mechanics", "settlement_style", "cash settled"),
        ("mechanics", "settlement_style", "Cash"),
        ("mechanics", "exercise_style", "American-style"),
        ("mechanics", "exercise_style", "american"),
        ("contract", "exercise_style", "American-style"),
        ("contract", "exercise_style", "European "),
        ("contract", "exercise_style", "unknown"),
    ],
)
def test_schema_rejects_noncanonical_security_mechanics_before_evaluation(
    location: str, field: str, variant: str
) -> None:
    raw = fixture()
    target = (
        candidate(raw)["contracts"][0] if location == "contract" else candidate(raw)["mechanics"]
    )
    target[field] = variant
    with pytest.raises(SchemaError, match="must be exactly"):
        parse_run(raw)


@pytest.mark.parametrize(
    ("location", "field", "blank"),
    [
        ("contract", "exercise_style", ""),
        ("contract", "settlement_style", " "),
        ("mechanics", "exercise_style", ""),
        ("mechanics", "settlement_style", " "),
    ],
)
def test_schema_rejects_blank_security_mechanics_labels(
    location: str, field: str, blank: str
) -> None:
    raw = fixture()
    target = (
        candidate(raw)["contracts"][0] if location == "contract" else candidate(raw)["mechanics"]
    )
    target[field] = blank
    with pytest.raises(SchemaError, match="non-empty"):
        parse_run(raw)


def test_physical_settlement_alias_on_all_selected_legs_cannot_reach_actionable_evaluation() -> (
    None
):
    raw = fixture()
    candidate(raw)["mechanics"]["settlement_style"] = "physically settled"
    for contract in candidate(raw)["contracts"]:
        contract["settlement_style"] = "physically settled"
    with pytest.raises(SchemaError, match="contract.settlement_style must be exactly"):
        parse_run(raw)


@pytest.mark.parametrize("field", ["underlying_invalidation", "thesis_invalidation", "volatility_invalidation", "time_exit_at", "time_exit_rationale", "rapid_double_response", "loss_50_response", "direction_correct_iv_wrong_response", "sector_only_response", "expiration_management", "assignment_management", "roll_policy"])
def test_exit_plan_requires_each_typed_contingency(field: str) -> None:
    raw = fixture()
    del candidate(raw)["structure_plan"]["exit_plan"][field]
    with pytest.raises(SchemaError):
        parse_run(raw)


@pytest.mark.parametrize("time_exit", ["2026-08-11T14:00:00+00:00", "2026-08-22T00:00:00+00:00"])
def test_exit_time_must_be_after_run_and_no_later_than_expiration(time_exit: str) -> None:
    raw = fixture()
    candidate(raw)["structure_plan"]["exit_plan"]["time_exit_at"] = time_exit
    with pytest.raises(SchemaError, match="time_exit_at"):
        parse_run(raw)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "historical_quote_provenance",
            lambda raw: candidate(raw)["quote_provenance"].update({"source": "HISTORICAL"}),
        ),
        ("negative_bid", lambda raw: candidate(raw)["contracts"][0]["quote"].update({"bid": "-1"})),
        (
            "mark_outside_bid_ask",
            lambda raw: candidate(raw)["contracts"][0]["quote"].update({"mark": "2"}),
        ),
        ("negative_iv", lambda raw: candidate(raw)["contracts"][0]["quote"].update({"iv": "-0.5"})),
        (
            "delta_outside_domain",
            lambda raw: candidate(raw)["contracts"][0]["quote"].update({"delta": "2.0"}),
        ),
        (
            "negative_gamma",
            lambda raw: candidate(raw)["contracts"][0]["quote"].update({"gamma": "-0.01"}),
        ),
        (
            "negative_vega",
            lambda raw: candidate(raw)["contracts"][0]["quote"].update({"vega": "-0.01"}),
        ),
        ("zero_strike", lambda raw: candidate(raw)["contracts"][0].update({"strike": "0"})),
        (
            "surface_zero_iv",
            lambda raw: candidate(raw)["surface_nodes"][0].update({"implied_volatility": "0"}),
        ),
        (
            "surface_delta_outside_domain",
            lambda raw: candidate(raw)["surface_nodes"][0].update({"delta": "2"}),
        ),
        ("zero_atm_iv", lambda raw: candidate(raw)["iv_snapshot"].update({"atm_iv": "0"})),
        (
            "zero_term_iv",
            lambda raw: candidate(raw)["iv_snapshot"].update({"term_structure": ["0"]}),
        ),
        (
            "negative_implied_move",
            lambda raw: candidate(raw)["volatility"].update({"implied_move_pct": "-0.01"}),
        ),
        (
            "negative_realized_volatility",
            lambda raw: candidate(raw)["volatility"].update({"realized_volatility": "-0.01"}),
        ),
    ],
)
def test_schema_rejects_contradictory_provenance_and_impossible_quote_volatility_domains(
    label: str, mutate: object
) -> None:
    raw = fixture()
    mutate(raw)  # type: ignore[operator]
    with pytest.raises(SchemaError):
        parse_run(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: candidate(raw)["sources"][0].update({"unexpected": "x"}),
        lambda raw: candidate(raw).update({"tool": "anything"}),
        lambda raw: candidate(raw)["contracts"][0].update({"strike": 100}),
        lambda raw: candidate(raw)["contracts"][0].update({"strike": "NaN"}),
        lambda raw: candidate(raw)["structures"][0].update({"quantity": True}),
        lambda raw: candidate(raw).update({"underlying_as_of": "2026-08-11T15:00:00"}),
        lambda raw: candidate(raw)["sources"][0].update(
            {"retrieved_at": "2026-08-11T14:00:00+00:00"}
        ),
        lambda raw: candidate(raw)["contracts"].append(
            copy.deepcopy(candidate(raw)["contracts"][0])
        ),
        lambda raw: candidate(raw)["claims"][0].update({"source_ids": ["missing"]}),
        lambda raw: candidate(raw)["structures"][0]["legs"][0].update({"contract_id": "missing"}),
        lambda raw: candidate(raw)["distribution"]["scenarios"][0].update({"probability": "1.2"}),
        lambda raw: candidate(raw)["distribution"]["scenarios"][1].update({"probability": "0.4"}),
        lambda raw: candidate(raw).update({"contracts": [candidate(raw)["contracts"][0]] * 501}),
        lambda raw: candidate(raw)["contracts"][1].update({"multiplier": 10}),
        lambda raw: candidate(raw)["contracts"][1].update({"settlement_style": "physical"}),
    ],
)
def test_schema_rejects_malformed_untrusted_nested_input(mutate: object) -> None:
    raw = fixture()
    mutate(raw)  # type: ignore[operator]
    with pytest.raises(SchemaError):
        parse_run(raw)
