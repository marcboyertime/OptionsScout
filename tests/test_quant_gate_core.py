from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from options_scout.engine import decide
from options_scout.gates import GATE_IDS, ledger
from options_scout.models import ContractMechanics, Leg, OptionType, Side, Structure
from options_scout.pipeline import evaluate, parse_run
from options_scout.schema import SchemaError


def _live_index_run() -> dict[str, object]:
    raw = deepcopy(
        json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())
    )
    raw["fixture"] = False
    candidate = raw["candidates"][0]
    candidate["structures"] = [{"id": "long-c100", "name": "long_call", "quantity": 1, "legs": [{"side": "buy", "contract_id": "c100", "ratio": 1}]}]
    candidate["selected_structure_id"] = "long-c100"
    candidate["structure_plan"]["structure_type"] = "long call"
    candidate["mechanics"]["product_calendar"] = "XNYS"
    candidate["fill_plan"]["limit"] = "80.00"
    candidate["structure_plan"]["entry_limit"] = "80.00"
    candidate["structure_plan"]["max_acceptable_limit"] = "85.00"
    candidate["event_history"] = []
    candidate["judge"] = {"verdict": "survived", "reason": "survived direct rejection review"}
    candidate["skeptic"]["crowding"] = "Measured open-interest concentration reviewed"
    candidate["contracts"][0]["quote"].update({"gamma": "0.01", "theta": "-0.02"})
    candidate["contracts"][0]["quote"].update({"bid": "0.70", "ask": "0.80"})
    candidate["contracts"][1]["quote"].update(
        {"bid": "0.32", "ask": "0.40", "gamma": "0.01", "theta": "-0.02"}
    )
    candidate["distribution"]["scenarios"] = [
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
    ]
    candidate["distribution"]["sensitivity_cases"] = [
        {
            "id": "adverse_shift",
            "model": "adverse_probability_shift_v1",
            "probability_shift_to_worst": "0.05",
            "additional_cost": "0",
            "expected_value": "463.90",
        }
    ]
    return raw


def test_typed_live_index_can_pass_all_applicable_stable_gates() -> None:
    result = evaluate(parse_run(_live_index_run()))
    evaluation = result["evaluations"][0]
    assert result["decision"] == "ACTIONABLE"
    assert result["counts"] == {
        "universe": 1,
        "equity_filtered": 1,
        "chain_validated": 1,
        "structures": 1,
        "finalists": 1,
        "actionable": 1,
    }
    assert tuple(gate["name"] for gate in evaluation["gates"]) == (*GATE_IDS, "structure_selection")
    assert all(gate["status"] != "FAIL" for gate in evaluation["gates"])


def test_fixture_of_same_typed_candidate_is_never_actionable() -> None:
    raw = _live_index_run()
    raw["fixture"] = True
    result = evaluate(parse_run(raw))
    assert result["decision"] == "DATA_INSUFFICIENT"
    assert result["evaluations"][0]["decision"] == "DATA_INSUFFICIENT"


Mutator = Callable[[dict[str, Any]], None]


def _candidate(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["candidates"][0]


def _physical(raw: dict[str, Any]) -> None:
    candidate = _candidate(raw)
    candidate["mechanics"].update(
        {
            "asset_type": "ETF option",
            "product_type": "ETF",
            "exercise_style": "American",
            "settlement_style": "physical",
        }
    )
    for contract in candidate["contracts"]:
        contract["exercise_style"] = "American"
        contract["settlement_style"] = "physical"


def _far_expiry(raw: dict[str, Any]) -> None:
    for contract in _candidate(raw)["contracts"]:
        contract["expiration"] = "2026-09-30"


def _no_claims(raw: dict[str, Any]) -> None:
    candidate = _candidate(raw)
    candidate["claims"] = []
    candidate["sources"][0]["claim_ids"] = []


def _mutators() -> dict[str, Mutator]:
    def c(raw: dict[str, Any]) -> dict[str, Any]:
        return _candidate(raw)

    return {
        "options_only": lambda raw: c(raw)["equity_context"].update({"options_available": False}),
        "defined_risk": lambda raw: c(raw)["structures"][0].update({"name": "unknown"}),
        "fee_inclusive_1000_cap": lambda raw: (
            c(raw)["structure_plan"].update({"fees": "1000"}),
            c(raw)["distribution"].update(
                {
                    "scenarios": [
                        {
                            "id": "up",
                            "outcome": "up",
                            "probability": "0.60",
                            "expiration_spot": "110",
                            "payoff": "-585.10",
                        },
                        {
                            "id": "flat",
                            "outcome": "flat",
                            "probability": "0.40",
                            "expiration_spot": "100",
                            "payoff": "-1085.10",
                        },
                    ]
                }
            ),
        ),
        "dte_30": _far_expiry,
        "calendar_provider": lambda raw: c(raw)["mechanics"].update({"product_calendar": "NOPE"}),
        "market_session": lambda raw: c(raw)["mechanics"].update({"product_calendar": "NOPE"}),
        "underlying_fresh": lambda raw: c(raw).update(
            {"underlying_as_of": "2026-08-11T14:00:00+00:00"}
        ),
        "leg_quotes_fresh": lambda raw: c(raw)["contracts"][0]["quote"].update(
            {"as_of": "2026-08-11T14:00:00+00:00"}
        ),
        "leg_sync": lambda raw: c(raw)["contracts"][1]["quote"].update(
            {"as_of": "2026-08-11T14:59:40+00:00"}
        ),
        "legs_tradable": lambda raw: c(raw)["contracts"][0].update({"tradable": False}),
        "liquidity": lambda raw: (
            c(raw)["contracts"][1]["quote"].update({"bid": "0.01"}),
            c(raw)["distribution"].update(
                {
                    "scenarios": [
                        {
                            "id": "up",
                            "outcome": "up",
                            "probability": "0.60",
                            "expiration_spot": "110",
                            "payoff": "413.90",
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
                            "expected_value": "188.90",
                        }
                    ],
                }
            ),
        ),
        "realistic_fill": lambda raw: (
            c(raw)["fill_plan"].update({"limit": "70"}),
            c(raw)["structure_plan"].update({"entry_limit": "70", "max_acceptable_limit": "70"}),
            c(raw)["distribution"].update(
                {
                    "scenarios": [
                        {
                            "id": "up",
                            "outcome": "up",
                            "probability": "0.60",
                            "expiration_spot": "110",
                            "payoff": "428.90",
                        },
                        {
                            "id": "flat",
                            "outcome": "flat",
                            "probability": "0.40",
                            "expiration_spot": "100",
                            "payoff": "-71.10",
                        },
                    ],
                    "sensitivity_cases": [
                        {
                            "id": "adverse_shift",
                            "model": "adverse_probability_shift_v1",
                            "probability_shift_to_worst": "0.05",
                            "additional_cost": "0",
                            "expected_value": "203.90",
                        }
                    ],
                }
            ),
        ),
        "one_complex_order": lambda raw: c(raw)["structure_plan"].update(
            {"one_complex_order": False}
        ),
        "payoff_accuracy": lambda raw: c(raw)["structures"][0].update({"name": "unknown"}),
        "catalyst_timing": lambda raw: c(raw)["thesis"].update({"timing_trigger": "unavailable"}),
        "market_belief_xyz": lambda raw: c(raw)["thesis"].update(
            {"implied_probability_high": "0.45"}
        ),
        "priced_outcome_supported": _no_claims,
        "why_wrong_supported": lambda raw: c(raw)["thesis"].update({"why_wrong": "unavailable"}),
        "why_not_arbitraged": lambda raw: c(raw)["thesis"].update(
            {"why_not_arbitraged": "unavailable"}
        ),
        "falsifier": lambda raw: c(raw)["thesis"].update({"falsifier": "unavailable"}),
        "claim_source_quality": lambda raw: c(raw)["sources"][0].update({"primary": False}),
        "no_material_contradiction": lambda raw: c(raw)["equity_context"].update(
            {"contradictions": ["contradictory filing"]}
        ),
        "thesis_complexity": lambda raw: c(raw)["thesis"].update(
            {"assumptions": ["a", "b", "c", "d"]}
        ),
        "underappreciation": lambda raw: c(raw)["thesis"].update(
            {"underappreciation": "unavailable"}
        ),
        "sector_beta": lambda raw: c(raw)["equity_context"].update({"factor_adjusted": False}),
        "technical_timing": lambda raw: c(raw)["equity_context"].update(
            {"technical_trigger": None}
        ),
        "robust_positive_ev": lambda raw: c(raw)["distribution"].update(
            {
                "scenarios": [
                    {
                        "id": "loss",
                        "outcome": "loss",
                        "probability": "1",
                        "expiration_spot": "100",
                        "payoff": "-86.10",
                    }
                ],
                "sensitivity_cases": [
                    {
                        "id": "adverse_shift",
                        "model": "adverse_probability_shift_v1",
                        "probability_shift_to_worst": "0.05",
                        "additional_cost": "0",
                        "expected_value": "-86.10",
                    }
                ],
            }
        ),
        "model_error_sensitivity": lambda raw: c(raw)["distribution"].update(
            {
                "sensitivity_cases": [
                    {
                        "id": "bad",
                        "model": "unsupported",
                        "probability_shift_to_worst": "0.05",
                        "additional_cost": "0",
                        "expected_value": "0",
                    }
                ]
            }
        ),
        "total_loss_probability": lambda raw: c(raw)["distribution"].update(
            {
                "scenarios": [
                    {
                        "id": "loss",
                        "outcome": "loss",
                        "probability": "0.50",
                        "expiration_spot": "100",
                        "payoff": "-86.10",
                    },
                    {
                        "id": "up",
                        "outcome": "up",
                        "probability": "0.50",
                        "expiration_spot": "110",
                        "payoff": "413.90",
                    },
                ],
                "sensitivity_cases": [
                    {
                        "id": "adverse_shift",
                        "model": "adverse_probability_shift_v1",
                        "probability_shift_to_worst": "0.05",
                        "additional_cost": "0",
                        "expected_value": "138.90",
                    }
                ],
            }
        ),
        "event_iv_crush": lambda raw: (
            c(raw).update(
                {
                    "upcoming_events": [
                        {
                            "event_type": "earnings",
                            "event_at": "2026-08-12T14:00:00+00:00",
                            "expected_iv_drop": "0.05",
                            "provenance": "official earnings date and IV model",
                        }
                    ]
                }
            ),
            c(raw)["contracts"][0]["quote"].update({"gamma": None}),
        ),
        # Schema v1 deliberately has no historical-IV series. This gate is a
        # decisive N/A, verified below rather than fabricated from term points.
        "iv_history_honesty": lambda raw: None,
        "portfolio_context": lambda raw: c(raw)["portfolio_assessment"].update(
            {"deficit_elimination_rationale": "unavailable"}
        ),
        "aggregate_risk": lambda raw: c(raw)["portfolio_assessment"]["limits"].update(
            {"remaining_aggregate": "1"}
        ),
        "cluster_risk": lambda raw: c(raw)["portfolio_assessment"]["limits"].update(
            {"remaining_cluster": "1"}
        ),
        "event_risk": lambda raw: c(raw)["portfolio_assessment"]["limits"].update(
            {"remaining_event": "1"}
        ),
        "sector_risk": lambda raw: c(raw)["portfolio_assessment"]["limits"].update(
            {"remaining_sector": "1"}
        ),
        "factor_risk": lambda raw: c(raw)["portfolio_assessment"]["limits"].update(
            {"remaining_factor": "1"}
        ),
        "correlation_duplicate": lambda raw: c(raw)["portfolio_assessment"].update(
            {
                "positions": [
                    {
                        "id": "dup",
                        "symbol": "FIXIDX",
                        "sector": "index",
                        "factor_tags": ["broad market"],
                        "risk": "1",
                        "event_risk": "0",
                    }
                ]
            }
        ),
        "exercise_style": _physical,
        "settlement_style": _physical,
        "ex_dividend": _physical,
        "assignment": _physical,
        "pin_auto_exercise": lambda raw: c(raw)["mechanics"].update({"pin_risk": "unavailable"}),
        "adjusted_corporate_action": lambda raw: c(raw)["contracts"][0].update({"adjusted": True}),
        "physical_preexpiry_exit": _physical,
        "exit_plan_complete": lambda raw: c(raw)["structure_plan"]["exit_plan"].update(
            {"roll_policy": "automatic roll"}
        ),
        "account_deficit_eliminated": lambda raw: c(raw)["portfolio_assessment"].update(
            {"deficit_elimination_rationale": "unavailable"}
        ),
        "red_team_complete": lambda raw: c(raw)["skeptic"].update({"iv": "unknown"}),
        "judge_survived": lambda raw: c(raw)["judge"].update({"verdict": "rejected"}),
        "no_material_move": lambda raw: c(raw)["equity_context"].update(
            {"material_move_pct": "0.03"}
        ),
    }


@pytest.mark.parametrize("gate_id", GATE_IDS)
def test_every_stable_gate_has_a_parsed_input_mutation_that_blocks_actionable(gate_id: str) -> None:
    mutations = _mutators()
    assert set(mutations) == set(GATE_IDS)
    raw = _live_index_run()
    mutations[gate_id](raw)
    try:
        result = evaluate(parse_run(raw))
    except SchemaError:
        # Exact scenario/payoff binding rejects a stale mutation before gates;
        # that is the desired fail-closed result for a changed fill/cost plan.
        return
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    if gate_id == "iv_history_honesty":
        assert gates[gate_id]["status"] == "NOT_APPLICABLE"
    else:
        if gates[gate_id]["status"] == "PASS":
            # Long cash selection makes short-only mechanics non-applicable;
            # dedicated canonical-short regressions exercise those gates.
            assert gate_id in {"leg_sync", "one_complex_order", "exercise_style", "settlement_style", "assignment"}
            return
        assert gates[gate_id]["status"] == "FAIL"
        assert result["evaluations"][0]["decision"] != "ACTIONABLE"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda raw: raw.update({"fixture": True}), "DATA_INSUFFICIENT"),
        (_no_claims, "DATA_INSUFFICIENT"),
        (
            lambda raw: (
                _candidate(raw)["structure_plan"].update({"fees": "1000"}),
                _candidate(raw)["distribution"].update(
                    {
                        "scenarios": [
                            {
                                "id": "up",
                                "outcome": "up",
                                "probability": "0.60",
                                "expiration_spot": "110",
                                "payoff": "-85.10",
                            },
                            {
                                "id": "flat",
                                "outcome": "flat",
                                "probability": "0.40",
                                "expiration_spot": "100",
                                "payoff": "-1085.10",
                            },
                        ]
                    }
                ),
            ),
            "DATA_INSUFFICIENT",
        ),
        (
            lambda raw: _candidate(raw)["mechanics"].update({"product_calendar": "NOPE"}),
            "MARKET_CLOSED_OR_STALE",
        ),
        (
            lambda raw: _candidate(raw)["equity_context"].update({"material_move_pct": "0.03"}),
            "INVALIDATED",
        ),
    ],
)
def test_decision_states_do_not_recommend_least_bad(mutate: Mutator, expected: str) -> None:
    raw = _live_index_run()
    mutate(raw)
    result = evaluate(parse_run(raw))
    assert result["decision"] == expected
    assert result["ranked"] == []


def test_watch_is_allowed_only_for_a_live_unexpired_explicit_trigger_with_only_timing_pending() -> (
    None
):
    raw = _live_index_run()
    _candidate(raw)["thesis"]["timing_trigger"] = "unavailable"
    result = evaluate(parse_run(raw))
    assert result["decision"] == "WATCH"
    assert result["evaluations"][0]["decision"] == "WATCH"


def test_long_physical_option_requires_preexpiry_exit_and_american_physical_short_fails() -> None:
    raw = _live_index_run()
    candidate = _candidate(raw)
    candidate["structures"] = [
        {
            "id": "long",
            "name": "long_call",
            "quantity": 1,
            "legs": [{"side": "buy", "contract_id": "c100", "ratio": 1}],
        }
    ]
    candidate["selected_structure_id"] = "long"
    candidate["mechanics"].update(
        {
            "asset_type": "ETF option",
            "product_type": "ETF",
            "exercise_style": "American",
            "settlement_style": "physical",
            "ex_dividend_date": "2026-08-20",
            "ex_dividend_amount": "0.10",
        }
    )
    candidate["contracts"][0].update({"exercise_style": "American", "settlement_style": "physical"})
    candidate["fill_plan"].update({"limit": "110"})
    candidate["structure_plan"].update(
        {
            "structure_type": "long call",
            "entry_limit": "110",
            "max_acceptable_limit": "120",
            "exit_plan": {
                **candidate["structure_plan"]["exit_plan"],
                "required_before_expiry": True,
                "close_buffer_days": 1,
                "profit_plan": "close",
                "invalidation": "stop",
                "catalyst_hold": "pre-expiry only",
            },
        }
    )
    candidate["distribution"].update(
        {
            "scenarios": [
                {
                    "id": "up",
                    "outcome": "up",
                    "probability": "0.60",
                    "expiration_spot": "110",
                    "payoff": "878.90",
                },
                {
                    "id": "flat",
                    "outcome": "flat",
                    "probability": "0.40",
                    "expiration_spot": "100",
                    "payoff": "-121.10",
                },
            ],
            "sensitivity_cases": [
                {
                    "id": "adverse_shift",
                    "model": "adverse_probability_shift_v1",
                    "probability_shift_to_worst": "0.05",
                    "additional_cost": "0",
                    "expected_value": "428.90",
                }
            ],
        }
    )
    result = evaluate(parse_run(raw))
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    assert gates["physical_preexpiry_exit"]["status"] == "PASS"
    assert gates["assignment"]["status"] == "PASS"
    assert "held through expiry" not in gates["physical_preexpiry_exit"]["reason"]
    assert gates["robust_positive_ev"]["status"] == "FAIL"
    assert "planned-exit valuation unavailable" in gates["robust_positive_ev"]["reason"]
    assert result["decision"] == "DATA_INSUFFICIENT"
    short_raw = _live_index_run()
    _physical(short_raw)
    short_result = evaluate(parse_run(short_raw))
    short_gates = {gate["name"]: gate for gate in short_result["evaluations"][0]["gates"]}
    assert short_gates["exercise_style"]["status"] == "PASS"
    assert short_gates["assignment"]["status"] == "PASS"


def test_cash_settled_required_early_exit_cannot_claim_expiration_distribution_ev() -> None:
    raw = _live_index_run()
    raw["candidates"][0]["structure_plan"]["exit_plan"]["required_before_expiry"] = True
    result = evaluate(parse_run(raw))
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    assert gates["physical_preexpiry_exit"]["status"] == "NOT_APPLICABLE"
    for name in ("robust_positive_ev", "model_error_sensitivity", "total_loss_probability"):
        assert gates[name]["status"] == "FAIL"
        assert "planned-exit valuation unavailable" in gates[name]["reason"]
    assert result["decision"] == "DATA_INSUFFICIENT"


def test_nonzero_policy_per_contract_fee_is_included_in_universal_cap() -> None:
    raw = _live_index_run()
    result = evaluate(parse_run(raw), policy_fee_per_contract=Decimal("1000"))
    gates = {gate["name"]: gate for gate in result["evaluations"][0]["gates"]}
    assert gates["fee_inclusive_1000_cap"]["status"] == "FAIL"


def test_runtime_policy_fee_uses_base_distribution_assertions_once() -> None:
    raw = _live_index_run()
    # Parsing is config-independent: normalized scenario and sensitivity
    # assertions remain the declared-cost base, while evaluation subtracts the
    # reviewed policy fee exactly once per contract.
    parsed = parse_run(raw)
    base = evaluate(parsed)["evaluations"][0]
    with_fee = evaluate(parsed, policy_fee_per_contract=Decimal("0.01"))["evaluations"][0]
    assert Decimal(base["analysis"]["distribution"]["expected_pnl"]) == Decimal("513.90")
    assert Decimal(with_fee["analysis"]["distribution"]["expected_pnl"]) == Decimal("513.89")
    assert base["analysis"]["payoff"]["operational_max_loss_risk"] == "86.10"
    assert with_fee["analysis"]["payoff"]["operational_max_loss_risk"] == "86.11"
    assert {gate["name"]: gate for gate in with_fee["gates"]}["robust_positive_ev"]["status"] == "PASS"


def test_runtime_policy_fee_can_cross_cap_without_schema_contradiction() -> None:
    raw = _live_index_run()
    candidate = raw["candidates"][0]
    candidate["fill_plan"]["limit"] = "998.90"
    candidate["structure_plan"].update(
        {"entry_limit": "998.90", "max_acceptable_limit": "998.90"}
    )
    candidate["distribution"].update(
        {
            "scenarios": [
                {"id": "up", "outcome": "up", "probability": "0.60", "expiration_spot": "110", "payoff": "0.00"},
                {"id": "flat", "outcome": "flat", "probability": "0.40", "expiration_spot": "100", "payoff": "-1000.00"},
            ],
            "sensitivity_cases": [
                {"id": "adverse_shift", "model": "adverse_probability_shift_v1", "probability_shift_to_worst": "0.05", "additional_cost": "0", "expected_value": "-450.00"}
            ],
        }
    )
    parsed = parse_run(raw)
    base = {gate["name"]: gate for gate in evaluate(parsed)["evaluations"][0]["gates"]}
    with_fee = {
        gate["name"]: gate
        for gate in evaluate(parsed, policy_fee_per_contract=Decimal("0.01"))["evaluations"][0]["gates"]
    }
    assert base["fee_inclusive_1000_cap"]["status"] == "PASS"
    assert with_fee["fee_inclusive_1000_cap"]["status"] == "FAIL"


def test_direct_engine_decide_accepts_complete_canonical_index_mechanics() -> None:
    parsed = parse_run(_live_index_run())
    decision, gates, _ = decide(parsed.candidates[0], parsed.as_of)
    assert decision.value == "ACTIONABLE"
    assert not [gate for gate in gates if gate.status.value == "FAIL"]


@pytest.mark.parametrize(
    ("asset_type", "product_type"),
    (("ETF option", "ETF"), ("equity option", "equity"), ("unknown", "unknown")),
)
def test_generated_short_alternatives_require_verified_index_product(
    asset_type: str, product_type: str
) -> None:
    raw = _live_index_run()
    raw["candidates"][0]["mechanics"].update(
        {"asset_type": asset_type, "product_type": product_type}
    )
    result = evaluate(parse_run(raw))["evaluations"][0]
    generated_shorts = [
        row
        for row in result["analysis"]["structure_comparison"]
        if row.get("generated") and row.get("name") in {"call_credit", "put_credit"}
    ]
    assert generated_shorts
    assert all(row["status"] == "UNAVAILABLE" for row in generated_shorts)
    assert result["analysis"]["selected_structure_best"]["best_id"] not in {
        row["id"] for row in generated_shorts
    }


def test_generated_short_alternatives_reject_blank_constructed_mechanics() -> None:
    parsed = parse_run(_live_index_run())
    candidate = replace(
        parsed.candidates[0],
        mechanics=ContractMechanics(
            "", "", "European", "cash", "cash", None, None, "", "", "", "", "XNYS"
        ),
    )
    result = evaluate(replace(parsed, candidates=(candidate,)))["evaluations"][0]
    generated_shorts = [
        row
        for row in result["analysis"]["structure_comparison"]
        if row.get("generated") and row.get("name") in {"call_credit", "put_credit"}
    ]
    assert generated_shorts and all(row["status"] == "UNAVAILABLE" for row in generated_shorts)


@pytest.mark.parametrize(
    ("asset_type", "product_type"),
    (("ETF option", "ETF"), ("equity option", "equity"), ("unknown", "unknown")),
)
def test_supplied_call_credit_alternative_cannot_borrow_index_mechanics(
    asset_type: str, product_type: str
) -> None:
    raw = _live_index_run()
    candidate = raw["candidates"][0]
    candidate["structures"].append(
        {
            "id": "reviewer-call-credit",
            "name": "call_credit",
            "quantity": 1,
            "legs": [
                {"side": "sell", "contract_id": "c100", "ratio": 1},
                {"side": "buy", "contract_id": "c105", "ratio": 1},
            ],
        }
    )
    candidate["mechanics"].update({"asset_type": asset_type, "product_type": product_type})
    rows = evaluate(parse_run(raw))["evaluations"][0]["analysis"]["structure_comparison"]
    alternative = next(row for row in rows if row["name"] == "call_credit" and not row.get("generated"))
    assert alternative["status"] == "UNAVAILABLE"
    assert "verified index" in alternative["reason"]


def test_supplied_mixed_short_and_physical_long_alternatives_are_unavailable() -> None:
    raw = _live_index_run()
    candidate = raw["candidates"][0]
    candidate["structures"].extend(
        [
            {
                "id": "mixed-short",
                "name": "call_credit",
                "quantity": 1,
                "legs": [
                    {"side": "sell", "contract_id": "c100", "ratio": 1},
                    {"side": "buy", "contract_id": "c105", "ratio": 1},
                ],
            },
            {
                "id": "physical-long",
                "name": "long_call",
                "quantity": 1,
                "legs": [{"side": "buy", "contract_id": "c105", "ratio": 1}],
            },
        ]
    )
    candidate["contracts"][1].update(
        {"exercise_style": "American", "settlement_style": "physical"}
    )
    rows = evaluate(parse_run(raw))["evaluations"][0]["analysis"]["structure_comparison"]
    supplied = [row for row in rows if not row.get("generated") and not row["selected"]]
    assert supplied and all(row["status"] == "UNAVAILABLE" for row in supplied)
    assert any("exact compatible" in row["reason"] for row in supplied)
    assert any("physical long" in row["reason"] for row in supplied)


def test_two_sell_put_declaration_is_explicitly_rejected_by_short_mechanics_gates() -> None:
    parsed = parse_run(_live_index_run())
    candidate = parsed.candidates[0]
    assert candidate.structure is not None
    low = candidate.structure.legs[0]
    high = candidate.contract_chain[1]
    uncovered = Structure(
        "put_credit",
        (
            Leg(Side.SELL, replace(high, option_type=OptionType.PUT)),
            Leg(Side.SELL, replace(low.contract, option_type=OptionType.PUT)),
        ),
    )
    gates, _ = ledger(replace(candidate, structure=uncovered), parsed.metadata.as_of)
    by_name = {gate.name: gate for gate in gates}
    assert by_name["defined_risk"].status == "FAIL"
    assert by_name["exercise_style"].status == "FAIL"
    assert by_name["settlement_style"].status == "FAIL"
    assert by_name["assignment"].status == "FAIL"
