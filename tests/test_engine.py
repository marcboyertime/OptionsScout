from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from options_scout.engine import (
    analyze_structure,
    decide,
    distribution_ev,
    implied_move,
    iv_crush_matrix,
    payoff_at_expiration,
    preflight,
    validate_quotes,
)
from options_scout.models import (
    Candidate,
    ClaimRecord,
    Contract,
    ContractMechanics,
    Decision,
    Leg,
    OptionType,
    Quote,
    Side,
    SourceLabel,
    Structure,
    ThesisRecord,
)
from options_scout.structures import generate_supported_structures

NOW = datetime(2026, 8, 11, 15, tzinfo=UTC)


def contract(
    identifier: str,
    strike: str,
    kind: OptionType,
    bid: str = "1.00",
    ask: str = "1.10",
    **attrs: object,
) -> Contract:
    quote = Quote(
        Decimal(bid),
        Decimal(ask),
        attrs.pop("as_of", NOW),
        SourceLabel.LIVE,
        Decimal("1.05"),
        Decimal("0.40"),
        Decimal("0.5"),
        Decimal("0.01"),
        Decimal("-0.02"),
        Decimal("0.10"),
        50,
        1000,
    )
    return Contract(
        identifier,
        "IDX",
        (NOW + timedelta(days=14)).date().isoformat(),
        Decimal(strike),
        kind,
        quote,
        **attrs,
    )


def candidate(structure: Structure, **attrs: object) -> Candidate:
    values: dict[str, object] = dict(
        symbol="IDX",
        underlying=Decimal("100"),
        underlying_as_of=NOW,
        source=SourceLabel.LIVE,
        structure=structure,
        thesis_record=ThesisRecord(
            Decimal("0.5"),
            Decimal("0.5"),
            "outcome",
            "because",
            "arbitrage",
            "falsifier",
            (),
            "catalyst",
            "timing",
            (),
        ),
        claim_records=(ClaimRecord("c", "evidence", (), "test", Decimal("1")),),
        mechanics_evidence=("test",),
        portfolio_evidence=("test",),
        deficit_evidence=("test",),
    )
    values.update(attrs)
    return Candidate(**values)


def test_payoffs_long_call_put_and_spreads() -> None:
    call = Structure("long_call", (Leg(Side.BUY, contract("c", "100", OptionType.CALL)),))
    put = Structure("long_put", (Leg(Side.BUY, contract("p", "100", OptionType.PUT)),))
    debit = Structure(
        "call_debit",
        (
            Leg(Side.BUY, contract("c1", "100", OptionType.CALL)),
            Leg(Side.SELL, contract("c2", "105", OptionType.CALL, "0.40", "0.50")),
        ),
    )
    credit = Structure(
        "put_credit",
        (
            Leg(
                Side.SELL,
                contract(
                    "p1",
                    "100",
                    OptionType.PUT,
                    "1.00",
                    "1.10",
                    exercise_style="European",
                    settlement_style="cash",
                ),
            ),
            Leg(
                Side.BUY,
                contract(
                    "p2",
                    "95",
                    OptionType.PUT,
                    "0.40",
                    "0.50",
                    exercise_style="European",
                    settlement_style="cash",
                ),
            ),
        ),
    )
    assert payoff_at_expiration(call, Decimal("110")) > 0
    assert payoff_at_expiration(put, Decimal("90")) > 0
    assert analyze_structure(debit).max_loss > 0
    assert analyze_structure(credit).max_loss > 0


def test_iron_condor_and_quantity_loss() -> None:
    legs = (
        Leg(
            Side.BUY,
            contract(
                "p90",
                "90",
                OptionType.PUT,
                "0.20",
                "0.30",
                exercise_style="European",
                settlement_style="cash",
            ),
        ),
        Leg(
            Side.SELL,
            contract(
                "p95",
                "95",
                OptionType.PUT,
                "0.70",
                "0.80",
                exercise_style="European",
                settlement_style="cash",
            ),
        ),
        Leg(
            Side.SELL,
            contract(
                "c105",
                "105",
                OptionType.CALL,
                "0.70",
                "0.80",
                exercise_style="European",
                settlement_style="cash",
            ),
        ),
        Leg(
            Side.BUY,
            contract(
                "c110",
                "110",
                OptionType.CALL,
                "0.20",
                "0.30",
                exercise_style="European",
                settlement_style="cash",
            ),
        ),
    )
    payoff = analyze_structure(Structure("iron_condor", legs))
    assert payoff.max_loss <= Decimal("1000")
    assert payoff.max_gain is not None


def test_quote_failures() -> None:
    stale = contract("x", "100", OptionType.CALL, as_of=NOW - timedelta(minutes=3))
    crossed = contract("y", "105", OptionType.CALL, "2", "1")
    result = validate_quotes(Structure("bad", (Leg(Side.BUY, stale), Leg(Side.SELL, crossed))), NOW)
    assert {gate.name for gate in result} >= {"freshness", "two_sided_quote"}


def test_volatility_distribution_and_iv_crush() -> None:
    assert implied_move(Decimal("100"), Decimal("5")) == Decimal("0.05")
    assert (
        len(iv_crush_matrix(Decimal("100"), Decimal("0.5"), Decimal("0.1"), Decimal("0.2"))) == 30
    )
    ev = distribution_ev([Decimal("-100"), Decimal("50"), Decimal("300")], [0.2, 0.5, 0.3])
    assert ev["expected_pnl"] > 0 and ev["total_loss_probability"] == 0.2
    with pytest.raises(ValueError):
        distribution_ev([Decimal("1")], [0.5])


def test_iv_crush_matrix_is_direction_aware_cost_inclusive_and_uses_implied_move() -> None:
    kwargs = {
        "implied_move_pct": Decimal("0.20"), "net_gamma": Decimal("0.01"),
        "net_theta": Decimal("-0.02"), "current_signed_complex_value": Decimal("100"),
        "signed_entry": Decimal("120"), "operational_costs": Decimal("15"),
    }
    bullish = iv_crush_matrix(Decimal("100"), Decimal("0.5"), Decimal("0.1"), Decimal("0.2"), **kwargs)
    bearish = iv_crush_matrix(Decimal("100"), Decimal("-0.5"), Decimal("0.1"), Decimal("0.2"), **kwargs)
    selection = {"spot_case": "equal_implied_move", "time_case": "immediate", "iv_case": "mild"}
    bull_correct = next(row for row in bullish if all(row[key] == value for key, value in selection.items()))
    bear_correct = next(row for row in bearish if all(row[key] == value for key, value in selection.items()))
    assert bull_correct["direction"] == "bullish" and bear_correct["direction"] == "bearish"
    assert Decimal(bull_correct["estimated_pnl"]) > Decimal()
    assert Decimal(bear_correct["estimated_pnl"]) > Decimal()
    conservative = next(row for row in bullish if row["spot_case"] == "smaller_than_implied" and row["time_case"] == "delayed" and row["iv_case"] == "severe")
    assert Decimal(conservative["estimated_pnl"]) < Decimal(bull_correct["estimated_pnl"])


def test_iv_crush_scales_quantity_once_and_uses_vega_percentage_points_for_debit_and_credit() -> None:
    common = {
        "implied_move_pct": Decimal("0.10"), "net_gamma": Decimal(), "net_theta": Decimal(),
        "current_signed_complex_value": Decimal("200"), "operational_costs": Decimal(),
    }
    one = iv_crush_matrix(Decimal("100"), Decimal("1"), Decimal("1"), Decimal("0.15"), signed_entry=Decimal("200"), quantity=1, **common)
    two = iv_crush_matrix(Decimal("100"), Decimal("2"), Decimal("2"), Decimal("0.15"), signed_entry=Decimal("400"), quantity=2, **{**common, "current_signed_complex_value": Decimal("400")})
    key = {"spot_case": "equal_implied_move", "time_case": "immediate", "iv_case": "typical"}
    one_value = Decimal(next(row for row in one if all(row[k] == v for k, v in key.items()))["estimated_pnl"])
    two_value = Decimal(next(row for row in two if all(row[k] == v for k, v in key.items()))["estimated_pnl"])
    assert two_value == one_value * 2
    credit = iv_crush_matrix(Decimal("100"), Decimal("1"), Decimal("0.1"), Decimal("0.15"), signed_entry=Decimal("-100"), current_signed_complex_value=Decimal("-80"), implied_move_pct=Decimal("0.10"), net_gamma=Decimal(), net_theta=Decimal(), operational_costs=Decimal())
    assert Decimal(next(row for row in credit if all(row[k] == v for k, v in key.items()))["estimated_pnl"]).is_finite()


def test_generate_supported_structures_is_canonical_deterministic_and_bounded() -> None:
    contracts = (contract("g100", "100", OptionType.CALL), contract("g105", "105", OptionType.CALL))
    generated = generate_supported_structures(contracts, cap=3)
    assert generated == generate_supported_structures(tuple(reversed(contracts)), cap=3)
    assert len(generated) <= 3
    assert {item.name for item in generated} <= {"long_call", "call_debit", "call_credit"}


def test_fixture_and_missing_portfolio_block_actionable() -> None:
    long = Structure(
        "long",
        (
            Leg(
                Side.BUY,
                contract(
                    "l",
                    "100",
                    OptionType.CALL,
                    exercise_style="American",
                    settlement_style="physical",
                ),
            ),
        ),
    )
    decision, gates, _ = decide(candidate(long, fixture=True), NOW)
    assert decision is not decision.ACTIONABLE
    assert any(gate.name == "fixture_mode" for gate in gates)


def test_1000_fee_inclusive_cap_and_american_short_rejection() -> None:
    expensive = Structure(
        "long",
        (
            Leg(
                Side.BUY,
                contract(
                    "e",
                    "100",
                    OptionType.CALL,
                    "11",
                    "12",
                    exercise_style="European",
                    settlement_style="cash",
                ),
            ),
        ),
    )
    decision, gates, _ = decide(candidate(expensive), NOW, fees=Decimal("1"))
    assert decision is not decision.ACTIONABLE
    assert any(gate.name == "universal_total_loss_cap" for gate in gates)
    american_short = Structure(
        "spread",
        (
            Leg(
                Side.SELL,
                contract(
                    "s",
                    "100",
                    OptionType.CALL,
                    exercise_style="American",
                    settlement_style="physical",
                ),
            ),
            Leg(
                Side.BUY,
                contract(
                    "b",
                    "105",
                    OptionType.CALL,
                    exercise_style="American",
                    settlement_style="physical",
                ),
            ),
        ),
    )
    _, gates, _ = decide(candidate(american_short), NOW)
    assert any(gate.name == "american_short_leg" for gate in gates)


def test_physical_expiry_and_cash_index_allowance() -> None:
    physical = Structure(
        "physical",
        (
            Leg(
                Side.BUY,
                contract(
                    "a",
                    "100",
                    OptionType.CALL,
                    exercise_style="American",
                    settlement_style="physical",
                ),
            ),
        ),
    )
    _, physical_gates, _ = decide(candidate(physical), NOW)
    assert any(gate.name == "physical_expiration" for gate in physical_gates)
    cash = Structure(
        "cash_spread",
        (
            Leg(
                Side.SELL,
                contract(
                    "s", "100", OptionType.CALL, exercise_style="European", settlement_style="cash"
                ),
            ),
            Leg(
                Side.BUY,
                contract(
                    "b", "105", OptionType.CALL, exercise_style="European", settlement_style="cash"
                ),
            ),
        ),
    )
    mechanics = ContractMechanics(
        "index option", "index", "European", "cash", "cash", None, None,
        "contained", "contained", "contained", "none", "XNYS",
    )
    decision, gates, payoff = decide(candidate(cash, mechanics=mechanics), NOW)
    # Direct compatibility candidates lack the evidence required by the full
    # ledger. Verified index mechanics remain necessary, but no longer bypass
    # unrelated failed hard gates.
    assert decision is decision.DATA_INSUFFICIENT
    assert payoff is not None and payoff.possibility_of_account_deficit == "ELIMINATED"
    assert {gate.name for gate in gates if not gate.passed}


@pytest.mark.parametrize(
    "mechanics",
    (
        None,
        ContractMechanics("equity option", "equity", "European", "cash", "cash", None, None, "", "", "", "", ""),
        ContractMechanics("ETF option", "ETF", "European", "cash", "cash", None, None, "", "", "", "", ""),
    ),
)
def test_decide_compatibility_path_never_actionable_for_unverified_short_mechanics(
    mechanics: ContractMechanics | None,
) -> None:
    short = Structure(
        "call_credit",
        (Leg(Side.SELL, contract("s", "100", OptionType.CALL, exercise_style="European", settlement_style="cash")), Leg(Side.BUY, contract("b", "105", OptionType.CALL, exercise_style="European", settlement_style="cash"))),
    )
    kwargs: dict[str, object] = {"mechanics": mechanics} if mechanics is not None else {}
    decision, _, _ = decide(candidate(short, **kwargs), NOW)
    assert decision is not decision.ACTIONABLE


def test_decide_compatibility_path_rejects_mixed_short_leg_mechanics() -> None:
    short = Structure(
        "call_credit",
        (
            Leg(Side.SELL, contract("s", "100", OptionType.CALL, exercise_style="American", settlement_style="cash")),
            Leg(Side.BUY, contract("b", "105", OptionType.CALL, exercise_style="European", settlement_style="cash")),
        ),
    )
    mechanics = ContractMechanics("index option", "index", "European", "cash", "cash", None, None, "", "", "", "", "")
    decision, _, _ = decide(candidate(short, mechanics=mechanics), NOW)
    assert decision is not decision.ACTIONABLE


@pytest.mark.parametrize("name", ("put_credit", "iron_condor"))
def test_decide_compatibility_retains_malformed_short_topology_failures(name: str) -> None:
    """A declared name cannot turn uncovered two-short puts into a trade."""
    malformed = Structure(
        name,
        (
            Leg(Side.SELL, contract("p100", "100", OptionType.PUT, exercise_style="European", settlement_style="cash")),
            Leg(Side.SELL, contract("p95", "95", OptionType.PUT, exercise_style="European", settlement_style="cash")),
        ),
    )
    mechanics = ContractMechanics(
        "index option", "index", "European", "cash", "cash", None, None,
        "contained", "contained", "contained", "none", "XNYS",
    )
    decision, gates, _ = decide(candidate(malformed, mechanics=mechanics), NOW)
    assert decision is not Decision.ACTIONABLE
    assert any(gate.name == "defined_risk" and not gate.passed for gate in gates)


def test_preflight_retains_malformed_short_topology_failure() -> None:
    malformed = Structure(
        "put_credit",
        (
            Leg(Side.SELL, contract("p100", "100", OptionType.PUT, exercise_style="European", settlement_style="cash")),
            Leg(Side.SELL, contract("p95", "95", OptionType.PUT, exercise_style="European", settlement_style="cash")),
        ),
    )
    mechanics = ContractMechanics(
        "index option", "index", "European", "cash", "cash", None, None,
        "contained", "contained", "contained", "none", "XNYS",
    )
    decision, gates = preflight(Decimal("100"), candidate(malformed, mechanics=mechanics), NOW)
    assert decision is not Decision.ACTIONABLE
    assert any(gate.name == "defined_risk" and not gate.passed for gate in gates)


def test_preflight_more_than_two_percent_invalidates() -> None:
    cash = Structure(
        "cash",
        (
            Leg(
                Side.BUY,
                contract(
                    "a", "100", OptionType.CALL, exercise_style="European", settlement_style="cash"
                ),
            ),
        ),
    )
    refreshed = candidate(cash, underlying=Decimal("103"))
    decision, _ = preflight(Decimal("100"), refreshed, NOW)
    assert decision.value == "INVALIDATED"
