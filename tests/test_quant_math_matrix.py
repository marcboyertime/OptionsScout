from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from options_scout.engine import analyze_structure, payoff_at_expiration, validate_quotes
from options_scout.models import Contract, Leg, OptionType, Quote, Side, SourceLabel, Structure
from options_scout.portfolio import PortfolioLimits, size_quantity
from options_scout.structures import classify_structure, compare_structures, validate_structure_plan
from options_scout.volatility import (
    forward_event_variance,
    implied_move,
    iv_expected_move,
    iv_rank,
    weighted_metrics,
)

NOW = datetime(2026, 8, 11, 15, tzinfo=UTC)


def _contract(
    identifier: str, strike: str, kind: OptionType, price: str, **changes: object
) -> Contract:
    quote = Quote(
        Decimal(price),
        Decimal(price),
        changes.pop("as_of", NOW),
        SourceLabel.LIVE,
        iv=Decimal("0.2"),
        delta=Decimal("0.4"),
        gamma=Decimal("0.01"),
        theta=Decimal("-0.02"),
        vega=Decimal("0.1"),
        volume=100,
        open_interest=1000,
    )
    return Contract(
        identifier,
        "IDX",
        (NOW + timedelta(days=10)).date().isoformat(),
        Decimal(strike),
        kind,
        quote,
        exercise_style="European",
        settlement_style="cash",
        **changes,
    )


@pytest.mark.parametrize(
    ("structure", "loss", "gain", "breakevens"),
    [
        (
            Structure("long_call", (Leg(Side.BUY, _contract("c", "100", OptionType.CALL, "2")),)),
            Decimal("200"),
            None,
            (Decimal("102"),),
        ),
        (
            Structure("long_put", (Leg(Side.BUY, _contract("p", "100", OptionType.PUT, "2")),)),
            Decimal("200"),
            Decimal("9800"),
            (Decimal("98"),),
        ),
        (
            Structure(
                "call_debit",
                (
                    Leg(Side.BUY, _contract("c1", "100", OptionType.CALL, "2")),
                    Leg(Side.SELL, _contract("c2", "105", OptionType.CALL, "0.5")),
                ),
            ),
            Decimal("150"),
            Decimal("350"),
            (Decimal("101.5"),),
        ),
        (
            Structure(
                "put_debit",
                (
                    Leg(Side.BUY, _contract("p1", "105", OptionType.PUT, "2")),
                    Leg(Side.SELL, _contract("p2", "100", OptionType.PUT, "0.5")),
                ),
            ),
            Decimal("150"),
            Decimal("350"),
            (Decimal("103.5"),),
        ),
        (
            Structure(
                "call_credit",
                (
                    Leg(Side.SELL, _contract("cc1", "100", OptionType.CALL, "2")),
                    Leg(Side.BUY, _contract("cc2", "105", OptionType.CALL, "0.5")),
                ),
            ),
            Decimal("350"),
            Decimal("150"),
            (Decimal("101.5"),),
        ),
        (
            Structure(
                "put_credit",
                (
                    Leg(Side.SELL, _contract("pc1", "105", OptionType.PUT, "2")),
                    Leg(Side.BUY, _contract("pc2", "100", OptionType.PUT, "0.5")),
                ),
            ),
            Decimal("350"),
            Decimal("150"),
            (Decimal("103.5"),),
        ),
        (
            Structure(
                "iron_condor",
                (
                    Leg(Side.BUY, _contract("ip90", "90", OptionType.PUT, "0.25")),
                    Leg(Side.SELL, _contract("ip95", "95", OptionType.PUT, "1")),
                    Leg(Side.SELL, _contract("ic105", "105", OptionType.CALL, "1")),
                    Leg(Side.BUY, _contract("ic110", "110", OptionType.CALL, "0.25")),
                ),
            ),
            Decimal("350"),
            Decimal("150"),
            (Decimal("93.5"), Decimal("106.5")),
        ),
        (
            Structure(
                "iron_butterfly",
                (
                    Leg(Side.BUY, _contract("ibp", "90", OptionType.PUT, "0.25")),
                    Leg(Side.SELL, _contract("ibs1", "100", OptionType.PUT, "1")),
                    Leg(Side.SELL, _contract("ibs2", "100", OptionType.CALL, "1")),
                    Leg(Side.BUY, _contract("ibc", "110", OptionType.CALL, "0.25")),
                ),
            ),
            Decimal("850"),
            Decimal("150"),
            (Decimal("98.5"), Decimal("101.5")),
        ),
        (
            Structure(
                "butterfly",
                (
                    Leg(Side.BUY, _contract("b1", "90", OptionType.CALL, "2")),
                    Leg(Side.SELL, _contract("b2", "100", OptionType.CALL, "1"), 2),
                    Leg(Side.BUY, _contract("b3", "110", OptionType.CALL, "2")),
                ),
            ),
            Decimal("200"),
            Decimal("800"),
            (Decimal("92"), Decimal("108")),
        ),
    ],
)
def test_exact_expiration_payoff_matrix(
    structure: Structure, loss: Decimal, gain: Decimal | None, breakevens: tuple[Decimal, ...]
) -> None:
    payoff = analyze_structure(structure)
    assert payoff.max_loss == loss
    assert payoff.max_gain == gain
    assert payoff.breakevens == breakevens
    assert payoff.table[0][0] == Decimal()


def test_piecewise_tails_quantity_multiplier_ratio_and_expiry_table() -> None:
    long_call = Structure(
        "long_call", (Leg(Side.BUY, _contract("lc", "100", OptionType.CALL, "2")),)
    )
    naked_short = Structure(
        "short", (Leg(Side.SELL, _contract("sc", "100", OptionType.CALL, "1")),)
    )
    assert analyze_structure(long_call).max_loss == Decimal("200")
    assert analyze_structure(long_call).max_gain is None
    assert analyze_structure(naked_short).max_loss is None
    sized = Structure(
        "long_call",
        (Leg(Side.BUY, _contract("q", "100", OptionType.CALL, "2", multiplier=50), 2),),
        quantity=3,
    )
    assert payoff_at_expiration(sized, Decimal("110")) == Decimal("2400")
    assert dict(analyze_structure(long_call).table)[Decimal("100")] == Decimal("-200")


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"bid": None}, "two_sided_quote"),
        ({"ask": None}, "two_sided_quote"),
        ({"bid": Decimal("0")}, "two_sided_quote"),
        ({"bid": Decimal("2"), "ask": Decimal("1")}, "two_sided_quote"),
        ({"bid": Decimal("0.1"), "ask": Decimal("1")}, "liquidity"),
        ({"as_of": NOW - timedelta(minutes=2)}, "freshness"),
    ],
)
def test_quote_price_failure_matrix(changes: dict[str, object], failure: str) -> None:
    contract = _contract("q", "100", OptionType.CALL, "1")
    quote = contract.quote.__class__(**{**contract.quote.__dict__, **changes})
    mutated = contract.__class__(**{**contract.__dict__, "quote": quote})
    assert failure in {
        item.name
        for item in validate_quotes(Structure("long_call", (Leg(Side.BUY, mutated),)), NOW)
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("iv", Decimal("-0.5")),
        ("delta", Decimal("2")),
        ("gamma", Decimal("-0.01")),
        ("vega", Decimal("-0.01")),
    ],
)
def test_quote_validation_defends_constructed_candidates_against_impossible_domains(
    field: str, value: Decimal
) -> None:
    contract = _contract("domain", "100", OptionType.CALL, "1")
    quote = contract.quote.__class__(**{**contract.quote.__dict__, field: value})
    mutated = contract.__class__(**{**contract.__dict__, "quote": quote})
    assert "quote_domain" in {
        item.name
        for item in validate_quotes(Structure("long_call", (Leg(Side.BUY, mutated),)), NOW)
    }


@pytest.mark.parametrize("field", ["iv", "delta", "gamma", "theta", "vega"])
def test_missing_each_iv_or_greek_is_a_contract_failure(field: str) -> None:
    contract = _contract("g", "100", OptionType.CALL, "1")
    quote = contract.quote.__class__(**{**contract.quote.__dict__, field: None})
    mutated = contract.__class__(**{**contract.__dict__, "quote": quote})
    assert "greeks_iv" in {
        item.name
        for item in validate_quotes(Structure("long_call", (Leg(Side.BUY, mutated),)), NOW)
    }


def test_duplicate_adjusted_wrong_multiplier_and_untradable_are_separate_integrity_failures() -> (
    None
):
    contract = _contract("same", "100", OptionType.CALL, "1")
    bad = contract.__class__(
        **{**contract.__dict__, "adjusted": True, "multiplier": 50, "tradable": False}
    )
    failures = {
        item.name
        for item in validate_quotes(
            Structure("x", (Leg(Side.BUY, contract), Leg(Side.BUY, bad))), NOW
        )
    }
    assert {"duplicate_contract", "contract_integrity"} <= failures


def test_structure_classifier_rejects_calendar_diagonal_and_compares_supported_complexes() -> None:
    call = Structure("long_call", (Leg(Side.BUY, _contract("a", "100", OptionType.CALL, "1")),))
    put = Structure("long_put", (Leg(Side.BUY, _contract("b", "100", OptionType.PUT, "1")),))
    assert classify_structure(call).supported and classify_structure(put).defined_risk
    assert classify_structure(Structure("calendar", call.legs)).pre_expiration_only
    assert validate_structure_plan(Structure("diagonal", call.legs))
    assert compare_structures(
        (call, put), {"long_call": Decimal("2"), "long_put": Decimal("1")}
    ) == [call, put]


@pytest.mark.parametrize(
    "structure",
    [
        Structure(
            "call_debit",
            (
                Leg(Side.BUY, _contract("a", "100", OptionType.CALL, "1")),
                Leg(Side.BUY, _contract("b", "105", OptionType.CALL, "1")),
            ),
        ),
        Structure(
            "call_credit",
            (
                Leg(Side.SELL, _contract("a", "105", OptionType.CALL, "1")),
                Leg(Side.BUY, _contract("b", "100", OptionType.CALL, "1")),
            ),
        ),
        Structure(
            "put_debit",
            (
                Leg(Side.BUY, _contract("a", "100", OptionType.PUT, "1")),
                Leg(Side.SELL, _contract("b", "105", OptionType.PUT, "1")),
            ),
        ),
        # Sol's two-sell-put probe: finite sampled economics must not bless an
        # uncovered declaration as a put credit spread.
        Structure(
            "put_credit",
            (
                Leg(Side.SELL, _contract("a", "105", OptionType.PUT, "1")),
                Leg(Side.SELL, _contract("b", "100", OptionType.PUT, "1")),
            ),
        ),
        Structure(
            "butterfly",
            (
                Leg(Side.BUY, _contract("a", "90", OptionType.CALL, "1")),
                Leg(Side.SELL, _contract("b", "100", OptionType.CALL, "1"), 2),
                Leg(Side.BUY, _contract("c", "115", OptionType.CALL, "1")),
            ),
        ),
        Structure(
            "iron_butterfly",
            (
                Leg(Side.BUY, _contract("a", "90", OptionType.PUT, "1")),
                Leg(Side.SELL, _contract("b", "100", OptionType.PUT, "1"), 2),
                Leg(Side.SELL, _contract("c", "100", OptionType.CALL, "1")),
                Leg(Side.BUY, _contract("d", "110", OptionType.CALL, "1")),
            ),
        ),
    ],
)
def test_named_structure_topology_rejects_same_side_wrong_strike_ratio_and_uncovered_shorts(
    structure: Structure,
) -> None:
    classification = classify_structure(structure)
    assert not classification.supported
    assert not classification.defined_risk


def test_decimal_distribution_volatility_and_portfolio_lower_of_matrix() -> None:
    metrics = weighted_metrics(
        [Decimal("-100"), Decimal("0"), Decimal("100"), Decimal("500")],
        [Decimal("0.1"), Decimal("0.2"), Decimal("0.4"), Decimal("0.3")],
        Decimal("100"),
    )
    assert metrics.expected_pnl == Decimal("180")
    assert metrics.median_pnl == Decimal("100")
    assert metrics.probability_profit == Decimal("0.7")
    assert metrics.probability_any_loss == Decimal("0.1")
    assert metrics.probability_total_loss == Decimal("0.1")
    assert (metrics.probability_2x, metrics.probability_3x, metrics.probability_5x) == (
        Decimal("0.3"),
        Decimal("0.3"),
        Decimal("0.3"),
    )
    assert metrics.expected_shortfall == Decimal("-100") and metrics.return_on_max_risk == Decimal(
        "1.8"
    )
    assert implied_move(Decimal("100"), Decimal("5")) == Decimal("0.05")
    assert iv_expected_move(Decimal("100"), Decimal("0.2"), 30) > Decimal()
    assert forward_event_variance(Decimal("0.2"), 10, Decimal("0.3"), 20) is not None
    assert iv_rank(Decimal("0.2"), [Decimal("0.2")]) == (None, None)
    limits = PortfolioLimits(
        Decimal("1000"),
        Decimal("10000"),
        Decimal("0.05"),
        Decimal("300"),
        Decimal("200"),
        Decimal("150"),
        Decimal("125"),
        Decimal("110"),
    )
    assert size_quantity(Decimal("100"), limits) == 1


def test_total_loss_probability_uses_typed_operational_max_loss_not_worst_submitted_case() -> None:
    metrics = weighted_metrics(
        [Decimal("-50"), Decimal("100")], [Decimal("0.4"), Decimal("0.6")], Decimal("100")
    )
    assert metrics.probability_total_loss == Decimal("0")
