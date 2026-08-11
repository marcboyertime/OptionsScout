from decimal import Decimal

import pytest

from options_scout.portfolio import PortfolioLimits, clusters, size_quantity
from options_scout.volatility import (
    forward_event_variance,
    implied_move,
    iv_expected_move,
    iv_rank,
    weighted_metrics,
)


def test_weighted_median_expected_shortfall_and_probability_strings() -> None:
    metrics = weighted_metrics(
        [Decimal("-100"), Decimal("10"), Decimal("200")],
        [Decimal("0.2"), Decimal("0.5"), Decimal("0.3")],
        Decimal("100"),
    )
    assert metrics.median_pnl == Decimal("10") and metrics.expected_shortfall < 0
    with pytest.raises(ValueError):
        weighted_metrics([Decimal("1")], [Decimal("0.5")], Decimal("1"))


def test_iv_history_and_forward_variance() -> None:
    assert implied_move(Decimal("100"), Decimal("5")) == Decimal("0.05")
    assert iv_expected_move(Decimal("100"), Decimal("0.2"), 30) > 0
    assert forward_event_variance(Decimal("0.2"), 10, Decimal("0.3"), 20) is not None
    assert iv_rank(Decimal("0.2"), [Decimal("0.2")]) == (None, None)


def test_portfolio_lower_of_and_shared_catalyst_cluster() -> None:
    limits = PortfolioLimits(
        bankroll=Decimal("10000"),
        max_bankroll_pct=Decimal("0.05"),
        remaining_aggregate=Decimal("300"),
        remaining_cluster=Decimal("200"),
    )
    assert size_quantity(Decimal("100"), limits) == 2
    grouped = clusters(
        [("A", {"earnings", "tech"}, Decimal("100")), ("B", {"earnings"}, Decimal("200"))]
    )
    assert len(grouped) == 1 and grouped[0]["risk"] == Decimal("300")
