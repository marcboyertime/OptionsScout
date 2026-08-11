from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt

from .models import SurfaceNode

ZERO = Decimal()


@dataclass(frozen=True)
class DistributionMetrics:
    expected_pnl: Decimal
    median_pnl: Decimal
    probability_profit: Decimal
    probability_any_loss: Decimal
    probability_total_loss: Decimal
    probability_2x: Decimal
    probability_3x: Decimal
    probability_5x: Decimal
    expected_shortfall: Decimal
    best: Decimal
    worst: Decimal
    return_on_max_risk: Decimal | None = None


def implied_move(spot: Decimal, straddle: Decimal) -> Decimal:
    if spot <= ZERO or straddle < ZERO:
        raise ValueError("invalid ATM straddle/spot")
    return straddle / spot


def iv_expected_move(spot: Decimal, iv: Decimal, days: int) -> Decimal:
    if spot <= ZERO or iv < ZERO or days < 0:
        raise ValueError("invalid IV expected-move inputs")
    return spot * iv * Decimal(str(sqrt(days / 365)))


def forward_event_variance(
    front_iv: Decimal, front_days: int, back_iv: Decimal, back_days: int
) -> Decimal | None:
    if front_days <= 0 or back_days <= front_days or min(front_iv, back_iv) < ZERO:
        return None
    value = (back_iv * back_iv * back_days - front_iv * front_iv * front_days) / Decimal(
        back_days - front_days
    )
    return value if value >= ZERO else None


def iv_rank(current: Decimal, history: list[Decimal]) -> tuple[Decimal | None, Decimal | None]:
    if len(history) < 2:
        return None, None
    low, high = min(history), max(history)
    rank = ZERO if high == low else (current - low) / (high - low)
    percentile = Decimal(sum(item <= current for item in history)) / Decimal(len(history))
    return rank, percentile


def weighted_metrics(
    payoffs: list[Decimal], probabilities: list[Decimal], max_risk: Decimal
) -> DistributionMetrics:
    if (
        len(payoffs) != len(probabilities)
        or not payoffs
        or sum(probabilities) != Decimal("1")
        or any(p < ZERO for p in probabilities)
    ):
        raise ValueError("probabilities must be exact Decimal strings summing to 1")
    if max_risk < ZERO:
        raise ValueError("max risk cannot be negative")
    pairs = sorted(zip(payoffs, probabilities, strict=True), key=lambda item: item[0])
    cumulative = ZERO
    median = pairs[-1][0]
    for pnl, probability in pairs:
        cumulative += probability
        if cumulative >= Decimal("0.5"):
            median = pnl
            break
    worst = min(payoffs)
    # Expected shortfall is the exact worst decile, including a fractional boundary mass.
    remaining = Decimal("0.10")
    tail_sum = ZERO
    for pnl, probability in pairs:
        used = min(remaining, probability)
        tail_sum += pnl * used
        remaining -= used
        if remaining == ZERO:
            break
    shortfall = tail_sum / Decimal("0.10")
    # ``max_risk`` is the positive, typed operational loss (entry plus every
    # declared cost).  Scenario P/L is an exact Decimal from the same
    # deterministic expiration calculation, so total loss is exact equality
    # with ``-max_risk`` -- no float epsilon and no "worst submitted case"
    # proxy.  A zero-risk payoff has no loss event.
    total_loss = -max_risk
    return DistributionMetrics(
        sum((pnl * probability for pnl, probability in pairs), ZERO),
        median,
        sum((prob for pnl, prob in pairs if pnl > ZERO), ZERO),
        sum((prob for pnl, prob in pairs if pnl < ZERO), ZERO),
        sum((prob for pnl, prob in pairs if max_risk > ZERO and pnl == total_loss), ZERO),
        sum((prob for pnl, prob in pairs if pnl >= max_risk * 2), ZERO),
        sum((prob for pnl, prob in pairs if pnl >= max_risk * 3), ZERO),
        sum((prob for pnl, prob in pairs if pnl >= max_risk * 5), ZERO),
        shortfall,
        max(payoffs),
        worst,
        (sum((pnl * probability for pnl, probability in pairs), ZERO) / max_risk)
        if max_risk > ZERO
        else None,
    )


def surface_summary(
    spot: Decimal,
    nodes: tuple[SurfaceNode, ...],
    snapshot_iv: Decimal,
) -> dict[str, Decimal | int | str]:
    """A small, typed-data-only summary; it never invents an IV rank/history."""
    if spot <= ZERO or snapshot_iv < ZERO:
        raise ValueError("invalid surface summary inputs")
    ordered = sorted(nodes, key=lambda node: (node.expiration, abs(node.strike - spot)))
    nearest = ordered[0] if ordered else None
    expiries = sorted({node.expiration for node in ordered})
    atm = nearest.implied_volatility if nearest is not None else snapshot_iv
    skew = max((node.implied_volatility for node in ordered), default=atm) - min(
        (node.implied_volatility for node in ordered), default=atm
    )
    return {
        "atm_iv": atm,
        "snapshot_iv": snapshot_iv,
        "surface_skew_range": skew,
        "expirations": len(expiries),
        "term_points": len(ordered),
        "iv_rank": "UNAVAILABLE_WITHOUT_HISTORICAL_SNAPSHOTS",
    }
