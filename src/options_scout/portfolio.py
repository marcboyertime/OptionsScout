from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .models import PortfolioAssessment


@dataclass(frozen=True)
class PortfolioLimits:
    hard_cap: Decimal = Decimal("1000")
    bankroll: Decimal | None = None
    max_bankroll_pct: Decimal | None = None
    remaining_aggregate: Decimal | None = None
    remaining_cluster: Decimal | None = None
    remaining_event: Decimal | None = None
    remaining_sector: Decimal | None = None
    remaining_factor: Decimal | None = None


def size_quantity(per_contract_risk: Decimal, limits: PortfolioLimits) -> int:
    if per_contract_risk <= 0:
        return 0
    caps = [limits.hard_cap]
    if limits.bankroll is not None and limits.max_bankroll_pct is not None:
        caps.append(limits.bankroll * limits.max_bankroll_pct)
    for value in (
        limits.remaining_aggregate,
        limits.remaining_cluster,
        limits.remaining_event,
        limits.remaining_sector,
        limits.remaining_factor,
    ):
        if value is not None:
            caps.append(value)
    return max(0, int(min(caps) // per_contract_risk))


def clusters(records: list[tuple[str, set[str], Decimal]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for symbol, tags, risk in records:
        match = next((entry for entry in result if set(entry["tags"]) & tags), None)
        if match is None:
            result.append({"symbols": [symbol], "tags": sorted(tags), "risk": risk})
        else:
            match["symbols"].append(symbol)
            match["tags"] = sorted(set(match["tags"]) | tags)
            match["risk"] += risk
    return result


def assessment_limits_pass(
    assessment: PortfolioAssessment, incremental_risk: Decimal, event_trade: bool
) -> dict[str, bool]:
    """Evaluate the submitted portfolio context, never a guessed account balance.

    The typed assessment carries already-calculated worst-case buckets.  Adding the
    candidate's operational risk makes each gate independently auditable.
    """
    if incremental_risk < 0:
        return {name: False for name in ("aggregate", "cluster", "event", "sector", "factor")}
    limits = assessment.limits
    current = {
        "aggregate": assessment.aggregate_risk,
        "cluster": assessment.cluster_risk,
        "event": assessment.event_risk,
        "sector": assessment.sector_risk,
        "factor": assessment.factor_risk,
    }
    remaining = {
        "aggregate": limits.remaining_aggregate,
        "cluster": limits.remaining_cluster,
        "event": limits.remaining_event,
        "sector": limits.remaining_sector,
        "factor": limits.remaining_factor,
    }
    result = {
        key: incremental_risk <= remaining[key]
        and current[key] + incremental_risk <= limits.hard_cap
        for key in current
    }
    if not event_trade:
        # The event bucket remains checked as a true portfolio constraint, but is
        # not allowed to create a fictitious event concentration for this trade.
        result["event"] = assessment.event_risk <= limits.remaining_event
    return result


def duplicate_expression(
    assessment: PortfolioAssessment,
    symbol: str,
    factor_tags: set[str],
    correlation_floor: Decimal = Decimal("0.80"),
) -> bool:
    """Suppress same-symbol and strongly correlated factor expressions."""
    if any(position.symbol == symbol for position in assessment.positions):
        return True
    correlated = {
        record.position_id
        for record in assessment.correlations
        if record.correlation >= correlation_floor and record.rationale.strip()
    }
    return any(
        position.id in correlated and factor_tags.intersection(position.factor_tags)
        for position in assessment.positions
    )
