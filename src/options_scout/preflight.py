"""Structured, conservative preflight comparison of immutable and refreshed analyses."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def _at(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _legs(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    legs = _at(evaluation, "candidate", "structure", "legs")
    if not isinstance(legs, list):
        return {}
    return {str(_at(leg, "contract", "id") or index): leg for index, leg in enumerate(legs)}


def _leg_list(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selected legs in declared complex-order order, including duplicates."""
    legs = _at(evaluation, "candidate", "structure", "legs")
    return legs if isinstance(legs, list) and all(isinstance(leg, dict) for leg in legs) else []


def _changed(before: Any, after: Any) -> bool:
    return bool(before != after)


def _record(
    changes: list[dict[str, Any]], family: str, field: str, before: Any, after: Any
) -> None:
    if _changed(before, after):
        changes.append({"family": family, "field": field, "before": before, "after": after})


def compare_evaluations(original: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    """Compare only decision-relevant data; absence is itself an explicit mismatch."""
    changes: list[dict[str, Any]] = []
    # A refreshed recommendation must be the same proposed trade, not merely
    # another currently-valid candidate with the same candidate/contract IDs.
    # These are immutable order-identity terms and always invalidate when they
    # differ; ordinary quote/Greek refreshes remain separately auditable.
    for identity_path in (
        ("candidate", "structure", "name"),
        ("structure",),
        ("analysis", "structure", "kind"),
        ("candidate", "structure", "quantity"),
        ("candidate", "fill_plan", "order_type"),
        ("candidate", "structure_plan", "structure_type"),
        ("candidate", "structure_plan", "one_complex_order"),
    ):
        _record(
            changes,
            "trade_identity",
            ".".join(identity_path),
            _at(original, *identity_path),
            _at(refreshed, *identity_path),
        )
    original_leg_list, refreshed_leg_list = _leg_list(original), _leg_list(refreshed)
    _record(
        changes,
        "trade_identity",
        "legs.count",
        len(original_leg_list),
        len(refreshed_leg_list),
    )
    _record(
        changes,
        "trade_identity",
        "legs.contract_id_set",
        sorted(str(_at(leg, "contract", "id")) for leg in original_leg_list),
        sorted(str(_at(leg, "contract", "id")) for leg in refreshed_leg_list),
    )
    for index in range(max(len(original_leg_list), len(refreshed_leg_list))):
        before = original_leg_list[index] if index < len(original_leg_list) else None
        after = refreshed_leg_list[index] if index < len(refreshed_leg_list) else None
        for field in ("side", "ratio"):
            _record(
                changes,
                "trade_identity",
                f"legs[{index}].{field}",
                _at(before, field),
                _at(after, field),
            )
        for field in (
            "id",
            "symbol",
            "expiration",
            "strike",
            "option_type",
            "multiplier",
            "exercise_style",
            "settlement_style",
            "tradable",
            "adjusted",
        ):
            _record(
                changes,
                "trade_identity",
                f"legs[{index}].contract.{field}",
                _at(before, "contract", field),
                _at(after, "contract", field),
            )
    _record(
        changes,
        "underlying",
        "price",
        _at(original, "candidate", "underlying"),
        _at(refreshed, "candidate", "underlying"),
    )
    old_underlying, new_underlying = (
        _at(original, "candidate", "underlying"),
        _at(refreshed, "candidate", "underlying"),
    )
    move: str | None = None
    try:
        move = format(
            abs(
                (Decimal(str(new_underlying)) - Decimal(str(old_underlying)))
                / Decimal(str(old_underlying))
            ),
            "f",
        )
    except (InvalidOperation, ZeroDivisionError):
        changes.append(
            {
                "family": "underlying",
                "field": "movement",
                "before": old_underlying,
                "after": new_underlying,
                "reason": "missing or invalid underlying",
            }
        )
    for contract_id in sorted(set(_legs(original)) | set(_legs(refreshed))):
        before, after = _legs(original).get(contract_id), _legs(refreshed).get(contract_id)
        for field in ("bid", "ask", "mark", "iv", "delta", "gamma", "theta", "vega", "as_of"):
            _record(
                changes,
                "quotes_and_greeks",
                f"{contract_id}.{field}",
                _at(before, "contract", "quote", field),
                _at(after, "contract", "quote", field),
            )
    comparison_paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "fills_and_payoff": (
            ("analysis", "payoff", "fills", "natural_entry"),
            ("analysis", "payoff", "fills", "realistic_limit_entry"),
            ("candidate", "fill_plan", "limit"),
            ("candidate", "structure_plan", "entry_limit"),
            ("candidate", "structure_plan", "max_acceptable_limit"),
            ("analysis", "payoff", "breakevens"),
            ("analysis", "payoff", "theoretical_max_loss"),
            ("analysis", "payoff", "operational_max_loss_risk"),
            ("analysis", "payoff", "max_gain"),
        ),
        "volatility": (
            ("candidate", "volatility_context", "implied_move_pct"),
            ("analysis", "volatility"),
            ("analysis", "iv_crush"),
        ),
        "mechanics": (("candidate", "mechanics"),),
        "portfolio": (("candidate", "portfolio_assessment"),),
        "thesis_and_evidence": (
            ("candidate", "thesis_record"),
            ("candidate", "claim_records"),
            ("candidate", "source_records"),
        ),
    }
    for family, paths in comparison_paths.items():
        for comparison_path in paths:
            _record(
                changes,
                family,
                ".".join(comparison_path),
                _at(original, *comparison_path),
                _at(refreshed, *comparison_path),
            )
    failed_gates = [
        item.get("name", "unnamed")
        for item in refreshed.get("gates", [])
        if item.get("status") == "FAIL" or ("status" not in item and item.get("passed") is False)
    ]
    refreshed_legs = _legs(refreshed)
    crossed_or_missing = False
    for leg in refreshed_legs.values():
        quote = _at(leg, "contract", "quote")
        bid, ask = _at(quote, "bid"), _at(quote, "ask")
        try:
            crossed_or_missing = (
                crossed_or_missing
                or bid is None
                or ask is None
                or Decimal(str(bid)) <= 0
                or Decimal(str(ask)) <= 0
                or Decimal(str(bid)) > Decimal(str(ask))
            )
        except InvalidOperation:
            crossed_or_missing = True
    old_risk = _at(original, "analysis", "payoff", "operational_max_loss_risk")
    new_risk = _at(refreshed, "analysis", "payoff", "operational_max_loss_risk")
    economics_exceeded = False
    try:
        economics_exceeded = Decimal(str(new_risk)) > Decimal(str(old_risk))
    except (InvalidOperation, TypeError):
        economics_exceeded = True
    critical_families = {"trade_identity", "mechanics", "portfolio", "thesis_and_evidence"}
    critical_changes = [item for item in changes if item["family"] in critical_families]
    # Quotes, Greeks, and IV are expected to refresh.  They remain in the
    # auditable diff but invalidate only by rerunning into a hard gate failure,
    # crossed/missing data, a >2% underlying move, or worse economics.
    invalid = bool(failed_gates or crossed_or_missing or economics_exceeded or critical_changes)
    if move is not None and Decimal(move) > Decimal("0.02"):
        invalid = True
    return {
        "id": original.get("id"),
        "symbol": original.get("symbol"),
        "underlying_move_pct": move,
        "changes": changes,
        "failed_refreshed_gates": failed_gates,
        "crossed_or_missing": crossed_or_missing,
        "economics_exceeded_original_risk": economics_exceeded,
        "critical_changes": critical_changes,
        "invalid": invalid,
    }


def compare_payloads(original: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    original_items = original.get("evaluations", [])
    refreshed_items = refreshed.get("evaluations", [])
    original_by_id = {
        str(item.get("id")): item
        for item in original_items
        if isinstance(item, dict) and item.get("id")
    }
    refreshed_by_id = {
        str(item.get("id")): item
        for item in refreshed_items
        if isinstance(item, dict) and item.get("id")
    }
    invalid_id_sets = len(original_by_id) != len(original_items) or len(refreshed_by_id) != len(
        refreshed_items
    )
    comparisons: list[dict[str, Any]] = []
    for candidate_id in sorted(set(original_by_id) | set(refreshed_by_id)):
        before, after = original_by_id.get(candidate_id), refreshed_by_id.get(candidate_id)
        if before is None or after is None:
            comparisons.append(
                {
                    "id": candidate_id,
                    "symbol": (before or after or {}).get("symbol"),
                    "changes": [
                        {
                            "family": "candidate",
                            "field": "presence",
                            "before": before is not None,
                            "after": after is not None,
                        }
                    ],
                    "failed_refreshed_gates": [],
                    "invalid": True,
                }
            )
        else:
            comparisons.append(compare_evaluations(before, after))
    return {
        "reran_all_gates": True,
        "candidate_ids_match": not invalid_id_sets and set(original_by_id) == set(refreshed_by_id),
        "comparisons": comparisons,
        "invalid": invalid_id_sets
        or set(original_by_id) != set(refreshed_by_id)
        or any(item["invalid"] for item in comparisons),
    }
