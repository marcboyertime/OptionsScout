from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from .calendar import session_status
from .engine import (
    DEFAULT_LIQUIDITY_RULES,
    LiquidityRules,
    Payoff,
    analyze_structure,
    expiration_pnl,
    fill_economics,
    iv_crush_matrix,
    planned_exit_intrinsic_pnl,
)
from .gates import decision_from_ledger, ledger
from .models import Candidate, Decision, GateResult, GateStatus, jsonable
from .portfolio import duplicate_expression
from .schema import ParsedRun, SchemaError, load_run, parse_run
from .structures import classify_structure, generate_supported_structures
from .volatility import surface_summary, weighted_metrics

__all__ = ["ParsedRun", "SchemaError", "evaluate", "load_run", "parse_run"]


def _active_watch(candidate: Candidate, now: datetime) -> str | None:
    for trigger in candidate.watch_triggers:
        if trigger.expires_at >= now and trigger.condition.strip() and trigger.action.strip():
            return trigger.condition
    return None


def _distribution(
    candidate: Candidate, payoff: Payoff | None, policy_fee_per_contract: Decimal = Decimal()
) -> dict[str, Any] | None:
    planned_exit = candidate.planned_exit_valuation
    use_planned_exit = (
        candidate.structure_plan.exit_plan.required_before_expiry and planned_exit is not None
    )
    scenarios = planned_exit.scenarios if use_planned_exit and planned_exit is not None else candidate.distribution.scenarios
    if not scenarios or payoff is None or candidate.structure is None:
        return None
    try:
        policy_fees = policy_fee_per_contract * Decimal(
            sum(leg.ratio for leg in candidate.structure.legs) * candidate.structure.quantity
        )
        if use_planned_exit:
            assert planned_exit is not None
            payoffs = [
                planned_exit_intrinsic_pnl(
                    candidate.structure,
                    scenario.underlying_spot,
                    candidate.structure_plan.fees + policy_fees,
                    candidate.structure_plan.commissions,
                    candidate.structure_plan.exit_slippage,
                    candidate.fill_plan.max_slippage,
                    entry=candidate.structure_plan.max_acceptable_limit,
                )
                for scenario in planned_exit.scenarios
            ]
        else:
            payoffs = [
                expiration_pnl(
                    candidate.structure,
                    scenario.expiration_spot,
                    candidate.structure_plan.fees + policy_fees,
                    candidate.structure_plan.commissions,
                    candidate.structure_plan.exit_slippage,
                    candidate.fill_plan.max_slippage,
                    entry=candidate.structure_plan.max_acceptable_limit,
                )
                for scenario in candidate.distribution.scenarios
            ]
        metrics = weighted_metrics(
            payoffs,
            [scenario.probability for scenario in scenarios],
            payoff.operational_max_loss_risk or Decimal(),
        )
    except ValueError as error:
        return {"status": "UNAVAILABLE", "reason": str(error)}
    result = cast(dict[str, Any], jsonable(metrics))
    result["valuation_basis"] = "planned_exit_post_event" if use_planned_exit else "expiration"
    return result


def _nonselected_alternative_mechanics_reason(
    candidate: Candidate, structure: Any
) -> str | None:
    """Return a containment reason before an alternative can compete.

    A supplied alternative is untrusted proposal data just like a generated
    one.  It cannot borrow the selected structure's verified mechanics or
    planned-exit valuation.
    """
    series = {
        (
            leg.contract.symbol,
            leg.contract.expiration,
            leg.contract.multiplier,
            leg.contract.exercise_style,
            leg.contract.settlement_style,
        )
        for leg in structure.legs
    }
    shorts = [leg for leg in structure.legs if leg.side.value == "sell"]
    if len(series) != 1:
        return (
            "alternative short legs lack exact compatible symbol/expiration/multiplier/exercise/settlement "
            "European cash mechanics"
            if shorts
            else "alternative legs lack exact compatible symbol/expiration/multiplier/exercise/settlement mechanics"
        )
    if shorts and (
        candidate.mechanics.product_type != "index"
        or candidate.mechanics.asset_type != "index option"
        or any(
            leg.contract.exercise_style != "European"
            or leg.contract.settlement_style != "cash"
            for leg in structure.legs
        )
    ):
        return "alternative short leg requires verified index/index-option European cash mechanics"
    if not shorts and any(leg.contract.settlement_style == "physical" for leg in structure.legs):
        # AlternativeEconomics intentionally contains costs only. Until an
        # alternative has its own typed exit plan and conservative valuation,
        # no selected-plan early-exit evidence may be reused.
        return "physical long alternative requires its own typed pre-expiry exit plan and conservative valuation"
    return None


def _structure_comparison(
    candidate: Candidate, policy_fee_per_contract: Decimal = Decimal()
) -> list[dict[str, Any]]:
    """Derive comparable maximum-fill economics for every typed alternative."""
    alternatives: list[dict[str, Any]] = []
    independent = {item.structure_id: item for item in candidate.alternative_economics}
    for index, structure in enumerate(candidate.structure_set):
        policy_fees = policy_fee_per_contract * Decimal(
            sum(leg.ratio for leg in structure.legs) * structure.quantity
        )
        classification = classify_structure(structure)
        item: dict[str, Any] = {
            "id": f"alternative-{index}",
            "name": structure.name,
            "classification": jsonable(classification),
            "selected": structure == candidate.structure,
        }
        if not classification.supported:
            item.update({"status": "REJECTED", "reason": classification.reason})
            alternatives.append(item)
            continue
        if structure != candidate.structure:
            mechanics_reason = _nonselected_alternative_mechanics_reason(candidate, structure)
            if mechanics_reason is not None:
                item.update({"status": "UNAVAILABLE", "reason": mechanics_reason})
                alternatives.append(item)
                continue
        try:
            fills = fill_economics(structure)
            if structure == candidate.structure:
                maximum_fill = candidate.structure_plan.max_acceptable_limit
                fees = candidate.structure_plan.fees
                commissions = candidate.structure_plan.commissions
                exit_slippage = candidate.structure_plan.exit_slippage
                entry_slippage = candidate.fill_plan.max_slippage
                if candidate.structure_plan.exit_plan.required_before_expiry:
                    if candidate.planned_exit_valuation is None:
                        item.update({"status": "UNAVAILABLE", "reason": "planned-exit valuation unavailable"})
                        alternatives.append(item)
                        continue
                    valuation_basis = "selected_planned_exit_valuation"
                else:
                    valuation_basis = "selected_reviewed_max_fill"
            else:
                economics = independent.get(structure.identifier)
                maximum_fill = fills.natural_entry
                fees = economics.fees if economics is not None else Decimal()
                commissions = economics.commissions if economics is not None else Decimal()
                exit_slippage = (
                    economics.exit_slippage if economics is not None else fills.expected_exit_slippage
                )
                entry_slippage = (
                    economics.entry_slippage
                    if economics is not None
                    else abs(fills.natural_entry - fills.realistic_limit_entry)
                )
                valuation_basis = (
                    "alternative_reviewed_costs_natural_max_fill"
                    if economics is not None
                    else "alternative_quote_derived_costs_natural_max_fill"
                )
            payoff = analyze_structure(structure, maximum_fill)
            if valuation_basis == "selected_planned_exit_valuation":
                assert candidate.planned_exit_valuation is not None
                terminal = [
                    planned_exit_intrinsic_pnl(structure, scenario.underlying_spot, fees + policy_fees, commissions, exit_slippage, entry_slippage, entry=maximum_fill)
                    for scenario in candidate.planned_exit_valuation.scenarios
                ]
                probabilities = [
                    scenario.probability for scenario in candidate.planned_exit_valuation.scenarios
                ]
            else:
                terminal = [
                    expiration_pnl(structure, scenario.expiration_spot, fees + policy_fees, commissions, exit_slippage, entry_slippage, entry=maximum_fill)
                    for scenario in candidate.distribution.scenarios
                ]
                probabilities = [scenario.probability for scenario in candidate.distribution.scenarios]
            metrics = weighted_metrics(
                terminal,
                probabilities,
                max(-min(terminal), Decimal()),
            )
        except ValueError as error:
            # Parsing rejects a supported alternative that cannot be valued;
            # this defensive branch protects public Candidate construction.
            item.update({"status": "REJECTED", "reason": f"inconsistent valuation: {error}"})
            alternatives.append(item)
            continue
        item.update(
            {
                "status": "VALUED",
                "max_fill_payoff": jsonable(payoff),
                "fill_economics": jsonable(fills),
                "maximum_fill": maximum_fill,
                "costs": {
                    "fees": fees + policy_fees,
                    "commissions": commissions,
                    "exit_slippage": exit_slippage,
                    "entry_slippage": entry_slippage,
                    "policy_fees": policy_fees,
                },
                "valuation_basis": valuation_basis,
                "expected_value": metrics.expected_pnl,
                "breakevens": payoff.breakevens,
                "operational_max_loss_risk": (
                    max(
                        -min(terminal),
                        (payoff.max_loss or Decimal())
                        + fees
                        + commissions
                        + exit_slippage
                        + entry_slippage
                        + policy_fees,
                    )
                ),
            }
        )
        alternatives.append(item)
    supplied_signatures = {
        (classify_structure(structure).kind, tuple((leg.side.value, leg.contract.id, leg.ratio) for leg in structure.legs))
        for structure in candidate.structure_set
    }
    for structure in generate_supported_structures(candidate.contract_chain):
        signature = (
            classify_structure(structure).kind,
            tuple((leg.side.value, leg.contract.id, leg.ratio) for leg in structure.legs),
        )
        if signature in supplied_signatures:
            continue
        selected_expiration = candidate.structure.legs[0].contract.expiration if candidate.structure else ""
        mechanics_reason = _nonselected_alternative_mechanics_reason(candidate, structure)
        if structure.legs[0].contract.expiration != selected_expiration:
            alternatives.append({"id": structure.identifier, "name": structure.name, "classification": jsonable(classify_structure(structure)), "selected": False, "generated": True, "status": "UNAVAILABLE", "reason": "different expiration requires independently sourced horizon distribution"})
            continue
        if mechanics_reason is not None:
            alternatives.append({"id": structure.identifier, "name": structure.name, "classification": jsonable(classify_structure(structure)), "selected": False, "generated": True, "status": "UNAVAILABLE", "reason": mechanics_reason})
            continue
        if candidate.structure_plan.exit_plan.required_before_expiry:
            alternatives.append({"id": structure.identifier, "name": structure.name, "classification": jsonable(classify_structure(structure)), "selected": False, "generated": True, "status": "UNAVAILABLE", "reason": "generated alternative lacks its own planned-exit valuation"})
            continue
        try:
            fills = fill_economics(structure)
            maximum_fill = fills.natural_entry
            entry_slippage = abs(fills.natural_entry - fills.realistic_limit_entry)
            exit_slippage = fills.expected_exit_slippage
            policy_fees = policy_fee_per_contract * Decimal(
                sum(leg.ratio for leg in structure.legs) * structure.quantity
            )
            payoff = analyze_structure(structure, maximum_fill)
            terminal = [
                expiration_pnl(
                    structure,
                    scenario.expiration_spot,
                    policy_fees,
                    Decimal(),
                    exit_slippage,
                    entry_slippage,
                    entry=maximum_fill,
                )
                for scenario in candidate.distribution.scenarios
            ]
            metrics = weighted_metrics(
                terminal,
                [scenario.probability for scenario in candidate.distribution.scenarios],
                max(-min(terminal), Decimal()),
            )
            alternatives.append(
                {
                    "id": structure.identifier,
                    "name": structure.name,
                    "classification": jsonable(classify_structure(structure)),
                    "selected": False,
                    "generated": True,
                    "status": "VALUED",
                    "valuation_basis": "generated_quote_derived_costs_natural_max_fill",
                    "max_fill_payoff": jsonable(payoff),
                    "fill_economics": jsonable(fills),
                    "maximum_fill": maximum_fill,
                    "costs": {"fees": policy_fees, "commissions": Decimal(), "exit_slippage": exit_slippage, "entry_slippage": entry_slippage, "policy_fees": policy_fees},
                    "expected_value": metrics.expected_pnl,
                    "breakevens": payoff.breakevens,
                    "operational_max_loss_risk": max(
                        -min(terminal),
                        (payoff.max_loss or Decimal()) + policy_fees + exit_slippage + entry_slippage,
                    ),
                }
            )
        except ValueError as error:
            alternatives.append({"id": structure.identifier, "name": structure.name, "classification": jsonable(classify_structure(structure)), "selected": False, "generated": True, "status": "UNAVAILABLE", "reason": f"generated topology lacks defensible economics: {error}"})
    return alternatives


def evaluate(
    parsed: ParsedRun,
    liquidity_rules: LiquidityRules = DEFAULT_LIQUIDITY_RULES,
    policy_fee_per_contract: Decimal = Decimal(),
) -> dict[str, Any]:
    """Evaluate each unique candidate once and rank only genuine actionable finalists."""
    counts = {
        "universe": len(parsed.candidates),
        "equity_filtered": 0,
        "chain_validated": 0,
        "structures": 0,
        "finalists": 0,
        "actionable": 0,
    }
    evaluations: list[dict[str, Any]] = []
    for candidate in parsed.candidates:
        if candidate.structure is None:
            # Schema v1 never permits this, but Candidate remains publicly constructible.
            continue
        if (
            candidate.equity_context.tradable
            and candidate.equity_context.options_available
            and candidate.underlying is not None
        ):
            counts["equity_filtered"] += 1
        session = session_status(parsed.as_of, candidate.mechanics.product_calendar or "XNYS")
        gates, payoff = ledger(
            candidate,
            parsed.as_of,
            live=candidate.source.value == "LIVE",
            fixture=parsed.fixture or candidate.fixture,
            session_override=session,
            liquidity_rules=liquidity_rules,
            policy_fee_per_contract=policy_fee_per_contract,
        )
        if not any(
            item.status is GateStatus.FAIL
            for item in gates
            if item.name in {"leg_quotes_fresh", "leg_sync"}
        ):
            counts["chain_validated"] += 1
        classification = classify_structure(candidate.structure)
        if classification.supported:
            counts["structures"] += 1
        trigger = _active_watch(candidate, parsed.as_of)
        comparison = _structure_comparison(candidate, policy_fee_per_contract)
        selected_item = next((item for item in comparison if item.get("selected")), None)
        valued = [
            item
            for item in comparison
            if item.get("status") == "VALUED"
        ]
        best_id = (
            max(valued, key=lambda item: (Decimal(str(item["expected_value"])), str(item["id"])))
            if valued
            else None
        )
        selected_is_best = bool(
            selected_item is not None and best_id is not None and selected_item["id"] == best_id["id"]
        )
        gates.append(
            GateResult(
                "structure_selection",
                selected_is_best,
                "selected structure is the highest independently derived expected-value compatible alternative after policy fees"
                if selected_is_best
                else "selected structure is dominated by an independently derived compatible alternative",
                status=GateStatus.PASS if selected_is_best else GateStatus.FAIL,
                trace=f"selected={selected_item.get('id') if selected_item else None}; best={best_id.get('id') if best_id else None}",
            )
        )
        decision = decision_from_ledger(
            gates,
            candidate.source.value == "LIVE",
            parsed.fixture or candidate.fixture,
            trigger,
            abs(candidate.equity_context.material_move_pct) > Decimal("0.02"),
        )
        analysis: dict[str, Any] = {
            "session": session,
            "payoff": jsonable(payoff),
            "structure": jsonable(classification),
            "distribution": _distribution(candidate, payoff, policy_fee_per_contract),
            "volatility": jsonable(
                surface_summary(
                    candidate.underlying,
                    candidate.surface_nodes,
                    candidate.volatility_context.iv_snapshot.atm_iv,
                )
            )
            if candidate.underlying is not None
            else {"status": "UNAVAILABLE"},
            "red_team": jsonable(candidate.skeptic),
            "structure_comparison": comparison,
            "selected_structure_best": {
                "passed": selected_is_best,
                "selected_id": selected_item.get("id") if selected_item else None,
                "best_id": best_id.get("id") if best_id else None,
                "reason": "selected structure is the highest independently derived expected-value alternative after policy fees" if selected_is_best else "selected structure is not the highest independently derived expected-value alternative after policy fees",
            },
        }
        spanning_events = [
            event
            for event in candidate.volatility_context.upcoming_events
            if parsed.as_of < event.event_at
            and event.event_at.date().isoformat()
            <= max(leg.contract.expiration for leg in candidate.structure.legs)
        ]
        if (
            payoff is not None
            and candidate.underlying is not None
            and payoff.net_delta is not None
            and payoff.net_vega is not None
            and payoff.net_gamma is not None
            and payoff.net_theta is not None
            and payoff.fills.midpoint_entry is not None
            and abs(payoff.net_delta) >= Decimal("0.0001")
            and spanning_events
        ):
            # This is deliberately the explicitly sourced expected IV drop, not
            # the unrelated price implied-move percentage.
            analysis["iv_crush"] = iv_crush_matrix(
                candidate.underlying,
                payoff.net_delta,
                payoff.net_vega,
                max(event.expected_iv_drop for event in spanning_events),
                implied_move_pct=candidate.volatility_context.implied_move_pct,
                net_gamma=payoff.net_gamma,
                net_theta=payoff.net_theta,
                multiplier=candidate.structure.legs[0].contract.multiplier,
                quantity=candidate.structure.quantity,
                current_signed_complex_value=payoff.fills.midpoint_entry or Decimal(),
                signed_entry=candidate.structure_plan.max_acceptable_limit,
                operational_costs=(
                    candidate.structure_plan.fees
                    + candidate.structure_plan.commissions
                    + candidate.fill_plan.max_slippage
                    + policy_fee_per_contract
                    * Decimal(
                        sum(leg.ratio for leg in candidate.structure.legs)
                        * candidate.structure.quantity
                    )
                    + candidate.structure_plan.exit_slippage
                ),
            )
        evaluation = {
            "id": candidate.id,
            "symbol": candidate.symbol,
            "structure": candidate.structure.name,
            "decision": decision.value,
            "gates": jsonable(gates),
            "analysis": analysis,
            "candidate": _safe_candidate(candidate),
            "operational": {
                "one_complex_order": candidate.structure_plan.one_complex_order,
                "classification": classification.kind,
                "watch_trigger": trigger,
                "pre_expiration_only": classification.pre_expiration_only,
            },
        }
        evaluations.append(evaluation)
        if decision is Decision.ACTIONABLE:
            counts["finalists"] += 1
            counts["actionable"] += 1
    ranked = sorted(
        (item for item in evaluations if item["decision"] == Decision.ACTIONABLE.value),
        key=lambda item: (
            -Decimal(str(item["analysis"]["distribution"]["expected_pnl"])),
            -sum(1 for source in item["candidate"]["source_records"] if source["primary"]),
            -Decimal(str(item["candidate"]["equity_context"]["underlying_liquidity"])),
            Decimal(str(item["analysis"]["payoff"]["operational_max_loss_risk"])),
            len(item["candidate"]["thesis_record"]["assumptions"]),
            item["id"],
        ),
    )
    all_session = session_status(parsed.as_of)
    if ranked:
        overall = Decision.ACTIONABLE
    elif any(item["decision"] == Decision.INVALIDATED.value for item in evaluations):
        overall = Decision.INVALIDATED
    elif any(item["decision"] == Decision.MARKET_CLOSED_OR_STALE.value for item in evaluations):
        overall = Decision.MARKET_CLOSED_OR_STALE
    elif any(item["decision"] == Decision.DATA_INSUFFICIENT.value for item in evaluations):
        overall = Decision.DATA_INSUFFICIENT
    elif any(item["decision"] == Decision.WATCH.value for item in evaluations):
        overall = Decision.WATCH
    else:
        overall = Decision.NO_TRADE
    return {
        "decision": overall.value,
        "counts": counts,
        "evaluations": evaluations,
        "ranked": ranked,
        "session": all_session,
    }


def _safe_candidate(candidate: Candidate) -> dict[str, Any]:
    """Serialize decision evidence without account/position identifiers."""
    value = jsonable(candidate)
    assessment = candidate.portfolio_assessment
    factor_tags = {
        item
        for item in (candidate.equity_context.sector_behavior, candidate.equity_context.factor_behavior)
        if item
    }
    value["portfolio_assessment"] = {
        "limits": jsonable(assessment.limits),
        "aggregate_risk": assessment.aggregate_risk,
        "cluster_risk": assessment.cluster_risk,
        "event_risk": assessment.event_risk,
        "sector_risk": assessment.sector_risk,
        "factor_risk": assessment.factor_risk,
        # Portfolio prose and record identifiers are deliberately transient.
        # The persisted decision needs the gate-relevant result, not account-
        # adjacent free text that may contain a symbol or position identifier.
        "deficit_elimination_verified": bool(
            assessment.deficit_elimination_rationale.strip()
        ),
        "position_count": len(assessment.positions),
        "correlation_count": len(assessment.correlations),
        "duplicate_or_correlated_expression": duplicate_expression(
            assessment, candidate.symbol, factor_tags
        ),
    }
    return cast(dict[str, Any], value)
