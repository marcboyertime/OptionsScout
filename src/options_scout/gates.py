from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from .calendar import session_status
from .engine import (
    DEFAULT_LIQUIDITY_RULES,
    LiquidityRules,
    Payoff,
    analyze_structure,
    dte,
    expiration_pnl,
    iv_crush_matrix,
    planned_exit_intrinsic_pnl,
    validate_quotes,
)
from .models import Candidate, CatalystType, Decision, GateResult, GateStatus, SensitivityCase, Side
from .portfolio import assessment_limits_pass, duplicate_expression
from .structures import classify_structure
from .volatility import weighted_metrics

ZERO = Decimal()
CAP = Decimal("1000")

GATE_IDS = (
    "options_only",
    "defined_risk",
    "fee_inclusive_1000_cap",
    "dte_30",
    "calendar_provider",
    "market_session",
    "underlying_fresh",
    "leg_quotes_fresh",
    "leg_sync",
    "legs_tradable",
    "liquidity",
    "realistic_fill",
    "one_complex_order",
    "payoff_accuracy",
    "catalyst_timing",
    "market_belief_xyz",
    "priced_outcome_supported",
    "why_wrong_supported",
    "why_not_arbitraged",
    "falsifier",
    "claim_source_quality",
    "no_material_contradiction",
    "thesis_complexity",
    "underappreciation",
    "sector_beta",
    "technical_timing",
    "robust_positive_ev",
    "model_error_sensitivity",
    "total_loss_probability",
    "event_iv_crush",
    "iv_history_honesty",
    "portfolio_context",
    "aggregate_risk",
    "cluster_risk",
    "event_risk",
    "sector_risk",
    "factor_risk",
    "correlation_duplicate",
    "exercise_style",
    "settlement_style",
    "ex_dividend",
    "assignment",
    "pin_auto_exercise",
    "adjusted_corporate_action",
    "physical_preexpiry_exit",
    "exit_plan_complete",
    "account_deficit_eliminated",
    "red_team_complete",
    "judge_survived",
    "no_material_move",
)


def _result(
    name: str, status: GateStatus, reason: str, evidence: tuple[str, ...] = (), trace: str = ""
) -> GateResult:
    return GateResult(
        name, status is GateStatus.PASS, reason, status=status, evidence_ids=evidence, trace=trace
    )


def _pass(name: str, reason: str, evidence: tuple[str, ...] = (), trace: str = "") -> GateResult:
    return _result(name, GateStatus.PASS, reason, evidence, trace)


def _fail(name: str, reason: str, evidence: tuple[str, ...] = (), trace: str = "") -> GateResult:
    return _result(name, GateStatus.FAIL, reason, evidence, trace)


def _na(name: str, reason: str, trace: str = "") -> GateResult:
    return _result(name, GateStatus.NOT_APPLICABLE, reason, trace=trace)


def _text(value: str) -> bool:
    return value.strip().casefold() not in {
        "",
        "none",
        "n/a",
        "na",
        "unknown",
        "unavailable",
        "missing",
    }


def _source_evidence(candidate: Candidate) -> tuple[str, ...]:
    return tuple(source.id for source in candidate.source_records)


def _claim_evidence(candidate: Candidate) -> tuple[str, ...]:
    return tuple(claim.id for claim in candidate.claim_records)


def _quote_result(
    name: str, quote_fail: set[str], failures: set[str], passed_reason: str
) -> GateResult:
    hit = quote_fail.intersection(failures)
    return (
        _fail(name, "quote validation failed: " + ", ".join(sorted(hit)))
        if hit
        else _pass(name, passed_reason)
    )


def _full_distribution(
    candidate: Candidate,
    payoff: Payoff,
    *,
    expiration_valuation_allowed: bool = True,
    policy_fees: Decimal = ZERO,
) -> tuple[GateResult, GateResult, GateResult, dict[str, Decimal]]:
    planned_exit = candidate.planned_exit_valuation
    if not expiration_valuation_allowed and planned_exit is None:
        reason = "planned-exit valuation unavailable: expiration-payoff scenarios cannot value a required pre-expiry exit"
        return (
            _fail("robust_positive_ev", reason),
            _fail("model_error_sensitivity", reason),
            _fail("total_loss_probability", reason),
            {},
        )
    distribution = candidate.distribution
    using_planned_exit = planned_exit is not None and not expiration_valuation_allowed
    scenarios = planned_exit.scenarios if using_planned_exit and planned_exit is not None else distribution.scenarios
    if not scenarios:
        unavailable = _fail("robust_positive_ev", "distribution scenarios unavailable")
        return (
            unavailable,
            _fail("model_error_sensitivity", "sensitivity cases unavailable"),
            _fail("total_loss_probability", "distribution scenarios unavailable"),
            {},
        )
    assert candidate.structure is not None
    if using_planned_exit and any(leg.side is Side.SELL for leg in candidate.structure.legs):
        reason = "intrinsic_close_v1 is only a conservative long-option lower bound; short-leg extrinsic close liability is unavailable"
        return (
            _fail("robust_positive_ev", reason),
            _fail("model_error_sensitivity", reason),
            _fail("total_loss_probability", reason),
            {},
        )
    # Submitted distribution figures are normalized, config-independent base
    # economics: declared structure-plan costs only. Runtime policy fees are
    # deliberately applied after that assertion, exactly once, so one valid
    # captured input remains parseable under every reviewed fee policy.
    if using_planned_exit:
        assert planned_exit is not None
        base_payoffs = [
            planned_exit_intrinsic_pnl(
                candidate.structure,
                item.underlying_spot,
                candidate.structure_plan.fees,
                candidate.structure_plan.commissions,
                candidate.structure_plan.exit_slippage,
                candidate.fill_plan.max_slippage,
                entry=candidate.structure_plan.max_acceptable_limit,
            )
            for item in planned_exit.scenarios
        ]
    else:
        scenarios = distribution.scenarios
        base_payoffs = [
            expiration_pnl(
                candidate.structure,
                item.expiration_spot,
                candidate.structure_plan.fees,
                candidate.structure_plan.commissions,
                candidate.structure_plan.exit_slippage,
                candidate.fill_plan.max_slippage,
                entry=candidate.structure_plan.max_acceptable_limit,
            )
            for item in scenarios
        ]
    # The policy fee is a non-negative fixed whole-position charge and applies
    # to every terminal P/L once, after base assertion validation.
    derived_payoffs = [item - policy_fees for item in base_payoffs]
    if not using_planned_exit and any(
        item.payoff != derived
        for item, derived in zip(distribution.scenarios, base_payoffs, strict=True)
    ):
        invalid = _fail(
            "robust_positive_ev",
            "scenario payoff does not equal exact selected-structure expiration economics",
        )
        return (
            invalid,
            _fail("model_error_sensitivity", "distribution is not auditable"),
            _fail("total_loss_probability", "distribution is not auditable"),
            {},
        )
    try:
        metrics = weighted_metrics(
            derived_payoffs,
            [item.probability for item in scenarios],
            payoff.operational_max_loss_risk or ZERO,
        )
    except ValueError as error:
        invalid = _fail("robust_positive_ev", f"invalid Decimal distribution: {error}")
        return (
            invalid,
            _fail("model_error_sensitivity", "distribution invalid"),
            _fail("total_loss_probability", "distribution invalid"),
            {},
        )
    trace = f"EV={metrics.expected_pnl}; worst10={metrics.expected_shortfall}; loss={metrics.probability_any_loss}; total={metrics.probability_total_loss}"
    ev_gate = (
        _pass(
            "robust_positive_ev",
            "conservative full-distribution EV is positive",
            tuple(item.id for item in scenarios),
            trace,
        )
        if metrics.expected_pnl > ZERO
        else _fail(
            "robust_positive_ev",
            "conservative full-distribution EV is not positive",
            tuple(item.id for item in scenarios),
            trace,
        )
    )
    cases = planned_exit.sensitivity_cases if using_planned_exit and planned_exit is not None else distribution.sensitivity_cases
    if not cases:
        sensitivity = _fail("model_error_sensitivity", "sensitivity cases unavailable")
    else:

        def adverse_ev(payoffs: list[Decimal], case: SensitivityCase) -> Decimal | None:
            # Shift actual probability mass from the best terminals to the
            # worst terminal.  This makes the supplied sensitivity auditable
            # rather than an independently asserted expected-value string.
            shift = case.probability_shift_to_worst
            probabilities = [item.probability for item in scenarios]
            payoffs = list(payoffs)
            worst_index = min(range(len(payoffs)), key=payoffs.__getitem__)
            available = sum(
                (
                    probabilities[index]
                    for index, value in enumerate(payoffs)
                    if value > payoffs[worst_index]
                ),
                ZERO,
            )
            if shift > available:
                return None
            remaining = shift
            for index in sorted(
                (index for index in range(len(payoffs)) if payoffs[index] > payoffs[worst_index]),
                key=payoffs.__getitem__,
                reverse=True,
            ):
                moved = min(probabilities[index], remaining)
                probabilities[index] -= moved
                probabilities[worst_index] += moved
                remaining -= moved
                if remaining == ZERO:
                    break
            return (
                sum(
                    (
                        value * probability
                        for value, probability in zip(payoffs, probabilities, strict=True)
                    ),
                    ZERO,
                )
                - case.additional_cost
            )

        invalid_cases: list[str] = []
        runtime_cases: dict[str, Decimal] = {}
        for case in cases:
            base = adverse_ev(base_payoffs, case)
            runtime = adverse_ev(derived_payoffs, case)
            if (
                case.model != "adverse_probability_shift_v1"
                or base is None
                or runtime is None
                or case.expected_value != base
                or runtime <= ZERO
            ):
                invalid_cases.append(case.id)
            elif runtime is not None:
                runtime_cases[case.id] = runtime
        sensitivity = (
            _fail(
                "model_error_sensitivity",
                "base sensitivity must exactly match declared-cost economics and remain positive after runtime policy fees",
                tuple(invalid_cases),
                "; ".join(
                    f"{case.id}=base:{case.expected_value};runtime:{runtime_cases.get(case.id)}"
                    for case in cases
                ),
            )
            if invalid_cases
            else _pass(
                "model_error_sensitivity",
                "documented base sensitivity remains positive after runtime policy fees",
                tuple(case.id for case in cases),
                "; ".join(
                    f"{case.id}=base:{case.expected_value};runtime:{runtime_cases[case.id]}"
                    for case in cases
                ),
            )
        )
    unusually_strong = (
        len(candidate.source_records) >= 2
        and any(source.primary for source in candidate.source_records)
        and candidate.thesis_record is not None
        and _text(candidate.thesis_record.underappreciation)
        and metrics.expected_pnl >= (payoff.operational_max_loss_risk or ZERO) * Decimal("2")
        and metrics.best >= abs(metrics.worst) * Decimal("3")
    )
    if metrics.probability_total_loss > Decimal("0.65"):
        total = _fail(
            "total_loss_probability",
            "total-loss probability exceeds the absolute 65% ceiling",
            tuple(item.id for item in scenarios),
            trace,
        )
    elif metrics.probability_total_loss <= Decimal("0.40"):
        total = _pass(
            "total_loss_probability",
            "total-loss probability <= 40%",
            tuple(item.id for item in scenarios),
            trace,
        )
    elif unusually_strong:
        total = _pass(
            "total_loss_probability",
            "total-loss probability >40% has documented primary evidence and 3:1 terminal/2x-risk asymmetry",
            _source_evidence(candidate),
            trace,
        )
    else:
        total = _fail(
            "total_loss_probability",
            "total-loss probability >40% lacks documented 3:1 terminal/2x-risk asymmetry",
            tuple(item.id for item in scenarios),
            trace,
        )
    return (
        ev_gate,
        sensitivity,
        total,
        {
            "expected_pnl": metrics.expected_pnl,
            "median_pnl": metrics.median_pnl,
            "loss_probability": metrics.probability_any_loss,
            "total_loss_probability": metrics.probability_total_loss,
            "expected_shortfall": metrics.expected_shortfall,
            "best": metrics.best,
            "worst": metrics.worst,
        },
    )


def ledger(
    candidate: Candidate,
    now: datetime,
    fees: Decimal = ZERO,
    policy_fee_per_contract: Decimal = ZERO,
    live: bool = True,
    fixture: bool = False,
    session_override: Mapping[str, object] | None = None,
    liquidity_rules: LiquidityRules = DEFAULT_LIQUIDITY_RULES,
) -> tuple[list[GateResult], Payoff | None]:
    """Return the fixed ordered hard-gate ledger, sourced only from typed evidence."""
    if candidate.structure is None or not candidate.structure.legs:
        return [_fail(name, "structure unavailable") for name in GATE_IDS], None
    if candidate.thesis_record is None:
        return [_fail(name, "typed thesis record unavailable") for name in GATE_IDS], None
    try:
        quote_payoff = analyze_structure(candidate.structure)
        operational_entry = max(
            candidate.structure_plan.max_acceptable_limit, quote_payoff.fills.realistic_limit_entry
        )
        payoff = analyze_structure(candidate.structure, operational_entry)
    except ValueError as error:
        return [_fail(name, f"payoff unavailable: {error}") for name in GATE_IDS], None
    structure = candidate.structure
    classification = classify_structure(structure)
    session = session_override or session_status(
        now, candidate.mechanics.product_calendar or "XNYS"
    )
    quote_fail = {
        item.name
        for item in validate_quotes(
            structure,
            now,
            underlying_liquidity=candidate.equity_context.underlying_liquidity,
            early_exit_required=candidate.structure_plan.exit_plan.required_before_expiry,
            liquidity_rules=liquidity_rules,
        )
        if not item.passed
    }
    shorts = [leg for leg in structure.legs if leg.side is Side.SELL]
    physical = (candidate.mechanics.settlement_style or "").casefold() == "physical" or any(
        (leg.contract.settlement_style or "").casefold() == "physical" for leg in structure.legs
    )
    verified_index_short = (
        candidate.mechanics.product_type == "index"
        and candidate.mechanics.asset_type == "index option"
    )
    european_cash_shorts = (
        bool(shorts)
        and verified_index_short
        and all(
            leg.contract.exercise_style == "European" and leg.contract.settlement_style == "cash"
            for leg in structure.legs
        )
    )
    policy_fees = policy_fee_per_contract * Decimal(
        sum(leg.ratio for leg in structure.legs) * structure.quantity
    )
    costs = (
        candidate.structure_plan.fees,
        candidate.structure_plan.commissions,
        candidate.structure_plan.exit_slippage,
        candidate.fill_plan.max_slippage,
        fees,
        policy_fees,
    )
    total_cost = (
        payoff.max_loss + sum(costs, ZERO)
        if payoff.max_loss is not None and all(cost >= ZERO for cost in costs)
        else None
    )
    if candidate.structure_plan.exit_plan.required_before_expiry:
        planned = candidate.planned_exit_valuation
        if planned is not None:
            planned_worst_risk = max(
                (
                    -planned_exit_intrinsic_pnl(
                        structure,
                        scenario.underlying_spot,
                        candidate.structure_plan.fees + policy_fees,
                        candidate.structure_plan.commissions,
                        candidate.structure_plan.exit_slippage,
                        candidate.fill_plan.max_slippage,
                        entry=candidate.structure_plan.max_acceptable_limit,
                    )
                    for scenario in planned.scenarios
                ),
                default=ZERO,
            )
            total_cost = max(total_cost or ZERO, planned_worst_risk)
    # Payoff extrema are expiration economics before operational costs.  The
    # reported operational classification is the intended whole-complex loss,
    # including every typed fee, commission and close slippage amount.
    if total_cost is not None:
        payoff = replace(payoff, operational_max_loss_risk=total_cost)
    claims = candidate.claim_records
    sources = candidate.source_records
    thesis = candidate.thesis_record
    valid_claim_links = bool(claims) and all(
        claim.source_ids and set(claim.source_ids) <= {source.id for source in sources}
        for claim in claims
    )
    primary_count = sum(source.primary for source in sources)
    complex_thesis = len(thesis.assumptions) > 3 or len(thesis.narratives) > 2
    latest_expiry = max(leg.contract.expiration for leg in structure.legs)
    event_note = tuple(
        event
        for event in candidate.volatility_context.upcoming_events
        if now < event.event_at and event.event_at.date().isoformat() <= latest_expiry
    )
    binary_catalyst_spans = (
        thesis.catalyst_type in {CatalystType.EARNINGS, CatalystType.FDA, CatalystType.MACRO}
        and thesis.catalyst_at is not None
        and now < thesis.catalyst_at
        and thesis.catalyst_at.date().isoformat() <= latest_expiry
    )
    matching_catalyst_event = tuple(
        event
        for event in event_note
        if event.event_type is thesis.catalyst_type and event.event_at == thesis.catalyst_at
    )
    risk = payoff.operational_max_loss_risk or Decimal("1000000000")
    limits = assessment_limits_pass(
        candidate.portfolio_assessment, risk, bool(event_note) or binary_catalyst_spans
    )
    factor_tags = set(
        filter(
            None,
            (candidate.equity_context.sector_behavior, candidate.equity_context.factor_behavior),
        )
    )
    duplicate = duplicate_expression(candidate.portfolio_assessment, candidate.symbol, factor_tags)
    long_physical = physical and not shorts
    # Any required early exit—not just physical delivery—needs a typed exit
    # valuation. Expiration-only scenarios cannot support ACTIONABLE EV.
    ev_gate, sensitivity_gate, total_loss_gate, _ = _full_distribution(
        candidate,
        payoff,
        expiration_valuation_allowed=not candidate.structure_plan.exit_plan.required_before_expiry,
        policy_fees=policy_fees,
    )
    all_greeks = not quote_fail.intersection({"greeks_iv"})
    crush_rows = (
        iv_crush_matrix(
            candidate.underlying,
            payoff.net_delta or ZERO,
            payoff.net_vega or ZERO,
            max(event.expected_iv_drop for event in (matching_catalyst_event or event_note)),
            implied_move_pct=candidate.volatility_context.implied_move_pct,
            net_gamma=payoff.net_gamma,
            net_theta=payoff.net_theta,
            multiplier=structure.legs[0].contract.multiplier,
            quantity=structure.quantity,
            current_signed_complex_value=payoff.fills.midpoint_entry or ZERO,
            signed_entry=candidate.structure_plan.max_acceptable_limit,
            operational_costs=(
                candidate.structure_plan.fees
                + candidate.structure_plan.commissions
                + candidate.fill_plan.max_slippage
                + policy_fees
                + candidate.structure_plan.exit_slippage
            ),
        )
        if candidate.underlying is not None
        and payoff.net_delta is not None
        and payoff.net_vega is not None
        and payoff.net_gamma is not None
        and payoff.net_theta is not None
        and payoff.fills.midpoint_entry is not None
        and abs(payoff.net_delta) >= Decimal("0.0001")
        and candidate.volatility_context.implied_move_pct > ZERO
        and (matching_catalyst_event or event_note)
        else []
    )
    crush_pnls = [Decimal(str(row["estimated_pnl"])) for row in crush_rows]
    conservative_crush = next(
        (
            Decimal(str(row["estimated_pnl"]))
            for row in crush_rows
            if row["spot_case"] == "smaller_than_implied"
            and row["time_case"] == "delayed"
            and row["iv_case"] == "severe"
        ),
        None,
    )
    exit_plan = candidate.structure_plan.exit_plan
    canonical_short = not shorts or classification.supported
    no_move = abs(candidate.equity_context.material_move_pct) <= Decimal("0.02")
    values: dict[str, GateResult] = {
        "options_only": _pass(
            "options_only",
            "typed equity context confirms options availability",
            tuple(leg.contract.id for leg in structure.legs),
        )
        if candidate.equity_context.options_available
        else _fail("options_only", "typed equity context does not confirm options availability"),
        "defined_risk": _pass(
            "defined_risk",
            "canonical topology has finite exact expiration loss",
            trace=f"max_loss={payoff.max_loss}",
        )
        if classification.supported and payoff.max_loss is not None
        else _fail(
            "defined_risk",
            "noncanonical/unsupported or unbounded payoff",
            trace=f"structure={classification.kind}; {classification.reason}; max_loss={payoff.max_loss}",
        ),
        "fee_inclusive_1000_cap": _pass(
            "fee_inclusive_1000_cap",
            "fee-inclusive intended worst case <= $1,000",
            trace=f"{total_cost}",
        )
        if total_cost is not None and total_cost <= CAP
        else _fail(
            "fee_inclusive_1000_cap",
            "fee-inclusive intended worst case exceeds $1,000 or is unbounded",
            trace=f"{total_cost}",
        ),
        "dte_30": _pass(
            "dte_30",
            "all contract expirations are 0..30 days",
            tuple(leg.contract.id for leg in structure.legs),
            ",".join(str(dte(leg.contract.expiration, now.date())) for leg in structure.legs),
        )
        if all(0 <= dte(leg.contract.expiration, now.date()) <= 30 for leg in structure.legs)
        else _fail("dte_30", "contract expiration outside 0..30 days"),
        "calendar_provider": _pass(
            "calendar_provider",
            "typed product calendar resolved",
            trace=str(session.get("open", "")),
        )
        if bool(session.get("available"))
        else _fail(
            "calendar_provider",
            str(session.get("reason", "calendar unavailable")),
            trace=candidate.mechanics.product_calendar,
        ),
        "market_session": _pass("market_session", "regular product session")
        if bool(session.get("regular"))
        else _fail("market_session", str(session.get("session", session.get("reason", "closed")))),
        "underlying_fresh": _pass(
            "underlying_fresh",
            "underlying quote <=90 seconds",
            (candidate.quote_provenance.source_id,),
            f"as_of={candidate.underlying_as_of.isoformat()}",
        )
        if candidate.underlying is not None
        and candidate.underlying_as_of is not None
        and candidate.underlying_as_of <= now
        and (now - candidate.underlying_as_of).total_seconds() <= 90
        else _fail("underlying_fresh", "underlying missing, future-dated, or stale"),
        "leg_quotes_fresh": _quote_result(
            "leg_quotes_fresh", quote_fail, {"freshness"}, "all option quotes <=90 seconds"
        ),
        "leg_sync": _quote_result(
            "leg_sync", quote_fail, {"synchronization"}, "leg quote timestamps are synchronized"
        ),
        "legs_tradable": _quote_result(
            "legs_tradable",
            quote_fail,
            {"contract_integrity", "duplicate_contract", "greeks_iv", "quote_domain"},
            "tradable, unadjusted standard contracts with valid IV/Greeks",
        ),
        "liquidity": _quote_result(
            "liquidity",
            quote_fail,
            {"liquidity"},
            "premium-band absolute/relative spread, volume/OI, underlying-liquidity, leg-count, and early-exit tests pass",
        ),
        "realistic_fill": _pass(
            "realistic_fill",
            "signed whole-complex realistic cash, declared target, and maximum are consistent",
            tuple(leg.contract.id for leg in structure.legs),
            f"realistic_cash={payoff.fills.realistic_limit_entry}; fill_target={candidate.fill_plan.limit}; entry_target={candidate.structure_plan.entry_limit}; max_cash={candidate.structure_plan.max_acceptable_limit}",
        )
        if not quote_fail.intersection({"two_sided_quote"})
        and candidate.fill_plan.order_type.casefold() == "limit"
        and candidate.structure_plan.one_complex_order
        and candidate.fill_plan.limit == candidate.structure_plan.entry_limit
        and payoff.fills.realistic_limit_entry
        <= candidate.structure_plan.entry_limit
        <= candidate.structure_plan.max_acceptable_limit
        else _fail(
            "realistic_fill",
            "signed whole-complex fill must satisfy realistic_entry <= target == declared fill <= maximum",
        ),
        "one_complex_order": _pass(
            "one_complex_order", "single-leg or declared one complex net-limit order"
        )
        if len(structure.legs) == 1 or candidate.structure_plan.one_complex_order
        else _fail("one_complex_order", "multi-leg structure may not be legged"),
        "payoff_accuracy": _pass(
            "payoff_accuracy",
            "exact piecewise payoff evaluated at S=0, strikes and right tail",
            tuple(leg.contract.id for leg in structure.legs),
            f"knots={payoff.table}; breakevens={payoff.breakevens}; slope tail classified",
        )
        if classification.supported
        else _fail(
            "payoff_accuracy",
            "declared structure cannot be exactly classified",
            trace=classification.reason,
        ),
        "catalyst_timing": _pass(
            "catalyst_timing",
            "typed catalyst classification/date and timing trigger present",
            trace=f"{thesis.catalyst_type}:{thesis.catalyst_at}; {thesis.timing_trigger}",
        )
        if _text(thesis.catalyst)
        and thesis.catalyst_at is not None
        and _text(thesis.timing_trigger)
        else _fail(
            "catalyst_timing", "typed catalyst classification/date or timing trigger missing"
        ),
        "market_belief_xyz": _pass(
            "market_belief_xyz",
            "implied probability range and outcome supplied",
            trace=f"{thesis.implied_probability_low}..{thesis.implied_probability_high}",
        )
        if thesis.implied_probability_low < thesis.implied_probability_high
        and _text(thesis.outcome)
        else _fail("market_belief_xyz", "market belief probability range/outcome missing"),
        "priced_outcome_supported": _pass(
            "priced_outcome_supported",
            "claim evidence supports outcome",
            _claim_evidence(candidate),
        )
        if valid_claim_links
        else _fail("priced_outcome_supported", "claims/sources are missing or unlinked"),
        "why_wrong_supported": _pass(
            "why_wrong_supported", "typed disconfirming case supplied", trace=thesis.why_wrong
        )
        if _text(thesis.why_wrong)
        else _fail("why_wrong_supported", "why-wrong case missing"),
        "why_not_arbitraged": _pass(
            "why_not_arbitraged",
            "typed non-arbitrage rationale supplied",
            trace=thesis.why_not_arbitraged,
        )
        if _text(thesis.why_not_arbitraged)
        else _fail("why_not_arbitraged", "non-arbitrage rationale missing"),
        "falsifier": _pass("falsifier", "typed falsifier supplied", trace=thesis.falsifier)
        if _text(thesis.falsifier)
        else _fail("falsifier", "falsifier missing"),
        "claim_source_quality": _pass(
            "claim_source_quality",
            "linked sources include required primary support",
            _source_evidence(candidate),
        )
        if valid_claim_links and primary_count >= (2 if complex_thesis else 1)
        else _fail(
            "claim_source_quality",
            "source links/primary support insufficient for thesis complexity",
        ),
        "no_material_contradiction": _pass(
            "no_material_contradiction", "typed equity context has no material contradictions"
        )
        if not candidate.equity_context.contradictions
        else _fail(
            "no_material_contradiction",
            "material contradiction recorded",
            candidate.equity_context.contradictions,
        ),
        "thesis_complexity": _pass(
            "thesis_complexity",
            "complexity supported by source/claim depth",
            _claim_evidence(candidate),
        )
        if not complex_thesis or (len(claims) >= 3 and primary_count >= 2)
        else _fail(
            "thesis_complexity", "complex thesis requires >=3 claims and >=2 primary sources"
        ),
        "underappreciation": _pass(
            "underappreciation",
            "typed underappreciation rationale linked to claims",
            _claim_evidence(candidate),
            thesis.underappreciation,
        )
        if _text(thesis.underappreciation) and valid_claim_links
        else _fail("underappreciation", "underappreciation rationale/evidence missing"),
        "sector_beta": _pass(
            "sector_beta",
            "factor-adjusted sector exposure assessed",
            trace=f"beta={candidate.equity_context.sector_beta}",
        )
        if candidate.equity_context.factor_adjusted
        and _text(candidate.equity_context.sector_behavior)
        and _text(candidate.equity_context.factor_behavior)
        else _fail("sector_beta", "sector/factor-adjusted assessment missing"),
        "technical_timing": _pass(
            "technical_timing",
            "typed technical trigger supplied",
            trace=candidate.equity_context.technical_trigger or "",
        )
        if candidate.equity_context.technical_trigger
        and _text(candidate.equity_context.technical_trigger)
        else _fail("technical_timing", "typed technical trigger missing"),
        "robust_positive_ev": ev_gate,
        "model_error_sensitivity": sensitivity_gate,
        "total_loss_probability": total_loss_gate,
        "event_iv_crush": _pass(
            "event_iv_crush",
            "upcoming binary event has typed expected IV-drop provenance and complete Greeks",
            tuple(f"event:{event.event_type}" for event in (matching_catalyst_event or event_note)),
            "; ".join(
                f"drop={event.expected_iv_drop}; {event.provenance}"
                for event in (matching_catalyst_event or event_note)
            ),
        )
        if (matching_catalyst_event if binary_catalyst_spans else event_note)
        and all_greeks
        and payoff.net_gamma is not None
        and payoff.net_theta is not None
        and candidate.volatility_context.implied_move_pct > ZERO
        and any(value > ZERO for value in crush_pnls)
        and conservative_crush is not None
        and conservative_crush > ZERO
        else (
            _fail(
                "event_iv_crush",
                "event IV-crush matrix lacks complete unit-consistent inputs, has no positive case, or its delayed direction-correct smaller-than-implied severe-crush case is nonpositive after all costs",
                trace=f"conservative={conservative_crush}; rows={crush_rows}",
            )
            if binary_catalyst_spans or event_note
            else _na(
                "event_iv_crush",
                "no upcoming binary event spans expiration; historical events remain historical",
            )
        ),
        "iv_history_honesty": _na(
            "iv_history_honesty",
            "schema provides one local IV snapshot, not historical snapshots; rank/percentile is not claimed",
            candidate.volatility_context.iv_snapshot.as_of.isoformat(),
        ),
        "portfolio_context": _pass(
            "portfolio_context",
            "typed portfolio limits and causal correlation context supplied",
            trace=(
                f"positions={len(candidate.portfolio_assessment.positions)}; "
                f"correlations={len(candidate.portfolio_assessment.correlations)}; "
                "deficit rationale verified"
            ),
        )
        if _text(candidate.portfolio_assessment.deficit_elimination_rationale)
        else _fail("portfolio_context", "typed portfolio/deficit context missing"),
        "aggregate_risk": _pass(
            "aggregate_risk", "incremental worst case fits aggregate limit", trace=str(risk)
        )
        if limits["aggregate"]
        else _fail("aggregate_risk", "aggregate worst case exceeds typed limit", trace=str(risk)),
        "cluster_risk": _pass(
            "cluster_risk", "cluster worst case fits typed limit", trace=str(risk)
        )
        if limits["cluster"]
        else _fail("cluster_risk", "cluster worst case exceeds typed limit", trace=str(risk)),
        "event_risk": _pass("event_risk", "event bucket fits typed limit", trace=str(risk))
        if limits["event"]
        else _fail("event_risk", "event bucket exceeds typed limit", trace=str(risk)),
        "sector_risk": _pass("sector_risk", "sector bucket fits typed limit", trace=str(risk))
        if limits["sector"]
        else _fail("sector_risk", "sector bucket exceeds typed limit", trace=str(risk)),
        "factor_risk": _pass("factor_risk", "factor bucket fits typed limit", trace=str(risk))
        if limits["factor"]
        else _fail("factor_risk", "factor bucket exceeds typed limit", trace=str(risk)),
        "correlation_duplicate": _fail(
            "correlation_duplicate",
            "duplicate or strongly correlated factor expression",
            trace=f"correlations={len(candidate.portfolio_assessment.correlations)}",
        )
        if duplicate
        else _pass(
            "correlation_duplicate", "no same-symbol/strongly-correlated duplicate expression"
        ),
        "exercise_style": _pass(
            "exercise_style",
            "canonical short structure is verified index with all legs European cash-settled"
            if shorts
            else "no short exercise path",
        )
        if not shorts or (canonical_short and european_cash_shorts)
        else _fail(
            "exercise_style",
            "short legs require a canonical covered topology plus verified index/all-leg European cash settlement",
        ),
        "settlement_style": _pass(
            "settlement_style",
            "canonical short structure is verified index with all legs European cash-settled"
            if shorts
            else "long-only settlement classified",
        )
        if not shorts or (canonical_short and european_cash_shorts)
        else _fail(
            "settlement_style",
            "noncanonical/uncovered or ETF/equity-labelled/incompatible short is prohibited",
        ),
        "ex_dividend": _na("ex_dividend", "cash-settled index has no equity dividend delivery")
        if not physical
        else (
            _pass(
                "ex_dividend",
                "physical product explicitly assessed for ex-dividend",
                trace=str(candidate.mechanics.ex_dividend_date),
            )
            if candidate.mechanics.ex_dividend_date is not None
            and candidate.mechanics.ex_dividend_amount is not None
            else _fail("ex_dividend", "physical product lacks explicit ex-dividend assessment")
        ),
        "assignment": _pass("assignment", "no short physical/early-assignment path")
        if not shorts or (canonical_short and european_cash_shorts)
        else _fail(
            "assignment", "noncanonical/uncovered or incompatible short assignment path exists"
        ),
        "pin_auto_exercise": _pass(
            "pin_auto_exercise",
            "pin/auto-exercise mechanics explicitly documented",
            trace=f"pin={candidate.mechanics.pin_risk}; auto={candidate.mechanics.auto_exercise}",
        )
        if _text(candidate.mechanics.pin_risk) and _text(candidate.mechanics.auto_exercise)
        else _fail("pin_auto_exercise", "pin/auto-exercise mechanics missing"),
        "adjusted_corporate_action": _pass(
            "adjusted_corporate_action",
            "unadjusted contracts and explicit corporate-action assessment",
            trace=candidate.mechanics.corporate_action,
        )
        if not any(leg.contract.adjusted for leg in structure.legs)
        and _text(candidate.mechanics.corporate_action)
        else _fail(
            "adjusted_corporate_action", "adjusted contract/corporate-action assessment failure"
        ),
        "physical_preexpiry_exit": _pass(
            "physical_preexpiry_exit",
            "long physical option has required pre-expiry exit and >=1 trading-day buffer",
            trace=str(exit_plan.close_buffer_days),
        )
        if long_physical and exit_plan.required_before_expiry and exit_plan.close_buffer_days >= 1
        else (
            _fail(
                "physical_preexpiry_exit",
                "physical option requires verified pre-expiry exit with >=1 day buffer",
            )
            if physical
            else _na("physical_preexpiry_exit", "cash-settled structure")
        ),
        "exit_plan_complete": _pass(
            "exit_plan_complete",
            "typed exit contingencies and explicit no-auto-roll policy are complete",
        )
        if all(
            _text(str(value))
            for key, value in exit_plan.__dict__.items()
            if key not in {"required_before_expiry", "close_buffer_days", "time_exit_at"}
        )
        and "new full analysis required" in str(exit_plan.__dict__["roll_policy"]).casefold()
        else _fail("exit_plan_complete", "exit plan is incomplete or permits automatic rolling"),
        "account_deficit_eliminated": _pass(
            "account_deficit_eliminated",
            "operational max loss is finite and typed deficit rationale is present",
            trace=f"operational={payoff.operational_max_loss_risk}",
        )
        if payoff.operational_max_loss_risk is not None
        and total_cost is not None
        and total_cost <= CAP
        and _text(candidate.portfolio_assessment.deficit_elimination_rationale)
        and (not shorts or (canonical_short and european_cash_shorts))
        else _fail(
            "account_deficit_eliminated",
            "noncanonical/assignment/physical/unbounded/fee cap deficit path remains",
        ),
        "red_team_complete": _pass("red_team_complete", "all skeptic fields are populated")
        if all(_text(value) for value in candidate.skeptic.__dict__.values())
        else _fail("red_team_complete", "one or more skeptic fields missing"),
        "judge_survived": _pass(
            "judge_survived",
            "explicit reject-resistant judge verdict",
            trace=candidate.judge.reason,
        )
        if candidate.judge.verdict.casefold().strip()
        in {"survived", "survives", "reject-resistant", "reject_resistant"}
        and _text(candidate.judge.reason)
        else _fail(
            "judge_survived",
            "judge has not explicitly survived red-team review",
            trace=candidate.judge.verdict,
        ),
        "no_material_move": _pass(
            "no_material_move",
            "typed material move <=2%",
            trace=str(candidate.equity_context.material_move_pct),
        )
        if no_move
        else _fail(
            "no_material_move",
            "typed material move >2%; reanalysis required",
            trace=str(candidate.equity_context.material_move_pct),
        ),
    }
    return [values[name] for name in GATE_IDS], payoff


def decision_from_ledger(
    items: list[GateResult],
    live: bool,
    fixture: bool,
    watch_trigger: str | None = None,
    invalidated: bool = False,
) -> Decision:
    by_name = {item.name: item for item in items}
    if invalidated or by_name.get("no_material_move", _pass("", "")).status is GateStatus.FAIL:
        return Decision.INVALIDATED
    if fixture or not live:
        return Decision.DATA_INSUFFICIENT
    failed = {item.name for item in items if item.status is GateStatus.FAIL}
    if failed.intersection(
        {"calendar_provider", "market_session", "underlying_fresh", "leg_quotes_fresh", "leg_sync"}
    ):
        return Decision.MARKET_CLOSED_OR_STALE
    if watch_trigger and failed == {"catalyst_timing"}:
        return Decision.WATCH
    missing = {
        "priced_outcome_supported",
        "claim_source_quality",
        "portfolio_context",
        "robust_positive_ev",
        "model_error_sensitivity",
        "total_loss_probability",
    }
    if failed.intersection(missing):
        return Decision.DATA_INSUFFICIENT
    return Decision.NO_TRADE if failed else Decision.ACTIONABLE
