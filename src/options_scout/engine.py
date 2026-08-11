from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

from .models import Candidate, Decision, GateResult, GateStatus, Leg, OptionType, Side, Structure

ZERO = Decimal()


@dataclass(frozen=True)
class FillEconomics:
    """Cash values for the whole complex, including ratio, multiplier and quantity."""

    natural_entry: Decimal
    midpoint_entry: Decimal | None
    realistic_limit_entry: Decimal
    expected_exit_slippage: Decimal


@dataclass(frozen=True)
class LiquidityBand:
    max_premium: Decimal | None
    max_relative_spread: Decimal
    max_absolute_spread: Decimal


@dataclass(frozen=True)
class LiquidityRules:
    underlying_minimum_usd: Decimal
    single_leg_min_volume: int
    single_leg_min_open_interest: int
    complex_or_early_exit_min_volume: int
    complex_or_early_exit_min_open_interest: int
    premium_bands: tuple[LiquidityBand, ...]


DEFAULT_LIQUIDITY_RULES = LiquidityRules(
    Decimal("100000"),
    1,
    25,
    10,
    100,
    (
        LiquidityBand(Decimal("1"), Decimal("0.25"), Decimal("0.20")),
        LiquidityBand(Decimal("5"), Decimal("0.20"), Decimal("0.50")),
        LiquidityBand(None, Decimal("0.15"), Decimal("1.00")),
    ),
)


def parse_liquidity_rules(raw: object) -> LiquidityRules:
    """Parse the policy's finite, ordered liquidity thresholds fail-closed."""
    if not isinstance(raw, dict):
        raise ValueError("liquidity policy must be an object")
    fields = {
        "underlying_minimum_usd",
        "single_leg_min_volume",
        "single_leg_min_open_interest",
        "complex_or_early_exit_min_volume",
        "complex_or_early_exit_min_open_interest",
        "premium_bands",
    }
    if set(raw) != fields or not isinstance(raw["premium_bands"], list) or not raw["premium_bands"]:
        raise ValueError("liquidity policy fields are invalid")

    def decimal(field: str) -> Decimal:
        value = raw[field]
        if not isinstance(value, str):
            raise ValueError(f"liquidity {field} must be an exact Decimal string")
        parsed = Decimal(value)
        if not parsed.is_finite() or parsed < ZERO:
            raise ValueError(f"liquidity {field} cannot be negative or non-finite")
        return parsed

    def integer(field: str) -> int:
        value = raw[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"liquidity {field} must be a non-negative integer")
        return value

    bands: list[LiquidityBand] = []
    previous = ZERO
    for index, item in enumerate(raw["premium_bands"]):
        if not isinstance(item, dict) or set(item) != {
            "max_premium",
            "max_relative_spread",
            "max_absolute_spread",
        }:
            raise ValueError("liquidity premium band fields are invalid")
        maximum = item["max_premium"]
        if maximum is None:
            if index != len(raw["premium_bands"]) - 1:
                raise ValueError("only the final liquidity band may be unbounded")
            parsed_maximum = None
        elif isinstance(maximum, str):
            parsed_maximum = Decimal(maximum)
            if not parsed_maximum.is_finite() or parsed_maximum <= previous:
                raise ValueError("liquidity premium bands must be strictly increasing")
            previous = parsed_maximum
        else:
            raise ValueError("liquidity max_premium must be Decimal string or null")
        relative = item["max_relative_spread"]
        absolute = item["max_absolute_spread"]
        if not isinstance(relative, str) or not isinstance(absolute, str):
            raise ValueError("liquidity spread caps must be exact Decimal strings")
        relative_value, absolute_value = Decimal(relative), Decimal(absolute)
        if (
            not relative_value.is_finite()
            or not absolute_value.is_finite()
            or relative_value <= ZERO
            or absolute_value <= ZERO
        ):
            raise ValueError("liquidity spread caps must be positive finite values")
        bands.append(LiquidityBand(parsed_maximum, relative_value, absolute_value))
    return LiquidityRules(
        decimal("underlying_minimum_usd"),
        integer("single_leg_min_volume"),
        integer("single_leg_min_open_interest"),
        integer("complex_or_early_exit_min_volume"),
        integer("complex_or_early_exit_min_open_interest"),
        tuple(bands),
    )


@dataclass(frozen=True)
class Payoff:
    entry: Decimal
    max_loss: Decimal | None
    max_gain: Decimal | None
    breakevens: tuple[Decimal, ...]
    table: tuple[tuple[Decimal, Decimal], ...]
    net_delta: Decimal | None
    net_vega: Decimal | None
    fills: FillEconomics
    theoretical_max_loss: Decimal | None
    operational_max_loss_risk: Decimal | None
    assignment_risk: str
    physical_settlement_risk: str
    possibility_of_account_deficit: str
    net_gamma: Decimal | None = None
    net_theta: Decimal | None = None


def _multiplier(structure: Structure) -> int:
    multipliers = {leg.contract.multiplier for leg in structure.legs}
    if len(multipliers) != 1 or next(iter(multipliers)) <= 0:
        raise ValueError("structure contracts must use one positive compatible multiplier")
    return next(iter(multipliers))


def _cash_scale(structure: Structure) -> Decimal:
    return Decimal(_multiplier(structure) * structure.quantity)


def _price(leg: Leg, mode: Literal["natural", "mid", "realistic"]) -> Decimal | None:
    quote = leg.contract.quote
    if mode == "natural":
        return quote.ask if leg.side is Side.BUY else quote.bid
    if quote.bid is None or quote.ask is None:
        return None
    midpoint = (quote.bid + quote.ask) / Decimal("2")
    if mode == "mid":
        return midpoint
    # A conservative executable limit gives up half the distance from mid to natural.
    return (
        (midpoint + quote.ask) / Decimal("2")
        if leg.side is Side.BUY
        else (midpoint + quote.bid) / Decimal("2")
    )


def fill_economics(structure: Structure) -> FillEconomics:
    values: dict[str, Decimal | None] = {}
    for mode in ("natural", "mid", "realistic"):
        net = ZERO
        for leg in structure.legs:
            price = _price(leg, mode)
            if price is None:
                values[mode] = None
                break
            net += price * leg.ratio * (1 if leg.side is Side.BUY else -1)
        else:
            values[mode] = net * _cash_scale(structure)
    natural, realistic = values["natural"], values["realistic"]
    if natural is None or realistic is None:
        raise ValueError("natural and realistic complex fills require two-sided quotes")
    return FillEconomics(natural, values["mid"], realistic, abs(natural - realistic))


def _intrinsic_per_share(leg: Leg, spot: Decimal) -> Decimal:
    intrinsic = (
        max(ZERO, spot - leg.contract.strike)
        if leg.contract.option_type is OptionType.CALL
        else max(ZERO, leg.contract.strike - spot)
    )
    return intrinsic * leg.ratio * (1 if leg.side is Side.BUY else -1)


def payoff_at_expiration(
    structure: Structure, spot: Decimal, entry: Decimal | None = None
) -> Decimal:
    if spot < ZERO:
        raise ValueError("expiration payoff domain is spot >= 0")
    cost = entry if entry is not None else fill_economics(structure).realistic_limit_entry
    return (
        sum((_intrinsic_per_share(leg, spot) for leg in structure.legs), ZERO)
        * _cash_scale(structure)
        - cost
    )


def expiration_pnl(
    structure: Structure,
    spot: Decimal,
    fees: Decimal = ZERO,
    commissions: Decimal = ZERO,
    exit_slippage: Decimal = ZERO,
    entry_slippage: Decimal = ZERO,
    *,
    entry: Decimal | None = None,
) -> Decimal:
    """Whole-complex expiration cash P/L at a declared entry, including costs."""
    if min(fees, commissions, exit_slippage, entry_slippage) < ZERO:
        raise ValueError("operational costs cannot be negative")
    return (
        payoff_at_expiration(structure, spot, entry)
        - fees
        - commissions
        - exit_slippage
        - entry_slippage
    )


def planned_exit_intrinsic_pnl(
    structure: Structure,
    underlying_spot: Decimal,
    fees: Decimal = ZERO,
    commissions: Decimal = ZERO,
    exit_slippage: Decimal = ZERO,
    entry_slippage: Decimal = ZERO,
    *,
    entry: Decimal | None = None,
) -> Decimal:
    """Conservative pre-expiry close trace using only intrinsic liquidation.

    This deliberately does not trust a caller supplied P/L or borrow the
    expiration distribution.  It values an immediate close at the scenario
    spot with zero remaining time value, then subtracts every typed cost.
    """
    return expiration_pnl(
        structure,
        underlying_spot,
        fees,
        commissions,
        exit_slippage,
        entry_slippage,
        entry=entry,
    )


def _slope(structure: Structure, probe: Decimal) -> Decimal:
    """The cash payoff slope on an open segment containing ``probe``."""
    per_share = ZERO
    for leg in structure.legs:
        sign = Decimal(leg.ratio if leg.side is Side.BUY else -leg.ratio)
        if leg.contract.option_type is OptionType.CALL and probe > leg.contract.strike:
            per_share += sign
        elif leg.contract.option_type is OptionType.PUT and probe < leg.contract.strike:
            per_share -= sign
    return per_share * _cash_scale(structure)


def _greek(structure: Structure, name: str) -> Decimal | None:
    values = [getattr(leg.contract.quote, name) for leg in structure.legs]
    if any(value is None for value in values):
        return None
    return sum(
        (
            Decimal(value) * leg.ratio * (1 if leg.side is Side.BUY else -1) * structure.quantity
            for leg, value in zip(structure.legs, values, strict=True)
        ),
        ZERO,
    )


def analyze_structure(structure: Structure, entry: Decimal | None = None) -> Payoff:
    """Exact expiration payoff extrema/roots at an explicit whole-complex entry.

    ``fills`` always preserves quote-derived realistic economics.  Supplying an
    entry is used for the operational maximum-permitted limit, while
    ``theoretical_max_loss`` remains quote-derived at the realistic fill.
    """
    if not structure.legs or structure.quantity < 1:
        raise ValueError("a structure needs positive integer quantity")
    fills = fill_economics(structure)
    selected_entry = fills.realistic_limit_entry if entry is None else entry
    strikes = tuple(sorted({leg.contract.strike for leg in structure.legs}))
    knots = (ZERO, *strikes)
    values = tuple((spot, payoff_at_expiration(structure, spot, selected_entry)) for spot in knots)
    right_slope = _slope(structure, strikes[-1] + Decimal("1"))
    bounded_loss, bounded_gain = right_slope >= ZERO, right_slope <= ZERO
    max_loss = max(ZERO, -min(value for _, value in values)) if bounded_loss else None
    max_gain = max(ZERO, max(value for _, value in values)) if bounded_gain else None
    roots: list[Decimal] = [spot for spot, value in values if value == ZERO]
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        value = payoff_at_expiration(structure, left, selected_entry)
        slope = _slope(structure, (left + right) / Decimal("2"))
        if slope:
            root = left - value / slope
            if left < root < right:
                roots.append(root)
    # The final ray is part of the domain too: long calls often break even here.
    last = strikes[-1]
    last_value = payoff_at_expiration(structure, last, selected_entry)
    if right_slope:
        root = last - last_value / right_slope
        if root > last:
            roots.append(root)
    shorts = [leg for leg in structure.legs if leg.side is Side.SELL]
    physical = any(
        (leg.contract.settlement_style or "").casefold() == "physical" for leg in structure.legs
    )
    short_physical = bool(shorts and physical)
    assignment = "PRESENT" if short_physical else "NONE_VERIFIED"
    quote_values = tuple(
        payoff_at_expiration(structure, spot, fills.realistic_limit_entry) for spot in knots
    )
    theoretical_max_loss = max(ZERO, -min(quote_values)) if bounded_loss else None
    operational = None if short_physical else max_loss
    deficit = "POSSIBLE" if operational is None else "ELIMINATED"
    return Payoff(
        selected_entry,
        max_loss,
        max_gain,
        tuple(sorted(set(roots))),
        values,
        _greek(structure, "delta"),
        _greek(structure, "vega"),
        fills,
        theoretical_max_loss,
        operational,
        assignment,
        "PRESENT" if physical else "NONE_VERIFIED",
        deficit,
        _greek(structure, "gamma"),
        _greek(structure, "theta"),
    )


def dte(expiration: str, today: date | None = None) -> int:
    return (date.fromisoformat(expiration) - (today or datetime.now(UTC).date())).days


def validate_quotes(
    structure: Structure,
    now: datetime,
    max_age_seconds: int = 90,
    sync_seconds: int = 15,
    underlying_liquidity: Decimal | None = None,
    early_exit_required: bool = False,
    liquidity_rules: LiquidityRules = DEFAULT_LIQUIDITY_RULES,
) -> list[GateResult]:
    """Validate each quote once; callers map these concrete failures into the stable ledger."""
    results: list[GateResult] = []
    seen: set[str] = set()
    times: list[datetime] = []
    for leg in structure.legs:
        contract, quote = leg.contract, leg.contract.quote
        domain_nonfinite = not contract.strike.is_finite() or any(
            value is not None and not value.is_finite()
            for value in (
                quote.bid,
                quote.ask,
                quote.mark,
                quote.iv,
                quote.delta,
                quote.gamma,
                quote.theta,
                quote.vega,
            )
        )
        domain_invalid = (
            domain_nonfinite
            or contract.strike <= ZERO
            or any(
                value is not None and value < ZERO for value in (quote.bid, quote.ask, quote.mark)
            )
            or (quote.bid is not None and quote.ask is not None and quote.bid > quote.ask)
            or (
                quote.mark is not None
                and quote.bid is not None
                and quote.ask is not None
                and not quote.bid <= quote.mark <= quote.ask
            )
            or (quote.iv is not None and quote.iv <= ZERO)
            or (
                quote.delta is not None
                and (quote.delta < Decimal("-1") or quote.delta > Decimal("1"))
            )
            or (
                quote.delta is not None
                and contract.option_type is OptionType.CALL
                and quote.delta < ZERO
            )
            or (
                quote.delta is not None
                and contract.option_type is OptionType.PUT
                and quote.delta > ZERO
            )
            or (quote.gamma is not None and quote.gamma < ZERO)
            or (quote.vega is not None and quote.vega < ZERO)
            or (quote.theta is not None and not quote.theta.is_finite())
        )
        if domain_invalid:
            results.append(
                GateResult(
                    "quote_domain", False, "impossible quote/volatility domain", trace=contract.id
                )
            )
        if contract.id in seen:
            results.append(
                GateResult("duplicate_contract", False, "duplicate contract", trace=contract.id)
            )
        seen.add(contract.id)
        if not contract.tradable or contract.adjusted or contract.multiplier != 100:
            results.append(
                GateResult(
                    "contract_integrity",
                    False,
                    "untradable, adjusted, or nonstandard multiplier",
                    trace=contract.id,
                )
            )
        if quote.bid is None or quote.ask is None or quote.bid <= ZERO or quote.ask <= ZERO:
            results.append(
                GateResult("two_sided_quote", False, "missing or zero bid/ask", trace=contract.id)
            )
        elif quote.bid > quote.ask:
            results.append(
                GateResult("two_sided_quote", False, "crossed market", trace=contract.id)
            )
        else:
            premium = (quote.ask + quote.bid) / Decimal("2")
            spread = quote.ask - quote.bid
            band = next(
                (
                    item
                    for item in liquidity_rules.premium_bands
                    if item.max_premium is None or premium <= item.max_premium
                ),
                None,
            )
            if band is None:
                raise ValueError("liquidity policy has no terminal premium band")
            relative_cap, absolute_cap = band.max_relative_spread, band.max_absolute_spread
            complex_or_early = len(structure.legs) > 1 or early_exit_required
            minimum_oi = (
                liquidity_rules.complex_or_early_exit_min_open_interest
                if complex_or_early
                else liquidity_rules.single_leg_min_open_interest
            )
            minimum_volume = (
                liquidity_rules.complex_or_early_exit_min_volume
                if complex_or_early
                else liquidity_rules.single_leg_min_volume
            )
            if (
                spread / premium > relative_cap
                or spread > absolute_cap
                or quote.open_interest is None
                or quote.open_interest < minimum_oi
                or quote.volume is None
                or quote.volume < minimum_volume
                or (
                    underlying_liquidity is not None
                    and underlying_liquidity < liquidity_rules.underlying_minimum_usd
                )
            ):
                results.append(
                    GateResult(
                        "liquidity",
                        False,
                        "premium-band spread, volume/OI, underlying-liquidity, or early-exit liquidity test failed",
                        trace=contract.id,
                    )
                )
        if any(
            getattr(quote, field) is None for field in ("iv", "delta", "gamma", "theta", "vega")
        ):
            results.append(GateResult("greeks_iv", False, "missing IV or Greek", trace=contract.id))
        if quote.as_of > now or (now - quote.as_of).total_seconds() > max_age_seconds:
            results.append(GateResult("freshness", False, "stale option quote", trace=contract.id))
        times.append(quote.as_of)
    if times and (max(times) - min(times)).total_seconds() > sync_seconds:
        results.append(GateResult("synchronization", False, "leg timestamps are not synchronized"))
    return results


def implied_move(spot: Decimal, atm_straddle: Decimal) -> Decimal:
    if spot <= ZERO or atm_straddle < ZERO:
        raise ValueError("invalid implied move inputs")
    return atm_straddle / spot


def iv_crush_matrix(
    spot: Decimal,
    net_delta: Decimal,
    net_vega: Decimal,
    event_iv_drop: Decimal,
    *,
    implied_move_pct: Decimal | None = None,
    net_gamma: Decimal | None = None,
    net_theta: Decimal | None = None,
    multiplier: int = 100,
    quantity: int = 1,
    current_signed_complex_value: Decimal = ZERO,
    signed_entry: Decimal = ZERO,
    operational_costs: Decimal = ZERO,
) -> list[dict[str, str]]:
    """Direction-aware, unit-consistent post-event Greek stress matrix.

    The compatibility defaults retain a diagnostic matrix for legacy callers;
    actionability callers must provide all typed inputs and costs. ``theta``
    is applied for elapsed days, gamma uses the square of the dollar spot
    change, and all Greek outputs are converted exactly once by multiplier and
    complex quantity is already included in the aggregate Greeks produced by
    ``_greek``; this function scales position Greeks by multiplier exactly
    once. Vega is conventionally per one IV percentage point, while the typed
    IV drop is a decimal fraction and is converted to percentage points.
    """
    if (
        spot <= ZERO
        or event_iv_drop < ZERO
        or multiplier <= 0
        or quantity <= 0
        or operational_costs < ZERO
    ):
        raise ValueError("invalid IV-crush valuation inputs")
    move_size = implied_move_pct if implied_move_pct is not None else Decimal("0.05")
    gamma = net_gamma if net_gamma is not None else ZERO
    theta = net_theta if net_theta is not None else ZERO
    if move_size <= ZERO:
        raise ValueError("implied move must be positive")
    if abs(net_delta) < Decimal("0.0001"):
        raise ValueError("near-zero delta has no direction-correct IV-crush scenario")
    correct_sign = Decimal("1") if net_delta > ZERO else Decimal("-1")
    scale = Decimal(multiplier)
    rows: list[dict[str, str]] = []
    for time_name, days_elapsed in (("immediate", Decimal()), ("delayed", Decimal("2"))):
        for move_name, move in (
            ("wrong_direction", -correct_sign * move_size),
            ("no_move", ZERO),
            ("smaller_than_implied", correct_sign * move_size / Decimal("2")),
            ("equal_implied_move", correct_sign * move_size),
            ("larger_move", correct_sign * move_size * Decimal("1.25")),
        ):
            for crush_name, crush in (
                ("mild", event_iv_drop / 2),
                ("typical", event_iv_drop),
                ("severe", event_iv_drop * 2),
            ):
                dollar_move = spot * move
                pnl = (
                    net_delta * dollar_move
                    + Decimal("0.5") * gamma * dollar_move * dollar_move
                    + net_vega * (-(crush * Decimal("100")))
                    + theta * days_elapsed
                ) * scale + current_signed_complex_value - signed_entry - operational_costs
                rows.append(
                    {
                        "spot_case": move_name,
                        "time_case": time_name,
                        "iv_case": crush_name,
                        "estimated_pnl": format(pnl, "f"),
                        "label": "ESTIMATED",
                        "limitations": "Greek stress estimate; not an executable quote",
                        "direction": "bullish" if correct_sign > ZERO else "bearish",
                        "implied_move_pct": format(move_size, "f"),
                        "days_elapsed": format(days_elapsed, "f"),
                    }
                )
    return rows


def distribution_ev(payoffs: list[Decimal], probabilities: list[float]) -> dict[str, float]:
    if len(payoffs) != len(probabilities) or not payoffs or abs(sum(probabilities) - 1) > 1e-8:
        raise ValueError("invalid full-distribution probabilities")
    pairs = list(zip(payoffs, probabilities, strict=True))
    # Legacy float helper only: mirror typed semantics by measuring the actual
    # maximum loss magnitude, rather than probability of the worst submitted
    # scenario.  Typed decision paths use Decimal ``weighted_metrics``.
    worst = min(payoffs)
    max_risk = max(0.0, -float(worst))
    loss_probabilities = [probability for pnl, probability in pairs if pnl < ZERO]
    # This legacy helper retains float output; typed paths use volatility.weighted_metrics.
    return {
        "expected_pnl": sum(float(pnl) * probability for pnl, probability in pairs),
        "median_proxy": float(sorted(pairs, key=lambda item: item[0])[len(pairs) // 2][0]),
        "loss_probability": sum(loss_probabilities),
        "total_loss_probability": sum(
            probability for pnl, probability in pairs if max_risk > 0 and float(pnl) == -max_risk
        ),
        "expected_shortfall": sum(
            float(pnl) * probability for pnl, probability in pairs if pnl < ZERO
        )
        / sum(loss_probabilities)
        if loss_probabilities
        else 0.0,
        "two_x_probability": sum(
            probability for pnl, probability in pairs if pnl >= abs(worst) * 2
        ),
        "three_x_probability": sum(
            probability for pnl, probability in pairs if pnl >= abs(worst) * 3
        ),
        "five_x_probability": sum(
            probability for pnl, probability in pairs if pnl >= abs(worst) * 5
        ),
    }


def decide(
    candidate: Candidate, now: datetime, risk_cap: Decimal = Decimal("1000"), fees: Decimal = ZERO
) -> tuple[Decision, list[GateResult], Payoff | None]:
    """Compatibility API; the full state machine lives in gates.decision_from_ledger."""
    from .gates import decision_from_ledger, ledger

    items, payoff = ledger(candidate, now, fees=fees)
    # Preserve the small legacy diagnostic API without contaminating the stable
    # 49-entry pipeline ledger.  These records are aliases for the corresponding
    # typed hard gates, not a second decision mechanism.
    legacy: list[GateResult] = []
    if candidate.fixture:
        legacy.append(
            GateResult("fixture_mode", False, "fixture/demo data is permanently non-live")
        )
    if payoff is None or payoff.max_loss is None or (payoff.max_loss + fees) > risk_cap:
        legacy.append(
            GateResult(
                "universal_total_loss_cap",
                False,
                "unbounded or fee-inclusive defined loss exceeds $1,000",
            )
        )
    if candidate.structure is not None:
        shorts = [leg for leg in candidate.structure.legs if leg.side is Side.SELL]
        if any(
            (
                leg.contract.exercise_style == "American"
                and (leg.contract.settlement_style or "").casefold() == "physical"
            )
            for leg in shorts
        ):
            legacy.append(
                GateResult(
                    "american_short_leg",
                    False,
                    "American-style physically settled short legs are categorically rejected",
                )
            )
        if any(
            (leg.contract.settlement_style or "").casefold() == "physical"
            for leg in candidate.structure.legs
        ):
            legacy.append(
                GateResult(
                    "physical_expiration",
                    False,
                    "physical options require a verified pre-expiry close plan",
                )
            )
    strict_decision = decision_from_ledger(
        items, candidate.source.value == "LIVE", candidate.fixture
    )
    # Older callers construct Candidate directly (before the strict parsed-run
    # schema existed).  Keep that API useful while requiring the same cash,
    # bounded-loss and evidence flags it historically supplied. Parsed inputs
    # always take the full 50-gate path above.
    if (
        not candidate.source_records
        and not candidate.fixture
        and payoff is not None
        and candidate.structure is not None
    ):
        quote_fail = validate_quotes(candidate.structure, now)
        shorts = [leg for leg in candidate.structure.legs if leg.side is Side.SELL]
        verified_cash_index_short = not shorts or (
            candidate.mechanics.product_type == "index"
            and candidate.mechanics.asset_type == "index option"
            and all(
                leg.contract.exercise_style == "European"
                and leg.contract.settlement_style == "cash"
                for leg in candidate.structure.legs
            )
        )
        compat = (
            not legacy
            # Compatibility must never discard a failed typed hard gate.  It
            # is only a public convenience API, not a second safety policy.
            and not any(item.status is GateStatus.FAIL for item in items)
            and payoff.operational_max_loss_risk is not None
            and payoff.operational_max_loss_risk + fees <= risk_cap
            and not quote_fail
            and bool(candidate.claim_records)
            and bool(candidate.mechanics_evidence)
            and bool(candidate.portfolio_evidence)
            and bool(candidate.deficit_evidence)
            and verified_cash_index_short
            and all(
                dte(leg.contract.expiration, now.date()) <= 30 for leg in candidate.structure.legs
            )
        )
        if compat:
            return Decision.ACTIONABLE, legacy + items, payoff
    return strict_decision, legacy + items, payoff


def preflight(
    original_underlying: Decimal, refreshed: Candidate, now: datetime
) -> tuple[Decision, list[GateResult]]:
    if refreshed.underlying is None:
        return Decision.DATA_INSUFFICIENT, [GateResult("underlying", False, "refresh unavailable")]
    if original_underlying <= ZERO or abs(refreshed.underlying / original_underlying - 1) > Decimal(
        "0.02"
    ):
        return Decision.INVALIDATED, [
            GateResult(
                "price_change", False, "underlying moved more than 2%; complete reanalysis required"
            )
        ]
    decision, gates, _ = decide(refreshed, now)
    return decision, gates
