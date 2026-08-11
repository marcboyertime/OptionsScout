from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    ACTIONABLE = "ACTIONABLE"
    WATCH = "WATCH"
    NO_TRADE = "NO_TRADE"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    INVALIDATED = "INVALIDATED"
    MARKET_CLOSED_OR_STALE = "MARKET_CLOSED_OR_STALE"


class SourceLabel(StrEnum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    CACHED = "CACHED"
    HISTORICAL = "HISTORICAL"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CatalystType(StrEnum):
    EARNINGS = "earnings"
    FDA = "fda"
    MACRO = "macro"
    OTHER = "other"


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    as_of: datetime
    retrieved_at: datetime
    config_version: str
    model_version: str
    prompt_version: str


@dataclass(frozen=True)
class CaptureBinding:
    capture_id: str
    tool: str
    schema_identity: str
    parameter_schema_sha256: str
    response_schema_sha256: str
    normalized_projection_schema_sha256: str
    payload_hash: str
    normalized_input_hash: str
    source_label: SourceLabel
    as_of: datetime
    retrieved_at: datetime


@dataclass(frozen=True)
class SourceRecord:
    id: str
    title: str
    publisher: str
    url: str
    published_at: datetime
    event_at: datetime | None
    retrieved_at: datetime
    primary: bool
    claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimRecord:
    id: str
    text: str
    source_ids: tuple[str, ...]
    inference: str
    confidence: Decimal


@dataclass(frozen=True)
class ThesisRecord:
    implied_probability_low: Decimal
    implied_probability_high: Decimal
    outcome: str
    why_wrong: str
    why_not_arbitraged: str
    falsifier: str
    assumptions: tuple[str, ...]
    catalyst: str
    timing_trigger: str
    narratives: tuple[str, ...]
    underappreciation: str = ""
    catalyst_type: CatalystType = CatalystType.OTHER
    catalyst_at: datetime | None = None


@dataclass(frozen=True)
class EquityContext:
    tradable: bool
    options_available: bool
    underlying_liquidity: Decimal
    returns: tuple[Decimal, ...]
    volume: Decimal
    realized_volatility: Decimal
    sector_behavior: str
    factor_behavior: str
    sector_beta: Decimal
    factor_adjusted: bool
    material_move_pct: Decimal
    technical_trigger: str | None
    contradictions: tuple[str, ...]


@dataclass(frozen=True)
class QuoteProvenance:
    source_id: str
    retrieved_at: datetime
    as_of: datetime
    source: SourceLabel
    methodology: str


@dataclass(frozen=True)
class ContractMechanics:
    asset_type: str
    product_type: str
    exercise_style: str
    settlement_style: str
    deliverable: str
    ex_dividend_date: date | None
    ex_dividend_amount: Decimal | None
    assignment_risk: str
    pin_risk: str
    auto_exercise: str
    corporate_action: str
    product_calendar: str


@dataclass(frozen=True)
class FillPlan:
    order_type: str
    limit: Decimal
    rationale: str
    max_slippage: Decimal


@dataclass(frozen=True)
class ExitPlan:
    required_before_expiry: bool
    close_buffer_days: int
    profit_plan: str
    invalidation: str
    catalyst_hold: str
    underlying_invalidation: str
    thesis_invalidation: str
    volatility_invalidation: str
    time_exit_at: datetime
    time_exit_rationale: str
    rapid_double_response: str
    loss_50_response: str
    direction_correct_iv_wrong_response: str
    sector_only_response: str
    expiration_management: str
    assignment_management: str
    roll_policy: str


@dataclass(frozen=True)
class StructurePlan:
    structure_type: str
    one_complex_order: bool
    entry_limit: Decimal
    max_acceptable_limit: Decimal
    fees: Decimal
    commissions: Decimal
    exit_slippage: Decimal
    exit_plan: ExitPlan


@dataclass(frozen=True)
class SurfaceNode:
    expiration: date
    strike: Decimal
    option_type: OptionType
    implied_volatility: Decimal
    delta: Decimal | None


@dataclass(frozen=True)
class IVSnapshot:
    as_of: datetime
    atm_iv: Decimal
    skew: Decimal
    term_structure: tuple[Decimal, ...]


@dataclass(frozen=True)
class EventHistory:
    event_type: str
    event_at: datetime
    observed_move_pct: Decimal
    notes: str


@dataclass(frozen=True)
class UpcomingEvent:
    """A dated binary event that can affect an open option position.

    Historical observations are deliberately not repurposed as a forecast.
    """

    event_type: CatalystType
    event_at: datetime
    expected_iv_drop: Decimal
    provenance: str


@dataclass(frozen=True)
class VolatilityContext:
    implied_move_pct: Decimal
    realized_volatility: Decimal
    event_history: tuple[EventHistory, ...]
    upcoming_events: tuple[UpcomingEvent, ...]
    iv_snapshot: IVSnapshot


@dataclass(frozen=True)
class Scenario:
    id: str
    outcome: str
    probability: Decimal
    expiration_spot: Decimal
    payoff: Decimal


@dataclass(frozen=True)
class SensitivityCase:
    id: str
    model: str
    probability_shift_to_worst: Decimal
    additional_cost: Decimal
    expected_value: Decimal


@dataclass(frozen=True)
class PlannedExitScenario:
    id: str
    outcome: str
    probability: Decimal
    underlying_spot: Decimal


@dataclass(frozen=True)
class PlannedExitValuation:
    """Auditable post-event close valuation, never an expiration surrogate."""

    method: str
    valuation_at: datetime
    source_id: str
    scenarios: tuple[PlannedExitScenario, ...]
    sensitivity_cases: tuple[SensitivityCase, ...]


@dataclass(frozen=True)
class DistributionSet:
    scenarios: tuple[Scenario, ...]
    sensitivity_cases: tuple[SensitivityCase, ...]
    scenario_model: str
    provenance: str


@dataclass(frozen=True)
class AlternativeEconomics:
    """Independently declared costs, never an alternative probability model.

    The whole-complex maximum fill is derived from this structure's own
    conservative natural fill. Same-horizon alternatives share one canonical
    underlying distribution so a submitter cannot manufacture an attractive
    comparison by injecting probabilities.
    """

    structure_id: str
    fees: Decimal
    commissions: Decimal
    exit_slippage: Decimal
    entry_slippage: Decimal


@dataclass(frozen=True)
class PortfolioPosition:
    id: str
    symbol: str
    sector: str
    factor_tags: tuple[str, ...]
    risk: Decimal
    event_risk: Decimal


@dataclass(frozen=True)
class PortfolioLimits:
    hard_cap: Decimal
    remaining_aggregate: Decimal
    remaining_cluster: Decimal
    remaining_event: Decimal
    remaining_sector: Decimal
    remaining_factor: Decimal


@dataclass(frozen=True)
class CorrelationRecord:
    position_id: str
    correlation: Decimal
    rationale: str


@dataclass(frozen=True)
class PortfolioAssessment:
    positions: tuple[PortfolioPosition, ...]
    limits: PortfolioLimits
    correlations: tuple[CorrelationRecord, ...]
    aggregate_risk: Decimal
    cluster_risk: Decimal
    event_risk: Decimal
    sector_risk: Decimal
    factor_risk: Decimal
    deficit_elimination_rationale: str


@dataclass(frozen=True)
class SkepticRecord:
    opposing_view: str
    weakest_evidence: str
    key_assumption: str
    crowding: str
    beta: str
    iv: str
    liquidity: str
    total_loss: str


@dataclass(frozen=True)
class JudgeRecord:
    verdict: str
    reason: str


@dataclass(frozen=True)
class WatchTrigger:
    condition: str
    expires_at: datetime
    action: str


def money(value: str | Decimal) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("money must be an exact decimal string") from error
    if not amount.is_finite():
        raise ValueError("non-finite decimal is prohibited")
    return amount


@dataclass(frozen=True)
class Quote:
    bid: Decimal | None
    ask: Decimal | None
    as_of: datetime
    source: SourceLabel
    mark: Decimal | None = None
    iv: Decimal | None = None
    delta: Decimal | None = None
    gamma: Decimal | None = None
    theta: Decimal | None = None
    vega: Decimal | None = None
    volume: int | None = None
    open_interest: int | None = None


@dataclass(frozen=True)
class Contract:
    id: str
    symbol: str
    expiration: str
    strike: Decimal
    option_type: OptionType
    quote: Quote
    tradable: bool = True
    multiplier: int = 100
    adjusted: bool = False
    exercise_style: str | None = None
    settlement_style: str | None = None


@dataclass(frozen=True)
class Leg:
    side: Side
    contract: Contract
    ratio: int = 1


@dataclass(frozen=True)
class Structure:
    name: str
    legs: tuple[Leg, ...]
    quantity: int = 1
    identifier: str = ""


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    severity: str = "hard"
    status: GateStatus | None = None
    evidence_ids: tuple[str, ...] = ()
    trace: str = ""


@dataclass(frozen=True, init=False)
class Candidate:
    id: str
    symbol: str
    underlying: Decimal | None
    underlying_as_of: datetime | None
    source: SourceLabel
    structure: Structure | None
    structure_set: tuple[Structure, ...]
    contract_chain: tuple[Contract, ...]
    alternative_economics: tuple[AlternativeEconomics, ...]
    thesis_record: ThesisRecord | None
    source_records: tuple[SourceRecord, ...]
    claim_records: tuple[ClaimRecord, ...]
    equity_context: EquityContext
    quote_provenance: QuoteProvenance
    mechanics: ContractMechanics
    fill_plan: FillPlan
    structure_plan: StructurePlan
    surface_nodes: tuple[SurfaceNode, ...]
    volatility_context: VolatilityContext
    distribution: DistributionSet
    planned_exit_valuation: PlannedExitValuation | None
    portfolio_assessment: PortfolioAssessment
    skeptic: SkepticRecord
    judge: JudgeRecord
    watch_triggers: tuple[WatchTrigger, ...]
    fixture: bool = False

    def __init__(
        self,
        symbol: str,
        underlying: Decimal | None,
        underlying_as_of: datetime | None,
        source: SourceLabel,
        structure: Structure | None = None,
        thesis_record: ThesisRecord | None = None,
        claim_records: tuple[ClaimRecord, ...] = (),
        event: bool = False,
        mechanics_evidence: tuple[str, ...] = (),
        portfolio_evidence: tuple[str, ...] = (),
        fixture: bool = False,
        deficit_evidence: tuple[str, ...] = (),
        *,
        id: str | None = None,
        structure_set: tuple[Structure, ...] = (),
        contract_chain: tuple[Contract, ...] = (),
        alternative_economics: tuple[AlternativeEconomics, ...] = (),
        source_records: tuple[SourceRecord, ...] = (),
        equity_context: EquityContext | None = None,
        quote_provenance: QuoteProvenance | None = None,
        mechanics: ContractMechanics | None = None,
        fill_plan: FillPlan | None = None,
        structure_plan: StructurePlan | None = None,
        surface_nodes: tuple[SurfaceNode, ...] = (),
        volatility_context: VolatilityContext | None = None,
        distribution: DistributionSet | None = None,
        planned_exit_valuation: PlannedExitValuation | None = None,
        portfolio_assessment: PortfolioAssessment | None = None,
        skeptic: SkepticRecord | None = None,
        judge: JudgeRecord | None = None,
        watch_triggers: tuple[WatchTrigger, ...] = (),
    ) -> None:
        point = underlying_as_of or datetime.fromtimestamp(0).astimezone()
        amount = underlying or Decimal()
        default_thesis = ThesisRecord(Decimal(), Decimal("1"), "", "", "", "", (), "", "", ())
        default_equity = EquityContext(
            False,
            False,
            Decimal(),
            (),
            Decimal(),
            Decimal(),
            "",
            "",
            Decimal(),
            False,
            Decimal(),
            None,
            (),
        )
        default_provenance = QuoteProvenance(
            "compat", point, point, source, "compatibility constructor"
        )
        default_mechanics = ContractMechanics(
            "", "", "", "", "", None, None, "", "", "", "", "compatibility"
        )
        default_fill = FillPlan("limit", Decimal(), "", Decimal())
        default_exit = ExitPlan(
            False, 0, "", "", "", "", "", "", point, "", "", "", "", "", "", "", ""
        )
        default_structure = StructurePlan(
            "", False, Decimal(), Decimal(), Decimal(), Decimal(), Decimal(), default_exit
        )
        default_snapshot = IVSnapshot(point, Decimal(), Decimal(), ())
        default_event = EventHistory("compatibility", point, Decimal(), "")
        default_volatility = VolatilityContext(
            Decimal(), Decimal(), (default_event,) if event else (), (), default_snapshot
        )
        default_distribution = DistributionSet(
            (), (), "bounded_terminal_spot_v1", "compatibility constructor"
        )
        default_limits = PortfolioLimits(
            Decimal("1000"), Decimal(), Decimal(), Decimal(), Decimal(), Decimal()
        )
        compat_positions = (
            (PortfolioPosition("compatibility", symbol, "", (), Decimal(), Decimal()),)
            if portfolio_evidence
            else ()
        )
        default_portfolio = PortfolioAssessment(
            compat_positions,
            default_limits,
            (),
            Decimal(),
            Decimal(),
            Decimal(),
            Decimal(),
            Decimal(),
            "compatibility evidence" if deficit_evidence else "",
        )
        object.__setattr__(self, "id", id or symbol)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "underlying", amount)
        object.__setattr__(self, "underlying_as_of", point)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "structure", structure or Structure("", ()))
        object.__setattr__(self, "structure_set", structure_set or ((structure,) if structure else ()))
        object.__setattr__(
            self,
            "contract_chain",
            contract_chain
            or tuple(
                {leg.contract.id: leg.contract for item in (structure_set or ((structure,) if structure else ())) for leg in item.legs}.values()
            ),
        )
        object.__setattr__(self, "alternative_economics", alternative_economics)
        object.__setattr__(self, "thesis_record", thesis_record or default_thesis)
        object.__setattr__(self, "source_records", source_records)
        object.__setattr__(self, "claim_records", claim_records)
        object.__setattr__(self, "equity_context", equity_context or default_equity)
        object.__setattr__(self, "quote_provenance", quote_provenance or default_provenance)
        object.__setattr__(
            self,
            "mechanics",
            mechanics
            or (
                ContractMechanics(
                    "", "", "", "", "", None, None, "", "", "", "", mechanics_evidence[0]
                )
                if mechanics_evidence
                else default_mechanics
            ),
        )
        object.__setattr__(self, "fill_plan", fill_plan or default_fill)
        object.__setattr__(self, "structure_plan", structure_plan or default_structure)
        object.__setattr__(self, "surface_nodes", surface_nodes)
        object.__setattr__(self, "volatility_context", volatility_context or default_volatility)
        object.__setattr__(self, "distribution", distribution or default_distribution)
        object.__setattr__(self, "planned_exit_valuation", planned_exit_valuation)
        object.__setattr__(self, "portfolio_assessment", portfolio_assessment or default_portfolio)
        object.__setattr__(
            self, "skeptic", skeptic or SkepticRecord("", "", "", "", "", "", "", "")
        )
        object.__setattr__(self, "judge", judge or JudgeRecord("", ""))
        object.__setattr__(self, "watch_triggers", watch_triggers)
        object.__setattr__(self, "fixture", fixture)

    @property
    def event(self) -> bool:
        return bool(
            self.volatility_context.event_history or self.volatility_context.upcoming_events
        )

    @property
    def mechanics_evidence(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in (self.mechanics.product_calendar, self.mechanics.assignment_risk)
            if item
        )

    @property
    def portfolio_evidence(self) -> tuple[str, ...]:
        return ("typed portfolio assessment",) if self.portfolio_assessment.positions else ()

    @property
    def deficit_evidence(self) -> tuple[str, ...]:
        return (
            ("typed deficit assessment",)
            if self.portfolio_assessment.deficit_elimination_rationale
            else ()
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value
