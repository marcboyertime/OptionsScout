from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .engine import expiration_pnl, fill_economics
from .models import (
    AlternativeEconomics,
    Candidate,
    CaptureBinding,
    CatalystType,
    ClaimRecord,
    Contract,
    ContractMechanics,
    CorrelationRecord,
    DistributionSet,
    EquityContext,
    EventHistory,
    ExitPlan,
    FillPlan,
    IVSnapshot,
    JudgeRecord,
    Leg,
    OptionType,
    PlannedExitScenario,
    PlannedExitValuation,
    PortfolioAssessment,
    PortfolioLimits,
    PortfolioPosition,
    Quote,
    QuoteProvenance,
    RunMetadata,
    Scenario,
    SensitivityCase,
    Side,
    SkepticRecord,
    SourceLabel,
    SourceRecord,
    Structure,
    StructurePlan,
    SurfaceNode,
    ThesisRecord,
    UpcomingEvent,
    VolatilityContext,
    WatchTrigger,
    money,
)
from .structures import classify_structure


class SchemaError(ValueError):
    """The untrusted normalized-run document is not schema version 1."""


_FORBIDDEN_KEY_PARTS = ("command", "path", "tool", "instruction")
_MAX_CANDIDATES = 500
_MAX_CONTRACTS = 500
_MAX_STRUCTURES = 20
_MAX_RECORDS = 100
# These mechanics labels are security-relevant controls, not display text.
# Accepting aliases or case-folded variants lets a physically settled contract
# evade exact downstream containment checks.
EXERCISE_STYLES = frozenset({"American", "European"})
SETTLEMENT_STYLES = frozenset({"physical", "cash"})


def _obj(raw: object, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SchemaError(f"{name} must be an object")
    for key in raw:
        if not isinstance(key, str) or any(part in key.casefold() for part in _FORBIDDEN_KEY_PARTS):
            raise SchemaError("executable or path-like normalized-run field is prohibited")
    return raw


def _only(raw: object, allowed: set[str], required: set[str], name: str) -> dict[str, Any]:
    record = _obj(raw, name)
    keys = set(record)
    if keys != keys & allowed or not required <= keys:
        raise SchemaError(f"unknown or missing {name} fields")
    return record


def _str(raw: object, name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return raw


def _bool(raw: object, name: str) -> bool:
    if type(raw) is not bool:
        raise SchemaError(f"{name} must be a boolean")
    return raw


def _int(raw: object, name: str, minimum: int = 0) -> int:
    if type(raw) is not int or raw < minimum:
        raise SchemaError(f"{name} must be a non-bool integer >= {minimum}")
    return raw


def _decimal(raw: object, name: str, optional: bool = False) -> Decimal | None:
    if raw is None and optional:
        return None
    if not isinstance(raw, str):
        raise SchemaError(f"{name} must be an exact Decimal string")
    try:
        return money(raw)
    except ValueError as error:
        raise SchemaError(f"invalid {name}") from error


def _time(raw: object, name: str) -> datetime:
    value = _str(raw, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SchemaError(f"invalid {name}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _date(raw: object, name: str, optional: bool = False) -> date | None:
    if raw is None and optional:
        return None
    value = _str(raw, name)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise SchemaError(f"invalid {name}") from error


def _strings(raw: object, name: str, cap: int = _MAX_RECORDS) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) > cap:
        raise SchemaError(f"{name} must be a bounded list")
    return tuple(_str(item, name) for item in raw)


def _source_label(raw: object, name: str) -> SourceLabel:
    try:
        return SourceLabel(_str(raw, name))
    except ValueError as error:
        raise SchemaError(f"invalid {name}") from error


def _canonical_label(raw: object, name: str, allowed: frozenset[str], description: str) -> str:
    value = _str(raw, name)
    if value not in allowed:
        raise SchemaError(f"{name} must be exactly {description}")
    return value


def _nonnegative(value: Decimal | None, name: str) -> Decimal | None:
    if value is not None and value < Decimal():
        raise SchemaError(f"{name} must be non-negative")
    return value


def _positive(value: Decimal, name: str) -> Decimal:
    if value <= Decimal():
        raise SchemaError(f"{name} must be positive")
    return value


def _delta(value: Decimal | None, name: str, option_type: OptionType) -> Decimal | None:
    if value is None:
        return None
    if value < Decimal("-1") or value > Decimal("1"):
        raise SchemaError(f"{name} must be within [-1, 1]")
    if (option_type is OptionType.CALL and value < Decimal()) or (
        option_type is OptionType.PUT and value > Decimal()
    ):
        raise SchemaError(f"{name} direction is incompatible with option type")
    return value


def _metadata(raw: object) -> RunMetadata:
    value = _only(
        raw,
        {"run_id", "as_of", "retrieved_at", "config_version", "model_version", "prompt_version"},
        {"run_id", "as_of", "retrieved_at", "config_version", "model_version", "prompt_version"},
        "metadata",
    )
    as_of = _time(value["as_of"], "metadata.as_of")
    retrieved = _time(value["retrieved_at"], "metadata.retrieved_at")
    if retrieved < as_of:
        raise SchemaError("metadata retrieval precedes as_of")
    return RunMetadata(
        _str(value["run_id"], "metadata.run_id"),
        as_of,
        retrieved,
        _str(value["config_version"], "metadata.config_version"),
        _str(value["model_version"], "metadata.model_version"),
        _str(value["prompt_version"], "metadata.prompt_version"),
    )


def _capture_binding(raw: object | None) -> CaptureBinding | None:
    if raw is None:
        return None
    # The broker tool identity is inert provenance, not an executable field.
    # It is allowed only in this fixed typed binding and later must match the
    # immutable capture record plus the positive policy allowlist.
    if not isinstance(raw, dict):
        raise SchemaError("capture_binding must be an object")
    allowed = {
        "capture_id",
        "tool",
        "schema_identity",
        "parameter_schema_sha256",
        "response_schema_sha256",
        "normalized_projection_schema_sha256",
        "payload_hash",
        "normalized_input_hash",
        "source_label",
        "as_of",
        "retrieved_at",
    }
    if set(raw) != allowed:
        raise SchemaError("unknown or missing capture_binding fields")
    value = raw
    as_of, retrieved_at = (
        _time(value["as_of"], "capture_binding.as_of"),
        _time(value["retrieved_at"], "capture_binding.retrieved_at"),
    )
    if retrieved_at < as_of:
        raise SchemaError("capture binding retrieval precedes as_of")
    return CaptureBinding(
        _str(value["capture_id"], "capture_binding.capture_id"),
        _str(value["tool"], "capture_binding.tool"),
        _str(value["schema_identity"], "capture_binding.schema_identity"),
        _str(value["parameter_schema_sha256"], "capture_binding.parameter_schema_sha256"),
        _str(value["response_schema_sha256"], "capture_binding.response_schema_sha256"),
        _str(
            value["normalized_projection_schema_sha256"],
            "capture_binding.normalized_projection_schema_sha256",
        ),
        _str(value["payload_hash"], "capture_binding.payload_hash"),
        _str(value["normalized_input_hash"], "capture_binding.normalized_input_hash"),
        _source_label(value["source_label"], "capture_binding.source_label"),
        as_of,
        retrieved_at,
    )


def _quote(raw: object, source: SourceLabel, option_type: OptionType) -> Quote:
    value = _only(
        raw,
        {
            "bid",
            "ask",
            "mark",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "as_of",
            "volume",
            "open_interest",
        },
        {"bid", "ask", "as_of"},
        "quote",
    )
    bid = _nonnegative(_decimal(value.get("bid"), "quote.bid", True), "quote.bid")
    ask = _nonnegative(_decimal(value.get("ask"), "quote.ask", True), "quote.ask")
    mark = _nonnegative(_decimal(value.get("mark"), "quote.mark", True), "quote.mark")
    if bid is not None and ask is not None and bid > ask:
        raise SchemaError("quote bid exceeds ask")
    if mark is not None and bid is not None and ask is not None and not bid <= mark <= ask:
        raise SchemaError("quote mark must be within an uncrossed bid-ask")
    iv = _decimal(value.get("iv"), "quote.iv", True)
    if iv is not None:
        _positive(iv, "quote.iv")
    gamma = _nonnegative(_decimal(value.get("gamma"), "quote.gamma", True), "quote.gamma")
    vega = _nonnegative(_decimal(value.get("vega"), "quote.vega", True), "quote.vega")
    return Quote(
        bid,
        ask,
        _time(value["as_of"], "quote.as_of"),
        source,
        mark,
        iv,
        _delta(_decimal(value.get("delta"), "quote.delta", True), "quote.delta", option_type),
        gamma,
        _decimal(value.get("theta"), "quote.theta", True),
        vega,
        _int(value["volume"], "quote.volume") if "volume" in value else None,
        _int(value["open_interest"], "quote.open_interest") if "open_interest" in value else None,
    )


def _contracts(
    raw: object, symbol: str, source: SourceLabel, as_of: datetime
) -> dict[str, Contract]:
    if not isinstance(raw, list) or len(raw) > _MAX_CONTRACTS:
        raise SchemaError("contracts must be a list of at most 500")
    result: dict[str, Contract] = {}
    for item in raw:
        value = _only(
            item,
            {
                "id",
                "symbol",
                "expiration",
                "strike",
                "type",
                "quote",
                "tradable",
                "multiplier",
                "adjusted",
                "exercise_style",
                "settlement_style",
            },
            {
                "id",
                "symbol",
                "expiration",
                "strike",
                "type",
                "quote",
                "tradable",
                "multiplier",
                "adjusted",
                "exercise_style",
                "settlement_style",
            },
            "contract",
        )
        identifier, contract_symbol = (
            _str(value["id"], "contract.id"),
            _str(value["symbol"], "contract.symbol"),
        )
        if identifier in result or contract_symbol != symbol:
            raise SchemaError("duplicate contract ID or incompatible contract symbol")
        expiration = _date(value["expiration"], "contract.expiration")
        assert expiration is not None
        if expiration < as_of.date():
            raise SchemaError("contract expiration precedes run as_of")
        try:
            option_type = OptionType(_str(value["type"], "contract.type"))
        except ValueError as error:
            raise SchemaError("invalid contract option type") from error
        multiplier = _int(value["multiplier"], "contract.multiplier", 1)
        result[identifier] = Contract(
            identifier,
            contract_symbol,
            expiration.isoformat(),
            _positive(_decimal(value["strike"], "contract.strike") or Decimal(), "contract.strike"),
            option_type,
            _quote(value["quote"], source, option_type),
            _bool(value["tradable"], "contract.tradable"),
            multiplier,
            _bool(value["adjusted"], "contract.adjusted"),
            _canonical_label(
                value["exercise_style"],
                "contract.exercise_style",
                EXERCISE_STYLES,
                "American or European",
            ),
            _canonical_label(
                value["settlement_style"],
                "contract.settlement_style",
                SETTLEMENT_STYLES,
                "physical or cash",
            ),
        )
    return result


def _structures(
    raw: object, contracts: dict[str, Contract]
) -> tuple[dict[str, Structure], tuple[Structure, ...]]:
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_STRUCTURES:
        raise SchemaError("structures must contain 1..20 records")
    map_: dict[str, Structure] = {}
    all_: list[Structure] = []
    for item in raw:
        value = _only(
            item,
            {"id", "name", "quantity", "legs"},
            {"id", "name", "quantity", "legs"},
            "structure",
        )
        identifier = _str(value["id"], "structure.id")
        if identifier in map_ or not isinstance(value["legs"], list) or not value["legs"]:
            raise SchemaError("duplicate structure ID or invalid legs")
        legs: list[Leg] = []
        for leg_raw in value["legs"]:
            leg = _only(
                leg_raw, {"side", "contract_id", "ratio"}, {"side", "contract_id", "ratio"}, "leg"
            )
            contract_id = _str(leg["contract_id"], "leg.contract_id")
            if contract_id not in contracts:
                raise SchemaError("broken leg contract reference")
            try:
                side = Side(_str(leg["side"], "leg.side"))
            except ValueError as error:
                raise SchemaError("invalid leg side") from error
            legs.append(Leg(side, contracts[contract_id], _int(leg["ratio"], "leg.ratio", 1)))
        structure = Structure(
            _str(value["name"], "structure.name"),
            tuple(legs),
            _int(value["quantity"], "structure.quantity", 1),
            identifier,
        )
        map_[identifier] = structure
        all_.append(structure)
    return map_, tuple(all_)


def _sources_claims(
    raw_sources: object, raw_claims: object, metadata: RunMetadata
) -> tuple[tuple[SourceRecord, ...], tuple[ClaimRecord, ...]]:
    if (
        not isinstance(raw_sources, list)
        or not isinstance(raw_claims, list)
        or len(raw_sources) > _MAX_RECORDS
        or len(raw_claims) > _MAX_RECORDS
    ):
        raise SchemaError("sources and claims must be lists of at most 100")
    source_ids: set[str] = set()
    sources: list[SourceRecord] = []
    for item in raw_sources:
        value = _only(
            item,
            {
                "id",
                "title",
                "publisher",
                "url",
                "published_at",
                "event_at",
                "retrieved_at",
                "primary",
                "claim_ids",
            },
            {
                "id",
                "title",
                "publisher",
                "url",
                "published_at",
                "event_at",
                "retrieved_at",
                "primary",
                "claim_ids",
            },
            "source",
        )
        identifier = _str(value["id"], "source.id")
        published, event, retrieved = (
            _time(value["published_at"], "source.published_at"),
            (
                _time(value["event_at"], "source.event_at")
                if value["event_at"] is not None
                else None
            ),
            _time(value["retrieved_at"], "source.retrieved_at"),
        )
        if (
            identifier in source_ids
            or retrieved < published
            or (event is not None and event > retrieved)
            or retrieved > metadata.retrieved_at
        ):
            raise SchemaError("duplicate source ID or invalid source event/retrieval ordering")
        source_ids.add(identifier)
        sources.append(
            SourceRecord(
                identifier,
                _str(value["title"], "source.title"),
                _str(value["publisher"], "source.publisher"),
                _str(value["url"], "source.url"),
                published,
                event,
                retrieved,
                _bool(value["primary"], "source.primary"),
                _strings(value["claim_ids"], "source.claim_ids"),
            )
        )
    claim_ids: set[str] = set()
    claims: list[ClaimRecord] = []
    for item in raw_claims:
        value = _only(
            item,
            {"id", "text", "source_ids", "inference", "confidence"},
            {"id", "text", "source_ids", "inference", "confidence"},
            "claim",
        )
        identifier, references = (
            _str(value["id"], "claim.id"),
            _strings(value["source_ids"], "claim.source_ids"),
        )
        confidence = _decimal(value["confidence"], "claim.confidence")
        assert confidence is not None
        if (
            identifier in claim_ids
            or not references
            or not set(references) <= source_ids
            or confidence < Decimal()
            or confidence > Decimal("1")
        ):
            raise SchemaError("duplicate/broken claim source reference or invalid confidence")
        claim_ids.add(identifier)
        claims.append(
            ClaimRecord(
                identifier,
                _str(value["text"], "claim.text"),
                references,
                _str(value["inference"], "claim.inference"),
                confidence,
            )
        )
    if any(not set(source.claim_ids) <= claim_ids for source in sources):
        raise SchemaError("broken source claim reference")
    return tuple(sources), tuple(claims)


def _thesis(raw: object, as_of: datetime) -> ThesisRecord:
    value = _only(
        raw,
        {
            "implied_probability_low",
            "implied_probability_high",
            "outcome",
            "why_wrong",
            "why_not_arbitraged",
            "falsifier",
            "assumptions",
            "catalyst",
            "catalyst_type",
            "catalyst_at",
            "timing_trigger",
            "narratives",
            "underappreciation",
        },
        {
            "implied_probability_low",
            "implied_probability_high",
            "outcome",
            "why_wrong",
            "why_not_arbitraged",
            "falsifier",
            "assumptions",
            "catalyst",
            "catalyst_type",
            "catalyst_at",
            "timing_trigger",
            "narratives",
            "underappreciation",
        },
        "thesis",
    )
    low, high = (
        _decimal(value["implied_probability_low"], "thesis.implied_probability_low"),
        _decimal(value["implied_probability_high"], "thesis.implied_probability_high"),
    )
    assert low is not None and high is not None
    if low < Decimal() or high > Decimal("1") or low > high:
        raise SchemaError("invalid thesis probability range")
    try:
        catalyst_type = CatalystType(_str(value["catalyst_type"], "thesis.catalyst_type"))
    except ValueError as error:
        raise SchemaError("invalid thesis catalyst type") from error
    catalyst_at = _time(value["catalyst_at"], "thesis.catalyst_at")
    if catalyst_at < as_of:
        raise SchemaError("thesis catalyst date precedes run as_of")
    return ThesisRecord(
        low,
        high,
        _str(value["outcome"], "thesis.outcome"),
        _str(value["why_wrong"], "thesis.why_wrong"),
        _str(value["why_not_arbitraged"], "thesis.why_not_arbitraged"),
        _str(value["falsifier"], "thesis.falsifier"),
        _strings(value["assumptions"], "thesis.assumptions"),
        _str(value["catalyst"], "thesis.catalyst"),
        _str(value["timing_trigger"], "thesis.timing_trigger"),
        _strings(value["narratives"], "thesis.narratives"),
        _str(value["underappreciation"], "thesis.underappreciation"),
        catalyst_type,
        catalyst_at,
    )


def _equity(raw: object) -> EquityContext:
    value = _only(
        raw,
        {
            "tradable",
            "options_available",
            "underlying_liquidity",
            "returns",
            "volume",
            "realized_volatility",
            "sector_behavior",
            "factor_behavior",
            "sector_beta",
            "factor_adjusted",
            "material_move_pct",
            "technical_trigger",
            "contradictions",
        },
        {
            "tradable",
            "options_available",
            "underlying_liquidity",
            "returns",
            "volume",
            "realized_volatility",
            "sector_behavior",
            "factor_behavior",
            "sector_beta",
            "factor_adjusted",
            "material_move_pct",
            "technical_trigger",
            "contradictions",
        },
        "equity_context",
    )
    if not isinstance(value["returns"], list) or len(value["returns"]) > _MAX_RECORDS:
        raise SchemaError("equity_context.returns must be bounded")
    return EquityContext(
        _bool(value["tradable"], "equity_context.tradable"),
        _bool(value["options_available"], "equity_context.options_available"),
        _nonnegative(
            _decimal(value["underlying_liquidity"], "equity_context.underlying_liquidity"),
            "equity_context.underlying_liquidity",
        )
        or Decimal(),
        tuple(_decimal(item, "equity_context.returns") or Decimal() for item in value["returns"]),
        _nonnegative(_decimal(value["volume"], "equity_context.volume"), "equity_context.volume")
        or Decimal(),
        _nonnegative(
            _decimal(value["realized_volatility"], "equity_context.realized_volatility"),
            "equity_context.realized_volatility",
        )
        or Decimal(),
        _str(value["sector_behavior"], "equity_context.sector_behavior"),
        _str(value["factor_behavior"], "equity_context.factor_behavior"),
        _decimal(value["sector_beta"], "equity_context.sector_beta") or Decimal(),
        _bool(value["factor_adjusted"], "equity_context.factor_adjusted"),
        _decimal(value["material_move_pct"], "equity_context.material_move_pct") or Decimal(),
        (
            _str(value["technical_trigger"], "equity_context.technical_trigger")
            if value["technical_trigger"] is not None
            else None
        ),
        _strings(value["contradictions"], "equity_context.contradictions"),
    )


def _mechanics(raw: object) -> ContractMechanics:
    keys = {
        "asset_type",
        "product_type",
        "exercise_style",
        "settlement_style",
        "deliverable",
        "ex_dividend_date",
        "ex_dividend_amount",
        "assignment_risk",
        "pin_risk",
        "auto_exercise",
        "corporate_action",
        "product_calendar",
    }
    value = _only(raw, keys, keys, "mechanics")
    ex_date = _date(value["ex_dividend_date"], "mechanics.ex_dividend_date", True)
    ex_amount = _decimal(value["ex_dividend_amount"], "mechanics.ex_dividend_amount", True)
    if (ex_date is None) != (ex_amount is None):
        raise SchemaError("ex-dividend date and amount must appear together")
    return ContractMechanics(
        _str(value["asset_type"], "mechanics.asset_type"),
        _str(value["product_type"], "mechanics.product_type"),
        _canonical_label(
            value["exercise_style"],
            "mechanics.exercise_style",
            EXERCISE_STYLES,
            "American or European",
        ),
        _canonical_label(
            value["settlement_style"],
            "mechanics.settlement_style",
            SETTLEMENT_STYLES,
            "physical or cash",
        ),
        _str(value["deliverable"], "mechanics.deliverable"),
        ex_date,
        ex_amount,
        _str(value["assignment_risk"], "mechanics.assignment_risk"),
        _str(value["pin_risk"], "mechanics.pin_risk"),
        _str(value["auto_exercise"], "mechanics.auto_exercise"),
        _str(value["corporate_action"], "mechanics.corporate_action"),
        _str(value["product_calendar"], "mechanics.product_calendar"),
    )


def _plans(
    raw_fill: object, raw_structure: object, as_of: datetime, expiration: str
) -> tuple[FillPlan, StructurePlan]:
    fill = _only(
        raw_fill,
        {"order_type", "limit", "rationale", "max_slippage"},
        {"order_type", "limit", "rationale", "max_slippage"},
        "fill_plan",
    )
    exit_raw = _only(
        _only(
            raw_structure,
            {
                "structure_type",
                "one_complex_order",
                "entry_limit",
                "max_acceptable_limit",
                "fees",
                "commissions",
                "exit_slippage",
                "exit_plan",
            },
            {
                "structure_type",
                "one_complex_order",
                "entry_limit",
                "max_acceptable_limit",
                "fees",
                "commissions",
                "exit_slippage",
                "exit_plan",
            },
            "structure_plan",
        )["exit_plan"],
        {
            "required_before_expiry",
            "close_buffer_days",
            "profit_plan",
            "invalidation",
            "catalyst_hold",
            "underlying_invalidation",
            "thesis_invalidation",
            "volatility_invalidation",
            "time_exit_at",
            "time_exit_rationale",
            "rapid_double_response",
            "loss_50_response",
            "direction_correct_iv_wrong_response",
            "sector_only_response",
            "expiration_management",
            "assignment_management",
            "roll_policy",
        },
        {
            "required_before_expiry",
            "close_buffer_days",
            "profit_plan",
            "invalidation",
            "catalyst_hold",
            "underlying_invalidation",
            "thesis_invalidation",
            "volatility_invalidation",
            "time_exit_at",
            "time_exit_rationale",
            "rapid_double_response",
            "loss_50_response",
            "direction_correct_iv_wrong_response",
            "sector_only_response",
            "expiration_management",
            "assignment_management",
            "roll_policy",
        },
        "exit_plan",
    )
    structure = _obj(raw_structure, "structure_plan")
    time_exit = _time(exit_raw["time_exit_at"], "exit_plan.time_exit_at")
    expiry_end = datetime.fromisoformat(f"{expiration}T23:59:59+00:00")
    if time_exit <= as_of or time_exit > expiry_end:
        raise SchemaError("exit_plan.time_exit_at must be after run and no later than expiration")
    if (
        _bool(exit_raw["required_before_expiry"], "exit_plan.required_before_expiry")
        and time_exit.date().isoformat() >= expiration
    ):
        raise SchemaError("required pre-expiry exit must be before expiration")
    exit_plan = ExitPlan(
        _bool(exit_raw["required_before_expiry"], "exit_plan.required_before_expiry"),
        _int(exit_raw["close_buffer_days"], "exit_plan.close_buffer_days"),
        _str(exit_raw["profit_plan"], "exit_plan.profit_plan"),
        _str(exit_raw["invalidation"], "exit_plan.invalidation"),
        _str(exit_raw["catalyst_hold"], "exit_plan.catalyst_hold"),
        _str(exit_raw["underlying_invalidation"], "exit_plan.underlying_invalidation"),
        _str(exit_raw["thesis_invalidation"], "exit_plan.thesis_invalidation"),
        _str(exit_raw["volatility_invalidation"], "exit_plan.volatility_invalidation"),
        time_exit,
        _str(exit_raw["time_exit_rationale"], "exit_plan.time_exit_rationale"),
        _str(exit_raw["rapid_double_response"], "exit_plan.rapid_double_response"),
        _str(exit_raw["loss_50_response"], "exit_plan.loss_50_response"),
        _str(
            exit_raw["direction_correct_iv_wrong_response"],
            "exit_plan.direction_correct_iv_wrong_response",
        ),
        _str(exit_raw["sector_only_response"], "exit_plan.sector_only_response"),
        _str(exit_raw["expiration_management"], "exit_plan.expiration_management"),
        _str(exit_raw["assignment_management"], "exit_plan.assignment_management"),
        _str(exit_raw["roll_policy"], "exit_plan.roll_policy"),
    )
    result = FillPlan(
        _str(fill["order_type"], "fill_plan.order_type"),
        _decimal(fill["limit"], "fill_plan.limit") or Decimal(),
        _str(fill["rationale"], "fill_plan.rationale"),
        _decimal(fill["max_slippage"], "fill_plan.max_slippage") or Decimal(),
    )
    plan = StructurePlan(
        _str(structure["structure_type"], "structure_plan.structure_type"),
        _bool(structure["one_complex_order"], "structure_plan.one_complex_order"),
        _decimal(structure["entry_limit"], "structure_plan.entry_limit") or Decimal(),
        _decimal(structure["max_acceptable_limit"], "structure_plan.max_acceptable_limit")
        or Decimal(),
        _decimal(structure["fees"], "structure_plan.fees") or Decimal(),
        _decimal(structure["commissions"], "structure_plan.commissions") or Decimal(),
        _decimal(structure["exit_slippage"], "structure_plan.exit_slippage") or Decimal(),
        exit_plan,
    )
    if min(result.max_slippage, plan.fees, plan.commissions, plan.exit_slippage) < Decimal():
        raise SchemaError("fees, commissions, and slippage must be non-negative")
    if (
        result.order_type.casefold() != "limit"
        or result.limit != plan.entry_limit
        or plan.entry_limit > plan.max_acceptable_limit
    ):
        raise SchemaError("fill limit, entry limit, and maximum acceptable limit are inconsistent")
    return result, plan


def _volatility(
    raw_surface: object,
    raw_iv: object,
    raw_events: object,
    raw_upcoming: object,
    raw_volatility: object,
    as_of: datetime,
) -> tuple[tuple[SurfaceNode, ...], VolatilityContext]:
    if (
        not isinstance(raw_surface, list)
        or len(raw_surface) > _MAX_RECORDS
        or not isinstance(raw_events, list)
        or len(raw_events) > _MAX_RECORDS
        or not isinstance(raw_upcoming, list)
        or len(raw_upcoming) > _MAX_RECORDS
    ):
        raise SchemaError("surface_nodes and event contexts must be bounded lists")
    surface: list[SurfaceNode] = []
    for item in raw_surface:
        value = _only(
            item,
            {"expiration", "strike", "option_type", "implied_volatility", "delta"},
            {"expiration", "strike", "option_type", "implied_volatility", "delta"},
            "surface_node",
        )
        expiration = _date(value["expiration"], "surface_node.expiration")
        assert expiration is not None
        if expiration < as_of.date():
            raise SchemaError("surface expiration precedes run as_of")
        try:
            option_type = OptionType(_str(value["option_type"], "surface_node.option_type"))
        except ValueError as error:
            raise SchemaError("invalid surface option type") from error
        surface.append(
            SurfaceNode(
                expiration,
                _positive(
                    _decimal(value["strike"], "surface_node.strike") or Decimal(),
                    "surface_node.strike",
                ),
                option_type,
                _positive(
                    _decimal(value["implied_volatility"], "surface_node.implied_volatility")
                    or Decimal(),
                    "surface_node.implied_volatility",
                ),
                _delta(
                    _decimal(value["delta"], "surface_node.delta", True),
                    "surface_node.delta",
                    option_type,
                ),
            )
        )
    iv = _only(
        raw_iv,
        {"as_of", "atm_iv", "skew", "term_structure"},
        {"as_of", "atm_iv", "skew", "term_structure"},
        "iv_snapshot",
    )
    if not isinstance(iv["term_structure"], list) or len(iv["term_structure"]) > _MAX_RECORDS:
        raise SchemaError("iv term structure must be bounded")
    snapshot = IVSnapshot(
        _time(iv["as_of"], "iv_snapshot.as_of"),
        _positive(_decimal(iv["atm_iv"], "iv_snapshot.atm_iv") or Decimal(), "iv_snapshot.atm_iv"),
        _nonnegative(_decimal(iv["skew"], "iv_snapshot.skew"), "iv_snapshot.skew") or Decimal(),
        tuple(
            _positive(
                _decimal(item, "iv_snapshot.term_structure") or Decimal(),
                "iv_snapshot.term_structure",
            )
            for item in iv["term_structure"]
        ),
    )
    events: list[EventHistory] = []
    for item in raw_events:
        value = _only(
            item,
            {"event_type", "event_at", "observed_move_pct", "notes"},
            {"event_type", "event_at", "observed_move_pct", "notes"},
            "event_history",
        )
        event_at = _time(value["event_at"], "event_history.event_at")
        if event_at > as_of:
            raise SchemaError("event history cannot follow run as_of")
        events.append(
            EventHistory(
                _str(value["event_type"], "event_history.event_type"),
                event_at,
                _decimal(value["observed_move_pct"], "event_history.observed_move_pct")
                or Decimal(),
                _str(value["notes"], "event_history.notes"),
            )
        )
    upcoming: list[UpcomingEvent] = []
    for item in raw_upcoming:
        value = _only(
            item,
            {"event_type", "event_at", "expected_iv_drop", "provenance"},
            {"event_type", "event_at", "expected_iv_drop", "provenance"},
            "upcoming_event",
        )
        event_at = _time(value["event_at"], "upcoming_event.event_at")
        expected_drop = (
            _decimal(value["expected_iv_drop"], "upcoming_event.expected_iv_drop") or Decimal()
        )
        if event_at <= as_of or expected_drop <= Decimal():
            raise SchemaError(
                "upcoming event must be future-dated with a positive expected IV-drop assumption"
            )
        try:
            event_type = CatalystType(_str(value["event_type"], "upcoming_event.event_type"))
        except ValueError as error:
            raise SchemaError("invalid upcoming event type") from error
        upcoming.append(
            UpcomingEvent(
                event_type,
                event_at,
                expected_drop,
                _str(value["provenance"], "upcoming_event.provenance"),
            )
        )
    volatility = _only(
        raw_volatility,
        {"implied_move_pct", "realized_volatility"},
        {"implied_move_pct", "realized_volatility"},
        "volatility",
    )
    return tuple(surface), VolatilityContext(
        _nonnegative(
            _decimal(volatility["implied_move_pct"], "volatility.implied_move_pct"),
            "volatility.implied_move_pct",
        )
        or Decimal(),
        _nonnegative(
            _decimal(volatility["realized_volatility"], "volatility.realized_volatility"),
            "volatility.realized_volatility",
        )
        or Decimal(),
        tuple(events),
        tuple(upcoming),
        snapshot,
    )


def _distribution(raw: object) -> DistributionSet:
    value = _only(
        raw,
        {"scenarios", "sensitivity_cases", "scenario_model", "provenance"},
        {"scenarios", "sensitivity_cases", "scenario_model", "provenance"},
        "distribution",
    )
    scenario_model = _str(value["scenario_model"], "distribution.scenario_model")
    if scenario_model != "bounded_terminal_spot_v1":
        raise SchemaError("distribution must use the bounded_terminal_spot_v1 scenario model")
    provenance = _str(value["provenance"], "distribution.provenance")
    scenarios_raw, sensitivity_raw = value["scenarios"], value["sensitivity_cases"]
    if (
        not isinstance(scenarios_raw, list)
        or not scenarios_raw
        or len(scenarios_raw) > _MAX_RECORDS
        or not isinstance(sensitivity_raw, list)
        or len(sensitivity_raw) > _MAX_RECORDS
    ):
        raise SchemaError("distribution lists are invalid")
    ids: set[str] = set()
    scenarios: list[Scenario] = []
    for item in scenarios_raw:
        record = _only(
            item,
            {"id", "outcome", "probability", "expiration_spot", "payoff"},
            {"id", "outcome", "probability", "expiration_spot", "payoff"},
            "scenario",
        )
        identifier, probability = (
            _str(record["id"], "scenario.id"),
            _decimal(record["probability"], "scenario.probability"),
        )
        assert probability is not None
        if identifier in ids or probability < Decimal() or probability > Decimal("1"):
            raise SchemaError("duplicate scenario ID or probability outside 0..1")
        ids.add(identifier)
        spot = _decimal(record["expiration_spot"], "scenario.expiration_spot") or Decimal()
        if spot < Decimal():
            raise SchemaError("scenario expiration spot must be non-negative")
        scenarios.append(
            Scenario(
                identifier,
                _str(record["outcome"], "scenario.outcome"),
                probability,
                spot,
                _decimal(record["payoff"], "scenario.payoff") or Decimal(),
            )
        )
    if sum(item.probability for item in scenarios) != Decimal("1"):
        raise SchemaError("scenario probabilities must sum exactly to 1")
    sensitivity: list[SensitivityCase] = []
    sensitivity_ids: set[str] = set()
    for item in sensitivity_raw:
        record = _only(
            item,
            {"id", "model", "probability_shift_to_worst", "additional_cost", "expected_value"},
            {"id", "model", "probability_shift_to_worst", "additional_cost", "expected_value"},
            "sensitivity_case",
        )
        identifier = _str(record["id"], "sensitivity_case.id")
        model = _str(record["model"], "sensitivity_case.model")
        shift = _decimal(
            record["probability_shift_to_worst"], "sensitivity_case.probability_shift_to_worst"
        )
        additional_cost = _decimal(record["additional_cost"], "sensitivity_case.additional_cost")
        assert shift is not None and additional_cost is not None
        if (
            identifier in sensitivity_ids
            or shift <= Decimal()
            or shift > Decimal("0.25")
            or additional_cost < Decimal()
        ):
            raise SchemaError(
                "sensitivity must be a bounded probability shift with non-negative cost"
            )
        sensitivity_ids.add(identifier)
        sensitivity.append(
            SensitivityCase(
                identifier,
                model,
                shift,
                additional_cost,
                _decimal(record["expected_value"], "sensitivity_case.expected_value") or Decimal(),
            )
        )
    return DistributionSet(tuple(scenarios), tuple(sensitivity), scenario_model, provenance)


def _alternative_economics(
    raw: object, structures: dict[str, Structure], selected_id: str
) -> tuple[AlternativeEconomics, ...]:
    """Parse independent, structure-specific operational costs only."""
    if not isinstance(raw, list) or len(raw) > _MAX_STRUCTURES:
        raise SchemaError("alternative_economics must be a list of at most 20 records")
    supported_ids = {
        identifier
        for identifier, structure in structures.items()
        if classify_structure(structure).supported
    }
    seen: set[str] = set()
    records: list[AlternativeEconomics] = []
    for item in raw:
        value = _only(
            item,
            {
                "structure_id", "fees", "commissions", "exit_slippage", "entry_slippage"
            },
            {
                "structure_id", "fees", "commissions", "exit_slippage", "entry_slippage"
            },
            "alternative_economics",
        )
        identifier = _str(value["structure_id"], "alternative_economics.structure_id")
        if identifier == selected_id or identifier not in supported_ids or identifier in seen:
            raise SchemaError("alternative economics must uniquely name a non-selected supported structure")
        costs = tuple(
            _decimal(value[key], f"alternative_economics.{key}")
            for key in ("fees", "commissions", "exit_slippage", "entry_slippage")
        )
        if any(cost is None or cost < Decimal() for cost in costs):
            raise SchemaError("alternative economics costs must be non-negative exact decimals")
        assert all(cost is not None for cost in costs)
        typed_costs = tuple(cost for cost in costs if cost is not None)
        try:
            fill_economics(structures[identifier])
        except ValueError as error:
            raise SchemaError("alternative lacks defensible two-sided maximum fill") from error
        records.append(AlternativeEconomics(identifier, typed_costs[0], typed_costs[1], typed_costs[2], typed_costs[3]))
        seen.add(identifier)
    # Missing records receive only quote-derived default execution costs at
    # evaluation; they never inherit the selected structure's plan.
    return tuple(records)


def _planned_exit_valuation(
    raw: object | None, exit_plan: ExitPlan, sources: tuple[SourceRecord, ...]
) -> PlannedExitValuation | None:
    if raw is None:
        return None
    value = _only(
        raw,
        {"method", "valuation_at", "source_id", "scenarios", "sensitivity_cases"},
        {"method", "valuation_at", "source_id", "scenarios", "sensitivity_cases"},
        "planned_exit_valuation",
    )
    valuation_at = _time(value["valuation_at"], "planned_exit_valuation.valuation_at")
    source_id = _str(value["source_id"], "planned_exit_valuation.source_id")
    scenarios_raw, sensitivity_raw = value["scenarios"], value["sensitivity_cases"]
    if (
        _str(value["method"], "planned_exit_valuation.method") != "intrinsic_close_v1"
        or valuation_at != exit_plan.time_exit_at
        or source_id not in {source.id for source in sources}
        or not isinstance(scenarios_raw, list)
        or not scenarios_raw
        or len(scenarios_raw) > _MAX_RECORDS
        or not isinstance(sensitivity_raw, list)
        or len(sensitivity_raw) > _MAX_RECORDS
    ):
        raise SchemaError("planned exit valuation must be typed, sourced, and timed exactly to the exit plan")
    ids: set[str] = set()
    scenarios: list[PlannedExitScenario] = []
    for item in scenarios_raw:
        record = _only(
            item,
            {"id", "outcome", "probability", "underlying_spot"},
            {"id", "outcome", "probability", "underlying_spot"},
            "planned_exit_scenario",
        )
        identifier = _str(record["id"], "planned_exit_scenario.id")
        probability = _decimal(record["probability"], "planned_exit_scenario.probability")
        assert probability is not None
        if identifier in ids or probability < Decimal() or probability > Decimal("1"):
            raise SchemaError("planned exit scenario ID/probability is invalid")
        ids.add(identifier)
        spot = _decimal(record["underlying_spot"], "planned_exit_scenario.underlying_spot")
        assert spot is not None
        if spot <= Decimal():
            raise SchemaError("planned exit underlying spot must be positive")
        scenarios.append(
            PlannedExitScenario(
                identifier,
                _str(record["outcome"], "planned_exit_scenario.outcome"),
                probability,
                spot,
            )
        )
    if sum(item.probability for item in scenarios) != Decimal("1"):
        raise SchemaError("planned exit scenario probabilities must sum exactly to 1")
    sensitivity: list[SensitivityCase] = []
    seen: set[str] = set()
    for item in sensitivity_raw:
        record = _only(item, {"id", "model", "probability_shift_to_worst", "additional_cost", "expected_value"}, {"id", "model", "probability_shift_to_worst", "additional_cost", "expected_value"}, "planned_exit_sensitivity")
        identifier = _str(record["id"], "planned_exit_sensitivity.id")
        shift = _decimal(record["probability_shift_to_worst"], "planned_exit_sensitivity.probability_shift_to_worst")
        cost = _decimal(record["additional_cost"], "planned_exit_sensitivity.additional_cost")
        assert shift is not None and cost is not None
        if identifier in seen or shift <= Decimal() or shift > Decimal("0.25") or cost < Decimal():
            raise SchemaError("planned exit sensitivity must be bounded and non-negative")
        seen.add(identifier)
        sensitivity.append(SensitivityCase(identifier, _str(record["model"], "planned_exit_sensitivity.model"), shift, cost, _decimal(record["expected_value"], "planned_exit_sensitivity.expected_value") or Decimal()))
    return PlannedExitValuation("intrinsic_close_v1", valuation_at, source_id, tuple(scenarios), tuple(sensitivity))


def _scenario_support_bounds(
    underlying: Decimal, metadata: RunMetadata, selected: Structure, volatility: VolatilityContext
) -> tuple[Decimal, Decimal]:
    """Auditable 30-day terminal support, deliberately capped before tail fantasy."""
    latest = max(date.fromisoformat(leg.contract.expiration) for leg in selected.legs)
    days = min(30, max(0, (latest - metadata.as_of.date()).days))
    horizon = Decimal(max(1, days)) / Decimal("365")
    realized_tail = volatility.realized_volatility * horizon.sqrt() * Decimal("4")
    implied_tail = volatility.implied_move_pct * Decimal("3")
    observed_tail = max(
        (abs(event.observed_move_pct) * Decimal("3") for event in volatility.event_history),
        default=Decimal(),
    )
    # A 10% floor avoids underrepresenting ordinary terminal uncertainty; 75% is
    # a hard Phase-1 ceiling rather than an invitation to invent unlimited tails.
    move = min(Decimal("0.75"), max(Decimal("0.10"), realized_tail, implied_tail, observed_tail))
    return max(Decimal(), underlying * (Decimal("1") - move)), underlying * (Decimal("1") + move)


def _portfolio(raw: object) -> PortfolioAssessment:
    keys = {
        "positions",
        "limits",
        "correlations",
        "aggregate_risk",
        "cluster_risk",
        "event_risk",
        "sector_risk",
        "factor_risk",
        "deficit_elimination_rationale",
    }
    value = _only(raw, keys, keys, "portfolio_assessment")
    if (
        not isinstance(value["positions"], list)
        or len(value["positions"]) > _MAX_RECORDS
        or not isinstance(value["correlations"], list)
        or len(value["correlations"]) > _MAX_RECORDS
    ):
        raise SchemaError("portfolio positions and correlations must be bounded lists")
    positions: list[PortfolioPosition] = []
    position_ids: set[str] = set()
    for item in value["positions"]:
        record = _only(
            item,
            {"id", "symbol", "sector", "factor_tags", "risk", "event_risk"},
            {"id", "symbol", "sector", "factor_tags", "risk", "event_risk"},
            "portfolio_position",
        )
        identifier = _str(record["id"], "portfolio_position.id")
        if identifier in position_ids:
            raise SchemaError("duplicate portfolio position ID")
        position_ids.add(identifier)
        risk, event_risk = (
            _decimal(record["risk"], "portfolio_position.risk") or Decimal(),
            _decimal(record["event_risk"], "portfolio_position.event_risk") or Decimal(),
        )
        if risk < Decimal() or event_risk < Decimal():
            raise SchemaError("portfolio position risk cannot be negative")
        positions.append(
            PortfolioPosition(
                identifier,
                _str(record["symbol"], "portfolio_position.symbol"),
                _str(record["sector"], "portfolio_position.sector"),
                _strings(record["factor_tags"], "portfolio_position.factor_tags"),
                risk,
                event_risk,
            )
        )
    limit = _only(
        value["limits"],
        {
            "hard_cap",
            "remaining_aggregate",
            "remaining_cluster",
            "remaining_event",
            "remaining_sector",
            "remaining_factor",
        },
        {
            "hard_cap",
            "remaining_aggregate",
            "remaining_cluster",
            "remaining_event",
            "remaining_sector",
            "remaining_factor",
        },
        "portfolio_limits",
    )
    limits = PortfolioLimits(
        *(
            _decimal(limit[key], f"portfolio_limits.{key}") or Decimal()
            for key in (
                "hard_cap",
                "remaining_aggregate",
                "remaining_cluster",
                "remaining_event",
                "remaining_sector",
                "remaining_factor",
            )
        )
    )
    if limits.hard_cap != Decimal("1000"):
        raise SchemaError("the universal fee-inclusive cap must be exactly 1000")
    if any(item < Decimal() for item in limits.__dict__.values()):
        raise SchemaError("portfolio limits cannot be negative")
    correlations: list[CorrelationRecord] = []
    correlated_ids: set[str] = set()
    for item in value["correlations"]:
        record = _only(
            item,
            {"position_id", "correlation", "rationale"},
            {"position_id", "correlation", "rationale"},
            "correlation",
        )
        position_id, correlation = (
            _str(record["position_id"], "correlation.position_id"),
            _decimal(record["correlation"], "correlation.correlation"),
        )
        assert correlation is not None
        if (
            position_id in correlated_ids
            or position_id not in position_ids
            or correlation < Decimal("-1")
            or correlation > Decimal("1")
        ):
            raise SchemaError("broken/duplicate correlation or value outside -1..1")
        correlated_ids.add(position_id)
        correlations.append(
            CorrelationRecord(
                position_id, correlation, _str(record["rationale"], "correlation.rationale")
            )
        )
    aggregate, cluster, event, sector, factor = (
        _decimal(value[f"{name}_risk"], f"portfolio_assessment.{name}_risk") or Decimal()
        for name in ("aggregate", "cluster", "event", "sector", "factor")
    )
    if min(aggregate, cluster, event, sector, factor) < Decimal():
        raise SchemaError("portfolio assessment risk cannot be negative")
    return PortfolioAssessment(
        tuple(positions),
        limits,
        tuple(correlations),
        aggregate,
        cluster,
        event,
        sector,
        factor,
        _str(
            value["deficit_elimination_rationale"],
            "portfolio_assessment.deficit_elimination_rationale",
        ),
    )


def _candidate(raw: object, metadata: RunMetadata, source: SourceLabel, fixture: bool) -> Candidate:
    required = {
        "id",
        "symbol",
        "underlying",
        "underlying_as_of",
        "contracts",
        "structures",
        "selected_structure_id",
        "thesis",
        "sources",
        "claims",
        "equity_context",
        "quote_provenance",
        "mechanics",
        "fill_plan",
        "structure_plan",
        "surface_nodes",
        "iv_snapshot",
        "event_history",
        "upcoming_events",
        "volatility",
        "distribution",
        "portfolio_assessment",
        "skeptic",
        "judge",
        "watch_triggers",
    }
    value = _only(
        raw,
        required | {"planned_exit_valuation", "alternative_economics"},
        required,
        "candidate",
    )
    identifier, symbol = (
        _str(value["id"], "candidate.id"),
        _str(value["symbol"], "candidate.symbol"),
    )
    underlying, underlying_as_of = (
        _decimal(value["underlying"], "candidate.underlying"),
        _time(value["underlying_as_of"], "candidate.underlying_as_of"),
    )
    assert underlying is not None
    if underlying <= Decimal() or underlying_as_of > metadata.retrieved_at:
        raise SchemaError("invalid underlying or its retrieval ordering")
    contracts = _contracts(value["contracts"], symbol, source, metadata.as_of)
    structures, _ = _structures(value["structures"], contracts)
    selected_id = _str(value["selected_structure_id"], "candidate.selected_structure_id")
    if selected_id not in structures:
        raise SchemaError("broken selected structure reference")
    sources, claims = _sources_claims(value["sources"], value["claims"], metadata)
    quote_raw = _only(
        value["quote_provenance"],
        {"source_id", "retrieved_at", "as_of", "source", "methodology"},
        {"source_id", "retrieved_at", "as_of", "source", "methodology"},
        "quote_provenance",
    )
    quote_source_id = _str(quote_raw["source_id"], "quote_provenance.source_id")
    if quote_source_id not in {item.id for item in sources}:
        raise SchemaError("broken quote provenance source reference")
    quote_source_record = next(item for item in sources if item.id == quote_source_id)
    quote_retrieved, quote_as_of = (
        _time(quote_raw["retrieved_at"], "quote_provenance.retrieved_at"),
        _time(quote_raw["as_of"], "quote_provenance.as_of"),
    )
    quote_source = _source_label(quote_raw["source"], "quote_provenance.source")
    if (
        quote_retrieved < quote_as_of
        or quote_retrieved > metadata.retrieved_at
        or quote_source is not source
    ):
        raise SchemaError("invalid quote provenance ordering")
    # A research article can remain useful historical background, but it may
    # not authorize a current LIVE quote.  The declared quote source must be
    # a fresh, typed source record linked to at least one claim; exact quote
    # timestamps are checked again at the operational capture boundary.
    if source is SourceLabel.LIVE and (
        quote_source_record.published_at > metadata.as_of
        or quote_source_record.retrieved_at > metadata.retrieved_at
        or (metadata.as_of - quote_source_record.published_at).total_seconds() > 86_400
    ):
        raise SchemaError("LIVE quote provenance must resolve to a fresh linked source record")
    mechanics = _mechanics(value["mechanics"])
    selected = structures[selected_id]
    if (
        any(
            leg.contract.multiplier != selected.legs[0].contract.multiplier for leg in selected.legs
        )
        or any(
            leg.contract.expiration != selected.legs[0].contract.expiration for leg in selected.legs
        )
        or any(
            leg.contract.exercise_style != mechanics.exercise_style
            or leg.contract.settlement_style != mechanics.settlement_style
            for leg in selected.legs
        )
    ):
        raise SchemaError("incompatible structure multiplier, expiration, exercise, or settlement")
    if mechanics.product_type.casefold() == "index" and (
        mechanics.exercise_style != "European" or mechanics.settlement_style != "cash"
    ):
        raise SchemaError("index product must be European cash-settled")
    fill, structure_plan = _plans(
        value["fill_plan"],
        value["structure_plan"],
        metadata.as_of,
        selected.legs[0].contract.expiration,
    )
    declared_topology = classify_structure(
        Structure(structure_plan.structure_type, selected.legs, selected.quantity)
    )
    selected_topology = classify_structure(selected)
    if (
        selected_topology.supported
        and (declared_topology.kind != selected_topology.kind or not declared_topology.supported)
    ):
        raise SchemaError("structure_plan.structure_type must exactly match selected canonical topology")
    surface_nodes, volatility = _volatility(
        value["surface_nodes"],
        value["iv_snapshot"],
        value["event_history"],
        value["upcoming_events"],
        value["volatility"],
        metadata.as_of,
    )
    distribution = _distribution(value["distribution"])
    planned_exit_valuation = _planned_exit_valuation(
        value.get("planned_exit_valuation"), structure_plan.exit_plan, sources
    )
    low_spot, high_spot = _scenario_support_bounds(underlying, metadata, selected, volatility)
    for scenario in distribution.scenarios:
        if not low_spot <= scenario.expiration_spot <= high_spot:
            raise SchemaError(
                "scenario expiration spot outside deterministic 30-day support bounds"
            )
        derived = expiration_pnl(
            selected,
            scenario.expiration_spot,
            structure_plan.fees,
            structure_plan.commissions,
            structure_plan.exit_slippage,
            fill.max_slippage,
            entry=structure_plan.max_acceptable_limit,
        )
        if scenario.payoff != derived:
            raise SchemaError(
                "scenario payoff must exactly match selected-structure maximum-fill expiration economics including costs"
            )
    alternative_economics = _alternative_economics(
        value.get("alternative_economics", []), structures, selected_id
    )
    skeptic_raw = _only(
        value["skeptic"],
        {
            "opposing_view",
            "weakest_evidence",
            "key_assumption",
            "crowding",
            "beta",
            "iv",
            "liquidity",
            "total_loss",
        },
        {
            "opposing_view",
            "weakest_evidence",
            "key_assumption",
            "crowding",
            "beta",
            "iv",
            "liquidity",
            "total_loss",
        },
        "skeptic",
    )
    judge_raw = _only(value["judge"], {"verdict", "reason"}, {"verdict", "reason"}, "judge")
    if not isinstance(value["watch_triggers"], list) or len(value["watch_triggers"]) > _MAX_RECORDS:
        raise SchemaError("watch_triggers must be a bounded list")
    triggers = tuple(
        WatchTrigger(
            _str(record["condition"], "watch_trigger.condition"),
            _time(record["expires_at"], "watch_trigger.expires_at"),
            _str(record["action"], "watch_trigger.action"),
        )
        for item in value["watch_triggers"]
        for record in [
            _only(
                item,
                {"condition", "expires_at", "action"},
                {"condition", "expires_at", "action"},
                "watch_trigger",
            )
        ]
    )
    if any(trigger.expires_at < metadata.as_of for trigger in triggers):
        raise SchemaError("watch trigger expires before run as_of")
    return Candidate(
        symbol,
        underlying,
        underlying_as_of,
        source,
        selected,
        _thesis(value["thesis"], metadata.as_of),
        claims,
        fixture=fixture,
        id=identifier,
        structure_set=tuple(structures[key] for key in sorted(structures)),
        contract_chain=tuple(contracts.values()),
        alternative_economics=alternative_economics,
        source_records=sources,
        equity_context=_equity(value["equity_context"]),
        quote_provenance=QuoteProvenance(
            quote_source_id,
            quote_retrieved,
            quote_as_of,
            quote_source,
            _str(quote_raw["methodology"], "quote_provenance.methodology"),
        ),
        mechanics=mechanics,
        fill_plan=fill,
        structure_plan=structure_plan,
        surface_nodes=surface_nodes,
        volatility_context=volatility,
        distribution=distribution,
        planned_exit_valuation=planned_exit_valuation,
        portfolio_assessment=_portfolio(value["portfolio_assessment"]),
        skeptic=SkepticRecord(
            *(
                _str(skeptic_raw[key], f"skeptic.{key}")
                for key in (
                    "opposing_view",
                    "weakest_evidence",
                    "key_assumption",
                    "crowding",
                    "beta",
                    "iv",
                    "liquidity",
                    "total_loss",
                )
            )
        ),
        judge=JudgeRecord(
            _str(judge_raw["verdict"], "judge.verdict"), _str(judge_raw["reason"], "judge.reason")
        ),
        watch_triggers=triggers,
    )


@dataclass(frozen=True)
class ParsedRun:
    metadata: RunMetadata
    fixture: bool
    source: SourceLabel
    candidates: tuple[Candidate, ...]
    capture_binding: CaptureBinding | None = None
    records: tuple[dict[str, Any], ...] = ()

    @property
    def as_of(self) -> datetime:
        return self.metadata.as_of

    @property
    def portfolio(self) -> None:
        return None


def parse_run(raw: dict[str, Any]) -> ParsedRun:
    value = _only(
        raw,
        {"schema_version", "metadata", "source_label", "fixture", "candidates", "capture_binding"},
        {"schema_version", "metadata", "source_label", "fixture", "candidates"},
        "normalized-run",
    )
    if (
        value["schema_version"] != "1"
        or type(value["fixture"]) is not bool
        or not isinstance(value["candidates"], list)
        or len(value["candidates"]) > _MAX_CANDIDATES
    ):
        raise SchemaError("invalid normalized-run version/types")
    metadata, source, fixture = (
        _metadata(value["metadata"]),
        _source_label(value["source_label"], "source_label"),
        value["fixture"],
    )
    candidates = tuple(_candidate(item, metadata, source, fixture) for item in value["candidates"])
    if len({item.id for item in candidates}) != len(candidates):
        raise SchemaError("duplicate candidate ID")
    binding = _capture_binding(value.get("capture_binding"))
    if binding is not None and (
        fixture or source is not SourceLabel.LIVE or binding.source_label is not SourceLabel.LIVE
    ):
        raise SchemaError("capture binding is only valid for non-fixture LIVE input")
    return ParsedRun(metadata, fixture, source, candidates, binding)


def load_run(path: Path) -> ParsedRun:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SchemaError("invalid normalized-run JSON") from error
    if not isinstance(raw, dict):
        raise SchemaError("normalized-run must be an object")
    return parse_run(raw)
