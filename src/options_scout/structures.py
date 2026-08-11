from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import Contract, Leg, OptionType, Side, Structure

SUPPORTED = {
    "long_call",
    "long_put",
    "call_debit",
    "put_debit",
    "call_credit",
    "put_credit",
    "iron_condor",
    "iron_butterfly",
    "butterfly",
}
PRE_EXPIRY_ONLY = {"calendar", "diagonal"}


@dataclass(frozen=True)
class StructureClassification:
    kind: str
    supported: bool
    defined_risk: bool
    pre_expiration_only: bool
    reason: str


def _canonical(name: str) -> str:
    value = name.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "longcall": "long_call",
        "longput": "long_put",
        "call_debit_spread": "call_debit",
        "put_debit_spread": "put_debit",
        "call_credit_spread": "call_credit",
        "put_credit_spread": "put_credit",
        "ironcondor": "iron_condor",
        "ironbutterfly": "iron_butterfly",
        "cash_call_debit": "call_debit",
    }
    return aliases.get(value, value)


def classify_structure(structure: Structure) -> StructureClassification:
    kind = _canonical(structure.name)
    if kind in PRE_EXPIRY_ONLY:
        return StructureClassification(
            kind, False, False, True, "requires compatible pre-expiration valuation"
        )
    if kind not in SUPPORTED:
        return StructureClassification(kind, False, False, False, "unsupported structure type")
    legs = structure.legs
    # A finite payoff sampled at a few knots is not proof that a declared
    # complex is covered.  Named structures are a deliberately small,
    # canonical grammar: one series, positive ratios, distinct contracts, and
    # the exact side/strike geometry below.  Quantity scales the whole
    # complex, so every per-complex ratio remains canonical.
    if not legs or any(type(leg.ratio) is not int or leg.ratio <= 0 for leg in legs):
        return StructureClassification(
            kind, False, False, False, "legs need positive integer ratios"
        )
    series = {
        (leg.contract.symbol, leg.contract.expiration, leg.contract.multiplier) for leg in legs
    }
    if len(series) != 1 or len({leg.contract.id for leg in legs}) != len(legs):
        return StructureClassification(
            kind, False, False, False, "legs must be distinct contracts in one compatible series"
        )

    def is_leg(leg_index: int, side: Side, option_type: OptionType, ratio: int = 1) -> bool:
        leg = legs[leg_index]
        return leg.side is side and leg.contract.option_type is option_type and leg.ratio == ratio

    if kind == "long_call":
        valid = len(legs) == 1 and is_leg(0, Side.BUY, OptionType.CALL)
    elif kind == "long_put":
        valid = len(legs) == 1 and is_leg(0, Side.BUY, OptionType.PUT)
    elif kind == "call_debit":
        valid = (
            len(legs) == 2
            and is_leg(0, Side.BUY, OptionType.CALL)
            and is_leg(1, Side.SELL, OptionType.CALL)
            and legs[0].contract.strike < legs[1].contract.strike
        )
    elif kind == "call_credit":
        valid = (
            len(legs) == 2
            and is_leg(0, Side.SELL, OptionType.CALL)
            and is_leg(1, Side.BUY, OptionType.CALL)
            and legs[0].contract.strike < legs[1].contract.strike
        )
    elif kind == "put_debit":
        valid = (
            len(legs) == 2
            and is_leg(0, Side.BUY, OptionType.PUT)
            and is_leg(1, Side.SELL, OptionType.PUT)
            and legs[1].contract.strike < legs[0].contract.strike
        )
    elif kind == "put_credit":
        valid = (
            len(legs) == 2
            and is_leg(0, Side.SELL, OptionType.PUT)
            and is_leg(1, Side.BUY, OptionType.PUT)
            and legs[1].contract.strike < legs[0].contract.strike
        )
    elif kind == "iron_condor":
        valid = (
            len(legs) == 4
            and is_leg(0, Side.BUY, OptionType.PUT)
            and is_leg(1, Side.SELL, OptionType.PUT)
            and is_leg(2, Side.SELL, OptionType.CALL)
            and is_leg(3, Side.BUY, OptionType.CALL)
            and legs[0].contract.strike
            < legs[1].contract.strike
            < legs[2].contract.strike
            < legs[3].contract.strike
        )
    elif kind == "iron_butterfly":
        valid = (
            len(legs) == 4
            and is_leg(0, Side.BUY, OptionType.PUT)
            and is_leg(1, Side.SELL, OptionType.PUT)
            and is_leg(2, Side.SELL, OptionType.CALL)
            and is_leg(3, Side.BUY, OptionType.CALL)
            and legs[0].contract.strike
            < legs[1].contract.strike
            == legs[2].contract.strike
            < legs[3].contract.strike
        )
    else:  # butterfly
        valid = (
            len(legs) == 3
            and legs[0].contract.option_type
            is legs[1].contract.option_type
            is legs[2].contract.option_type
            and is_leg(0, Side.BUY, legs[0].contract.option_type)
            and is_leg(1, Side.SELL, legs[1].contract.option_type, 2)
            and is_leg(2, Side.BUY, legs[2].contract.option_type)
            and legs[0].contract.strike < legs[1].contract.strike < legs[2].contract.strike
            and legs[1].contract.strike - legs[0].contract.strike
            == legs[2].contract.strike - legs[1].contract.strike
        )
    if not valid:
        return StructureClassification(
            kind, False, False, False, "legs do not match declared structure"
        )
    return StructureClassification(kind, True, True, False, "recognized defined-risk complex")


def validate_structure_plan(structure: Structure) -> list[str]:
    classification = classify_structure(structure)
    errors: list[str] = []
    if not classification.supported:
        errors.append(classification.reason)
    if len(structure.legs) > 1 and not structure.name:
        errors.append("multi-leg plan requires one complex net order")
    return errors


def compare_structures(
    structures: tuple[Structure, ...], expected_values: dict[str, Decimal]
) -> list[Structure]:
    """Deterministic, evidence-fed comparison; unsupported products never compete."""
    return sorted(
        (item for item in structures if classify_structure(item).supported),
        key=lambda item: (expected_values.get(item.name, Decimal("-Infinity")), item.name),
        reverse=True,
    )


def generate_supported_structures(
    contracts: tuple[Contract, ...], *, cap: int = 20
) -> tuple[Structure, ...]:
    """Return a bounded, canonical comparison set from a compatible chain.

    This function intentionally creates topology candidates only.  It does
    not invent commissions, exit assumptions or a probability distribution;
    callers must render those generated entries as UNAVAILABLE until reviewed
    independently.  Ordered contract IDs make the result deterministic.
    """
    if cap <= 0:
        return ()
    ordered = sorted(contracts, key=lambda item: (item.expiration, item.option_type.value, item.strike, item.id))
    generated: list[Structure] = []

    def add(name: str, legs: tuple[Leg, ...]) -> None:
        if len(generated) >= cap:
            return
        item = Structure(name, legs, identifier=f"generated:{name}:" + ":".join(leg.contract.id for leg in legs))
        if classify_structure(item).supported:
            generated.append(item)

    for contract in ordered:
        add("long_call" if contract.option_type is OptionType.CALL else "long_put", (Leg(Side.BUY, contract),))
    for option_type in (OptionType.CALL, OptionType.PUT):
        series = [item for item in ordered if item.option_type is option_type]
        for lower, upper in zip(series, series[1:], strict=False):
            if (lower.symbol, lower.expiration, lower.multiplier) != (upper.symbol, upper.expiration, upper.multiplier):
                continue
            if option_type is OptionType.CALL:
                add("call_debit", (Leg(Side.BUY, lower), Leg(Side.SELL, upper)))
                add("call_credit", (Leg(Side.SELL, lower), Leg(Side.BUY, upper)))
            else:
                add("put_credit", (Leg(Side.SELL, upper), Leg(Side.BUY, lower)))
                add("put_debit", (Leg(Side.BUY, upper), Leg(Side.SELL, lower)))
    return tuple(generated[:cap])
