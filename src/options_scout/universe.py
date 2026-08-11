"""Deterministic, local-only baseline universe planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_FIELDS = {"version", "max_universe", "categories"}
DEFAULT_UNIVERSE = {
    "version": "1",
    "max_universe": 40,
    "categories": {
        "broad_market": ["SPY", "QQQ", "IWM", "DIA"],
        "technology": ["XLK", "IGV"],
        "semiconductors": ["SMH", "SOXX"],
        "biotech": ["XBI", "IBB"],
        "energy": ["XLE", "USO"],
        "financials": ["XLF", "KRE"],
        "rates": ["TLT", "IEF"],
        "liquid_factors": ["GLD", "XLV", "XLY", "XLU", "XLRE"],
    },
}


class UniverseError(ValueError):
    pass


@dataclass(frozen=True)
class UniversePlan:
    max_universe: int
    categories: dict[str, tuple[str, ...]]

    def symbols(self, *additional_sources: tuple[str, ...]) -> tuple[str, ...]:
        """Merge deterministic sources, de-duplicate in first-seen order, then cap."""
        ordered = [symbol for values in self.categories.values() for symbol in values]
        ordered.extend(symbol for source in additional_sources for symbol in source)
        result: list[str] = []
        for symbol in ordered:
            if symbol not in result:
                result.append(symbol)
            if len(result) == self.max_universe:
                break
        return tuple(result)

    def health(self) -> dict[str, object]:
        return {
            "status": "LOCAL_DETERMINISTIC_CANDIDATE_PLAN",
            "max_universe": self.max_universe,
            "categories": {name: len(symbols) for name, symbols in self.categories.items()},
            "baseline_count": len(self.symbols()),
            "live_catalog_dependency": "NONE",
        }


def parse_universe(raw: object) -> UniversePlan:
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise UniverseError("universe config fields are unknown or incomplete")
    if (
        raw["version"] != "1"
        or type(raw["max_universe"]) is not int
        or not 1 <= raw["max_universe"] <= 500
    ):
        raise UniverseError("universe version or cap is invalid")
    categories_raw = raw["categories"]
    if not isinstance(categories_raw, dict) or not categories_raw:
        raise UniverseError("universe categories are invalid")
    categories: dict[str, tuple[str, ...]] = {}
    for name, symbols in categories_raw.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z_]{1,39}", name):
            raise UniverseError("universe category is invalid")
        if not isinstance(symbols, list) or not symbols or len(symbols) > 100:
            raise UniverseError("universe category symbols are invalid")
        if any(not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol) for symbol in symbols):
            raise UniverseError("universe symbol is invalid")
        categories[name] = tuple(symbols)
    return UniversePlan(raw["max_universe"], categories)


def load_universe(root: Path) -> UniversePlan:
    try:
        raw = json.loads((root / "config/universe.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UniverseError(f"invalid universe config: {error}") from error
    return parse_universe(raw)
