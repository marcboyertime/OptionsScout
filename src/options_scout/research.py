from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThesisCheck:
    passed: bool
    reason: str


def evaluate_thesis(
    thesis: dict[str, object], claim_ids: set[str], sector_beta: bool
) -> list[ThesisCheck]:
    required = (
        "implied_probability_range",
        "outcome",
        "why_wrong",
        "why_not_arbitraged",
        "falsifier",
        "catalyst",
        "timing_trigger",
    )
    checks = [
        ThesisCheck(
            all(thesis.get(key) for key in required),
            "complete X/Y/Z, arbitrage explanation, falsifier, and timing",
        )
    ]
    checks.append(ThesisCheck(bool(claim_ids), "claim-to-source evidence exists"))
    assumptions = thesis.get("primary_assumptions", [])
    checks.append(
        ThesisCheck(
            not isinstance(assumptions, list) or len(assumptions) <= 3 or len(claim_ids) >= 3,
            "complex thesis has stronger evidence",
        )
    )
    checks.append(ThesisCheck(not sector_beta, "single-name thesis is not sector beta"))
    return checks
