"""Local reports with a populated, review-only recommendation layout."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

REPORT_VERSION = "phase1-report-v1"


def _value(value: Any) -> str:
    if value is None or value == "":
        return "UNAVAILABLE"
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, dict | list)
        else str(value)
    )


def _at(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _failures(payload: dict[str, Any]) -> list[str]:
    records = list(payload.get("gates", []))
    for evaluation in payload.get("evaluations", []):
        records.extend(evaluation.get("gates", []))
    return [
        str(item.get("reason", "unspecified gate failure"))
        for item in records
        if item.get("status") == "FAIL" or ("status" not in item and item.get("passed") is False)
    ]


def _section(lines: list[str], number: int, title: str, content: Any) -> None:
    lines.extend([f"## {number}. {title}", "", _value(content), ""])


def _ticket(evaluation: dict[str, Any]) -> dict[str, Any]:
    candidate, analysis = evaluation.get("candidate", {}), evaluation.get("analysis", {})
    structure = candidate.get("structure", {})
    legs = [
        {
            "side": leg.get("side"),
            "contract_id": _at(leg, "contract", "id"),
            "strike": _at(leg, "contract", "strike"),
            "type": _at(leg, "contract", "option_type"),
        }
        for leg in structure.get("legs", [])
    ]
    payoff = analysis.get("payoff", {})
    return {
        "ticker": evaluation.get("symbol"),
        "expiration": [_at(leg, "contract", "expiration") for leg in structure.get("legs", [])],
        "buy_sell_legs": legs,
        "quantity": structure.get("quantity"),
        "realistic_entry": _at(payoff, "fills", "realistic_limit_entry"),
        "target_limit": _at(candidate, "fill_plan", "limit"),
        "declared_structure_entry_limit": _at(candidate, "structure_plan", "entry_limit"),
        "max_acceptable_limit": _at(candidate, "structure_plan", "max_acceptable_limit"),
        "theoretical_quote_derived_max_loss": payoff.get("theoretical_max_loss"),
        "operational_fee_slippage_max_fill_loss": payoff.get("operational_max_loss_risk"),
        "max_gain_at_max_acceptable_fill": payoff.get("max_gain"),
        "breakeven_at_max_acceptable_fill": payoff.get("breakevens"),
        "underlying": candidate.get("underlying"),
        "underlying_timestamp": candidate.get("underlying_as_of"),
        "leg_timestamps": [
            _at(leg, "contract", "quote", "as_of") for leg in structure.get("legs", [])
        ],
        "economics_invalidation": _at(candidate, "structure_plan", "exit_plan", "invalidation"),
        "trigger": _at(candidate, "thesis_record", "timing_trigger"),
        "classification_risks": {
            key: payoff.get(key)
            for key in (
                "theoretical_max_loss",
                "operational_max_loss_risk",
                "assignment_risk",
                "physical_settlement_risk",
                "possibility_of_account_deficit",
            )
        },
        "submission": "PREVIEW ONLY — NOT SUBMITTED",
    }


def markdown_report(trace_id: str, payload: dict[str, Any]) -> str:
    state = str(payload.get("decision", "DATA_INSUFFICIENT"))
    lines = [
        "# OptionsScout report",
        "",
        f"- Report version: `{REPORT_VERSION}`",
        f"- Trace: `{trace_id}`",
        f"- Decision: **{state}**",
        f"- Data label: `{payload.get('source_label', 'UNAVAILABLE')}`",
        f"- Freshness: {_value(payload.get('freshness'))}",
        f"- Normalized input hash: `{payload.get('normalized_input_hash', 'UNAVAILABLE')}`",
        f"- Funnel: `{json.dumps(payload.get('counts', {}), sort_keys=True)}`",
        "",
        "## Source health",
        "",
        _value(payload.get("source_health")),
        "",
    ]
    failures = _failures(payload)
    if failures:
        lines += ["## Rejections / blockers", "", *[f"- {reason}" for reason in failures], ""]
    if state != "ACTIONABLE":
        lines += ["No recommendation is available. Research-only; no order was submitted.", ""]
    else:
        evaluation: dict[str, Any] = next(
            (item for item in payload.get("ranked", []) if isinstance(item, dict)), {}
        ) or next(
            (
                item
                for item in payload.get("evaluations", [])
                if isinstance(item, dict) and item.get("decision") == "ACTIONABLE"
            ),
            {},
        )
        candidate, analysis = evaluation.get("candidate", {}), evaluation.get("analysis", {})
        thesis = candidate.get("thesis_record", {})
        _section(
            lines,
            1,
            "Executive summary",
            {
                "symbol": evaluation.get("symbol"),
                "structure": evaluation.get("structure"),
                "decision": evaluation.get("decision"),
            },
        )
        _section(
            lines,
            2,
            "Best trade",
            {
                "classification": _at(analysis, "structure", "kind"),
                "operational": evaluation.get("operational"),
            },
        )
        _section(
            lines,
            3,
            "Market belief, range, outcome, and why wrong",
            {
                "implied_probability_low": thesis.get("implied_probability_low"),
                "implied_probability_high": thesis.get("implied_probability_high"),
                "outcome": thesis.get("outcome"),
                "why_wrong": thesis.get("why_wrong"),
            },
        )
        _section(
            lines,
            4,
            "Narrative",
            {
                "why_not_arbitraged": thesis.get("why_not_arbitraged"),
                "underappreciation": thesis.get("underappreciation"),
            },
        )
        _section(
            lines,
            5,
            "Catalyst and timing",
            {
                "catalyst": thesis.get("catalyst"),
                "catalyst_type": thesis.get("catalyst_type"),
                "catalyst_at": thesis.get("catalyst_at"),
                "timing_trigger": thesis.get("timing_trigger"),
            },
        )
        _section(
            lines,
            6,
            "Fundamentals, claims, and sources",
            {"claims": candidate.get("claim_records"), "sources": candidate.get("source_records")},
        )
        _section(lines, 7, "Technical and factor context", candidate.get("equity_context"))
        _section(lines, 8, "Chain and quotes", _at(candidate, "structure", "legs"))
        _section(
            lines,
            9,
            "Volatility and IV crush",
            {"volatility": analysis.get("volatility"), "iv_crush": analysis.get("iv_crush")},
        )
        _section(
            lines,
            10,
            "Implied move versus breakeven",
            {
                "implied_move_pct": _at(candidate, "volatility_context", "implied_move_pct"),
                "breakevens": _at(analysis, "payoff", "breakevens"),
            },
        )
        _section(
            lines,
            11,
            "Structure comparison",
            {
                "selected_structure": analysis.get("structure"),
                "selected_payoff": analysis.get("payoff"),
                "comparison": analysis.get("structure_comparison"),
                "selection_rule": "selected structure must be the deterministic highest exact terminal EV among supplied supported alternatives",
            },
        )
        _section(lines, 12, "PREVIEW ONLY — NOT SUBMITTED", _ticket(evaluation))
        _section(lines, 13, "Distribution", analysis.get("distribution"))
        _section(lines, 14, "Portfolio and correlation", candidate.get("portfolio_assessment"))
        _section(lines, 15, "Exit and invalidation", _at(candidate, "structure_plan", "exit_plan"))
        _section(
            lines,
            16,
            "Alternatives",
            {"watch_triggers": candidate.get("watch_triggers"), "rejected": failures},
        )
        _section(lines, 17, "Red team", analysis.get("red_team"))
        _section(lines, 18, "Judge verdict", candidate.get("judge"))
        _section(
            lines,
            19,
            "Gate ledger and preflight requirements",
            {
                "gates": evaluation.get("gates"),
                "preflight": "Fresh quotes, mechanics, evidence, and portfolio inputs must match; compare with Robinhood review screen only.",
            },
        )
    lines += [
        "Every structure records: `THEORETICAL_MAX_LOSS`, `OPERATIONAL_MAX_LOSS_RISK`, `ASSIGNMENT_RISK`, `PHYSICAL_SETTLEMENT_RISK`, and `POSSIBILITY_OF_ACCOUNT_DEFICIT`.",
        "",
    ]
    return "\n".join(lines)


def html_report(trace_id: str, payload: dict[str, Any]) -> str:
    title = html.escape(f"OptionsScout {payload.get('decision', 'report')}")
    body = html.escape(markdown_report(trace_id, payload))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; script-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\"><meta name=\"referrer\" content=\"no-referrer\"><title>"
        + title
        + "</title><style>body{font:15px system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#17202a}pre{white-space:pre-wrap;word-break:break-word;background:#f5f7f8;padding:1rem;border-radius:.4rem}</style></head><body><h1>"
        + title
        + "</h1><p>Research-only; no order was submitted.</p><pre>"
        + body
        + "</pre></body></html>"
    )


def write_reports(root: Path, trace_id: str, payload: dict[str, Any]) -> dict[str, str]:
    directory = root / "artifacts" / "reports" / trace_id
    directory.mkdir(parents=True, exist_ok=True)
    document = {"report_version": REPORT_VERSION, **payload}
    json_path, markdown_path, html_path = (
        directory / "report.json",
        directory / "report.md",
        directory / "report.html",
    )
    json_path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, default=str)
    )
    markdown_path.write_text(markdown_report(trace_id, document))
    html_path.write_text(html_report(trace_id, document))
    return {"json": str(json_path), "markdown": str(markdown_path), "html": str(html_path)}
