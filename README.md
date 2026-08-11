# OptionsScout Phase 1

OptionsScout is a local-first, deterministic, **research-only** options scanner. It is designed to return `NO_TRADE` or `DATA_INSUFFICIENT` freely. It cannot submit, alter, or simulate an executable order.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
.venv/bin/options-scout init
.venv/bin/options-scout health --json
.venv/bin/options-scout scan --fixture --json
```

Major operations: `init`, `health`, `scan`, `analyze`, `capture-ingest`, `preflight`, `portfolio-check`, `report`, `history`, `source-health`, `safety-audit`, `calibration`, and `schedule-plan`. Use `--fixture` only for demo data; it is visibly non-live and cannot become actionable.

`scan --input FILE` and `analyze --input FILE` require a strict immutable-capture binding before an input may be treated as current LIVE. A self-labelled normalized JSON file is a research snapshot, not live authorization. The default repository policy has an intentionally empty reviewed allowlist, so non-fixture operational input returns `DATA_INSUFFICIENT`; use `--fixture` only for visibly non-live demonstrations. After a human has installed a strict reviewed read-only LIVE policy, `capture-ingest --capture-input REDACTED_ENVELOPE.json` validates and appends one already-redacted envelope without calling a broker. Review records pin canonical SHA-256 parameter, response, and complete normalized-projection schemas. A LIVE payload is exactly that fixed projection—no opaque authenticated response or nested extra object is accepted. It retains only schema-bounded public candidate and option-contract IDs so call/put legs bind unambiguously; account, user, order, and position identifiers remain forbidden. A later LIVE input must match both its immutable normalized-input hash and the capture's complete canonical quote/mechanics projection; a declared hash alone is never authorization. `preflight --decision-id ID --refreshed-input FILE`, `portfolio-check --portfolio-input FILE`, `report --latest`, `history --limit 20`, and `calibration --outcome-input FILE` are local append-only workflows. `--json` is accepted globally and on every subcommand.

`scan` without `--fixture` is deliberately fail-closed until this task exposes an authenticated Robinhood MCP schema matching the repository's intentionally empty-until-verified positive allowlist. A fresh authenticated Codex task must inspect its catalog and then make a redacted version-1 capture envelope. The Python package has no credentials, broker HTTP client, or unofficial endpoint.

`config/universe.json` is a deterministic, bounded baseline candidate plan covering broad market, technology, semiconductors, biotech, energy, financials, rates, and other liquid factors. `health --json` reports its category/count status. It is not live data and is never populated by the unavailable broker catalog; merge it with bounded dated catalysts, scans, and watchlist symbols before obtaining actual typed evidence.

The committed repository profile references the official endpoint `https://agent.robinhood.com/mcp/trading` but remains disabled. Robinhood's [Agentic Trading overview](https://robinhood.com/us/en/support/articles/agentic-trading-overview/) describes Codex setup and warns that the MCP can expose private account data and trading capability; that is why this repository keeps the allowlist empty. Follow [OpenAI MCP guidance](https://learn.chatgpt.com/docs/extend/mcp) and [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) for configuration context, but neither proves unattended Robinhood OAuth. Do not enable a schedule until authenticated access is empirically demonstrated.

## Hard mechanics boundary

Universal intended worst-case loss (premium, defined cash settlement, fees, commissions, planned slippage, and the maximum permitted whole-complex fill) is <= $1,000. No naked/uncovered option is accepted. Any short requires a verified index product with every leg European cash-settled; ETF/equity-labelled shorts remain prohibited. Physically settled stock/ETF options are never held through expiration; a long one requires an enforced pre-expiration close plan and typed post-event close valuation. Required early exits use that valuation—not expiration payoff—for EV, sensitivity, and total-loss gates. Supplied and deterministically generated canonical structures reuse one sourced underlying distribution, but each has its own quote-derived natural complex fill, slippage, leg-count policy fee, payoff, EV, breakevens, and risk; selection fails closed unless the selected topology is the policy-fee-aware best. Event candidates also fail if the direction-aware, cost-inclusive IV-crush matrix has no positive case or its required delayed/smaller-than-implied/severe-crush case is nonpositive.

Reports contain JSON, Markdown, and CSP-restricted local HTML. Money is exact-string Decimal serialization. Decision snapshots are append-only and hash-chained; all append-only payload tables, including outcomes, preflights, alerts, and run health, carry verified SHA-256 payload hashes.

Portfolio position and correlation records are transient gate inputs only. Persisted evaluations, reports, history, alerts, preflights, and SQLite decision records contain sanitized limits, aggregate risk, counts, and booleans—not position IDs, held symbols, sector/factor tags, or portfolio prose.

See [architecture](docs/ARCHITECTURE.md), [decision policy](docs/DECISION_POLICY.md), [provenance](docs/DATA_PROVENANCE.md), [scheduling](docs/SCHEDULING.md), [audit](docs/PHASE1_AUDIT.md), and [troubleshooting](docs/TROUBLESHOOTING.md).

For verification: `.venv/bin/options-scout --help`, `.venv/bin/ruff check .`, `.venv/bin/mypy src`, `.venv/bin/pytest`, and `.venv/bin/python -m compileall -q src`. On this Python 3.14/macOS environment use the non-editable local install above: the editable install may create a hidden `.pth` file that Python correctly ignores. The generated DB and reports are ignored; `fixtures/` and `artifacts/options-scout.schedule.sh` are intentional sanitized source artifacts.

If a legacy local database or report predates the portfolio-redaction boundary, it must be recoverably removed from its runtime location and regenerated from sanitized inputs; source changes alone never remove an existing artifact. This repository does not delete runtime artifacts during ordinary code work.
