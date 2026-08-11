import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from options_scout.cli import _alert_fingerprint, capture_ingest, main, persist, portfolio_check
from options_scout.normalizer import normalize_envelope
from options_scout.pipeline import evaluate
from options_scout.preflight import compare_evaluations
from options_scout.reporting import markdown_report, write_reports
from options_scout.safety import (
    SafetyError,
    assert_read_only_operation,
    normalized_projection_schema_sha256,
    policy,
    redact,
    static_safety_scan,
)
from options_scout.schema import parse_run
from options_scout.store import AuditStore
from options_scout.universe import UniverseError, load_universe, parse_universe

_PARAMETER_SCHEMA_HASH = "1" * 64
_RESPONSE_SCHEMA_HASH = "2" * 64


def _reviewed_schema(tool: str = "reviewed.market_quote", identity: str = "reviewed.market.quote/v1") -> dict[str, object]:
    return {
        "tool": tool,
        "schema_identity": identity,
        "source_label": "LIVE",
        "parameter_schema_sha256": _PARAMETER_SCHEMA_HASH,
        "response_schema_sha256": _RESPONSE_SCHEMA_HASH,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
    }


def test_safety_redaction_and_allowlist(tmp_path: Path) -> None:
    root = Path.cwd()
    assert redact({"token": "x", "symbol": "IDX"})["token"] == "[REDACTED]"
    with pytest.raises(SafetyError):
        assert_read_only_operation("unknown_market_tool", root)
    assert not static_safety_scan(root)


def test_redaction_preserves_unique_public_candidate_and_contract_ids_only() -> None:
    value = {
        "candidates": [{"id": "candidate-public", "contracts": [{"id": "call-public"}, {"id": "put-public"}]}],
        "account_id": "private-account", "user_id": "private-user", "order_id": "private-order", "position_id": "private-position",
    }
    redacted = redact(value)
    assert redacted["candidates"][0]["id"] == "candidate-public"
    assert [item["id"] for item in redacted["candidates"][0]["contracts"]] == ["call-public", "put-public"]
    assert all(redacted[key] == "[REDACTED]" for key in ("account_id", "user_id", "order_id", "position_id"))


def test_capture_hash_and_immutable_store(tmp_path: Path) -> None:
    payload = {"x": "1"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    raw = {
        "schema_version": "1",
        "tool": "fixture.normalized-run",
        "schema_identity": "options-scout.fixture.normalized-run/v1",
        "parameter_schema_sha256": _PARAMETER_SCHEMA_HASH,
        "response_schema_sha256": _RESPONSE_SCHEMA_HASH,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "redacted_arguments": {},
        "fixture_namespace": "options-scout.fixture.v1",
        "capture_id": "x",
        "as_of": datetime.now(UTC).isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "field_provenance": {},
        "source_label": "LIVE",
        "payload": payload,
        "payload_hash": hashlib.sha256(encoded).hexdigest(),
        "normalized_input_hash": hashlib.sha256(b"normalized").hexdigest(),
    }
    assert normalize_envelope(raw, set())["capture_id"] == "x"
    store = AuditStore(tmp_path / "a.sqlite3")
    store.initialize()
    store.append_decision("one", {"decision": "NO_TRADE"})
    with pytest.raises(sqlite3.DatabaseError):
        store.conn.execute("UPDATE decisions SET payload='x'")
    store.close()


def test_html_report_escapes_untrusted_content(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path,
        "trace",
        {
            "decision": "NO_TRADE",
            "source_label": "UNAVAILABLE",
            "freshness": "none",
            "counts": {},
            "gates": [{"passed": False, "reason": "<script>bad</script>"}],
        },
    )
    html = Path(paths["html"]).read_text()
    assert "Content-Security-Policy" in html and "&lt;script&gt;" in html


def test_static_audit_rejects_a_deliberate_executable_forbidden_fixture(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "bad.py").write_text('invoke_tool("place_order")\n')
    (source / "direct.py").write_text(
        "broker.place_order()\nplace_order()\n# place_order is denial prose\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "options-scout.schedule.sh").write_text(
        "# DISABLED enabled_tools=[] source-health\nrun place_order\n"
    )
    assert static_safety_scan(tmp_path) == [
        "src/bad.py:1: forbidden executable broker invocation",
        "src/direct.py:1: forbidden executable broker invocation",
        "src/direct.py:2: forbidden executable broker invocation",
        "artifacts/options-scout.schedule.sh:2: forbidden executable mutation token",
    ]


def test_preflight_diff_covers_quote_mechanics_portfolio_and_thesis() -> None:
    original = {
        "symbol": "IDX",
        "candidate": {
            "underlying": "100",
            "structure": {
                "legs": [
                    {
                        "contract": {
                            "id": "c1",
                            "quote": {
                                "bid": "1",
                                "ask": "2",
                                "iv": "0.2",
                                "delta": "0.3",
                                "gamma": "0.1",
                                "theta": "-0.1",
                                "vega": "0.1",
                                "as_of": "a",
                            },
                        }
                    }
                ]
            },
            "mechanics": {"settlement_style": "cash"},
            "portfolio_assessment": {"aggregate_risk": "10"},
            "thesis_record": {"outcome": "old"},
            "claim_records": ["c"],
            "source_records": ["s"],
        },
        "analysis": {
            "payoff": {
                "breakevens": ["101"],
                "theoretical_max_loss": "10",
                "operational_max_loss_risk": "10",
                "max_gain": "20",
                "fills": {"natural_entry": "1", "realistic_limit_entry": "1"},
            },
            "volatility": {"atm_iv": "0.2"},
        },
        "gates": [],
    }
    refreshed = json.loads(json.dumps(original))
    refreshed["candidate"]["underlying"] = "103"
    refreshed["candidate"]["structure"]["legs"][0]["contract"]["quote"]["bid"] = "1.1"
    refreshed["candidate"]["mechanics"]["settlement_style"] = "physical"
    refreshed["candidate"]["portfolio_assessment"]["aggregate_risk"] = "20"
    refreshed["candidate"]["thesis_record"]["outcome"] = "new"
    result = compare_evaluations(original, refreshed)
    assert result["invalid"] and result["underlying_move_pct"] == "0.03"
    assert {change["family"] for change in result["changes"]} >= {
        "underlying",
        "quotes_and_greeks",
        "mechanics",
        "portfolio",
        "thesis_and_evidence",
    }


def test_alert_dedup_records_no_trade_and_transition(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "a.sqlite3")
    store.initialize()
    store.append_decision("one", {"decision": "NO_TRADE"})
    assert store.alert_once("same", "one", "NO_TRADE")
    assert not store.alert_once("same", "one", "NO_TRADE")
    assert store.alert_once("same", "one", "INVALIDATED")
    store.close()


def test_append_only_alert_and_run_health_payload_hashes_detect_tampering(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "a.sqlite3")
    store.initialize()
    store.append_decision("one", {"decision": "NO_TRADE"})
    assert store.alert_once("fingerprint", "one", "NO_TRADE", {"gate": "unchanged"})
    store.record_health({"decision": "DATA_INSUFFICIENT"})
    assert store.integrity()
    # Simulate a database-level attacker bypassing the append-only trigger;
    # integrity must still detect every hashed child payload class.
    store.conn.execute("DROP TRIGGER alerts_no_update")
    store.conn.execute("DROP TRIGGER run_health_no_update")
    store.conn.execute("UPDATE alerts SET payload='{}'")
    store.conn.execute("UPDATE run_health SET payload='{}'")
    assert not store.integrity()
    store.close()


@pytest.mark.parametrize(
    "table",
    (
        "captures",
        "sources",
        "candidates",
        "research",
        "iv_snapshots",
        "structure_traces",
        "gate_results",
        "outcomes",
        "preflights",
        "alerts",
        "run_health",
    ),
)
def test_every_persisted_payload_class_detects_database_tampering(tmp_path: Path, table: str) -> None:
    """Hashes backstop each immutable trigger, including every child table."""
    store = AuditStore(tmp_path / f"{table}.sqlite3")
    store.initialize()
    store.append_capture({"capture_id": "cap", "source_label": "LIVE"})
    store.append_decision(
        "one",
        {
            "decision": "NO_TRADE",
            "evaluations": [
                {
                    "symbol": "IDX",
                    "gates": [],
                    "analysis": {"payoff": {}, "volatility": {}},
                    "candidate": {"claim_records": [], "source_records": [{"id": "source", "safe": True}]},
                }
            ],
        },
    )
    store.append_preflight("one", {"safe": True})
    store.append_outcome("one", {"safe": True})
    assert store.alert_once("fingerprint", "one", "NO_TRADE", {"safe": True})
    store.record_health({"safe": True})
    assert store.integrity()
    store.conn.execute(f"DROP TRIGGER {table}_no_update")
    store.conn.execute(f"UPDATE {table} SET payload='{{\"tampered\":true}}'")
    assert not store.integrity()
    store.close()


def test_candidate_update_and_delete_are_immutable_and_tamper_visible(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "candidate.sqlite3")
    store.initialize()
    store.append_decision(
        "one",
        {"evaluations": [{"symbol": "IDX", "gates": [], "analysis": {}, "candidate": {}}]},
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE candidates SET symbol='TAMPERED'")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM candidates")
    store.conn.execute("DROP TRIGGER candidates_no_delete")
    store.conn.execute("DELETE FROM candidates")
    assert not store.integrity()
    store.close()


def test_legacy_hash_exemption_cannot_be_forged_after_migration(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "legacy.sqlite3")
    store.initialize()
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO legacy_payload_hash_exemptions VALUES ('alerts',1,'now')"
        )
    store.conn.execute("DROP TRIGGER legacy_payload_hash_exemptions_no_insert")
    store.conn.execute(
        "INSERT INTO legacy_payload_hash_exemptions VALUES ('alerts',1,'now')"
    )
    assert not store.integrity()
    store.close()


def test_sources_are_decision_scoped_and_missing_source_is_detected(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "sources.sqlite3")
    store.initialize()
    payload = {"evaluations": [{"symbol": "IDX", "gates": [], "analysis": {}, "candidate": {"source_records": [{"id": "same-source", "title": "safe"}]}}]}
    store.append_decision("one", payload)
    store.append_decision("two", payload)
    assert store.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 2
    assert store.integrity()
    store.conn.execute("DROP TRIGGER sources_no_delete")
    store.conn.execute("DELETE FROM sources WHERE decision_id='one'")
    assert not store.integrity()
    store.close()


@pytest.mark.parametrize(
    "table",
    (
        "captures",
        "sources",
        "candidates",
        "research",
        "iv_snapshots",
        "structure_traces",
        "gate_results",
        "outcomes",
        "preflights",
        "alerts",
        "run_health",
    ),
)
def test_legacy_blank_hash_rows_are_backfilled_or_bounded_before_freeze(tmp_path: Path, table: str) -> None:
    store = AuditStore(tmp_path / f"{table}.sqlite3")
    store.initialize()
    # Reproduce a pre-freeze v3 row: the child table is immutable but lacks a
    # payload hash; the legacy exemption trigger did not yet exist.
    store.conn.execute(f"DROP TRIGGER {table}_no_update")
    store.conn.execute("DROP TRIGGER legacy_payload_hash_exemptions_no_insert")
    if table == "captures":
        store.conn.execute(
            "INSERT INTO captures (id,created_at,source_label,payload,payload_hash) VALUES ('legacy-cap','t','LIVE','{}','')"
        )
    elif table == "run_health":
        store.conn.execute("INSERT INTO run_health (created_at,payload,payload_hash) VALUES ('t','{}','')")
    elif table == "alerts":
        store.append_decision("one", {"evaluations": []})
        store.conn.execute("INSERT INTO alerts (fingerprint,created_at,decision_id,state,payload,payload_hash) VALUES ('a','t','one','NO_TRADE','{}','')")
    elif table in {"sources", "candidates", "research", "iv_snapshots", "structure_traces", "gate_results"}:
        store.append_decision(
            "one",
            {
                "evaluations": [
                    {
                        "symbol": "IDX",
                        "gates": [],
                        "analysis": {"payoff": {}, "volatility": {}},
                        "candidate": {"source_records": [{"id": "legacy-source"}]},
                    }
                ]
            },
        )
        store.conn.execute(f"UPDATE {table} SET payload_hash='' WHERE decision_id='one'")
    else:
        store.append_decision("one", {"evaluations": []})
        store.conn.execute(f"INSERT INTO {table} (decision_id,created_at,payload,payload_hash) VALUES ('one','t','{{}}','')")
    store.initialize()
    assert store.integrity()
    store.close()


def test_actionable_report_renders_populated_ticket_and_escape() -> None:
    evaluation = {
        "symbol": "IDX<script>",
        "structure": "debit",
        "decision": "ACTIONABLE",
        "candidate": {
            "underlying": "100",
            "underlying_as_of": "time",
            "structure": {
                "quantity": 1,
                "legs": [
                    {
                        "side": "buy",
                        "contract": {
                            "id": "c1",
                            "expiration": "2030-01-01",
                            "strike": "100",
                            "option_type": "call",
                            "quote": {"as_of": "time"},
                        },
                    }
                ],
            },
            "thesis_record": {
                "implied_probability_low": "0.4",
                "implied_probability_high": "0.5",
                "outcome": "outcome",
                "why_wrong": "reason",
                "catalyst": "catalyst",
                "timing_trigger": "trigger",
            },
            "structure_plan": {
                "entry_limit": "1",
                "max_acceptable_limit": "2",
                "exit_plan": {"invalidation": "stop"},
            },
        },
        "analysis": {
            "payoff": {
                "entry": "1",
                "breakevens": ["101"],
                "theoretical_max_loss": "100",
                "operational_max_loss_risk": "100",
                "max_gain": "300",
                "assignment_risk": "NONE",
                "physical_settlement_risk": "NONE",
                "possibility_of_account_deficit": "ELIMINATED",
            },
            "structure": {"kind": "call_debit"},
        },
        "gates": [],
    }
    report = markdown_report(
        "x", {"decision": "ACTIONABLE", "evaluations": [evaluation], "ranked": [evaluation]}
    )
    assert (
        "IDX<script>" in report
        and "PREVIEW ONLY — NOT SUBMITTED" in report
        and "2030-01-01" in report
    )


def test_cli_fixture_preflight_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPTIONS_SCOUT_ROOT", str(tmp_path))
    for args in (
        ("options-scout", "init", "--json"),
        ("options-scout", "scan", "--fixture", "--json"),
    ):
        monkeypatch.setattr("sys.argv", list(args))
        assert main() == 0
    store = AuditStore(tmp_path / "artifacts/options_scout.sqlite3")
    store.initialize()
    decision_id = store.history(1)[0]["id"]
    store.close()
    for args in (
        ("options-scout", "preflight", "--decision-id", decision_id, "--move", "0.03", "--json"),
        ("options-scout", "schedule-plan", "--json"),
    ):
        monkeypatch.setattr("sys.argv", list(args))
        assert main() == 0
    out = capsys.readouterr().out
    assert "NON" in out or "INVALIDATED" in out


def test_health_reports_fail_closed_research_status_and_schedule_test_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPTIONS_SCOUT_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["options-scout", "init", "--json"])
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["options-scout", "health", "--json"])
    assert main() == 0
    health = json.loads(capsys.readouterr().out)
    assert health["research"]["pipeline"] == "READY_LOCAL_TYPED_PROVENANCE"
    assert health["research"]["live_research_sources"].startswith("UNAVAILABLE")
    monkeypatch.setattr("sys.argv", ["options-scout", "schedule-plan", "--json"])
    assert main() == 0
    console = tmp_path / ".venv/bin/options-scout"
    console.parent.mkdir(parents=True)
    os.symlink(Path(sys.executable).parent / "options-scout", console)
    os.symlink(Path(sys.executable), console.parent / "python")
    script = tmp_path / "artifacts/options-scout.schedule.sh"
    text = script.read_text()
    assert all(
        token not in text
        for token in (
            "place_",
            "cancel_",
            "replace_",
            "exercise",
            "transfer",
            "deposit",
            "withdraw",
        )
    )
    assert 'export OPTIONS_SCOUT_ROOT="$ROOT"' in text and 'cd "$ROOT"' in text
    assert '"--test" ]; then' in text and '"source-health","--json"' in text
    assert 'mkdir "$LOCK"' in text and "runner.is_absolute()" in text
    deep_plist = (tmp_path / "artifacts/com.options-scout.disabled.plist").read_text()
    regular_plist = (
        tmp_path / "artifacts/com.options-scout.regular.disabled.plist"
    ).read_text()
    assert "<key>Disabled</key><true/>" in deep_plist
    assert "<key>StartCalendarInterval</key>" in deep_plist and "--deep" in deep_plist
    assert "<key>Disabled</key><true/>" in regular_plist
    assert "<key>StartInterval</key><integer>900</integer>" in regular_plist
    assert "--regular" in regular_plist and "timedelta(minutes=15)" in text
    assert subprocess.run([str(script), "--test"], cwd="/tmp", check=False).returncode == 0


def test_portfolio_identifiers_and_free_text_never_leave_transient_gate_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = json.loads((Path(__file__).parents[1] / "fixtures/normalized-run.json").read_text())
    assessment = raw["candidates"][0]["portfolio_assessment"]
    secrets = (
        "SECRET_POSITION_ID_9Z",
        "LEAKTICKER9Z",
        "SECRET_SECTOR_9Z",
        "SECRET_FACTOR_9Z",
        "SECRET_CORRELATION_RATIONALE_9Z",
        "SECRET_DEFICIT_RATIONALE_9Z",
    )
    assessment["positions"] = [
        {
            "id": secrets[0],
            "symbol": secrets[1],
            "sector": secrets[2],
            "factor_tags": [secrets[3]],
            "risk": "10",
            "event_risk": "1",
        }
    ]
    assessment["correlations"] = [
        {
            "position_id": secrets[0],
            "correlation": "0.25",
            "rationale": secrets[4],
        }
    ]
    assessment["deficit_elimination_rationale"] = secrets[5]

    payload = evaluate(parse_run(raw))
    serialized = json.dumps(payload, default=str)
    assert all(secret not in serialized for secret in secrets)
    safe = payload["evaluations"][0]["candidate"]["portfolio_assessment"]
    assert safe["position_count"] == 1
    assert safe["correlation_count"] == 1
    assert safe["deficit_elimination_verified"] is True
    assert safe["duplicate_or_correlated_expression"] is False

    persisted = persist(tmp_path, "privacy-case", payload)
    report_text = "\n".join(Path(path).read_text() for path in persisted["artifacts"].values())
    connection = sqlite3.connect(tmp_path / "artifacts/options_scout.sqlite3")
    try:
        database_dump = "\n".join(connection.iterdump())
    finally:
        connection.close()
    assert all(secret not in report_text and secret not in database_dump for secret in secrets)

    portfolio_input = tmp_path / "portfolio.json"
    portfolio_input.write_text(json.dumps(raw))
    monkeypatch.setenv("OPTIONS_SCOUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "options-scout",
            "portfolio-check",
            "--portfolio-input",
            str(portfolio_input),
            "--json",
        ],
    )
    assert main() == 0
    portfolio_output = capsys.readouterr().out
    assert all(secret not in portfolio_output for secret in secrets)


def test_portfolio_check_uses_proposed_context_not_nonempty_positions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def context(symbol: str, factors: list[str]) -> dict[str, object]:
        return {
            "max_risk_per_trade_usd": "1000",
            "remaining_aggregate_risk_usd": "1000",
            "remaining_cluster_risk_usd": "1000",
            "remaining_event_risk_usd": "1000",
            "remaining_sector_risk_usd": "1000",
            "remaining_factor_risk_usd": "1000",
            "trade_risk_usd": "10",
            "proposed_symbol": symbol,
            "proposed_factor_tags": factors,
            "positions": [
                {
                    "id": "PRIVATE_POSITION_9Z",
                    "symbol": "UNRELATED_9Z",
                    "sector": "unrelated",
                    "factor_tags": ["unrelated-factor"],
                    "risk": "10",
                    "event_risk": "0",
                }
            ],
            "correlations": [
                {
                    "position_id": "PRIVATE_POSITION_9Z",
                    "correlation": "0.95",
                    "rationale": "PRIVATE_CORRELATION_9Z",
                }
            ],
        }

    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps(context("PROPOSED_9Z", ["proposed-factor"])))
    assert portfolio_check(str(unrelated), True) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "VERIFIED_LOCAL"
    rendered = json.dumps(result)
    assert "PRIVATE_POSITION_9Z" not in rendered and "PRIVATE_CORRELATION_9Z" not in rendered

    related = tmp_path / "related.json"
    related.write_text(json.dumps(context("UNRELATED_9Z", ["proposed-factor"])))
    assert portfolio_check(str(related), True) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"

    unknown = context("PROPOSED_9Z", [])
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown))
    assert portfolio_check(str(unknown_path), True) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DATA_INSUFFICIENT"


def test_runtime_artifacts_have_no_sidecars_or_legacy_portfolio_identifiers() -> None:
    root = Path(__file__).parents[1]
    artifact_root = root / "artifacts"
    sqlite_files = [path for path in artifact_root.rglob("*") if path.is_file() and ".sqlite" in path.name]
    assert all(path.name == "options_scout.sqlite3" for path in sqlite_files)
    privacy_markers = (b"position_id", b"account_id", b"correlations\"", b"portfolio_position")
    for path in artifact_root.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert not any(marker in content for marker in privacy_markers), path
    ignored = (root / ".gitignore").read_text()
    assert "artifacts/*.sqlite3-*" in ignored and "artifacts/*.sqlite*" in ignored


def test_schedule_run_refuses_before_review_then_invokes_one_bounded_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPTIONS_SCOUT_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["options-scout", "init", "--json"])
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["options-scout", "schedule-plan", "--json"])
    assert main() == 0
    capsys.readouterr()

    console = tmp_path / ".venv/bin/options-scout"
    console.parent.mkdir(parents=True)
    os.symlink(Path(sys.executable).parent / "options-scout", console)
    os.symlink(Path(sys.executable), console.parent / "python")
    marker = tmp_path / "runner-invoked.json"
    stub = tmp_path / "bounded-codex-runner"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "prompt = pathlib.Path(sys.argv[1])\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{'argv': sys.argv[1:], 'prompt': prompt.read_text()}}))\n"
    )
    stub.chmod(0o700)
    script = tmp_path / "artifacts/options-scout.schedule.sh"
    environment = {**os.environ, "OPTIONS_SCOUT_CODEX_RUNNER": str(stub)}

    refused = subprocess.run([str(script), "--run"], cwd="/tmp", env=environment, check=False)
    assert refused.returncode != 0
    assert not marker.exists()

    config_path = tmp_path / "config/policy.json"
    config = json.loads(config_path.read_text())
    config["enabled_tools"] = ["reviewed.market_quote"]
    config["approved_capture_schemas"] = [_reviewed_schema()]
    config_path.write_text(json.dumps(config))
    mismatched_profile = subprocess.run(
        [str(script), "--run"], cwd="/tmp", env=environment, check=False
    )
    assert mismatched_profile.returncode != 0
    assert not marker.exists()
    (tmp_path / ".codex/config.toml").write_text(
        """[mcp_servers.robinhood-trading]
url = "https://agent.robinhood.com/mcp/trading"
enabled = true
enabled_tools = ["reviewed.market_quote"]

[options_scout]
mode = "read_only_research"
fail_closed_on_unknown_tools = true
"""
    )
    # `--test` validates reviewed-ready health but never needs or invokes the
    # runner; this is the documented activation preflight.
    ready_test = subprocess.run(
        [str(script), "--test"], cwd="/tmp", env={**os.environ}, check=False
    )
    assert ready_test.returncode == 0
    assert not marker.exists()
    invoked = subprocess.run([str(script), "--run"], cwd="/tmp", env=environment, check=False)
    assert invoked.returncode == 0
    invocation = json.loads(marker.read_text())
    assert invocation["argv"] == [str(tmp_path / "artifacts/options-scout.live-orchestration.md")]
    prompt = invocation["prompt"]
    assert prompt.index("capture-ingest") < prompt.index("scan --input")
    assert "exact positive-allowlisted read-only market-data tools" in prompt
    assert "Never submit or mutate anything" in prompt


def _reviewed_capture(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "normalized_projection": {
            "metadata": {
                "run_id": "reviewed-capture-fixture",
                "as_of": "2026-08-11T15:00:00+00:00",
                "retrieved_at": "2026-08-11T15:00:01+00:00",
            },
            "source_label": "LIVE",
            "candidates": [],
        }
    }
    return {
        "schema_version": "1",
        "capture_id": "reviewed-capture",
        "tool": "reviewed.market_quote",
        "schema_identity": "reviewed.market.quote/v1",
        "parameter_schema_sha256": _PARAMETER_SCHEMA_HASH,
        "response_schema_sha256": _RESPONSE_SCHEMA_HASH,
        "normalized_projection_schema_sha256": normalized_projection_schema_sha256(),
        "source_label": "LIVE",
        "as_of": "2026-08-11T15:00:00+00:00",
        "retrieved_at": "2026-08-11T15:00:01+00:00",
        "field_provenance": {},
        "redacted_arguments": {},
        "payload": payload,
        "payload_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "normalized_input_hash": "a" * 64,
    }


def test_capture_ingest_requires_reviewed_policy_and_persists_only_redacted_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = json.loads((Path(__file__).parents[1] / "config/policy.json").read_text())
    config.update(
        {
            "enabled_tools": ["reviewed.market_quote"],
            "approved_capture_schemas": [
                _reviewed_schema()
            ],
        }
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    envelope = _reviewed_capture(tmp_path)
    source = tmp_path / "envelope.json"
    source.write_text(json.dumps(envelope))
    assert capture_ingest(tmp_path, str(source), True) == 0
    assert json.loads(capsys.readouterr().out)["broker_invoked"] is False
    store = AuditStore(tmp_path / "artifacts/options_scout.sqlite3")
    store.initialize()
    assert store.capture_envelope("reviewed-capture")["normalized_input_hash"] == "a" * 64
    store.close()


def test_default_empty_policy_capture_ingest_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config/policy.json").write_text(
        (Path(__file__).parents[1] / "config/policy.json").read_text()
    )
    source = tmp_path / "envelope.json"
    source.write_text(json.dumps(_reviewed_capture(tmp_path)))
    assert capture_ingest(tmp_path, str(source), True) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["broker_invoked"] is False and "unapproved capture tool" in result["error"]


def test_policy_and_capture_ingest_reject_schema_drift_forbidden_and_unredacted_material(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = json.loads((Path(__file__).parents[1] / "config/policy.json").read_text())
    config.update(
        {
            "enabled_tools": ["place_order"],
            "approved_capture_schemas": [
                _reviewed_schema("place_order", "quote/v1")
            ],
        }
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    with pytest.raises(SafetyError, match="read-only"):
        policy(tmp_path)
    config["enabled_tools"] = ["reviewed.market_quote"]
    config["approved_capture_schemas"] = [
        {**_reviewed_schema("reviewed.market_quote", "quote/v1"), "drift": True}
    ]
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    with pytest.raises(SafetyError, match="schema fields"):
        policy(tmp_path)
    config["approved_capture_schemas"] = [
        _reviewed_schema("reviewed.market_quote", "quote/v1")
    ]
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    envelope = _reviewed_capture(tmp_path)
    envelope["schema_identity"] = "quote/v1"
    envelope["payload"] = {"token": "unredacted"}
    envelope["payload_hash"] = hashlib.sha256(
        json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source = tmp_path / "sensitive.json"
    source.write_text(json.dumps(envelope))
    assert capture_ingest(tmp_path, str(source), True) == 2
    assert "unredacted sensitive" in capsys.readouterr().out


def test_reviewed_projection_schema_hash_is_fixed_not_an_unused_path_list(tmp_path: Path) -> None:
    config = json.loads((Path(__file__).parents[1] / "config/policy.json").read_text())
    config["enabled_tools"] = ["reviewed.market_quote"]
    reviewed = _reviewed_schema()
    reviewed["normalized_projection_schema_sha256"] = "f" * 64
    config["approved_capture_schemas"] = [reviewed]
    (tmp_path / "config").mkdir()
    (tmp_path / "config/policy.json").write_text(json.dumps(config))
    with pytest.raises(SafetyError, match="reviewed LIVE"):
        policy(tmp_path)


def test_alert_fingerprint_changes_for_material_actionable_economics_but_deduplicates_same_no_trade() -> (
    None
):
    no_trade = {
        "decision": "NO_TRADE",
        "source_label": "LIVE",
        "evaluations": [
            {
                "id": "one",
                "symbol": "IDX",
                "decision": "NO_TRADE",
                "candidate": {"structure": {"name": "call_debit", "quantity": 1, "legs": []}},
                "analysis": {
                    "payoff": {
                        "entry": "10",
                        "operational_max_loss_risk": "100",
                        "max_gain": "200",
                        "breakevens": ["101"],
                    }
                },
                "gates": [{"name": "liquidity", "status": "FAIL"}],
            }
        ],
    }
    assert _alert_fingerprint(no_trade) == _alert_fingerprint(json.loads(json.dumps(no_trade)))
    actionable = json.loads(json.dumps(no_trade))
    actionable["decision"] = actionable["evaluations"][0]["decision"] = "ACTIONABLE"
    better = json.loads(json.dumps(actionable))
    better["evaluations"][0]["analysis"]["payoff"]["max_gain"] = "250"
    assert _alert_fingerprint(actionable) != _alert_fingerprint(better)


def test_persist_alerts_deduplicate_no_trade_quote_churn_but_realert_material_failure_and_state(
    tmp_path: Path,
) -> None:
    base = {
        "decision": "NO_TRADE",
        "source_label": "LIVE",
        "source_health": {
            "decision": "DATA_INSUFFICIENT",
            "catalog": "missing",
            "allowlist_status": "EMPTY",
            "live_quotes": "none",
            "freshness": "none",
        },
        "evaluations": [
            {
                "id": "one",
                "symbol": "IDX",
                "decision": "NO_TRADE",
                "candidate": {"structure": {"name": "call_debit", "quantity": 1, "legs": []}},
                "analysis": {
                    "payoff": {
                        "entry": "10",
                        "operational_max_loss_risk": "100",
                        "max_gain": "200",
                        "breakevens": [],
                    }
                },
                "gates": [{"name": "liquidity", "status": "FAIL"}],
            }
        ],
        "gates": [],
    }
    assert persist(tmp_path, "one", json.loads(json.dumps(base)))["alert_eligible"]
    quote_churn = json.loads(json.dumps(base))
    quote_churn["evaluations"][0]["analysis"]["payoff"]["max_gain"] = "999"
    assert not persist(tmp_path, "two", quote_churn)["alert_eligible"]
    changed_failure = json.loads(json.dumps(base))
    changed_failure["evaluations"][0]["gates"] = [
        {"name": "liquidity", "status": "FAIL"},
        {"name": "assignment", "status": "FAIL"},
    ]
    assert persist(tmp_path, "three", changed_failure)["alert_eligible"]
    transition = json.loads(json.dumps(base))
    transition["decision"] = transition["evaluations"][0]["decision"] = "ACTIONABLE"
    assert persist(tmp_path, "four", transition)["alert_eligible"]


def test_universe_coverage_dedupe_cap_and_strict_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    universe = load_universe(Path(__file__).parents[1])
    assert {
        "broad_market",
        "technology",
        "semiconductors",
        "biotech",
        "energy",
        "financials",
        "rates",
        "liquid_factors",
    } <= set(universe.categories)
    assert universe.symbols(("SPY", "NVDA", "NVDA"))[-1] == "NVDA"
    capped = parse_universe(
        {"version": "1", "max_universe": 2, "categories": {"broad_market": ["SPY", "QQQ", "SPY"]}}
    )
    assert capped.symbols(("IWM",)) == ("SPY", "QQQ")
    with pytest.raises(UniverseError):
        parse_universe({"version": "1", "max_universe": 2, "categories": {"bad-category": ["SPY"]}})
    monkeypatch.setenv("OPTIONS_SCOUT_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["options-scout", "init", "--json"])
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["options-scout", "health", "--json"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["universe"]["baseline_count"] > 0
