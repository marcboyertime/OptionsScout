"""Append-only, redacted local audit storage for OptionsScout."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_PORTFOLIO_FIELDS = frozenset(
    {
        "limits",
        "aggregate_risk",
        "cluster_risk",
        "event_risk",
        "sector_risk",
        "factor_risk",
        "deficit_elimination_verified",
        "position_count",
        "correlation_count",
        "duplicate_or_correlated_expression",
    }
)


def sanitize_persisted(value: Any) -> Any:
    """Project legacy decision records to the same public portfolio view.

    Old append-only rows cannot be rewritten without destroying their audit
    chain.  This read-time projection makes them safe for history, reports and
    preflight while preserving the stored record hash as historical evidence.
    """
    if isinstance(value, list):
        return [sanitize_persisted(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "portfolio_assessment" and isinstance(item, dict):
            result[key] = {
                field: sanitize_persisted(item[field])
                for field in _SAFE_PORTFOLIO_FIELDS
                if field in item
            }
            # Earlier public records have the old rationale string but no
            # explicit verification bit; preserve only the fact that it was
            # supplied, never its free text.
            if "deficit_elimination_verified" not in result[key]:
                result[key]["deficit_elimination_verified"] = bool(
                    item.get("deficit_elimination_rationale")
                )
            continue
        result[str(key)] = sanitize_persisted(item)
    return result


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class AuditStore:
    """SQLite store whose recommendations are immutable once written."""

    VERSION = 4

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        # The operator is deliberately single-process.  Rollback-journal mode
        # avoids persistent WAL/SHM sidecars that can retain copies of
        # redacted audit rows outside the one controlled database artifact.
        self.conn.execute("PRAGMA journal_mode=DELETE")

    def initialize(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_payload_hash_exemptions (
                    table_name TEXT NOT NULL, migration_rowid INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL, PRIMARY KEY(table_name,migration_rowid)
                );
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_label TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    input_hash TEXT NOT NULL, previous_hash TEXT NOT NULL, record_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decision_child_counts (
                    decision_id TEXT NOT NULL, table_name TEXT NOT NULL, expected_count INTEGER NOT NULL,
                    PRIMARY KEY(decision_id,table_name), FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, symbol TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS research (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS iv_snapshots (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, symbol TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL, FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS structure_traces (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL, FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS gate_results (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS preflights (
                    id INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    fingerprint TEXT PRIMARY KEY, created_at TEXT NOT NULL, decision_id TEXT NOT NULL,
                    state TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );
                CREATE TABLE IF NOT EXISTS run_health (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL
                );
                """
            )
            self._add_column_if_missing(
                "captures", "source_label", "TEXT NOT NULL DEFAULT 'UNAVAILABLE'"
            )
            self._add_column_if_missing("captures", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("decisions", "input_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("research", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            for table in ("candidates", "iv_snapshots", "structure_traces", "gate_results", "sources"):
                self._add_column_if_missing(table, "payload_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("iv_snapshots", "decision_id", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("outcomes", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("preflights", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("alerts", "payload", "TEXT NOT NULL DEFAULT '{}'")
            self._add_column_if_missing("alerts", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing("run_health", "payload_hash", "TEXT NOT NULL DEFAULT ''")
            for table in (
                "captures", "sources", "candidates", "research", "iv_snapshots", "structure_traces",
                "gate_results", "outcomes", "preflights", "alerts", "run_health",
            ):
                self._backfill_payload_hash(table)
            # A trigger list rather than an application convention makes accidental rewriting auditable.
            for table in (
                "legacy_payload_hash_exemptions",
                "captures",
                "sources",
                "decisions",
                "decision_child_counts",
                "candidates",
                "research",
                "iv_snapshots",
                "structure_traces",
                "gate_results",
                "outcomes",
                "preflights",
                "alerts",
                "run_health",
            ):
                self.conn.executescript(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT, 'immutable {table}'); END;"
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT, 'immutable {table}'); END;"
                )
            self.conn.execute(
                "CREATE TRIGGER IF NOT EXISTS legacy_payload_hash_exemptions_no_insert "
                "BEFORE INSERT ON legacy_payload_hash_exemptions "
                "BEGIN SELECT RAISE(ABORT, 'immutable legacy_payload_hash_exemptions'); END;"
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)",
                (self.VERSION, datetime.now(UTC).isoformat()),
            )

    def _add_column_if_missing(self, table: str, column: str, specification: str) -> None:
        columns = {str(row[1]) for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {specification}")

    def _backfill_payload_hash(self, table: str) -> None:
        """Hash legacy child rows before append-only triggers are installed."""
        for row in self.conn.execute(
            f"SELECT rowid AS migration_rowid,payload,payload_hash FROM {table} WHERE payload_hash=''"
        ):
            try:
                payload = json.loads(str(row["payload"]))
            except json.JSONDecodeError:
                continue
            try:
                self.conn.execute(
                    f"UPDATE {table} SET payload_hash=? WHERE rowid=?",
                    (digest(payload), row["migration_rowid"]),
                )
            except sqlite3.IntegrityError:
                # A prior version may already have installed the immutable
                # trigger.  Preserve that historical row rather than weakening
                # it; only newly appended records are hash-required.
                try:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO legacy_payload_hash_exemptions VALUES (?,?,?)",
                        (table, row["migration_rowid"], datetime.now(UTC).isoformat()),
                    )
                except sqlite3.IntegrityError:
                    # A previously frozen incomplete migration cannot be
                    # repaired by bypassing its own immutable table. Leave it
                    # tamper-visible until the legacy artifact is regenerated.
                    continue

    def append_capture(self, envelope: dict[str, Any]) -> str:
        ident = str(envelope["capture_id"])
        payload = canonical(envelope)
        with self.conn:
            self.conn.execute(
                "INSERT INTO captures VALUES (?,?,?,?,?)",
                (
                    ident,
                    datetime.now(UTC).isoformat(),
                    str(envelope["source_label"]),
                    payload,
                    digest(envelope),
                ),
            )
        return ident

    def capture_envelope(self, capture_id: str) -> dict[str, Any] | None:
        """Return the immutable, already-redacted capture envelope by identity."""
        row = self.conn.execute(
            "SELECT payload,payload_hash FROM captures WHERE id=?", (capture_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row["payload"]))
        except json.JSONDecodeError:
            return None
        return payload if digest(payload) == str(row["payload_hash"]) else None

    def append_decision(self, trace_id: str, payload: dict[str, Any]) -> str:
        payload = sanitize_persisted(payload)
        encoded = canonical(payload)
        row = self.conn.execute(
            "SELECT record_hash FROM decisions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        previous = str(row[0]) if row else "0" * 64
        record_hash = hashlib.sha256((previous + encoded).encode("utf-8")).hexdigest()
        input_hash = str(
            payload.get("normalized_input_hash") or digest(payload.get("evaluations", []))
        )
        with self.conn:
            self.conn.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
                (
                    trace_id,
                    datetime.now(UTC).isoformat(),
                    encoded,
                    input_hash,
                    previous,
                    record_hash,
                ),
            )
            for record in payload.get("evaluations", []):
                value = canonical(record)
                self.conn.execute(
                    "INSERT INTO candidates (decision_id,symbol,payload,payload_hash) VALUES (?,?,?,?)",
                    (trace_id, str(record.get("symbol", "")), value, digest(record)),
                )
                self.conn.execute(
                    "INSERT INTO gate_results (decision_id,payload,payload_hash) VALUES (?,?,?)",
                    (trace_id, canonical(record.get("gates", [])), digest(record.get("gates", []))),
                )
                analysis = record.get("analysis", {})
                self.conn.execute(
                    "INSERT INTO structure_traces (decision_id,created_at,payload,payload_hash) VALUES (?,?,?,?)",
                    (
                        trace_id,
                        datetime.now(UTC).isoformat(),
                        canonical(analysis.get("payoff", {})),
                        digest(analysis.get("payoff", {})),
                    ),
                )
                candidate = record.get("candidate", {})
                self.conn.execute(
                    "INSERT INTO research (decision_id,created_at,payload,payload_hash) VALUES (?,?,?,?)",
                    (
                        trace_id,
                        datetime.now(UTC).isoformat(),
                        canonical(candidate.get("claim_records", [])),
                        digest(candidate.get("claim_records", [])),
                    ),
                )
                self.conn.execute(
                    "INSERT INTO iv_snapshots (decision_id,symbol,created_at,payload,payload_hash) VALUES (?,?,?,?,?)",
                    (
                        trace_id,
                        str(record.get("symbol", "")),
                        datetime.now(UTC).isoformat(),
                        canonical(analysis.get("volatility", {})),
                        digest(analysis.get("volatility", {})),
                    ),
                )
                for source in record.get("candidate", {}).get("source_records", []):
                    source_value = canonical(source)
                    source_id = f"{trace_id}:{source.get('id', '')}"
                    self.conn.execute(
                        "INSERT INTO sources (id,decision_id,payload,payload_hash) VALUES (?,?,?,?)",
                        (source_id, trace_id, source_value, digest(source)),
                    )
            for table in ("candidates", "gate_results", "structure_traces", "research", "iv_snapshots"):
                self.conn.execute(
                    "INSERT INTO decision_child_counts (decision_id,table_name,expected_count) VALUES (?,?,?)",
                    (trace_id, table, len(payload.get("evaluations", []))),
                )
            self.conn.execute(
                "INSERT INTO decision_child_counts (decision_id,table_name,expected_count) VALUES (?,?,?)",
                (
                    trace_id,
                    "sources",
                    sum(
                        len(record.get("candidate", {}).get("source_records", []))
                        for record in payload.get("evaluations", [])
                    ),
                ),
            )
        return record_hash

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id,created_at,payload,input_hash,previous_hash,record_hash FROM decisions WHERE id=?",
            (decision_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "payload": sanitize_persisted(json.loads(row["payload"])),
            "input_hash": row["input_hash"],
            "previous_hash": row["previous_hash"],
            "hash": row["record_hash"],
        }

    def append_preflight(self, trace_id: str, payload: dict[str, Any]) -> None:
        self._append_child("preflights", trace_id, payload)

    def append_outcome(self, trace_id: str, payload: dict[str, Any]) -> None:
        self._append_child("outcomes", trace_id, payload)

    def _append_child(self, table: str, trace_id: str, payload: dict[str, Any]) -> None:
        payload = sanitize_persisted(payload)
        value = canonical(payload)
        with self.conn:
            self.conn.execute(
                f"INSERT INTO {table} (decision_id,created_at,payload,payload_hash) VALUES (?,?,?,?)",
                (trace_id, datetime.now(UTC).isoformat(), value, digest(payload)),
            )

    def outcomes(self) -> list[dict[str, Any]]:
        return [
            {
                "decision_id": row[0],
                "created_at": row[1],
                "payload": sanitize_persisted(json.loads(row[2])),
            }
            for row in self.conn.execute(
                "SELECT decision_id,created_at,payload FROM outcomes ORDER BY id"
            )
        ]

    def record_health(self, payload: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO run_health (created_at,payload,payload_hash) VALUES (?,?,?)",
                (
                    datetime.now(UTC).isoformat(),
                    canonical(sanitize_persisted(payload)),
                    digest(sanitize_persisted(payload)),
                ),
            )

    def verify_chain(self) -> bool:
        previous = "0" * 64
        for row in self.conn.execute(
            "SELECT payload,previous_hash,record_hash FROM decisions ORDER BY rowid"
        ):
            if (
                row["previous_hash"] != previous
                or hashlib.sha256((previous + row["payload"]).encode("utf-8")).hexdigest()
                != row["record_hash"]
            ):
                return False
            previous = row["record_hash"]
        return True

    def integrity(self) -> bool:
        return (
            self.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            and self._immutable_triggers_present()
            and self.verify_chain()
            and self._verify_child_hashes()
        )

    def _immutable_triggers_present(self) -> bool:
        tables = (
            "legacy_payload_hash_exemptions", "captures", "sources", "decisions", "decision_child_counts",
            "candidates", "research", "iv_snapshots", "structure_traces", "gate_results", "outcomes",
            "preflights", "alerts", "run_health",
        )
        present = {
            str(row[0])
            for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
        return all(
            f"{table}_no_update" in present and f"{table}_no_delete" in present
            for table in tables
        ) and "legacy_payload_hash_exemptions_no_insert" in present

    def _verify_child_hashes(self) -> bool:
        for table in ("captures", "sources", "candidates", "research", "iv_snapshots", "structure_traces", "gate_results", "outcomes", "preflights", "alerts", "run_health"):
            for row in self.conn.execute(
                f"SELECT rowid AS migration_rowid,payload,payload_hash FROM {table}"
            ):
                try:
                    if str(row["payload_hash"]) == "" and self.conn.execute(
                        "SELECT 1 FROM legacy_payload_hash_exemptions WHERE table_name=? AND migration_rowid=?",
                        (table, row["migration_rowid"]),
                    ).fetchone():
                        # Immutable legacy rows predate this migration and
                        # cannot be rewritten merely to add a hash.
                        continue
                    if str(row["payload_hash"]) == "":
                        return False
                    if digest(json.loads(str(row["payload"]))) != str(row["payload_hash"]):
                        return False
                except json.JSONDecodeError:
                    return False
        for row in self.conn.execute(
            "SELECT decision_id,table_name,expected_count FROM decision_child_counts"
        ):
            actual = self.conn.execute(
                f"SELECT COUNT(*) FROM {row['table_name']} WHERE decision_id=?", (row["decision_id"],)
            ).fetchone()[0]
            if actual != row["expected_count"]:
                return False
        return True

    def alert_once(
        self, fingerprint: str, trace_id: str, state: str, payload: dict[str, Any] | None = None
    ) -> bool:
        # `alerts` is append-only.  State is part of an alert event identity,
        # so a deterioration/invalidated transition is a new immutable row
        # rather than an update of an earlier notification.
        event_fingerprint = f"{fingerprint}:{state}"
        row = self.conn.execute(
            "SELECT state FROM alerts WHERE fingerprint=?", (event_fingerprint,)
        ).fetchone()
        if row is not None:
            return False
        with self.conn:
            self.conn.execute(
                "INSERT INTO alerts (fingerprint,created_at,decision_id,state,payload,payload_hash) VALUES (?,?,?,?,?,?)",
                (
                    event_fingerprint,
                    datetime.now(UTC).isoformat(),
                    trace_id,
                    state,
                    canonical(sanitize_persisted(payload or {})),
                    digest(sanitize_persisted(payload or {})),
                ),
            )
        return True

    def history(self, limit: int = 50, decision: str | None = None) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id,created_at,payload,input_hash,record_hash FROM decisions ORDER BY rowid DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )
        records = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "payload": sanitize_persisted(json.loads(row["payload"])),
                "input_hash": row["input_hash"],
                "hash": row["record_hash"],
            }
            for row in rows
        ]
        return [
            item
            for item in records
            if decision is None or item["payload"].get("decision") == decision
        ]

    def close(self) -> None:
        self.conn.close()
