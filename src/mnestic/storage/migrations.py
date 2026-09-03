"""Ordered schema migrations. Never edit an applied migration; append a new one.

Design: important entities are relational and queryable; JSON columns hold structured
sub-documents (state snapshots, patches, event payloads, decisions).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

Migration = Callable[[sqlite3.Connection, bool], None]


def run_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement script inside the caller's transaction.

    ``executescript`` would COMMIT first, so statements are split with
    ``sqlite3.complete_statement`` (trigger bodies contain semicolons).
    """
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            stmt = buf.strip()
            if stmt:
                conn.execute(stmt)
            buf = ""
    if buf.strip():
        conn.execute(buf.strip())


def _m001_initial(conn: sqlite3.Connection, fts_enabled: bool) -> None:
    run_script(conn, 
        """
        CREATE TABLE skills (
            skill_id      TEXT NOT NULL,
            version       TEXT NOT NULL,
            name          TEXT NOT NULL,
            description   TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            spec_json     TEXT NOT NULL,
            loaded_at     TEXT NOT NULL,
            PRIMARY KEY (skill_id, version)
        );

        CREATE TABLE runs (
            run_id             TEXT PRIMARY KEY,
            skill_id           TEXT NOT NULL,
            skill_version      TEXT NOT NULL,
            skill_content_hash TEXT NOT NULL,
            status             TEXT NOT NULL,
            workspace_root     TEXT NOT NULL,
            model_name         TEXT NOT NULL,
            task_input         TEXT NOT NULL,
            last_step          INTEGER NOT NULL DEFAULT 0,
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL,
            finished_at        TEXT,
            outcome_json       TEXT,
            FOREIGN KEY (skill_id, skill_version) REFERENCES skills(skill_id, version)
        );

        CREATE TABLE current_states (
            run_id        TEXT PRIMARY KEY REFERENCES runs(run_id),
            state_version INTEGER NOT NULL,
            status        TEXT NOT NULL,
            state_json    TEXT NOT NULL,
            state_bytes   INTEGER NOT NULL,
            updated_at    TEXT NOT NULL
        );

        CREATE TABLE state_versions (
            run_id      TEXT NOT NULL REFERENCES runs(run_id),
            version     INTEGER NOT NULL,
            step        INTEGER NOT NULL,
            patch_id    TEXT,
            state_json  TEXT NOT NULL,
            state_bytes INTEGER NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (run_id, version)
        );

        CREATE TABLE state_patches (
            patch_id          TEXT PRIMARY KEY,
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            step              INTEGER NOT NULL,
            expected_version  INTEGER NOT NULL,
            resulting_version INTEGER,
            status            TEXT NOT NULL,          -- applied | rejected
            error_code        TEXT,
            error             TEXT,
            patch_json        TEXT NOT NULL,
            changes_json      TEXT,
            created_at        TEXT NOT NULL
        );
        CREATE INDEX idx_state_patches_run_step ON state_patches(run_id, step);

        CREATE TABLE archive_events (
            event_id     TEXT PRIMARY KEY,
            run_id       TEXT NOT NULL REFERENCES runs(run_id),
            seq          INTEGER NOT NULL,
            step         INTEGER NOT NULL,
            event_type   TEXT NOT NULL,
            summary      TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_text TEXT NOT NULL,              -- flattened text for search
            ref_table    TEXT,
            ref_id       TEXT,
            created_at   TEXT NOT NULL,
            UNIQUE (run_id, seq)
        );
        CREATE INDEX idx_archive_events_run_type ON archive_events(run_id, event_type, seq);
        CREATE INDEX idx_archive_events_run_step ON archive_events(run_id, step);

        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY,
            run_id         TEXT NOT NULL REFERENCES runs(run_id),
            step           INTEGER NOT NULL,
            kind           TEXT NOT NULL,
            source         TEXT NOT NULL,
            content        TEXT NOT NULL,            -- bounded excerpt shown to the model
            full_content   TEXT NOT NULL,            -- complete content (archive)
            truncated      INTEGER NOT NULL DEFAULT 0,
            event_id       TEXT REFERENCES archive_events(event_id),
            data_json      TEXT,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX idx_observations_run_step ON observations(run_id, step);

        CREATE TABLE actions (
            action_id   TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL REFERENCES runs(run_id),
            step        INTEGER NOT NULL,
            kind        TEXT NOT NULL,
            action_json TEXT NOT NULL,
            status      TEXT NOT NULL,               -- pending | executing | executed | failed | interrupted
            event_id    TEXT REFERENCES archive_events(event_id),
            created_at  TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE INDEX idx_actions_run_step ON actions(run_id, step);

        CREATE TABLE tool_executions (
            execution_id   TEXT PRIMARY KEY,
            run_id         TEXT NOT NULL REFERENCES runs(run_id),
            step           INTEGER NOT NULL,
            action_id      TEXT REFERENCES actions(action_id),
            tool_name      TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            status         TEXT NOT NULL,            -- started | succeeded | failed | interrupted
            output         TEXT,
            error          TEXT,
            data_json      TEXT,
            started_at     TEXT NOT NULL,
            finished_at    TEXT,
            duration_ms    INTEGER,
            event_id       TEXT REFERENCES archive_events(event_id)
        );
        CREATE INDEX idx_tool_executions_run ON tool_executions(run_id, step);

        CREATE TABLE artifacts (
            artifact_id          TEXT PRIMARY KEY,
            run_id               TEXT NOT NULL REFERENCES runs(run_id),
            kind                 TEXT NOT NULL,
            locator              TEXT NOT NULL,
            description          TEXT NOT NULL,
            originating_event_id TEXT,
            created_at           TEXT NOT NULL
        );
        CREATE INDEX idx_artifacts_run ON artifacts(run_id);

        CREATE TABLE model_calls (
            call_id           TEXT PRIMARY KEY,
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            step              INTEGER NOT NULL,
            attempt           INTEGER NOT NULL DEFAULT 1,
            model_name        TEXT NOT NULL,
            context_json      TEXT NOT NULL,           -- exact ModelContext sent
            context_chars     INTEGER NOT NULL,
            approx_tokens     INTEGER NOT NULL,
            input_tokens      INTEGER,
            output_tokens     INTEGER,
            requests          INTEGER,
            status            TEXT NOT NULL,           -- ok | error
            error             TEXT,
            decision_json     TEXT,
            raw_messages_json TEXT,                    -- archived; never re-sent
            duration_ms       INTEGER,
            created_at        TEXT NOT NULL
        );
        CREATE INDEX idx_model_calls_run_step ON model_calls(run_id, step);

        CREATE TABLE memory_retrievals (
            retrieval_id TEXT PRIMARY KEY,
            run_id       TEXT NOT NULL REFERENCES runs(run_id),
            step         INTEGER NOT NULL,
            query_json   TEXT NOT NULL,
            result_json  TEXT NOT NULL,
            result_count INTEGER NOT NULL,
            event_id     TEXT REFERENCES archive_events(event_id),
            created_at   TEXT NOT NULL
        );

        CREATE TABLE semantic_memories (
            memory_id        TEXT PRIMARY KEY,
            key              TEXT NOT NULL UNIQUE,
            category         TEXT NOT NULL,
            content          TEXT NOT NULL,
            confidence       REAL NOT NULL DEFAULT 1.0,
            source_run_id    TEXT,
            source_event_ids TEXT NOT NULL,            -- JSON list
            promoted_by      TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE steps (
            run_id          TEXT NOT NULL REFERENCES runs(run_id),
            step            INTEGER NOT NULL,
            phase           TEXT NOT NULL,
            observation_id  TEXT REFERENCES observations(observation_id),
            retrieved_json  TEXT,                     -- MemoryResult(s) injected into this step's context
            decision_json   TEXT,
            patch_id        TEXT,
            action_id       TEXT,
            state_version_before INTEGER,
            state_version_after  INTEGER,
            started_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (run_id, step)
        );

        CREATE TABLE step_metrics (
            run_id            TEXT NOT NULL REFERENCES runs(run_id),
            step              INTEGER NOT NULL,
            context_chars     INTEGER,
            context_tokens_est INTEGER,
            input_tokens      INTEGER,
            output_tokens     INTEGER,
            state_bytes       INTEGER,
            model_calls       INTEGER NOT NULL DEFAULT 0,
            tool_calls        INTEGER NOT NULL DEFAULT 0,
            retrievals        INTEGER NOT NULL DEFAULT 0,
            patch_applied     INTEGER NOT NULL DEFAULT 0,
            elapsed_ms        INTEGER,
            created_at        TEXT NOT NULL,
            PRIMARY KEY (run_id, step)
        );

        CREATE TABLE errors (
            error_id   TEXT PRIMARY KEY,
            run_id     TEXT REFERENCES runs(run_id),
            step       INTEGER,
            kind       TEXT NOT NULL,
            message    TEXT NOT NULL,
            traceback  TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    if fts_enabled:
        run_script(conn, 
            """
            CREATE VIRTUAL TABLE archive_events_fts USING fts5(
                summary, payload_text, event_type UNINDEXED,
                content='archive_events', content_rowid='rowid'
            );
            CREATE TRIGGER archive_events_ai AFTER INSERT ON archive_events BEGIN
                INSERT INTO archive_events_fts(rowid, summary, payload_text, event_type)
                VALUES (new.rowid, new.summary, new.payload_text, new.event_type);
            END;
            CREATE TRIGGER archive_events_ad AFTER DELETE ON archive_events BEGIN
                INSERT INTO archive_events_fts(archive_events_fts, rowid, summary, payload_text, event_type)
                VALUES ('delete', old.rowid, old.summary, old.payload_text, old.event_type);
            END;
            """
        )


def _m002_cache_tokens(conn: sqlite3.Connection, fts_enabled: bool) -> None:
    run_script(
        conn,
        """
        ALTER TABLE model_calls ADD COLUMN cache_read_tokens INTEGER;
        ALTER TABLE model_calls ADD COLUMN cache_write_tokens INTEGER;
        ALTER TABLE step_metrics ADD COLUMN cache_read_tokens INTEGER;
        ALTER TABLE step_metrics ADD COLUMN cache_write_tokens INTEGER;
        """,
    )


MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "initial schema", _m001_initial),
    (2, "prompt-cache token accounting", _m002_cache_tokens),
]


def apply_migrations(conn: sqlite3.Connection, *, fts_enabled: bool) -> list[int]:
    """Apply pending migrations in order. Caller holds the transaction."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    newly: list[int] = []
    for version, name, fn in MIGRATIONS:
        if version in applied:
            continue
        fn(conn, fts_enabled)
        conn.execute(
            "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
            (version, name),
        )
        newly.append(version)
    return newly


def current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0] or 0)
