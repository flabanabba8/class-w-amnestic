"""Transactional repository facade over SQLite.

All writes that belong to one lifecycle phase are wrapped by the caller in
``store.transaction()`` so they commit atomically. State commits use optimistic
concurrency: ``UPDATE current_states ... WHERE state_version = ?`` must affect
exactly one row, otherwise ``StaleWriteError`` is raised and nothing is written.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import Any

from pydantic import Field

from mnestic.models.archive import ArchiveEvent, EventType, MemoryQuery, MemoryResult, RunMetadata
from mnestic.models.common import Record, new_id, utcnow
from mnestic.models.observation import Observation
from mnestic.models.patch import StatePatch
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ArtifactReference, ExecutionState
from mnestic.state.apply import state_size_bytes
from mnestic.storage.db import Database


class StaleWriteError(Exception):
    """A concurrent writer advanced the state; this write was rejected."""

    def __init__(self, run_id: str, expected: int, actual: int | None):
        self.run_id, self.expected, self.actual = run_id, expected, actual
        super().__init__(f"run {run_id}: expected state_version {expected}, found {actual}")


class StepRecord(Record):
    run_id: str
    step: int
    phase: str
    observation_id: str | None = None
    retrieved: list[MemoryResult] = Field(default_factory=list)
    decision: dict[str, Any] | None = None
    patch_id: str | None = None
    action_id: str | None = None
    state_version_before: int | None = None
    state_version_after: int | None = None
    started_at: datetime
    updated_at: datetime


class ToolExecutionRecord(Record):
    execution_id: str
    run_id: str
    step: int
    action_id: str | None
    tool_name: str
    arguments: dict[str, Any]
    status: str
    output: str | None = None
    error: str | None = None
    data: dict[str, Any] | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    event_id: str | None = None


class SemanticMemory(Record):
    memory_id: str
    key: str
    category: str
    content: str
    confidence: float
    source_run_id: str | None
    source_event_ids: list[str]
    promoted_by: str
    created_at: datetime
    updated_at: datetime


def _ts(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def flatten_text(value: Any, limit: int = 20_000) -> str:
    """Flatten a JSON payload into searchable text (keys and string scalars)."""
    parts: list[str] = []

    def walk(v: Any) -> None:
        if sum(len(p) for p in parts) > limit:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                parts.append(str(k))
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif v is not None:
            parts.append(str(v))

    walk(value)
    return " ".join(parts)[:limit]


class Store:
    def __init__(self, db: Database):
        self.db = db

    @property
    def transaction(self):
        return self.db.transaction

    # ---- skills ------------------------------------------------------------------------

    def upsert_skill(self, spec: SkillSpecification) -> None:
        self.db.execute(
            """INSERT INTO skills(skill_id, version, name, description, content_hash, spec_json, loaded_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(skill_id, version) DO UPDATE SET
                 name=excluded.name, description=excluded.description, content_hash=excluded.content_hash,
                 spec_json=excluded.spec_json, loaded_at=excluded.loaded_at""",
            (spec.skill_id, spec.version, spec.name, spec.description, spec.content_hash,
             spec.model_dump_json(exclude={"content_hash"}), _ts()),
        )

    def get_skill(self, skill_id: str, version: str) -> SkillSpecification | None:
        row = self.db.query_one("SELECT spec_json, content_hash FROM skills WHERE skill_id=? AND version=?", (skill_id, version))
        if row is None:
            return None
        data = json.loads(row["spec_json"])
        data.pop("content_hash", None)
        spec = SkillSpecification.model_validate(data)
        if spec.content_hash != row["content_hash"]:
            raise ValueError(f"stored skill {skill_id}@{version} does not match its recorded content hash (tampered?)")
        return spec

    # ---- runs --------------------------------------------------------------------------

    def create_run(self, meta: RunMetadata) -> None:
        self.db.execute(
            """INSERT INTO runs(run_id, skill_id, skill_version, skill_content_hash, status, workspace_root, model_name,
                                task_input, last_step, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta.run_id, meta.skill_id, meta.skill_version, meta.skill_content_hash, meta.status,
                meta.workspace_root, meta.model_name, meta.task_input, meta.last_step, _ts(meta.created_at), _ts(meta.updated_at),
            ),
        )

    def get_run(self, run_id: str) -> RunMetadata | None:
        row = self.db.query_one(
            "SELECT r.*, cs.state_version FROM runs r LEFT JOIN current_states cs USING(run_id) WHERE r.run_id=?", (run_id,)
        )
        return self._run_from_row(row) if row else None

    def list_runs(self, limit: int = 50) -> list[RunMetadata]:
        rows = self.db.query(
            "SELECT r.*, cs.state_version FROM runs r LEFT JOIN current_states cs USING(run_id) ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._run_from_row(r) for r in rows]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        last_step: int | None = None,
        finished_at: datetime | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        sets: list[str] = ["updated_at=?"]
        params: list[Any] = [_ts()]
        if status is not None:
            sets.append("status=?")
            params.append(status)
        if last_step is not None:
            sets.append("last_step=?")
            params.append(last_step)
        if finished_at is not None:
            sets.append("finished_at=?")
            params.append(_ts(finished_at))
        if outcome is not None:
            sets.append("outcome_json=?")
            params.append(_json(outcome))
        params.append(run_id)
        self.db.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?", tuple(params))

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunMetadata:
        return RunMetadata(
            run_id=row["run_id"], skill_id=row["skill_id"], skill_version=row["skill_version"],
            skill_content_hash=row["skill_content_hash"], status=row["status"], created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]), workspace_root=row["workspace_root"], model_name=row["model_name"],
            task_input=row["task_input"], last_step=row["last_step"], state_version=row["state_version"] or 0,
            finished_at=_dt(row["finished_at"]), outcome=json.loads(row["outcome_json"]) if row["outcome_json"] else None,
        )

    # ---- states ------------------------------------------------------------------------

    def init_state(self, state: ExecutionState) -> None:
        size = state_size_bytes(state)
        self.db.execute(
            "INSERT INTO current_states(run_id, state_version, status, state_json, state_bytes, updated_at) VALUES (?,?,?,?,?,?)",
            (state.run_id, state.state_version, state.status.value, state.model_dump_json(), size, _ts()),
        )
        self.db.execute(
            "INSERT INTO state_versions(run_id, version, step, patch_id, state_json, state_bytes, created_at) VALUES (?,?,?,?,?,?,?)",
            (state.run_id, state.state_version, 0, None, state.model_dump_json(), size, _ts()),
        )

    def get_state(self, run_id: str) -> ExecutionState:
        row = self.db.query_one("SELECT state_json FROM current_states WHERE run_id=?", (run_id,))
        if row is None:
            raise KeyError(f"no state for run {run_id}")
        return ExecutionState.model_validate_json(row["state_json"])

    def get_state_version_number(self, run_id: str) -> int | None:
        row = self.db.query_one("SELECT state_version FROM current_states WHERE run_id=?", (run_id,))
        return int(row["state_version"]) if row else None

    def commit_state(self, new_state: ExecutionState, *, expected_version: int, patch_id: str | None, step: int) -> None:
        """Atomically replace the current state iff it is still at ``expected_version``."""
        if new_state.state_version != expected_version + 1:
            raise ValueError("new_state.state_version must be expected_version + 1")
        size = state_size_bytes(new_state)
        with self.db.transaction():
            cur = self.db.execute(
                """UPDATE current_states SET state_version=?, status=?, state_json=?, state_bytes=?, updated_at=?
                   WHERE run_id=? AND state_version=?""",
                (new_state.state_version, new_state.status.value, new_state.model_dump_json(), size, _ts(),
                 new_state.run_id, expected_version),
            )
            if cur.rowcount != 1:
                actual = self.get_state_version_number(new_state.run_id)
                raise StaleWriteError(new_state.run_id, expected_version, actual)
            self.db.execute(
                "INSERT INTO state_versions(run_id, version, step, patch_id, state_json, state_bytes, created_at) VALUES (?,?,?,?,?,?,?)",
                (new_state.run_id, new_state.state_version, step, patch_id, new_state.model_dump_json(), size, _ts()),
            )
            self.db.execute("UPDATE runs SET status=?, updated_at=? WHERE run_id=?", (new_state.status.value, _ts(), new_state.run_id))

    def get_state_at_version(self, run_id: str, version: int) -> ExecutionState | None:
        row = self.db.query_one("SELECT state_json FROM state_versions WHERE run_id=? AND version=?", (run_id, version))
        return ExecutionState.model_validate_json(row["state_json"]) if row else None

    def list_state_versions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT version, step, patch_id, state_bytes, created_at FROM state_versions WHERE run_id=? ORDER BY version", (run_id,)
        )
        return [dict(r) for r in rows]

    def record_patch(
        self,
        run_id: str,
        step: int,
        patch: StatePatch,
        *,
        status: str,
        resulting_version: int | None = None,
        error_code: str | None = None,
        error: str | None = None,
        changes: list[str] | None = None,
    ) -> str:
        patch_id = new_id("patch")
        self.db.execute(
            """INSERT INTO state_patches(patch_id, run_id, step, expected_version, resulting_version, status, error_code, error,
                                         patch_json, changes_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (patch_id, run_id, step, patch.expected_state_version, resulting_version, status, error_code, error,
             patch.model_dump_json(), _json(changes) if changes is not None else None, _ts()),
        )
        return patch_id

    def list_patches(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT patch_id, step, expected_version, resulting_version, status, error_code, error, patch_json, changes_json, created_at "
            "FROM state_patches WHERE run_id=? ORDER BY created_at LIMIT ?",
            (run_id, limit),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["patch"] = json.loads(d.pop("patch_json"))
            d["changes"] = json.loads(d.pop("changes_json")) if d.get("changes_json") else None
            out.append(d)
        return out

    # ---- archive events ----------------------------------------------------------------

    def append_event(
        self,
        run_id: str,
        step: int,
        event_type: EventType,
        summary: str,
        payload: dict[str, Any] | None = None,
        *,
        ref_table: str | None = None,
        ref_id: str | None = None,
    ) -> ArchiveEvent:
        payload = payload or {}
        with self.db.transaction():
            row = self.db.query_one("SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM archive_events WHERE run_id=?", (run_id,))
            seq = int(row["n"]) if row else 0
            event = ArchiveEvent(run_id=run_id, seq=seq, step=step, event_type=event_type, summary=summary[:500],
                                 payload=payload, ref_table=ref_table, ref_id=ref_id)
            self.db.execute(
                """INSERT INTO archive_events(event_id, run_id, seq, step, event_type, summary, payload_json, payload_text,
                                              ref_table, ref_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (event.event_id, run_id, seq, step, event_type.value, event.summary, _json(payload), flatten_text(payload),
                 ref_table, ref_id, _ts(event.created_at)),
            )
        return event

    def get_event(self, event_id: str) -> ArchiveEvent | None:
        row = self.db.query_one("SELECT * FROM archive_events WHERE event_id=?", (event_id,))
        return self._event_from_row(row) if row else None

    def existing_event_ids(self, run_id: str, ids: set[str]) -> set[str]:
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        rows = self.db.query(
            f"SELECT event_id FROM archive_events WHERE run_id=? AND event_id IN ({placeholders})", (run_id, *ids)
        )
        return {r["event_id"] for r in rows}

    def count_events(self, run_id: str, event_type: str | None = None) -> int:
        if event_type:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM archive_events WHERE run_id=? AND event_type=?", (run_id, event_type))
        else:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM archive_events WHERE run_id=?", (run_id,))
        return int(row["n"]) if row else 0

    def list_events(
        self, run_id: str, *, limit: int = 50, offset: int = 0, event_type: str | None = None, step: int | None = None,
        newest_first: bool = False,
    ) -> list[ArchiveEvent]:
        sql = "SELECT * FROM archive_events WHERE run_id=?"
        params: list[Any] = [run_id]
        if event_type:
            sql += " AND event_type=?"
            params.append(event_type)
        if step is not None:
            sql += " AND step=?"
            params.append(step)
        sql += f" ORDER BY seq {'DESC' if newest_first else 'ASC'} LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [self._event_from_row(r) for r in self.db.query(sql, tuple(params))]

    def search_events(
        self, run_id: str, text: str, *, limit: int = 10, event_types: list[str] | None = None
    ) -> tuple[list[ArchiveEvent], int]:
        """Keyword search over summaries and flattened payloads. FTS5 when available, LIKE otherwise."""
        tokens = [t for t in re.findall(r"[\w\-./:@]+", text) if t.strip("-./:@")]
        if not tokens:
            return [], 0
        type_filter, type_params = "", []
        if event_types:
            type_filter = f" AND e.event_type IN ({','.join('?' * len(event_types))})"
            type_params = list(event_types)
        if self.db.fts_enabled:
            for joiner in (" AND ", " OR "):
                match = joiner.join(f'"{t.replace(chr(34), " ")}"' for t in tokens)
                sql = (
                    "SELECT e.* FROM archive_events_fts f JOIN archive_events e ON e.rowid = f.rowid "
                    f"WHERE archive_events_fts MATCH ? AND e.run_id=?{type_filter} ORDER BY bm25(archive_events_fts), e.seq DESC"
                )
                try:
                    rows = self.db.query(sql, (match, run_id, *type_params))
                except sqlite3.OperationalError:
                    rows = []
                if rows:
                    return [self._event_from_row(r) for r in rows[:limit]], len(rows)
            return [], 0
        clauses = " AND ".join("(e.summary LIKE ? OR e.payload_text LIKE ?)" for _ in tokens)
        like_params = [p for t in tokens for p in (f"%{t}%", f"%{t}%")]
        rows = self.db.query(
            f"SELECT e.* FROM archive_events e WHERE e.run_id=? AND {clauses}{type_filter} ORDER BY e.seq DESC",
            (run_id, *like_params, *type_params),
        )
        return [self._event_from_row(r) for r in rows[:limit]], len(rows)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> ArchiveEvent:
        return ArchiveEvent(
            event_id=row["event_id"], run_id=row["run_id"], seq=row["seq"], step=row["step"],
            event_type=EventType(row["event_type"]), summary=row["summary"], payload=json.loads(row["payload_json"]),
            ref_table=row["ref_table"], ref_id=row["ref_id"], created_at=_dt(row["created_at"]),
        )

    # ---- observations ------------------------------------------------------------------

    def save_observation(self, obs: Observation, full_content: str) -> None:
        self.db.execute(
            """INSERT INTO observations(observation_id, run_id, step, kind, source, content, full_content, truncated, event_id,
                                        data_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (obs.id, obs.run_id, obs.step, obs.kind.value, obs.source, obs.content, full_content, int(obs.truncated),
             obs.event_id, _json(obs.data) if obs.data is not None else None, _ts(obs.created_at)),
        )

    def get_observation(self, observation_id: str) -> Observation | None:
        row = self.db.query_one("SELECT * FROM observations WHERE observation_id=?", (observation_id,))
        return self._obs_from_row(row) if row else None

    def get_observation_full_content(self, observation_id: str) -> str | None:
        row = self.db.query_one("SELECT full_content FROM observations WHERE observation_id=?", (observation_id,))
        return row["full_content"] if row else None

    def list_observations(self, run_id: str, limit: int = 100) -> list[Observation]:
        rows = self.db.query("SELECT * FROM observations WHERE run_id=? ORDER BY step, created_at LIMIT ?", (run_id, limit))
        return [self._obs_from_row(r) for r in rows]

    @staticmethod
    def _obs_from_row(row: sqlite3.Row) -> Observation:
        return Observation(
            id=row["observation_id"], run_id=row["run_id"], step=row["step"], kind=row["kind"], source=row["source"],
            content=row["content"], truncated=bool(row["truncated"]), full_length=len(row["full_content"]),
            event_id=row["event_id"], data=json.loads(row["data_json"]) if row["data_json"] else None,
            created_at=_dt(row["created_at"]),
        )

    # ---- actions -----------------------------------------------------------------------

    def save_action(self, run_id: str, step: int, action: dict[str, Any], *, status: str, event_id: str | None) -> str:
        action_id = new_id("action")
        self.db.execute(
            "INSERT INTO actions(action_id, run_id, step, kind, action_json, status, event_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (action_id, run_id, step, action.get("kind", "?"), _json(action), status, event_id, _ts()),
        )
        return action_id

    def update_action(self, action_id: str, *, status: str) -> None:
        self.db.execute("UPDATE actions SET status=?, finished_at=? WHERE action_id=?", (status, _ts(), action_id))

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM actions WHERE action_id=?", (action_id,))
        if not row:
            return None
        d = dict(row)
        d["action"] = json.loads(d.pop("action_json"))
        return d

    # ---- tool executions ---------------------------------------------------------------

    def start_tool_execution(self, run_id: str, step: int, action_id: str | None, tool_name: str, arguments: dict[str, Any]) -> str:
        execution_id = new_id("exec")
        self.db.execute(
            """INSERT INTO tool_executions(execution_id, run_id, step, action_id, tool_name, arguments_json, status, started_at)
               VALUES (?,?,?,?,?,?,'started',?)""",
            (execution_id, run_id, step, action_id, tool_name, _json(arguments), _ts()),
        )
        return execution_id

    def finish_tool_execution(
        self, execution_id: str, *, status: str, output: str | None, error: str | None,
        data: dict[str, Any] | None, event_id: str | None, duration_ms: int | None,
    ) -> None:
        self.db.execute(
            """UPDATE tool_executions SET status=?, output=?, error=?, data_json=?, event_id=?, finished_at=?, duration_ms=?
               WHERE execution_id=?""",
            (status, output, error, _json(data) if data is not None else None, event_id, _ts(), duration_ms, execution_id),
        )

    def get_tool_execution(self, execution_id: str) -> ToolExecutionRecord | None:
        row = self.db.query_one("SELECT * FROM tool_executions WHERE execution_id=?", (execution_id,))
        return self._exec_from_row(row) if row else None

    def find_open_tool_execution(self, run_id: str, step: int) -> ToolExecutionRecord | None:
        row = self.db.query_one(
            "SELECT * FROM tool_executions WHERE run_id=? AND step=? AND status='started' ORDER BY started_at DESC LIMIT 1",
            (run_id, step),
        )
        return self._exec_from_row(row) if row else None

    def list_tool_executions(self, run_id: str, limit: int = 100) -> list[ToolExecutionRecord]:
        rows = self.db.query("SELECT * FROM tool_executions WHERE run_id=? ORDER BY started_at LIMIT ?", (run_id, limit))
        return [self._exec_from_row(r) for r in rows]

    @staticmethod
    def _exec_from_row(row: sqlite3.Row) -> ToolExecutionRecord:
        return ToolExecutionRecord(
            execution_id=row["execution_id"], run_id=row["run_id"], step=row["step"], action_id=row["action_id"],
            tool_name=row["tool_name"], arguments=json.loads(row["arguments_json"]), status=row["status"],
            output=row["output"], error=row["error"], data=json.loads(row["data_json"]) if row["data_json"] else None,
            started_at=_dt(row["started_at"]), finished_at=_dt(row["finished_at"]), duration_ms=row["duration_ms"],
            event_id=row["event_id"],
        )

    # ---- artifacts ---------------------------------------------------------------------

    def save_artifact(self, run_id: str, artifact: ArtifactReference) -> None:
        self.db.execute(
            """INSERT INTO artifacts(artifact_id, run_id, kind, locator, description, originating_event_id, created_at)
               VALUES (?,?,?,?,?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET locator=excluded.locator, description=excluded.description""",
            (artifact.id, run_id, artifact.kind, artifact.locator, artifact.description, artifact.originating_event_id,
             _ts(artifact.created_at)),
        )

    def list_artifacts(self, run_id: str, *, text: str | None = None, limit: int = 50) -> list[ArtifactReference]:
        if text:
            rows = self.db.query(
                "SELECT * FROM artifacts WHERE run_id=? AND (locator LIKE ? OR description LIKE ?) ORDER BY created_at LIMIT ?",
                (run_id, f"%{text}%", f"%{text}%", limit),
            )
        else:
            rows = self.db.query("SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at LIMIT ?", (run_id, limit))
        return [
            ArtifactReference(id=r["artifact_id"], kind=r["kind"], locator=r["locator"], description=r["description"],
                              originating_event_id=r["originating_event_id"], created_at=_dt(r["created_at"]))
            for r in rows
        ]

    # ---- model calls -------------------------------------------------------------------

    def save_model_call(
        self, run_id: str, step: int, *, attempt: int, model_name: str, context_json: str, context_chars: int,
        approx_tokens: int, input_tokens: int | None, output_tokens: int | None, requests: int | None, status: str,
        error: str | None, decision_json: str | None, raw_messages_json: str | None, duration_ms: int | None,
    ) -> str:
        call_id = new_id("call")
        self.db.execute(
            """INSERT INTO model_calls(call_id, run_id, step, attempt, model_name, context_json, context_chars, approx_tokens,
                                       input_tokens, output_tokens, requests, status, error, decision_json, raw_messages_json,
                                       duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (call_id, run_id, step, attempt, model_name, context_json, context_chars, approx_tokens, input_tokens,
             output_tokens, requests, status, error, decision_json, raw_messages_json, duration_ms, _ts()),
        )
        return call_id

    def get_model_calls(self, run_id: str, step: int | None = None) -> list[dict[str, Any]]:
        if step is None:
            rows = self.db.query("SELECT * FROM model_calls WHERE run_id=? ORDER BY created_at", (run_id,))
        else:
            rows = self.db.query("SELECT * FROM model_calls WHERE run_id=? AND step=? ORDER BY created_at", (run_id, step))
        return [dict(r) for r in rows]

    # ---- retrievals --------------------------------------------------------------------

    def save_retrieval(self, run_id: str, step: int, query: MemoryQuery, result: MemoryResult, event_id: str | None) -> str:
        retrieval_id = new_id("ret")
        self.db.execute(
            "INSERT INTO memory_retrievals(retrieval_id, run_id, step, query_json, result_json, result_count, event_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (retrieval_id, run_id, step, query.model_dump_json(), result.model_dump_json(), len(result.events), event_id, _ts()),
        )
        return retrieval_id

    # ---- semantic memory ---------------------------------------------------------------

    def upsert_semantic(
        self, *, key: str, category: str, content: str, confidence: float, source_run_id: str | None,
        source_event_ids: list[str], promoted_by: str,
    ) -> SemanticMemory:
        now = _ts()
        self.db.execute(
            """INSERT INTO semantic_memories(memory_id, key, category, content, confidence, source_run_id, source_event_ids,
                                             promoted_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET category=excluded.category, content=excluded.content,
                 confidence=excluded.confidence, source_run_id=excluded.source_run_id,
                 source_event_ids=excluded.source_event_ids, promoted_by=excluded.promoted_by, updated_at=excluded.updated_at""",
            (new_id("mem"), key, category, content, confidence, source_run_id, _json(source_event_ids), promoted_by, now, now),
        )
        mem = self.get_semantic(key)
        assert mem is not None
        return mem

    def get_semantic(self, key: str) -> SemanticMemory | None:
        row = self.db.query_one("SELECT * FROM semantic_memories WHERE key=?", (key,))
        return self._sem_from_row(row) if row else None

    def delete_semantic(self, key: str) -> bool:
        return self.db.execute("DELETE FROM semantic_memories WHERE key=?", (key,)).rowcount > 0

    def list_semantic(self, *, category: str | None = None, limit: int = 100) -> list[SemanticMemory]:
        if category:
            rows = self.db.query("SELECT * FROM semantic_memories WHERE category=? ORDER BY key LIMIT ?", (category, limit))
        else:
            rows = self.db.query("SELECT * FROM semantic_memories ORDER BY key LIMIT ?", (limit,))
        return [self._sem_from_row(r) for r in rows]

    def search_semantic(self, text: str, limit: int = 10) -> list[SemanticMemory]:
        tokens = re.findall(r"\w+", text)
        if not tokens:
            return []
        clauses = " AND ".join("(key LIKE ? OR content LIKE ? OR category LIKE ?)" for _ in tokens)
        params = [p for t in tokens for p in (f"%{t}%",) * 3]
        rows = self.db.query(f"SELECT * FROM semantic_memories WHERE {clauses} ORDER BY updated_at DESC LIMIT ?", (*params, limit))
        return [self._sem_from_row(r) for r in rows]

    @staticmethod
    def _sem_from_row(row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            memory_id=row["memory_id"], key=row["key"], category=row["category"], content=row["content"],
            confidence=row["confidence"], source_run_id=row["source_run_id"], source_event_ids=json.loads(row["source_event_ids"]),
            promoted_by=row["promoted_by"], created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
        )

    # ---- steps -------------------------------------------------------------------------

    def create_step(
        self, run_id: str, step: int, *, observation_id: str | None, retrieved: list[MemoryResult] | None,
        state_version_before: int, phase: str = "observation_ready",
    ) -> StepRecord:
        now = _ts()
        self.db.execute(
            """INSERT INTO steps(run_id, step, phase, observation_id, retrieved_json, state_version_before, started_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, step, phase, observation_id, _json([r.model_dump(mode="json") for r in retrieved]) if retrieved else None,
             state_version_before, now, now),
        )
        self.db.execute("UPDATE runs SET last_step=?, updated_at=? WHERE run_id=?", (step, now, run_id))
        rec = self.get_step(run_id, step)
        assert rec is not None
        return rec

    def update_step(
        self, run_id: str, step: int, *, phase: str, decision: dict[str, Any] | None = None, patch_id: str | None = None,
        action_id: str | None = None, state_version_after: int | None = None,
    ) -> None:
        sets: list[str] = ["phase=?", "updated_at=?"]
        params: list[Any] = [phase, _ts()]
        if decision is not None:
            sets.append("decision_json=?")
            params.append(_json(decision))
        if patch_id is not None:
            sets.append("patch_id=?")
            params.append(patch_id)
        if action_id is not None:
            sets.append("action_id=?")
            params.append(action_id)
        if state_version_after is not None:
            sets.append("state_version_after=?")
            params.append(state_version_after)
        params += [run_id, step]
        self.db.execute(f"UPDATE steps SET {', '.join(sets)} WHERE run_id=? AND step=?", tuple(params))

    def get_step(self, run_id: str, step: int) -> StepRecord | None:
        row = self.db.query_one("SELECT * FROM steps WHERE run_id=? AND step=?", (run_id, step))
        return self._step_from_row(row) if row else None

    def latest_step(self, run_id: str) -> StepRecord | None:
        row = self.db.query_one("SELECT * FROM steps WHERE run_id=? ORDER BY step DESC LIMIT 1", (run_id,))
        return self._step_from_row(row) if row else None

    def list_steps(self, run_id: str, limit: int = 1000) -> list[StepRecord]:
        rows = self.db.query("SELECT * FROM steps WHERE run_id=? ORDER BY step LIMIT ?", (run_id, limit))
        return [self._step_from_row(r) for r in rows]

    @staticmethod
    def _step_from_row(row: sqlite3.Row) -> StepRecord:
        retrieved = [MemoryResult.model_validate(r) for r in json.loads(row["retrieved_json"])] if row["retrieved_json"] else []
        return StepRecord(
            run_id=row["run_id"], step=row["step"], phase=row["phase"], observation_id=row["observation_id"],
            retrieved=retrieved, decision=json.loads(row["decision_json"]) if row["decision_json"] else None,
            patch_id=row["patch_id"], action_id=row["action_id"], state_version_before=row["state_version_before"],
            state_version_after=row["state_version_after"], started_at=_dt(row["started_at"]), updated_at=_dt(row["updated_at"]),
        )

    # ---- metrics / errors --------------------------------------------------------------

    METRIC_COLUMNS = ("context_chars", "context_tokens_est", "input_tokens", "output_tokens", "state_bytes", "model_calls",
                      "tool_calls", "retrievals", "patch_applied", "elapsed_ms")
    COUNTER_COLUMNS = frozenset({"model_calls", "tool_calls", "retrievals", "patch_applied"})

    def save_step_metrics(self, run_id: str, step: int, **metrics: int | None) -> None:
        """Merge metrics for a step: provided values win, missing ones keep their previous value."""
        with self.db.transaction():
            row = self.db.query_one("SELECT * FROM step_metrics WHERE run_id=? AND step=?", (run_id, step))
            merged = {c: (row[c] if row else (0 if c in self.COUNTER_COLUMNS else None)) for c in self.METRIC_COLUMNS}
            merged.update({k: v for k, v in metrics.items() if v is not None and k in self.METRIC_COLUMNS})
            cols = ", ".join(self.METRIC_COLUMNS)
            marks = ",".join("?" * len(self.METRIC_COLUMNS))
            self.db.execute(
                f"INSERT OR REPLACE INTO step_metrics(run_id, step, {cols}, created_at) VALUES (?,?,{marks},?)",
                (run_id, step, *[merged[c] for c in self.METRIC_COLUMNS], row["created_at"] if row else _ts()),
            )

    def run_metrics(self, run_id: str) -> dict[str, Any]:
        row = self.db.query_one(
            """SELECT COUNT(*) AS steps, SUM(model_calls) AS model_calls, SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens, SUM(tool_calls) AS tool_calls, SUM(retrievals) AS retrievals,
                      SUM(patch_applied) AS patches_applied, SUM(elapsed_ms) AS elapsed_ms, MAX(context_chars) AS max_context_chars,
                      AVG(context_chars) AS avg_context_chars, MAX(state_bytes) AS max_state_bytes
               FROM step_metrics WHERE run_id=?""",
            (run_id,),
        )
        metrics = dict(row) if row else {}
        metrics["archive_events"] = self.count_events(run_id)
        return metrics

    def list_step_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM step_metrics WHERE run_id=? ORDER BY step", (run_id,))]

    def save_error(self, run_id: str | None, step: int | None, kind: str, message: str, traceback: str | None = None) -> str:
        error_id = new_id("err")
        self.db.execute(
            "INSERT INTO errors(error_id, run_id, step, kind, message, traceback, created_at) VALUES (?,?,?,?,?,?,?)",
            (error_id, run_id, step, kind, message[:4000], traceback, _ts()),
        )
        return error_id
