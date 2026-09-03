"""Deterministic SQLite-backed archival retrieval.

Every query returns a bounded ``MemoryResult``; only that subset is injected into the
next context. Semantic/vector retrieval can be added as another ``Retriever`` later
without changing the runtime.
"""

from __future__ import annotations

import json
from typing import Any

from mnestic.models.archive import ArchiveEvent, EventType, MemoryQuery, MemoryResult, RetrievedEvent
from mnestic.storage.store import Store

OBSERVATION_TYPES = [EventType.OBSERVATION.value, EventType.TASK_INPUT.value, EventType.HUMAN_RESPONSE.value]
TOOL_TYPES = [EventType.TOOL_FINISHED.value, EventType.TOOL_FAILED.value, EventType.TOOL_STARTED.value, EventType.TOOL_INTERRUPTED.value]


class ArchiveRetriever:
    def __init__(self, store: Store, *, excerpt_chars: int = 1200):
        self.store = store
        self.excerpt_chars = excerpt_chars

    def retrieve(self, run_id: str, query: MemoryQuery) -> MemoryResult:
        q = query
        events: list[ArchiveEvent] = []
        total = 0
        note = ""
        if q.query_type == "recent":
            events = self.store.list_events(run_id, limit=q.limit, newest_first=True, event_type=q.event_type)
            total = self.store.count_events(run_id, q.event_type)
        elif q.query_type == "event":
            if not q.event_id:
                note = "event query requires event_id"
            else:
                ev = self.store.get_event(q.event_id)
                if ev and ev.run_id == run_id:
                    events, total = [ev], 1
                else:
                    note = f"event {q.event_id} not found"
        elif q.query_type == "events_by_type":
            if not q.event_type:
                note = "events_by_type requires event_type"
            else:
                events = self.store.list_events(run_id, limit=q.limit, newest_first=True, event_type=q.event_type)
                total = self.store.count_events(run_id, q.event_type)
        elif q.query_type == "search":
            if not q.text:
                note = "search requires text"
            else:
                events, total = self.store.search_events(run_id, q.text, limit=q.limit)
        elif q.query_type == "observations":
            if not q.text:
                events = self.store.list_events(run_id, limit=q.limit, newest_first=True, event_type=EventType.OBSERVATION.value)
                total = len(events)
            else:
                events, total = self.store.search_events(run_id, q.text, limit=q.limit, event_types=OBSERVATION_TYPES)
        elif q.query_type == "tool_executions":
            if not q.text:
                events = self.store.list_events(run_id, limit=q.limit, newest_first=True, event_type=EventType.TOOL_FINISHED.value)
                total = len(events)
            else:
                events, total = self.store.search_events(run_id, q.text, limit=q.limit, event_types=TOOL_TYPES)
        elif q.query_type == "artifacts":
            arts = self.store.list_artifacts(run_id, text=q.text, limit=q.limit)
            total = len(arts)
            return MemoryResult(
                query=q,
                events=[
                    RetrievedEvent(
                        event_id=a.originating_event_id or a.id, step=0, event_type="artifact", created_at=a.created_at,
                        summary=f"artifact {a.id} ({a.kind}) {a.locator}", excerpt=a.description[: self.excerpt_chars],
                    )
                    for a in arts
                ],
                total_matches=total,
            )
        elif q.query_type == "state_at_version":
            if q.version is None:
                note = "state_at_version requires version"
            else:
                st = self.store.get_state_at_version(run_id, q.version)
                if st is None:
                    note = f"no state version {q.version}"
                else:
                    payload = json.dumps(st.model_view(), separators=(",", ":"), ensure_ascii=False)
                    return MemoryResult(
                        query=q,
                        events=[
                            RetrievedEvent(
                                event_id=f"state@{q.version}", step=0, event_type="state_version", created_at=st.updated_at,
                                summary=f"execution state at version {q.version}", excerpt=payload[: self.excerpt_chars * 4],
                            )
                        ],
                        total_matches=1,
                        truncated=len(payload) > self.excerpt_chars * 4,
                    )
        elif q.query_type == "state_history":
            versions = self.store.list_state_versions(run_id)
            total = len(versions)
            recent = versions[-q.limit :]
            patches = {p["resulting_version"]: p for p in self.store.list_patches(run_id) if p["status"] == "applied"}
            out = []
            for v in recent:
                p = patches.get(v["version"])
                changes = ", ".join(p["changes"]) if p and p.get("changes") else "(initial)"
                out.append(
                    RetrievedEvent(
                        event_id=f"state@{v['version']}", step=v["step"], event_type="state_version",
                        created_at=v["created_at"], summary=f"version {v['version']} at step {v['step']}",
                        excerpt=changes[: self.excerpt_chars],
                    )
                )
            return MemoryResult(query=q, events=out, total_matches=total, truncated=total > len(out))
        elif q.query_type == "semantic":
            mems = self.store.search_semantic(q.text, limit=q.limit) if q.text else self.store.list_semantic(limit=q.limit)
            return MemoryResult(
                query=q,
                events=[
                    RetrievedEvent(
                        event_id=m.memory_id, step=0, event_type="semantic_memory", created_at=m.updated_at,
                        summary=f"[{m.category}] {m.key}", excerpt=m.content[: self.excerpt_chars],
                    )
                    for m in mems
                ],
                total_matches=len(mems),
            )
        else:  # pragma: no cover - Literal type prevents this
            note = f"unknown query_type {q.query_type}"

        retrieved = [self._to_retrieved(e) for e in events[: q.limit]]
        return MemoryResult(query=q, events=retrieved, total_matches=total, truncated=total > len(retrieved), note=note)

    def _to_retrieved(self, e: ArchiveEvent) -> RetrievedEvent:
        return RetrievedEvent(
            event_id=e.event_id, step=e.step, event_type=e.event_type.value, created_at=e.created_at,
            summary=e.summary, excerpt=_excerpt(e.payload, self.excerpt_chars),
        )


def _excerpt(payload: dict[str, Any], limit: int) -> str:
    """Prefer the human-relevant text fields, fall back to compact JSON."""
    for key in ("content", "full_content", "output", "text", "error", "question", "answer"):
        v = payload.get(key)
        if isinstance(v, str) and v:
            return v[:limit]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)[:limit]
