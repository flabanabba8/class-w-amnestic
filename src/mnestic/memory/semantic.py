"""Semantic (durable, cross-run) memory. Promotion is explicit and audited."""

from __future__ import annotations

from mnestic.models.archive import EventType
from mnestic.storage.store import SemanticMemory, Store


class SemanticMemoryStore:
    def __init__(self, store: Store):
        self.store = store

    def promote(
        self,
        *,
        key: str,
        content: str,
        category: str = "general",
        confidence: float = 1.0,
        source_run_id: str | None = None,
        source_event_ids: list[str] | None = None,
        promoted_by: str = "human",
        step: int = 0,
    ) -> SemanticMemory:
        """Explicitly promote knowledge into durable memory and record the promotion in the run archive."""
        source_event_ids = list(source_event_ids or [])
        if source_run_id:
            missing = set(source_event_ids) - self.store.existing_event_ids(source_run_id, set(source_event_ids))
            if missing:
                raise ValueError(f"source events not found in run {source_run_id}: {sorted(missing)}")
        with self.store.transaction():
            mem = self.store.upsert_semantic(
                key=key, category=category, content=content, confidence=confidence, source_run_id=source_run_id,
                source_event_ids=source_event_ids, promoted_by=promoted_by,
            )
            if source_run_id:
                self.store.append_event(
                    source_run_id, step, EventType.SEMANTIC_PROMOTED, f"semantic memory promoted: {key}",
                    {"key": key, "category": category, "content": content, "promoted_by": promoted_by,
                     "source_event_ids": source_event_ids},
                    ref_table="semantic_memories", ref_id=mem.memory_id,
                )
        return mem

    def get(self, key: str) -> SemanticMemory | None:
        return self.store.get_semantic(key)

    def forget(self, key: str) -> bool:
        return self.store.delete_semantic(key)

    def list_all(self, category: str | None = None) -> list[SemanticMemory]:
        return self.store.list_semantic(category=category)

    def search(self, text: str, limit: int = 10) -> list[SemanticMemory]:
        return self.store.search_semantic(text, limit=limit)
