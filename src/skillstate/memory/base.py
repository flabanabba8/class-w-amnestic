"""Retrieval protocol. Implementations may be lexical (v1, SQLite/FTS5) or semantic (later)."""

from __future__ import annotations

from typing import Protocol

from skillstate.models.archive import MemoryQuery, MemoryResult


class Retriever(Protocol):
    def retrieve(self, run_id: str, query: MemoryQuery) -> MemoryResult:
        """Return a bounded MemoryResult. Must never return more than ``query.limit`` events."""
        ...
