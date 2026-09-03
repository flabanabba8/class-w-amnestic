"""Archival memory search exposed as a tool (same bounded retriever as MemoryQuery)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from skillstate.models.archive import MemoryQuery
from skillstate.tools.base import Tool, ToolContext, ToolError, ToolResult


class ArchivalMemorySearchTool(Tool):
    name: ClassVar[str] = "archival_memory_search"
    description: ClassVar[str] = "Keyword-search this run's archive (observations, tool outputs, events). Prefer memory_query when possible."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=500)
        limit: int = Field(default=5, ge=1, le=20)

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        if ctx.retriever is None:
            raise ToolError("no retriever configured")
        result = ctx.retriever.retrieve(ctx.run_id, MemoryQuery(query_type="search", text=args.text, limit=args.limit))
        lines = [f"{len(result.events)} of {result.total_matches} matches"]
        for e in result.events:
            lines.append(f"[{e.event_id}] step {e.step} {e.event_type}: {e.summary}\n  {e.excerpt[:500]}")
        return ToolResult(ok=True, output="\n".join(lines), data={"event_ids": result.event_ids, "total": result.total_matches})
