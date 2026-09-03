"""Archive (episodic memory) records and retrieval request/response models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from mnestic.models.common import Record, StrictModel, new_id, utcnow


class EventType(StrEnum):
    RUN_CREATED = "run.created"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_PAUSED = "run.paused"
    TASK_INPUT = "task.input"
    OBSERVATION = "observation.captured"
    CONTEXT_BUILT = "context.built"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_ERROR = "model.error"
    DECISION_ACCEPTED = "decision.accepted"
    DECISION_REJECTED = "decision.rejected"
    PATCH_APPLIED = "patch.applied"
    PATCH_REJECTED = "patch.rejected"
    STATE_COMMITTED = "state.committed"
    STATE_COMPACTION = "state.compaction"
    ACTION_REQUESTED = "action.requested"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"
    TOOL_INTERRUPTED = "tool.interrupted"
    MEMORY_RETRIEVAL = "memory.retrieval"
    HUMAN_REQUEST = "human.request"
    HUMAN_RESPONSE = "human.response"
    SEMANTIC_PROMOTED = "semantic.promoted"
    ERROR = "error"


class ArchiveEvent(Record):
    """Append-only record. THE ARCHIVE IS NOT DEFAULT MODEL CONTEXT."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    seq: int = Field(ge=0, description="Monotonic per-run sequence")
    step: int = Field(ge=0)
    event_type: EventType
    summary: str = Field(default="", description="One-line human/searchable summary")
    payload: dict[str, Any] = Field(default_factory=dict)
    ref_table: str | None = None
    ref_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


MemoryQueryType = Literal[
    "recent",
    "event",
    "events_by_type",
    "search",
    "state_at_version",
    "state_history",
    "observations",
    "tool_executions",
    "artifacts",
    "semantic",
]


class MemoryQuery(StrictModel):
    """Explicit request to retrieve from archival memory. Bounded by ``limit``."""

    query_type: MemoryQueryType
    text: str | None = Field(default=None, max_length=500, description="search terms")
    event_id: str | None = Field(default=None, max_length=80)
    event_type: str | None = Field(default=None, max_length=80)
    version: int | None = Field(default=None, ge=0)
    limit: int = Field(default=5, ge=1, le=20)
    reason: str = Field(default="", max_length=300)


class RetrievedEvent(StrictModel):
    event_id: str
    step: int
    event_type: str
    created_at: datetime
    summary: str
    excerpt: str = Field(description="Bounded text excerpt of the payload")


class MemoryResult(StrictModel):
    query: MemoryQuery
    events: list[RetrievedEvent] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    note: str = ""

    @property
    def event_ids(self) -> list[str]:
        return [e.event_id for e in self.events]


class RunMetadata(Record):
    run_id: str
    skill_id: str
    skill_version: str
    skill_content_hash: str
    status: str
    created_at: datetime
    updated_at: datetime
    workspace_root: str
    model_name: str
    task_input: str
    last_step: int = 0
    state_version: int = 0
    finished_at: datetime | None = None
    outcome: dict[str, Any] | None = None
