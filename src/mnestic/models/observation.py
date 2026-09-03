"""Observation — the newest input to a reasoning step."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from mnestic.models.common import Record, new_id, utcnow


class ObservationKind(StrEnum):
    TASK_INPUT = "task_input"
    TOOL_RESULT = "tool_result"
    HUMAN_INPUT = "human_input"
    RUNTIME = "runtime"
    """Runtime feedback, e.g. 'your previous patch was rejected because ...'."""
    CONTINUE = "continue"
    """The model asked to reason again with no external action; nothing new happened."""


class Observation(Record):
    id: str = Field(default_factory=lambda: new_id("obs"))
    run_id: str
    step: int = Field(ge=0, description="The step at which this observation becomes the newest input")
    kind: ObservationKind
    source: str = Field(max_length=120, description="tool name, 'human', 'runtime', 'task'")
    content: str = Field(description="Bounded excerpt presented to the model")
    truncated: bool = False
    full_length: int = 0
    event_id: str | None = Field(default=None, description="Archive event id holding the full record")
    data: dict[str, Any] | None = Field(default=None, description="Small structured payload (exit code, path, …)")
    created_at: datetime = Field(default_factory=utcnow)
