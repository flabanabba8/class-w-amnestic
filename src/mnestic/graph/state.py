"""Graph-level state and dependency containers (separate from the semantic ExecutionState)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mnestic.agent.reasoner import Reasoner, UsageRecord
from mnestic.config import RuntimeConfig
from mnestic.context.builder import ContextBuilder, ModelContext
from mnestic.memory.base import Retriever
from mnestic.models.archive import MemoryResult
from mnestic.models.decision import AgentDecision
from mnestic.models.observation import Observation, ObservationKind
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ExecutionState, RunStatus
from mnestic.observability.logging import FieldsAdapter
from mnestic.storage.store import Store
from mnestic.tools.base import ToolRegistry, ToolResult


class RunOutcome(BaseModel):
    run_id: str
    status: RunStatus
    reason: str
    summary: str = ""
    steps: int
    state_version: int
    completion: dict[str, Any] | None = None


@dataclass
class RuntimeDeps:
    """Immutable-per-run dependencies injected into every node."""

    config: RuntimeConfig
    store: Store
    skill: SkillSpecification
    reasoner: Reasoner
    tools: ToolRegistry
    context_builder: ContextBuilder
    retriever: Retriever
    workspace_root: Path
    log: FieldsAdapter


@dataclass
class RuntimeGraphState:
    """Ephemeral controller state for one process invocation of the graph.

    Distinct from ``ExecutionState`` (semantic, durable, model-facing). Nothing here is
    ever rendered into a prompt except through ``ContextBuilder``.
    """

    run_id: str
    execution_state: ExecutionState
    step: int
    observation: Observation | None
    retrieved: list[MemoryResult] = field(default_factory=list)
    context: ModelContext | None = None
    decision: AgentDecision | None = None
    action_id: str | None = None
    tool_result: tuple[str, ToolResult, int] | None = None
    last_usage: UsageRecord | None = None
    consecutive_failures: int = 0
    consecutive_continues: int = 0
    recent_action_signatures: list[str] = field(default_factory=list)
    loop_trips: int = 0
    consecutive_timeouts: int = 0
    """Loop-guard trips this invocation; never reset, so persistent looping still fails the run."""
    steps_this_invocation: int = 0
    max_steps_this_invocation: int = 10_000
    max_observation_chars: int = 6000
    step_started: float | None = None
    model_calls_this_step: int = 0
    tool_calls_this_step: int = 0
    retrievals_this_step: int = 0
    patches_applied_this_step: int = 0

    def begin_step(self, step: int, observation: Observation, retrieved: list[MemoryResult]) -> None:
        self.step = step
        self.observation = observation
        self.retrieved = retrieved
        self.context = None
        self.decision = None
        self.action_id = None
        self.tool_result = None
        self.last_usage = None
        self.steps_this_invocation += 1
        self.model_calls_this_step = 0
        self.tool_calls_this_step = 0
        self.retrievals_this_step = 0
        self.patches_applied_this_step = 0
        if observation.kind != ObservationKind.CONTINUE:
            self.consecutive_continues = 0


