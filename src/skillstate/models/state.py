"""ExecutionState — the canonical operational (working) memory.

Everything in this module is what the model sees as "what is currently true".
It must stay small: raw evidence lives in the archive and is referenced by
event id.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from skillstate.models.common import Identifier, JsonScalar, ShortText, StrictModel, new_id, utcnow


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})

ALLOWED_STATUS_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_FOR_HUMAN,
            RunStatus.BLOCKED,
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.WAITING_FOR_HUMAN: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.BLOCKED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

MODEL_SETTABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.BLOCKED})
"""Statuses the model may request through a patch. Terminal statuses are set by the runtime."""


class Objective(StrictModel):
    statement: ShortText
    success_criteria: list[ShortText] = Field(default_factory=list, max_length=20)


class VerifiedFact(StrictModel):
    """Information supported by evidence in the archive."""

    id: Identifier = Field(default_factory=lambda: new_id("fact"))
    statement: ShortText
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_event_ids: list[str] = Field(min_length=1, max_length=20)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Hypothesis(StrictModel):
    """Something that might be true. Never silently becomes a fact."""

    id: Identifier = Field(default_factory=lambda: new_id("hyp"))
    statement: ShortText
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_event_ids: list[str] = Field(default_factory=list, max_length=20)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RejectedHypothesis(StrictModel):
    """An important possibility explicitly ruled out, with the reason."""

    id: Identifier
    statement: ShortText
    reason: ShortText
    evidence_event_ids: list[str] = Field(default_factory=list, max_length=20)
    rejected_at: datetime = Field(default_factory=utcnow)


class ArtifactReference(StrictModel):
    """A pointer to something large that lives outside working memory."""

    id: Identifier = Field(default_factory=lambda: new_id("art"))
    kind: str = Field(default="file", max_length=40)
    locator: str = Field(min_length=1, max_length=1000, description="Path, URI or identifier")
    description: ShortText
    originating_event_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


class PlanStep(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("plan"))
    description: ShortText
    status: PlanStepStatus = PlanStepStatus.PENDING


class PendingAction(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("act"))
    description: ShortText
    tool_name: str | None = Field(default=None, max_length=80)
    created_at: datetime = Field(default_factory=utcnow)


class Blocker(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("blk"))
    description: ShortText
    created_at: datetime = Field(default_factory=utcnow)


class OpenQuestion(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("q"))
    question: ShortText
    created_at: datetime = Field(default_factory=utcnow)


class Environment(StrictModel):
    """Small key/value description of the environment (cwd, OS, ports, versions…)."""

    properties: dict[str, JsonScalar] = Field(default_factory=dict)


class Counters(StrictModel):
    steps_completed: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    retrievals: int = 0
    patches_applied: int = 0
    patches_rejected: int = 0
    errors: int = 0
    consecutive_failures: int = 0


class Budgets(StrictModel):
    max_steps: int = Field(default=200, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_consecutive_failures: int = Field(default=3, ge=1)


class ExecutionState(StrictModel):
    """Canonical semantic state supplied to the model at every step.

    This is *not* the pydantic-graph runtime state (see ``graph/runtime.py``):
    that one is ephemeral controller bookkeeping. This one is what the agent
    believes and is durably versioned in SQLite.
    """

    run_id: str
    skill_id: str
    skill_version: str
    state_schema_version: str = "1"
    objective: Objective
    status: RunStatus = RunStatus.PENDING
    current_phase: str = Field(default="start", max_length=120)
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    rejected_hypotheses: list[RejectedHypothesis] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    constraints: list[ShortText] = Field(default_factory=list, max_length=50)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    current_plan: list[PlanStep] = Field(default_factory=list)
    pending_actions: list[PendingAction] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    unresolved_questions: list[OpenQuestion] = Field(default_factory=list)
    important_entities: dict[str, ShortText] = Field(default_factory=dict)
    counters: Counters = Field(default_factory=Counters)
    budgets: Budgets = Field(default_factory=Budgets)
    last_observation_summary: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)
    state_version: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check_integrity(self) -> ExecutionState:
        _check_unique_ids("verified_facts", [f.id for f in self.verified_facts])
        _check_unique_ids("active_hypotheses", [h.id for h in self.active_hypotheses])
        _check_unique_ids("rejected_hypotheses", [h.id for h in self.rejected_hypotheses])
        _check_unique_ids("artifacts", [a.id for a in self.artifacts])
        _check_unique_ids("current_plan", [p.id for p in self.current_plan])
        _check_unique_ids("pending_actions", [p.id for p in self.pending_actions])
        _check_unique_ids("blockers", [b.id for b in self.blockers])
        _check_unique_ids("unresolved_questions", [q.id for q in self.unresolved_questions])
        fact_ids = {f.id for f in self.verified_facts}
        hyp_ids = {h.id for h in self.active_hypotheses}
        if overlap := fact_ids & hyp_ids:
            raise ValueError(f"ids used for both facts and hypotheses: {sorted(overlap)}")
        fact_statements = {_norm(f.statement) for f in self.verified_facts}
        for h in self.active_hypotheses:
            if _norm(h.statement) in fact_statements:
                raise ValueError(f"hypothesis {h.id!r} duplicates a verified fact statement")
        return self

    # --- convenience -------------------------------------------------------------

    def find_fact(self, fact_id: str) -> VerifiedFact | None:
        return next((f for f in self.verified_facts if f.id == fact_id), None)

    def find_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.active_hypotheses if h.id == hypothesis_id), None)

    def model_view(self) -> dict[str, Any]:
        """The representation shown to the model: stable key order, no runtime-only noise."""
        data = self.model_dump(mode="json", exclude={"created_at", "updated_at", "metadata"})
        # Timestamps on nested items are not decision-relevant; drop to keep the state compact.
        for key in ("verified_facts", "active_hypotheses", "artifacts", "pending_actions", "blockers", "unresolved_questions"):
            for item in data.get(key, []):
                item.pop("created_at", None)
                item.pop("updated_at", None)
        for item in data.get("rejected_hypotheses", []):
            item.pop("rejected_at", None)
        return data


def _norm(statement: str) -> str:
    return " ".join(statement.lower().split())


def _check_unique_ids(field: str, ids: list[str]) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate id {i!r} in {field}")
        seen.add(i)
