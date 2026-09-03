"""StatePatch — the only way ExecutionState changes.

Each op is a strict model (unknown ops / unknown fields fail validation).
Application semantics live in ``mnestic.state.apply``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mnestic.models.common import Identifier, JsonScalar, ShortText, StrictModel
from mnestic.models.state import PlanStepStatus, RunStatus

EvidenceIds = Annotated[list[str], Field(max_length=20)]


class SetPhase(StrictModel):
    op: Literal["set_phase"] = "set_phase"
    phase: str = Field(min_length=1, max_length=120)


class SetStatus(StrictModel):
    op: Literal["set_status"] = "set_status"
    status: RunStatus
    reason: str = Field(default="", max_length=500)


class SetObjective(StrictModel):
    op: Literal["set_objective"] = "set_objective"
    statement: ShortText | None = None
    success_criteria: list[ShortText] | None = Field(default=None, max_length=20)


class AddFact(StrictModel):
    op: Literal["add_fact"] = "add_fact"
    id: Identifier | None = None
    statement: ShortText
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_event_ids: EvidenceIds = Field(description="Archive event ids that support this fact (required)")


class SupersedeFact(StrictModel):
    op: Literal["supersede_fact"] = "supersede_fact"
    fact_id: Identifier
    statement: ShortText
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_event_ids: EvidenceIds


class RemoveFact(StrictModel):
    op: Literal["remove_fact"] = "remove_fact"
    fact_id: Identifier
    reason: ShortText


class ArchiveFacts(StrictModel):
    """Explicit spill: move facts out of working state into the archive (recoverable by retrieval)."""

    op: Literal["archive_facts"] = "archive_facts"
    fact_ids: list[Identifier] = Field(min_length=1, max_length=100)
    reason: ShortText


class AddHypothesis(StrictModel):
    op: Literal["add_hypothesis"] = "add_hypothesis"
    id: Identifier | None = None
    statement: ShortText
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_event_ids: EvidenceIds = Field(default_factory=list)


class UpdateHypothesis(StrictModel):
    op: Literal["update_hypothesis"] = "update_hypothesis"
    hypothesis_id: Identifier
    statement: ShortText | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_event_ids: EvidenceIds | None = None


class RejectHypothesis(StrictModel):
    op: Literal["reject_hypothesis"] = "reject_hypothesis"
    hypothesis_id: Identifier
    reason: ShortText
    evidence_event_ids: EvidenceIds = Field(default_factory=list)


class PromoteHypothesis(StrictModel):
    """The only mechanism by which a hypothesis becomes a fact. Evidence is mandatory."""

    op: Literal["promote_hypothesis"] = "promote_hypothesis"
    hypothesis_id: Identifier
    evidence_event_ids: EvidenceIds = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AddArtifact(StrictModel):
    op: Literal["add_artifact"] = "add_artifact"
    id: Identifier | None = None
    kind: str = Field(default="file", max_length=40)
    locator: str = Field(min_length=1, max_length=1000)
    description: ShortText
    originating_event_id: str | None = None


class RemoveArtifact(StrictModel):
    op: Literal["remove_artifact"] = "remove_artifact"
    artifact_id: Identifier
    reason: ShortText


class SetPlan(StrictModel):
    """Replace the whole plan (canonical replacement, not append)."""

    op: Literal["set_plan"] = "set_plan"
    steps: list[ShortText] = Field(max_length=50)


class UpdatePlanStep(StrictModel):
    op: Literal["update_plan_step"] = "update_plan_step"
    step_id: Identifier
    status: PlanStepStatus
    description: ShortText | None = None


class AddPendingAction(StrictModel):
    op: Literal["add_pending_action"] = "add_pending_action"
    id: Identifier | None = None
    description: ShortText
    tool_name: str | None = Field(default=None, max_length=80)


class CompletePendingAction(StrictModel):
    op: Literal["complete_pending_action"] = "complete_pending_action"
    action_id: Identifier


class AddBlocker(StrictModel):
    op: Literal["add_blocker"] = "add_blocker"
    id: Identifier | None = None
    description: ShortText


class RemoveBlocker(StrictModel):
    op: Literal["remove_blocker"] = "remove_blocker"
    blocker_id: Identifier
    resolution: ShortText


class AddQuestion(StrictModel):
    op: Literal["add_question"] = "add_question"
    id: Identifier | None = None
    question: ShortText


class ResolveQuestion(StrictModel):
    op: Literal["resolve_question"] = "resolve_question"
    question_id: Identifier
    answer: ShortText


class SetEnvironment(StrictModel):
    op: Literal["set_environment"] = "set_environment"
    key: str = Field(min_length=1, max_length=80)
    value: JsonScalar


class ClearEnvironment(StrictModel):
    op: Literal["clear_environment"] = "clear_environment"
    key: str = Field(min_length=1, max_length=80)


class SetEntity(StrictModel):
    op: Literal["set_entity"] = "set_entity"
    name: str = Field(min_length=1, max_length=120)
    description: ShortText


class RemoveEntity(StrictModel):
    op: Literal["remove_entity"] = "remove_entity"
    name: str = Field(min_length=1, max_length=120)


class AddConstraint(StrictModel):
    op: Literal["add_constraint"] = "add_constraint"
    constraint: ShortText


class RemoveConstraint(StrictModel):
    op: Literal["remove_constraint"] = "remove_constraint"
    constraint: ShortText


class SetObservationSummary(StrictModel):
    """Replace (not append) the one-line summary of the newest observation."""

    op: Literal["set_observation_summary"] = "set_observation_summary"
    summary: str = Field(max_length=2000)


class SetMetadata(StrictModel):
    op: Literal["set_metadata"] = "set_metadata"
    key: str = Field(min_length=1, max_length=80)
    value: JsonScalar


PatchOp = Annotated[
    SetPhase
    | SetStatus
    | SetObjective
    | AddFact
    | SupersedeFact
    | RemoveFact
    | ArchiveFacts
    | AddHypothesis
    | UpdateHypothesis
    | RejectHypothesis
    | PromoteHypothesis
    | AddArtifact
    | RemoveArtifact
    | SetPlan
    | UpdatePlanStep
    | AddPendingAction
    | CompletePendingAction
    | AddBlocker
    | RemoveBlocker
    | AddQuestion
    | ResolveQuestion
    | SetEnvironment
    | ClearEnvironment
    | SetEntity
    | RemoveEntity
    | AddConstraint
    | RemoveConstraint
    | SetObservationSummary
    | SetMetadata,
    Field(discriminator="op"),
]


class StatePatch(StrictModel):
    """A validated, ordered list of mutations against a specific state version."""

    expected_state_version: int = Field(ge=0)
    ops: list[PatchOp] = Field(default_factory=list, max_length=50)

    @property
    def is_empty(self) -> bool:
        return not self.ops
