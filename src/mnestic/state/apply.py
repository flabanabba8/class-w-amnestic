"""Deterministic, validated application of a StatePatch to an ExecutionState.

Σ_{t+1} = Σ_t ⊕ ΔΣ_t   — never mutates the input state; returns a new validated state.

Rejection policy: any failing op rejects the *whole* patch (atomic). Errors carry the op
index so the runtime can hand the model actionable feedback as the next observation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, assert_never

from pydantic import ValidationError

from mnestic.config import StateLimits
from mnestic.models.common import utcnow
from mnestic.models.patch import (
    AddArtifact,
    AddBlocker,
    AddConstraint,
    AddFact,
    AddHypothesis,
    AddPendingAction,
    AddQuestion,
    ArchiveFacts,
    ClearEnvironment,
    CompletePendingAction,
    PromoteHypothesis,
    RejectHypothesis,
    RemoveArtifact,
    RemoveBlocker,
    RemoveConstraint,
    RemoveEntity,
    RemoveFact,
    ResolveQuestion,
    SetEntity,
    SetEnvironment,
    SetMetadata,
    SetObjective,
    SetObservationSummary,
    SetPhase,
    SetPlan,
    SetStatus,
    StatePatch,
    SupersedeFact,
    UpdateHypothesis,
    UpdatePlanStep,
)
from mnestic.models.state import (
    ALLOWED_STATUS_TRANSITIONS,
    MODEL_SETTABLE_STATUSES,
    ArtifactReference,
    Blocker,
    ExecutionState,
    Hypothesis,
    OpenQuestion,
    PendingAction,
    PlanStep,
    RejectedHypothesis,
    RunStatus,
    VerifiedFact,
)

EvidenceChecker = Callable[[Iterable[str]], set[str]]
"""Given candidate evidence event ids, return the subset that does NOT exist in the archive."""


class PatchRejected(Exception):
    """The patch is invalid against the current state. Nothing was changed."""

    def __init__(self, reason: str, *, code: str = "invalid", op_index: int | None = None):
        self.reason = reason
        self.code = code
        self.op_index = op_index
        prefix = f"op[{op_index}] " if op_index is not None else ""
        super().__init__(f"{prefix}{code}: {reason}")


class StaleStateError(PatchRejected):
    """The patch targets a state version that is no longer current."""

    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"patch expected state_version {expected} but current version is {actual}",
            code="stale_version",
        )
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class ArchivedItem:
    """Something removed from working memory that must be preserved in the archive."""

    kind: str  # fact | hypothesis | artifact | plan | pending_action | blocker | question | constraint | entity
    item: dict[str, Any]
    reason: str
    automatic: bool = False  # True when produced by limit-driven compaction


@dataclass
class PatchApplication:
    state: ExecutionState
    archived: list[ArchivedItem] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    @property
    def compactions(self) -> list[ArchivedItem]:
        return [a for a in self.archived if a.automatic]


def state_size_bytes(state: ExecutionState) -> int:
    return len(json.dumps(state.model_view(), separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def check_status_transition(current: RunStatus, new: RunStatus, *, by_model: bool) -> None:
    if by_model and new not in MODEL_SETTABLE_STATUSES:
        raise PatchRejected(
            f"status {new.value!r} cannot be set by a patch; only {sorted(s.value for s in MODEL_SETTABLE_STATUSES)} "
            "are model-settable. To finish the run, remove this set_status op and return the decision with "
            '`completion: {"outcome": "success", "summary": "...", "final_answer": "...", "artifact_ids": [...]}` '
            "and no action.",
            code="forbidden_status",
        )
    if new == current:
        return
    if new not in ALLOWED_STATUS_TRANSITIONS[current]:
        raise PatchRejected(f"impossible status transition {current.value} -> {new.value}", code="bad_transition")


def apply_patch(
    state: ExecutionState,
    patch: StatePatch,
    *,
    limits: StateLimits | None = None,
    evidence_checker: EvidenceChecker | None = None,
    now: datetime | None = None,
) -> PatchApplication:
    """Apply ``patch`` to ``state`` and return a new validated state (version + 1).

    Raises ``StaleStateError`` if the version does not match and ``PatchRejected`` for any
    other problem. The input ``state`` is never mutated.
    """
    limits = limits or StateLimits()
    now = now or utcnow()
    if patch.expected_state_version != state.state_version:
        raise StaleStateError(patch.expected_state_version, state.state_version)
    if state.status in ALLOWED_STATUS_TRANSITIONS and not ALLOWED_STATUS_TRANSITIONS[state.status] and patch.ops:
        raise PatchRejected(f"run is in terminal status {state.status.value}; no further patches accepted", code="terminal")

    _check_evidence(patch, evidence_checker)

    new = state.model_copy(deep=True)
    result = PatchApplication(state=new)
    for index, op in enumerate(patch.ops):
        try:
            _apply_op(new, op, result, now, limits)
        except PatchRejected as exc:
            raise PatchRejected(exc.reason, code=exc.code, op_index=index) from None
        except (ValueError, ValidationError) as exc:
            raise PatchRejected(str(exc), code="invalid", op_index=index) from None

    _compact(new, result, limits, now)
    _enforce_count_limits(new, limits)

    new.state_version = state.state_version + 1
    new.updated_at = now
    try:
        validated = ExecutionState.model_validate(new.model_dump())
    except ValidationError as exc:
        raise PatchRejected(f"resulting state failed validation: {exc}", code="invalid_state") from None

    size = state_size_bytes(validated)
    if size > limits.max_state_bytes:
        raise PatchRejected(
            f"resulting state would be {size} bytes, over the {limits.max_state_bytes} byte limit. "
            "Consolidate or archive facts (archive_facts), remove resolved plan steps/questions, "
            "and reference large content via artifacts instead of inlining it.",
            code="state_too_large",
        )
    result.state = validated
    return result


# --- helpers -----------------------------------------------------------------------------


def _check_evidence(patch: StatePatch, checker: EvidenceChecker | None) -> None:
    needed: dict[int, list[str]] = {}
    for index, op in enumerate(patch.ops):
        ids: list[str] = []
        if isinstance(op, (AddFact, SupersedeFact, PromoteHypothesis)):
            ids = op.evidence_event_ids
            if not ids:
                raise PatchRejected(
                    f"{op.op} requires at least one evidence_event_id (cite the observation/event that supports it)",
                    code="missing_evidence",
                    op_index=index,
                )
        elif isinstance(op, (AddHypothesis, UpdateHypothesis, RejectHypothesis)):
            ids = op.evidence_event_ids or []
        elif isinstance(op, AddArtifact) and op.originating_event_id:
            ids = [op.originating_event_id]
        if ids:
            needed[index] = ids
    if checker is None or not needed:
        return
    all_ids = {i for ids in needed.values() for i in ids}
    missing = checker(all_ids)
    if missing:
        for index, ids in needed.items():
            bad = sorted(set(ids) & missing)
            if bad:
                raise PatchRejected(
                    f"evidence event ids do not exist in the archive: {bad}",
                    code="unknown_evidence",
                    op_index=index,
                )


def _require(cond: bool, message: str, code: str = "not_found") -> None:
    if not cond:
        raise PatchRejected(message, code=code)


def _find[T](items: list[T], item_id: str, label: str) -> T:
    found = next((i for i in items if getattr(i, "id", None) == item_id), None)
    if found is None:
        raise PatchRejected(f"{label} {item_id!r} not found", code="not_found")
    return found


def _apply_op(
    s: ExecutionState,
    op: Any,
    result: PatchApplication,
    now: datetime,
    limits: StateLimits,
) -> None:
    if isinstance(op, SetPhase):
        s.current_phase = op.phase
        result.changes.append(f"phase -> {op.phase}")
    elif isinstance(op, SetStatus):
        check_status_transition(s.status, op.status, by_model=True)
        s.status = op.status
        result.changes.append(f"status -> {op.status.value}")
    elif isinstance(op, SetObjective):
        if op.statement is not None:
            s.objective.statement = op.statement
        if op.success_criteria is not None:
            s.objective.success_criteria = list(op.success_criteria)
        result.changes.append("objective updated")
    elif isinstance(op, AddFact):
        fact = VerifiedFact(
            statement=op.statement,
            confidence=op.confidence,
            evidence_event_ids=list(op.evidence_event_ids),
            created_at=now,
            updated_at=now,
            **({"id": op.id} if op.id else {}),
        )
        _require(_unique_id(s, fact.id), f"id {fact.id!r} already exists", "duplicate_id")
        for h in s.active_hypotheses:
            if _norm(h.statement) == _norm(op.statement):
                raise PatchRejected(
                    f"statement matches active hypothesis {h.id!r}; use promote_hypothesis with evidence instead",
                    code="hypothesis_promotion_required",
                )
        s.verified_facts.append(fact)
        result.changes.append(f"fact added {fact.id}")
    elif isinstance(op, SupersedeFact):
        old = _find(s.verified_facts, op.fact_id, "fact")
        result.archived.append(ArchivedItem("fact", old.model_dump(mode="json"), "superseded by new statement"))
        old.statement = op.statement
        old.confidence = op.confidence
        old.evidence_event_ids = list(op.evidence_event_ids)
        old.updated_at = now
        result.changes.append(f"fact superseded {old.id}")
    elif isinstance(op, RemoveFact):
        old = _find(s.verified_facts, op.fact_id, "fact")
        s.verified_facts.remove(old)
        result.archived.append(ArchivedItem("fact", old.model_dump(mode="json"), op.reason))
        result.changes.append(f"fact removed {old.id}")
    elif isinstance(op, ArchiveFacts):
        for fid in op.fact_ids:
            old = _find(s.verified_facts, fid, "fact")
            s.verified_facts.remove(old)
            result.archived.append(ArchivedItem("fact", old.model_dump(mode="json"), op.reason))
        result.changes.append(f"{len(op.fact_ids)} facts archived")
    elif isinstance(op, AddHypothesis):
        hyp = Hypothesis(
            statement=op.statement,
            confidence=op.confidence,
            evidence_event_ids=list(op.evidence_event_ids),
            created_at=now,
            updated_at=now,
            **({"id": op.id} if op.id else {}),
        )
        _require(_unique_id(s, hyp.id), f"id {hyp.id!r} already exists", "duplicate_id")
        s.active_hypotheses.append(hyp)
        result.changes.append(f"hypothesis added {hyp.id}")
    elif isinstance(op, UpdateHypothesis):
        hyp = _find(s.active_hypotheses, op.hypothesis_id, "hypothesis")
        if op.statement is not None:
            hyp.statement = op.statement
        if op.confidence is not None:
            hyp.confidence = op.confidence
        if op.evidence_event_ids is not None:
            hyp.evidence_event_ids = list(op.evidence_event_ids)
        hyp.updated_at = now
        result.changes.append(f"hypothesis updated {hyp.id}")
    elif isinstance(op, RejectHypothesis):
        hyp = _find(s.active_hypotheses, op.hypothesis_id, "hypothesis")
        s.active_hypotheses.remove(hyp)
        s.rejected_hypotheses.append(
            RejectedHypothesis(
                id=hyp.id,
                statement=hyp.statement,
                reason=op.reason,
                evidence_event_ids=list(op.evidence_event_ids) or list(hyp.evidence_event_ids),
                rejected_at=now,
            )
        )
        result.changes.append(f"hypothesis rejected {hyp.id}")
    elif isinstance(op, PromoteHypothesis):
        hyp = _find(s.active_hypotheses, op.hypothesis_id, "hypothesis")
        s.active_hypotheses.remove(hyp)
        s.verified_facts.append(
            VerifiedFact(
                id=hyp.id,
                statement=hyp.statement,
                confidence=op.confidence,
                evidence_event_ids=list(op.evidence_event_ids),
                created_at=now,
                updated_at=now,
            )
        )
        result.changes.append(f"hypothesis promoted to fact {hyp.id}")
    elif isinstance(op, AddArtifact):
        art = ArtifactReference(
            kind=op.kind,
            locator=op.locator,
            description=op.description,
            originating_event_id=op.originating_event_id,
            created_at=now,
            **({"id": op.id} if op.id else {}),
        )
        _require(_unique_id(s, art.id), f"id {art.id!r} already exists", "duplicate_id")
        s.artifacts.append(art)
        result.changes.append(f"artifact added {art.id}")
    elif isinstance(op, RemoveArtifact):
        art = _find(s.artifacts, op.artifact_id, "artifact")
        s.artifacts.remove(art)
        result.archived.append(ArchivedItem("artifact", art.model_dump(mode="json"), op.reason))
        result.changes.append(f"artifact removed {art.id}")
    elif isinstance(op, SetPlan):
        if s.current_plan:
            result.archived.append(
                ArchivedItem("plan", {"steps": [p.model_dump(mode="json") for p in s.current_plan]}, "plan replaced")
            )
        s.current_plan = [PlanStep(id=f"plan_{i + 1}", description=d) for i, d in enumerate(op.steps)]
        result.changes.append(f"plan set ({len(op.steps)} steps)")
    elif isinstance(op, UpdatePlanStep):
        step = _find(s.current_plan, op.step_id, "plan step")
        step.status = op.status
        if op.description is not None:
            step.description = op.description
        result.changes.append(f"plan step {step.id} -> {op.status.value}")
    elif isinstance(op, AddPendingAction):
        pa = PendingAction(description=op.description, tool_name=op.tool_name, created_at=now, **({"id": op.id} if op.id else {}))
        _require(_unique_id(s, pa.id), f"id {pa.id!r} already exists", "duplicate_id")
        s.pending_actions.append(pa)
        result.changes.append(f"pending action added {pa.id}")
    elif isinstance(op, CompletePendingAction):
        pa = _find(s.pending_actions, op.action_id, "pending action")
        s.pending_actions.remove(pa)
        result.archived.append(ArchivedItem("pending_action", pa.model_dump(mode="json"), "completed"))
        result.changes.append(f"pending action completed {pa.id}")
    elif isinstance(op, AddBlocker):
        b = Blocker(description=op.description, created_at=now, **({"id": op.id} if op.id else {}))
        _require(_unique_id(s, b.id), f"id {b.id!r} already exists", "duplicate_id")
        s.blockers.append(b)
        result.changes.append(f"blocker added {b.id}")
    elif isinstance(op, RemoveBlocker):
        b = _find(s.blockers, op.blocker_id, "blocker")
        s.blockers.remove(b)
        result.archived.append(ArchivedItem("blocker", b.model_dump(mode="json"), op.resolution))
        result.changes.append(f"blocker removed {b.id}")
    elif isinstance(op, AddQuestion):
        q = OpenQuestion(question=op.question, created_at=now, **({"id": op.id} if op.id else {}))
        _require(_unique_id(s, q.id), f"id {q.id!r} already exists", "duplicate_id")
        s.unresolved_questions.append(q)
        result.changes.append(f"question added {q.id}")
    elif isinstance(op, ResolveQuestion):
        q = _find(s.unresolved_questions, op.question_id, "question")
        s.unresolved_questions.remove(q)
        result.archived.append(ArchivedItem("question", {**q.model_dump(mode="json"), "answer": op.answer}, "resolved"))
        result.changes.append(f"question resolved {q.id}")
    elif isinstance(op, SetEnvironment):
        s.environment.properties[op.key] = op.value
        result.changes.append(f"env {op.key} set")
    elif isinstance(op, ClearEnvironment):
        _require(op.key in s.environment.properties, f"environment key {op.key!r} not set")
        result.archived.append(ArchivedItem("environment", {op.key: s.environment.properties.pop(op.key)}, "cleared"))
        result.changes.append(f"env {op.key} cleared")
    elif isinstance(op, SetEntity):
        s.important_entities[op.name] = op.description
        result.changes.append(f"entity {op.name} set")
    elif isinstance(op, RemoveEntity):
        _require(op.name in s.important_entities, f"entity {op.name!r} not found")
        result.archived.append(ArchivedItem("entity", {op.name: s.important_entities.pop(op.name)}, "removed"))
        result.changes.append(f"entity {op.name} removed")
    elif isinstance(op, AddConstraint):
        if op.constraint not in s.constraints:
            s.constraints.append(op.constraint)
        result.changes.append("constraint added")
    elif isinstance(op, RemoveConstraint):
        _require(op.constraint in s.constraints, "constraint not found")
        s.constraints.remove(op.constraint)
        result.archived.append(ArchivedItem("constraint", {"constraint": op.constraint}, "removed"))
        result.changes.append("constraint removed")
    elif isinstance(op, SetObservationSummary):
        s.last_observation_summary = op.summary or None
        result.changes.append("observation summary set")
    elif isinstance(op, SetMetadata):
        s.metadata[op.key] = op.value
        result.changes.append(f"metadata {op.key} set")
    else:
        assert_never(op)


def _compact(s: ExecutionState, result: PatchApplication, limits: StateLimits, now: datetime) -> None:
    """Automatic, archived compaction of bookkeeping lists (never facts/hypotheses/artifacts)."""
    overflow = len(s.rejected_hypotheses) - limits.max_rejected_hypotheses
    if overflow > 0:
        s.rejected_hypotheses.sort(key=lambda r: r.rejected_at)
        spilled, s.rejected_hypotheses = s.rejected_hypotheses[:overflow], s.rejected_hypotheses[overflow:]
        for r in spilled:
            result.archived.append(
                ArchivedItem("rejected_hypothesis", r.model_dump(mode="json"), "compaction: max_rejected_hypotheses", automatic=True)
            )
        result.changes.append(f"compaction: {overflow} rejected hypotheses spilled to archive")
    done = [p for p in s.current_plan if p.status.value in {"done", "skipped"}]
    if len(s.current_plan) > limits.max_plan_steps and done:
        for p in done:
            s.current_plan.remove(p)
            result.archived.append(ArchivedItem("plan_step", p.model_dump(mode="json"), "compaction: finished plan step", automatic=True))
        result.changes.append(f"compaction: {len(done)} finished plan steps spilled to archive")


def _enforce_count_limits(s: ExecutionState, limits: StateLimits) -> None:
    checks = [
        ("verified_facts", len(s.verified_facts), limits.max_facts, "archive_facts / supersede_fact to consolidate"),
        ("active_hypotheses", len(s.active_hypotheses), limits.max_hypotheses, "reject_hypothesis or promote_hypothesis"),
        ("artifacts", len(s.artifacts), limits.max_artifacts, "remove_artifact for obsolete artifacts"),
        ("current_plan", len(s.current_plan), limits.max_plan_steps, "set_plan with fewer steps"),
        ("pending_actions", len(s.pending_actions), limits.max_pending_actions, "complete_pending_action"),
        ("blockers", len(s.blockers), limits.max_blockers, "remove_blocker"),
        ("unresolved_questions", len(s.unresolved_questions), limits.max_questions, "resolve_question"),
        ("important_entities", len(s.important_entities), limits.max_entities, "remove_entity"),
        ("environment", len(s.environment.properties), limits.max_environment_keys, "clear_environment"),
        ("constraints", len(s.constraints), limits.max_constraints, "remove_constraint"),
        ("metadata", len(s.metadata), limits.max_metadata_keys, "fewer metadata keys"),
    ]
    for name, count, limit, hint in checks:
        if count > limit:
            raise PatchRejected(
                f"{name} would have {count} entries, over the limit of {limit}. Use {hint}.",
                code="limit_exceeded",
            )


def _unique_id(s: ExecutionState, candidate: str) -> bool:
    taken = (
        {f.id for f in s.verified_facts}
        | {h.id for h in s.active_hypotheses}
        | {r.id for r in s.rejected_hypotheses}
        | {a.id for a in s.artifacts}
        | {p.id for p in s.current_plan}
        | {p.id for p in s.pending_actions}
        | {b.id for b in s.blockers}
        | {q.id for q in s.unresolved_questions}
    )
    return candidate not in taken


def _norm(text: str) -> str:
    return " ".join(text.lower().split())
