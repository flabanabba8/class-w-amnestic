from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from mnestic.config import StateLimits
from mnestic.models.patch import StatePatch
from mnestic.models.state import ExecutionState, RunStatus
from mnestic.state.apply import PatchRejected, StaleStateError, apply_patch, state_size_bytes


def patch(v: int, *ops: dict) -> StatePatch:
    return StatePatch(expected_state_version=v, ops=list(ops))


def test_apply_does_not_mutate_input(base_state: ExecutionState):
    before = copy.deepcopy(base_state.model_dump())
    r = apply_patch(base_state, patch(0, {"op": "set_phase", "phase": "work"}))
    assert base_state.model_dump() == before
    assert r.state.current_phase == "work" and r.state.state_version == 1


def test_stale_version_rejected(base_state: ExecutionState):
    with pytest.raises(StaleStateError):
        apply_patch(base_state, patch(3, {"op": "set_phase", "phase": "x"}))


def test_whole_patch_is_atomic(base_state: ExecutionState):
    with pytest.raises(PatchRejected) as exc:
        apply_patch(base_state, patch(0, {"op": "set_phase", "phase": "ok"}, {"op": "remove_fact", "fact_id": "nope", "reason": "r"}))
    assert exc.value.op_index == 1 and exc.value.code == "not_found"
    assert base_state.current_phase == "start"


def test_evidence_checker_rejects_unknown_ids(base_state: ExecutionState):
    with pytest.raises(PatchRejected, match="unknown_evidence"):
        apply_patch(base_state, patch(0, {"op": "add_fact", "statement": "s", "evidence_event_ids": ["evt_missing"]}),
                    evidence_checker=lambda ids: set(ids))


def test_add_fact_requires_evidence(base_state: ExecutionState):
    with pytest.raises(PatchRejected, match="missing_evidence"):
        apply_patch(base_state, patch(0, {"op": "add_fact", "statement": "s", "evidence_event_ids": []}))


def test_hypothesis_lifecycle(base_state: ExecutionState):
    r = apply_patch(base_state, patch(0, {"op": "add_hypothesis", "id": "h1", "statement": "port is 8000", "confidence": 0.4}))
    s = r.state
    # cannot add a fact with the same statement: must promote with evidence
    with pytest.raises(PatchRejected, match="hypothesis_promotion_required"):
        apply_patch(s, patch(1, {"op": "add_fact", "statement": "Port is 8000", "evidence_event_ids": ["e1"]}))
    with pytest.raises(ValidationError):  # promote without evidence is not even representable
        patch(1, {"op": "promote_hypothesis", "hypothesis_id": "h1", "evidence_event_ids": []})
    r2 = apply_patch(s, patch(1, {"op": "promote_hypothesis", "hypothesis_id": "h1", "evidence_event_ids": ["e1"]}))
    assert [f.id for f in r2.state.verified_facts] == ["h1"] and not r2.state.active_hypotheses
    r3 = apply_patch(base_state, patch(0, {"op": "add_hypothesis", "id": "h2", "statement": "x"},
                                       {"op": "reject_hypothesis", "hypothesis_id": "h2", "reason": "disproved", "evidence_event_ids": ["e2"]}))
    assert r3.state.rejected_hypotheses[0].reason == "disproved" and not r3.state.active_hypotheses


def test_supersede_and_remove_fact_archive_old_values(base_state: ExecutionState):
    s = apply_patch(base_state, patch(0, {"op": "add_fact", "id": "f", "statement": "port 8000", "evidence_event_ids": ["e1"]})).state
    r = apply_patch(s, patch(1, {"op": "supersede_fact", "fact_id": "f", "statement": "port 9000", "evidence_event_ids": ["e2"]}))
    assert r.state.find_fact("f").statement == "port 9000"
    assert r.archived[0].kind == "fact" and r.archived[0].item["statement"] == "port 8000"
    r2 = apply_patch(r.state, patch(2, {"op": "remove_fact", "fact_id": "f", "reason": "obsolete"}))
    assert not r2.state.verified_facts and r2.archived[0].reason == "obsolete"


def test_status_rules(base_state: ExecutionState):
    with pytest.raises(PatchRejected, match="forbidden_status"):
        apply_patch(base_state, patch(0, {"op": "set_status", "status": "completed"}))
    r = apply_patch(base_state, patch(0, {"op": "set_status", "status": "blocked"}))
    assert r.state.status == RunStatus.BLOCKED
    done = base_state.model_copy(update={"status": RunStatus.COMPLETED})
    with pytest.raises(PatchRejected, match="terminal"):
        apply_patch(done, patch(0, {"op": "set_phase", "phase": "x"}))


def test_duplicate_id_rejected(base_state: ExecutionState):
    with pytest.raises(PatchRejected, match="duplicate_id"):
        apply_patch(base_state, patch(0, {"op": "add_blocker", "id": "b", "description": "x"}, {"op": "add_question", "id": "b", "question": "q"}))


def test_plan_replacement_archives_old_plan(base_state: ExecutionState):
    s = apply_patch(base_state, patch(0, {"op": "set_plan", "steps": ["a", "b"]})).state
    assert [p.id for p in s.current_plan] == ["plan_1", "plan_2"]
    r = apply_patch(s, patch(1, {"op": "update_plan_step", "step_id": "plan_1", "status": "done"}, {"op": "set_plan", "steps": ["c"]}))
    assert r.archived[0].kind == "plan" and len(r.state.current_plan) == 1


def test_environment_entities_questions_blockers(base_state: ExecutionState):
    r = apply_patch(base_state, patch(0,
        {"op": "set_environment", "key": "os", "value": "linux"}, {"op": "set_entity", "name": "db", "description": "sqlite"},
        {"op": "add_question", "id": "q1", "question": "why?"}, {"op": "resolve_question", "question_id": "q1", "answer": "because"},
        {"op": "add_blocker", "id": "b1", "description": "stuck"}, {"op": "remove_blocker", "blocker_id": "b1", "resolution": "unstuck"},
        {"op": "add_pending_action", "id": "pa", "description": "later"}, {"op": "complete_pending_action", "action_id": "pa"},
        {"op": "add_constraint", "constraint": "no network"}, {"op": "set_observation_summary", "summary": "saw stuff"},
        {"op": "set_metadata", "key": "k", "value": 1}, {"op": "set_objective", "statement": "new objective"},
    ))
    s = r.state
    assert s.environment.properties["os"] == "linux" and s.important_entities["db"] == "sqlite"
    assert not s.unresolved_questions and not s.blockers and not s.pending_actions
    assert s.constraints == ["no network"] and s.last_observation_summary == "saw stuff" and s.metadata["k"] == 1
    assert s.objective.statement == "new objective"
    assert {a.kind for a in r.archived} == {"question", "blocker", "pending_action"}


def test_count_limit_rejected_with_guidance(base_state: ExecutionState):
    limits = StateLimits(max_facts=2)
    ops = [{"op": "add_fact", "id": f"f{i}", "statement": f"fact {i}", "evidence_event_ids": ["e"]} for i in range(3)]
    with pytest.raises(PatchRejected, match="archive_facts") as exc:
        apply_patch(base_state, patch(0, *ops), limits=limits)
    assert exc.value.code == "limit_exceeded"


def test_state_too_large_rejected(base_state: ExecutionState):
    limits = StateLimits(max_state_bytes=1500)
    ops = [{"op": "add_fact", "id": f"f{i}", "statement": "x" * 500, "evidence_event_ids": ["e"]} for i in range(3)]
    with pytest.raises(PatchRejected, match="state_too_large"):
        apply_patch(base_state, patch(0, *ops), limits=limits)


def test_rejected_hypotheses_are_compacted_into_archive(base_state: ExecutionState):
    limits = StateLimits(max_rejected_hypotheses=2)
    ops = []
    for i in range(4):
        ops += [{"op": "add_hypothesis", "id": f"h{i}", "statement": f"h {i}"},
                {"op": "reject_hypothesis", "hypothesis_id": f"h{i}", "reason": "no"}]
    r = apply_patch(base_state, patch(0, *ops), limits=limits)
    assert len(r.state.rejected_hypotheses) == 2
    assert len(r.compactions) == 2 and all(c.kind == "rejected_hypothesis" for c in r.compactions)


def test_archive_facts_spill(base_state: ExecutionState):
    s = apply_patch(base_state, patch(0, {"op": "add_fact", "id": "f1", "statement": "old detail", "evidence_event_ids": ["e"]})).state
    r = apply_patch(s, patch(1, {"op": "archive_facts", "fact_ids": ["f1"], "reason": "no longer needed in working memory"}))
    assert not r.state.verified_facts and r.archived[0].item["statement"] == "old detail"


def test_state_size_bytes_tracks_content(base_state: ExecutionState):
    small = state_size_bytes(base_state)
    big = apply_patch(base_state, patch(0, {"op": "add_fact", "statement": "y" * 1000, "evidence_event_ids": ["e"]})).state
    assert state_size_bytes(big) > small + 900


def test_every_removal_op_archives_what_it_removes(base_state: ExecutionState):
    """Found by an agent run (Codex Luna, Q3): remove_constraint dropped data without archiving it."""
    s = apply_patch(base_state, patch(0, {"op": "add_constraint", "constraint": "no network"},
                                      {"op": "set_environment", "key": "os", "value": "linux"},
                                      {"op": "set_entity", "name": "db", "description": "sqlite"})).state
    r = apply_patch(s, patch(1, {"op": "remove_constraint", "constraint": "no network"},
                             {"op": "clear_environment", "key": "os"}, {"op": "remove_entity", "name": "db"}))
    assert {a.kind for a in r.archived} == {"constraint", "environment", "entity"}
    assert next(a for a in r.archived if a.kind == "constraint").item == {"constraint": "no network"}


def test_promote_and_reject_are_idempotent_and_not_found_hints(base_state: ExecutionState):
    """Gemma re-promoted an already-promoted hypothesis three steps running (bundled with its completion)."""
    s = apply_patch(base_state, patch(0, {"op": "add_hypothesis", "id": "h1", "statement": "x"},
                                      {"op": "promote_hypothesis", "hypothesis_id": "h1", "evidence_event_ids": ["e"]})).state
    r = apply_patch(s, patch(1, {"op": "promote_hypothesis", "hypothesis_id": "h1", "evidence_event_ids": ["e"]}, {"op": "set_phase", "phase": "done"}))
    assert r.state.current_phase == "done" and "already promoted" in r.changes[0]
    with pytest.raises(PatchRejected, match="that id is a verified fact"):
        apply_patch(r.state, patch(2, {"op": "update_hypothesis", "hypothesis_id": "h1", "confidence": 0.1}))
    s2 = apply_patch(base_state, patch(0, {"op": "add_hypothesis", "id": "h2", "statement": "y"},
                                       {"op": "reject_hypothesis", "hypothesis_id": "h2", "reason": "no"})).state
    r2 = apply_patch(s2, patch(1, {"op": "reject_hypothesis", "hypothesis_id": "h2", "reason": "again"}))
    assert "already rejected" in r2.changes[0] and len(r2.state.rejected_hypotheses) == 1
    with pytest.raises(PatchRejected, match="that id is a rejected hypothesis"):
        apply_patch(r2.state, patch(2, {"op": "promote_hypothesis", "hypothesis_id": "h2", "evidence_event_ids": ["e"]}))


def test_domain_path_ops_and_schema(base_state: ExecutionState):
    schema = {"type": "object", "properties": {"shelves": {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}}}}}
    s = apply_patch(base_state, patch(0, {"op": "set_path", "path": "shelves", "value": {"S01": {"bolt": 2}}}), domain_schema=schema).state
    r = apply_patch(s, patch(1, {"op": "adjust_path", "path": "shelves.S01.bolt", "delta": 3}, {"op": "adjust_path", "path": "shelves.S02.gear", "delta": 1}), domain_schema=schema)
    assert r.state.domain["shelves"] == {"S01": {"bolt": 5}, "S02": {"gear": 1}}
    r2 = apply_patch(r.state, patch(2, {"op": "adjust_path", "path": "shelves.S01.bolt", "delta": -5}), domain_schema=schema)
    assert r2.state.domain["shelves"]["S01"] == {}  # dropped at zero
    with pytest.raises(PatchRejected, match="domain_schema"):
        apply_patch(r2.state, patch(3, {"op": "set_path", "path": "shelves.S02.gear", "value": "lots"}), domain_schema=schema)
    with pytest.raises(PatchRejected, match="not numeric"):
        apply_patch(r2.state, patch(3, {"op": "set_path", "path": "note", "value": "x"}, {"op": "adjust_path", "path": "note", "delta": 1}))
    r3 = apply_patch(r2.state, patch(3, {"op": "delete_path", "path": "shelves.S02"}), domain_schema=schema)
    assert "S02" not in r3.state.domain["shelves"] and r3.archived[0].kind == "domain"
