from __future__ import annotations

import pytest
from pydantic import ValidationError

from mnestic.models.decision import AgentDecision
from mnestic.models.patch import StatePatch
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ExecutionState, Hypothesis, Objective, RunStatus, VerifiedFact


def _state(**kw):
    return ExecutionState(run_id="r", skill_id="s", skill_version="1", objective=Objective(statement="o"), status=RunStatus.RUNNING, **kw)


def test_duplicate_fact_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate id"):
        _state(verified_facts=[VerifiedFact(id="f1", statement="a", evidence_event_ids=["e"]),
                               VerifiedFact(id="f1", statement="b", evidence_event_ids=["e"])])


def test_id_shared_between_fact_and_hypothesis_rejected():
    with pytest.raises(ValidationError, match="both facts and hypotheses"):
        _state(verified_facts=[VerifiedFact(id="x", statement="a", evidence_event_ids=["e"])], active_hypotheses=[Hypothesis(id="x", statement="b")])


def test_hypothesis_duplicating_fact_statement_rejected():
    with pytest.raises(ValidationError, match="duplicates a verified fact"):
        _state(verified_facts=[VerifiedFact(id="f", statement="Port is 8000", evidence_event_ids=["e"])],
               active_hypotheses=[Hypothesis(id="h", statement="port is  8000")])


def test_fact_requires_evidence():
    with pytest.raises(ValidationError):
        VerifiedFact(statement="no evidence", evidence_event_ids=[])


def test_unknown_field_in_state_rejected():
    with pytest.raises(ValidationError):
        ExecutionState.model_validate({**_state().model_dump(), "messages": []})


def test_oversized_statement_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(statement="x" * 2001)


def test_model_view_drops_runtime_noise():
    view = _state(verified_facts=[VerifiedFact(id="f", statement="a", evidence_event_ids=["e"])]).model_view()
    assert "created_at" not in view and "metadata" not in view
    assert "created_at" not in view["verified_facts"][0]
    assert view["state_version"] == 0


def test_decision_requires_exactly_one_control():
    patch = {"expected_state_version": 0, "ops": []}
    with pytest.raises(ValidationError, match="exactly one"):
        AgentDecision(state_patch=patch)
    with pytest.raises(ValidationError, match="exactly one"):
        AgentDecision(state_patch=patch, action={"kind": "continue"}, completion={"outcome": "success", "summary": "s"})
    AgentDecision(state_patch=patch, action={"kind": "continue"})


def test_decision_rejects_extra_fields_and_long_rationale():
    patch = {"expected_state_version": 0, "ops": []}
    with pytest.raises(ValidationError):
        AgentDecision(state_patch=patch, action={"kind": "continue"}, thoughts="private chain of thought")
    with pytest.raises(ValidationError):
        AgentDecision(state_patch=patch, action={"kind": "continue"}, rationale_summary="x" * 601)


def test_unknown_patch_op_rejected():
    with pytest.raises(ValidationError):
        StatePatch(expected_state_version=0, ops=[{"op": "delete_everything"}])
    with pytest.raises(ValidationError):
        StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "x", "extra": 1}])
    with pytest.raises(ValidationError):
        StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": 123}])


def test_skill_specification_is_frozen_and_hash_stable():
    a = SkillSpecification(skill_id="s", name="S", version="1", description="d", instructions="i")
    b = SkillSpecification(skill_id="s", name="S", version="1", description="d", instructions="i")
    c = SkillSpecification(skill_id="s", name="S", version="1", description="d", instructions="i2")
    assert a.content_hash == b.content_hash != c.content_hash
    with pytest.raises(ValidationError):
        a.instructions = "mutated"  # type: ignore[misc]


def test_model_timeout_config(monkeypatch):
    from mnestic.config import RuntimeConfig

    assert RuntimeConfig().model_timeout_seconds == 300.0
    monkeypatch.setenv("MNESTIC_MODEL_TIMEOUT", "45")
    assert RuntimeConfig.from_env().model_timeout_seconds == 45.0


def test_pruned_decision_schema_is_much_smaller_and_still_validates():
    import json

    from mnestic.models.decision import decision_type_for

    full = len(json.dumps(AgentDecision.model_json_schema(), separators=(",", ":")))
    small_cls = decision_type_for(["set_entity", "set_observation_summary"])
    small = len(json.dumps(small_cls.model_json_schema(), separators=(",", ":")))
    assert small < full / 3, (small, full)
    d = small_cls.model_validate({"state_patch": {"expected_state_version": 0, "ops": [{"op": "set_entity", "name": "S01", "description": "empty"}]},
                                  "action": {"kind": "continue"}})
    assert isinstance(d, AgentDecision)
    with pytest.raises(ValidationError):  # ops outside the subset are rejected at the schema level
        small_cls.model_validate({"state_patch": {"expected_state_version": 0, "ops": [{"op": "add_fact", "statement": "x", "evidence_event_ids": ["e"]}]},
                                  "action": {"kind": "continue"}})
    with pytest.raises(ValueError, match="unknown patch ops"):
        decision_type_for(["nope"])
    assert "Exactly one of" not in json.dumps(AgentDecision.model_json_schema())  # class docstrings no longer in the schema
