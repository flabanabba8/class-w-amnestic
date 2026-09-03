"""§13 human input, §21 state-size control, §22 semantic memory."""

from __future__ import annotations

import pytest

from skillstate.config import StateLimits
from skillstate.memory.semantic import SemanticMemoryStore
from skillstate.models.archive import EventType, MemoryQuery
from skillstate.models.state import RunStatus
from tests.integration.helpers import complete, decision, obs_info


async def test_human_input_round_trip(make_runtime, store, simple_skill):
    contexts = []

    def script(ctx):
        contexts.append(ctx)
        kind, ev, text = obs_info(ctx)
        if kind == "task_input":
            return decision(ctx, action={"kind": "human_input", "question": "Which environment?", "options": ["staging", "prod"]})
        assert kind == "human_input" and text == "staging"
        return complete(ctx, [{"op": "add_fact", "id": "env", "statement": "target environment is staging", "evidence_event_ids": [ev]}], answer=text)

    rt = make_runtime(script)
    out = await rt.start(simple_skill, "deploy")
    assert out.status == RunStatus.WAITING_FOR_HUMAN and out.completion["question"] == "Which environment?"
    assert store.get_state(out.run_id).status == RunStatus.WAITING_FOR_HUMAN
    assert (await rt.resume(out.run_id)).status == RunStatus.WAITING_FOR_HUMAN  # still waiting without input
    out2 = await rt.resume(out.run_id, human_input="staging")
    assert out2.status == RunStatus.COMPLETED and out2.completion["final_answer"] == "staging"
    types = [e.event_type.value for e in store.list_events(out.run_id, limit=500)]
    assert "human.request" in types and "human.response" in types
    assert contexts[1].sections["latest_observation"].count("staging") == 1 and "Which environment" not in contexts[1].prompt


async def test_state_limits_reject_then_spill(make_runtime, store, simple_skill, config):
    cfg = config.model_copy(update={"state_limits": StateLimits(max_facts=3, max_rejected_hypotheses=2)})
    contexts = []

    def script(ctx):
        contexts.append(ctx)
        kind, ev, text = obs_info(ctx)
        if kind == "task_input":
            ops = [{"op": "add_fact", "id": f"f{i}", "statement": f"fact number {i}", "evidence_event_ids": [ev]} for i in range(5)]
            return decision(ctx, ops, action={"kind": "continue"})
        if kind == "runtime":
            assert "limit_exceeded" in text and "archive_facts" in text
            ops = [{"op": "add_fact", "id": f"f{i}", "statement": f"fact number {i}", "evidence_event_ids": [ev]} for i in range(3)]
            ops += [{"op": "archive_facts", "fact_ids": ["f0"], "reason": "spill"}]
            for i in range(4):
                ops += [{"op": "add_hypothesis", "id": f"h{i}", "statement": f"maybe {i}"}, {"op": "reject_hypothesis", "hypothesis_id": f"h{i}", "reason": "nope"}]
            return decision(ctx, ops, action={"kind": "continue"})
        return complete(ctx)

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "go")
    assert out.status == RunStatus.COMPLETED
    state = store.get_state(out.run_id)
    assert {f.id for f in state.verified_facts} == {"f1", "f2"} and len(state.rejected_hypotheses) == 2
    rejected = [p for p in store.list_patches(out.run_id) if p["status"] == "rejected"]
    assert len(rejected) == 1 and rejected[0]["error_code"] == "limit_exceeded"
    compactions = store.list_events(out.run_id, limit=500, event_type=EventType.STATE_COMPACTION.value)
    assert len(compactions) == 2 and {c.payload["item"]["id"] for c in compactions} == {"h0", "h1"}
    spilled = [e for e in store.list_events(out.run_id, limit=500, event_type=EventType.PATCH_APPLIED.value) if e.payload.get("kind") == "fact"]
    assert spilled[0].payload["item"]["statement"] == "fact number 0"
    from skillstate.memory.retrieval import ArchiveRetriever

    assert ArchiveRetriever(store).retrieve(out.run_id, MemoryQuery(query_type="search", text="fact number 0")).total_matches >= 1


async def test_state_too_large_feedback(make_runtime, store, simple_skill, config):
    cfg = config.model_copy(update={"state_limits": StateLimits(max_state_bytes=2500)})

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        if kind == "task_input":
            return decision(ctx, [{"op": "add_fact", "statement": "x" * 1900, "evidence_event_ids": [ev]}], action={"kind": "continue"})
        assert "state_too_large" in text
        return complete(ctx)

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "go")
    assert out.status == RunStatus.COMPLETED and not store.get_state(out.run_id).verified_facts


async def test_continue_loop_guard(make_runtime, store, simple_skill, config):
    cfg = config.model_copy(update={"max_consecutive_continues": 2, "max_decision_failures": 1})
    out = await make_runtime(lambda ctx: decision(ctx, action={"kind": "continue"}), cfg=cfg).start(simple_skill, "spin")
    assert out.status == RunStatus.FAILED and "consecutive" in out.summary


def test_semantic_memory_promotion_is_explicit_and_audited(store, simple_skill, base_state):
    from skillstate.models.archive import RunMetadata
    from skillstate.models.common import utcnow

    with store.transaction():
        store.upsert_skill(simple_skill)
        store.create_run(RunMetadata(run_id="run_test", skill_id=simple_skill.skill_id, skill_version=simple_skill.version, skill_content_hash=simple_skill.content_hash,
                                     status="running", created_at=utcnow(), updated_at=utcnow(), workspace_root="/tmp", model_name="mock", task_input="t"))
        store.init_state(base_state)
        ev = store.append_event("run_test", 0, EventType.OBSERVATION, "os", {"content": "uname: Linux 7.0"})
    sem = SemanticMemoryStore(store)
    with pytest.raises(ValueError, match="source events not found"):
        sem.promote(key="machine.kernel", content="Linux 7.0", source_run_id="run_test", source_event_ids=["evt_fake"])
    mem = sem.promote(key="machine.kernel", content="Linux 7.0", category="environment", source_run_id="run_test", source_event_ids=[ev.event_id], promoted_by="test")
    assert mem.source_event_ids == [ev.event_id]
    promoted = store.list_events("run_test", limit=10, event_type=EventType.SEMANTIC_PROMOTED.value)
    assert len(promoted) == 1 and promoted[0].payload["key"] == "machine.kernel"
    assert sem.search("kernel")[0].content == "Linux 7.0"
    assert sem.forget("machine.kernel") and not sem.list_all()
