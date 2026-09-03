"""§13 human input, §21 state-size control, §22 semantic memory."""

from __future__ import annotations

import pytest

from mnestic.config import StateLimits
from mnestic.memory.semantic import SemanticMemoryStore
from mnestic.models.archive import EventType, MemoryQuery
from mnestic.models.state import RunStatus
from tests.integration.helpers import complete, decision, obs_info, tool


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
    from mnestic.memory.retrieval import ArchiveRetriever

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
    from mnestic.models.archive import RunMetadata
    from mnestic.models.common import utcnow

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


async def test_repeated_identical_tool_action_guard(make_runtime, store, simple_skill, config):
    """A model that keeps re-running the same tool without updating state gets bounded feedback, then fails if it persists."""
    cfg = config.model_copy(update={"max_repeated_actions": 3, "max_decision_failures": 2})
    seen = []

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        seen.append(kind)
        if kind == "runtime":
            assert "3 times within the last" in text and "memory_query" in text
        return decision(ctx, action={"kind": "tool", "tool_name": "list_directory", "arguments": {"path": "."}})

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "loop forever")
    assert out.status == RunStatus.FAILED and out.reason == "too_many_failures" and "identical tool action" in out.summary
    assert seen.count("runtime") == 1  # feedback once; the second loop detection reaches max_decision_failures
    execs = store.list_tool_executions(out.run_id)
    # The first call answers the task input (different observation), so 3 identical (action, observation) pairs need
    # 3 executed listings before the guard fires; then 3 more before it fires again -> failure.
    assert len(execs) == 6
    assert all(e.status == "succeeded" for e in execs)


async def test_action_cycle_is_caught_by_window(make_runtime, store, simple_skill, config):
    """A -> B -> C -> A -> B -> C … (Luna's failure mode) trips the guard even though no two consecutive actions match."""
    cfg = config.model_copy(update={"max_repeated_actions": 3, "action_window": 12, "max_decision_failures": 1})
    paths = iter([".", "src", "nope", ".", "src", "nope", ".", "src", "nope", "."])

    def script(ctx):
        return decision(ctx, action={"kind": "tool", "tool_name": "list_directory", "arguments": {"path": next(paths)}})

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "cycle")
    assert out.status == RunStatus.FAILED and "3 times within the last 12" in out.summary
    assert len(store.list_tool_executions(out.run_id)) == 7  # '.' after the identical 'nope' failure repeats at actions 4, 7, 10 -> intercepted at 10


async def test_varied_tool_actions_do_not_trigger_guard(make_runtime, store, simple_skill, config):
    cfg = config.model_copy(update={"max_repeated_actions": 3, "action_window": 3})
    paths = iter([".", "src", ".", "src"])

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        try:
            return decision(ctx, action={"kind": "tool", "tool_name": "list_directory", "arguments": {"path": next(paths)}})
        except StopIteration:
            return complete(ctx)

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "alternate")
    assert out.status == RunStatus.COMPLETED and store.get_state(out.run_id).counters.errors == 0


async def test_non_consecutive_failures_do_not_accumulate(make_runtime, store, simple_skill, config):
    """A rejected patch, then several good steps, then another rejection must not add up to too_many_failures."""
    cfg = config.model_copy(update={"max_decision_failures": 2, "action_window": 1})  # window 1: loop guard cannot fire
    n = {"i": 0}

    def script(ctx):
        n["i"] += 1
        kind, ev, text = obs_info(ctx)
        if n["i"] in (1, 5):  # forbidden op -> rejection, twice, separated by successful steps
            return decision(ctx, [{"op": "set_status", "status": "completed"}], action={"kind": "continue"})
        if kind == "runtime":
            assert "completion:" in text
        if n["i"] >= 8:
            return complete(ctx)
        return tool(ctx, "list_directory", path="." if n["i"] % 2 else "src")

    out = await make_runtime(script, cfg=cfg).start(simple_skill, "go")
    assert out.status == RunStatus.COMPLETED
    assert store.get_state(out.run_id).counters.patches_rejected == 2


async def test_identical_actions_with_new_observations_are_not_a_loop(make_runtime, store, simple_skill, config):
    """Same tool+args in response to *different* results (e.g. a stream of identical orders) must not trip the guard."""
    from typing import ClassVar

    from pydantic import BaseModel, ConfigDict

    from mnestic.tools import default_registry
    from mnestic.tools.base import Tool, ToolContext, ToolResult

    class Counter(Tool):
        name: ClassVar[str] = "counter"
        description: ClassVar[str] = "returns a new number each call"

        class Args(BaseModel):
            model_config = ConfigDict(extra="forbid")

        n = 0

        async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
            Counter.n += 1
            return ToolResult(ok=True, output=f"tick {Counter.n}")

    reg = default_registry()
    reg.register(Counter())
    cfg = config.model_copy(update={"max_repeated_actions": 2, "action_window": 12, "max_decision_failures": 1})
    skill = simple_skill.model_copy(update={"required_tools": ["counter"]})

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        if "tick 6" in text:
            return complete(ctx)
        return tool(ctx, "counter")

    out = await make_runtime(script, tools=reg, cfg=cfg).start(skill, "count ticks")
    assert out.status == RunStatus.COMPLETED and store.get_state(out.run_id).counters.errors == 0


async def test_feedback_observation_repeats_the_original_input(make_runtime, store, simple_skill, config):
    """A rejected decision must not make the model lose the input it was answering (found on the warehouse benchmark)."""
    seen: list[str] = []
    bad = [{"op": "remove_fact", "fact_id": "nope", "reason": "x"}]

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        seen.append(kind)
        if kind == "runtime":
            assert "ORDER #7: ship 1 gear" in text and "repeated here" in text, text
        if seen.count("runtime") < 2:  # fail on the task input and on the first feedback; succeed on the second feedback
            return decision(ctx, bad, action={"kind": "continue"})
        return complete(ctx)

    cfg = config.model_copy(update={"max_decision_failures": 5})
    out = await make_runtime(script, cfg=cfg).start(simple_skill, "ORDER #7: ship 1 gear — choose a shelf")
    assert out.status == RunStatus.COMPLETED, out
    assert seen == ["task_input", "runtime", "runtime"]
