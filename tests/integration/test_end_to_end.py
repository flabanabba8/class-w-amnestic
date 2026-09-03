from __future__ import annotations

from skillstate.benchmarks.scripts import codebase_research_script, counter_script
from skillstate.models.state import RunStatus
from skillstate.skills.loader import SkillRegistry
from skillstate.storage.store import Store


async def test_deterministic_counter_completes(make_runtime, registry: SkillRegistry, store: Store):
    rt = make_runtime(counter_script)
    out = await rt.start(registry.get("deterministic-counter"), "count to 4")
    assert out.status == RunStatus.COMPLETED and out.completion["final_answer"] == "4"
    state = store.get_state(out.run_id)
    assert state.environment.properties["counter"] == 4 and state.status == RunStatus.COMPLETED
    assert state.counters.errors == 0 and state.counters.patches_applied == 5
    types = [e.event_type.value for e in store.list_events(out.run_id, limit=1000)]
    assert types[0] == "run.created" and types[1] == "task.input" and types[-1] == "run.completed"
    assert "context.built" in types and "patch.applied" in types and "state.committed" in types
    versions = store.list_state_versions(out.run_id)
    assert versions[0]["version"] == 0 and versions[-1]["version"] == out.state_version
    assert all(store.get_state_at_version(out.run_id, v["version"]) is not None for v in versions)


async def test_codebase_research_uses_tools_and_writes_report(make_runtime, registry: SkillRegistry, store: Store, workspace):
    out = await make_runtime(codebase_research_script).start(registry.get("codebase-research"), "Where is `PORT` configured?")
    assert out.status == RunStatus.COMPLETED and "src/app.py" in out.completion["final_answer"]
    assert (workspace / "RESEARCH_REPORT.md").exists()
    state = store.get_state(out.run_id)
    assert {a.locator for a in state.artifacts} == {"src/app.py", "RESEARCH_REPORT.md"}
    assert all(f.evidence_event_ids for f in state.verified_facts) and not state.active_hypotheses
    execs = store.list_tool_executions(out.run_id)
    assert [e.tool_name for e in execs] == ["list_directory", "search_text", "read_text_file", "write_workspace_file"]
    assert all(e.status == "succeeded" and e.output for e in execs)
    obs = store.list_observations(out.run_id)
    assert [o.kind.value for o in obs][:3] == ["task_input", "tool_result", "tool_result"]
    m = store.run_metrics(out.run_id)
    assert m["tool_calls"] == 4 and m["model_calls"] == 6 and m["max_context_chars"] > 0
    calls = store.get_model_calls(out.run_id)
    assert len(calls) == 6 and all(c["context_json"] for c in calls)


async def test_missing_required_tool_is_rejected_at_start(make_runtime, simple_skill):
    import pytest

    from skillstate.tools.base import ToolRegistry

    rt = make_runtime(lambda c: None, tools=ToolRegistry())
    with pytest.raises(ValueError, match="requires unavailable tools"):
        await rt.start(simple_skill, "x")
