from __future__ import annotations

from mnestic.benchmarks.scripts import codebase_research_script, counter_script
from mnestic.models.state import RunStatus
from mnestic.skills.loader import SkillRegistry
from mnestic.storage.store import Store


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

    from mnestic.tools.base import ToolRegistry

    rt = make_runtime(lambda c: None, tools=ToolRegistry())
    with pytest.raises(ValueError, match="requires unavailable tools"):
        await rt.start(simple_skill, "x")


async def test_tool_result_observation_records_request(make_runtime, registry, store):
    from mnestic.benchmarks.scripts import codebase_research_script

    out = await make_runtime(codebase_research_script).start(registry.get("codebase-research"), "Where is `PORT` configured?")
    obs = [o for o in store.list_observations(out.run_id) if o.kind.value == "tool_result"]
    assert obs and all(o.data["request"]["tool"] == o.source for o in obs)
    assert obs[0].data["request"]["arguments"] == {"path": "."}
    ctx = store.get_model_calls(out.run_id, step=1)[0]
    assert 'request=' in __import__("json").loads(ctx["context_json"])["sections"]["latest_observation"].split("\n", 1)[0]


async def test_initial_ops_seed_state_before_first_model_call(make_runtime, store, simple_skill):
    seen = []

    def script(ctx):
        from mnestic.benchmarks.scripts import parse_state

        seen.append(parse_state(ctx))
        from tests.integration.helpers import complete

        return complete(ctx)

    rt = make_runtime(script)
    out = await rt.start(simple_skill, "go", initial_ops=[{"op": "set_entity", "name": "S01", "description": "bolt=2"}, {"op": "set_environment", "key": "capacity", "value": 12}])
    assert out.status.value == "completed"
    assert seen[0]["important_entities"] == {"S01": "bolt=2"} and seen[0]["state_version"] == 1
    assert store.get_state_at_version(out.run_id, 0).important_entities == {}
    ev = store.list_events(out.run_id, limit=50, event_type="patch.applied")[0]
    assert ev.payload["source"] == "runtime:initial_ops"
