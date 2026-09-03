"""§24 crash/resume at several lifecycle points, §25 concurrency."""

from __future__ import annotations

import pytest

from skillstate.agent.reasoner import ScriptedReasoner
from skillstate.graph.runtime import Runtime
from skillstate.models.state import RunStatus
from skillstate.storage.db import Database
from skillstate.storage.store import Store
from skillstate.tools import default_registry
from tests.integration.helpers import CrashingTool, PhaseCrashStore, SimulatedCrash, complete, obs_info, tool


def three_step_script(ctx):
    """task -> list_directory -> read app.py -> complete. Robust to interrupted-tool observations (retries)."""
    kind, ev, text = obs_info(ctx)
    state = ctx.sections["execution_state"]
    if kind == "task_input":
        return tool(ctx, "list_directory", path=".")
    if "TOOL FAILED" in text:  # interrupted tool: retry the same tool
        return tool(ctx, "list_directory", path=".") if '"listed"' not in state else tool(ctx, "read_text_file", path="src/app.py")
    if '"listed"' not in state:
        return tool(ctx, "read_text_file", [{"op": "add_fact", "id": "listed", "statement": "workspace listed", "evidence_event_ids": [ev]}], path="src/app.py")
    return complete(ctx, [{"op": "add_fact", "id": "read", "statement": "app.py read", "evidence_event_ids": [ev]}], answer="ok")


@pytest.mark.parametrize("phase", ["context_built", "decision_recorded", "state_committed", "action_pending", "done"])
async def test_resume_after_crash_at_phase(config, phase):
    crash_store = PhaseCrashStore(config.db_path, phase, on_step=1)
    rt = Runtime(config, crash_store, reasoner=ScriptedReasoner(three_step_script), tools=default_registry())
    with pytest.raises(SimulatedCrash):
        await rt.start(config_skill(), "go")
    assert crash_store.crashed
    crash_store.db.close()

    # New process: fresh store, fresh runtime, resume from SQLite alone.
    store = Store(Database(config.db_path))
    run = store.list_runs()[0]
    assert run.status == "running"
    out = await Runtime(config, store, reasoner=ScriptedReasoner(three_step_script), tools=default_registry()).resume(run.run_id)
    assert out.status == RunStatus.COMPLETED, phase
    state = store.get_state(run.run_id)
    assert {f.id for f in state.verified_facts} == {"listed", "read"}
    # no duplicated patch application: each applied patch produced exactly one version
    applied = [p for p in store.list_patches(run.run_id) if p["status"] == "applied"]
    assert len({p["resulting_version"] for p in applied}) == len(applied)
    steps = store.list_steps(run.run_id)
    assert [s.step for s in steps] == list(range(len(steps)))
    events = store.list_events(run.run_id, limit=1000)
    assert any(e.event_type.value == "run.resumed" for e in events)
    assert [e.seq for e in events] == list(range(len(events)))  # archive is contiguous, append-only


async def test_resume_after_crash_during_tool_execution(config):
    CrashingTool.calls = 0
    reg = default_registry()
    reg.register(CrashingTool())
    skill = config_skill().model_copy(update={"required_tools": ["flaky"]})

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        if kind == "task_input" or "TOOL FAILED" in text:
            return tool(ctx, "flaky", value=1)
        return complete(ctx, [{"op": "add_fact", "id": "f", "statement": text[:50], "evidence_event_ids": [ev]}])

    store = Store(Database(config.db_path))
    with pytest.raises(SimulatedCrash):
        await Runtime(config, store, reasoner=ScriptedReasoner(script), tools=reg).start(skill, "go")
    run = store.list_runs()[0]
    assert store.latest_step(run.run_id).phase == "action_dispatched"
    assert store.find_open_tool_execution(run.run_id, 0).status == "started"
    store.db.close()

    store2 = Store(Database(config.db_path))
    out = await Runtime(config, store2, reasoner=ScriptedReasoner(script), tools=reg).resume(run.run_id)
    assert out.status == RunStatus.COMPLETED
    execs = store2.list_tool_executions(run.run_id)
    assert [e.status for e in execs] == ["interrupted", "succeeded"]
    obs = store2.list_observations(run.run_id)
    assert "interrupted" in obs[1].content and obs[1].data["ok"] is False
    assert any(e.event_type.value == "tool.interrupted" for e in store2.list_events(run.run_id, limit=500))


async def test_resume_rejects_terminal_and_modified_skill(config, store):
    rt = Runtime(config, store, reasoner=ScriptedReasoner(three_step_script), tools=default_registry())
    out = await rt.start(config_skill(), "go")
    with pytest.raises(RuntimeError, match="nothing to resume"):
        await rt.resume(out.run_id)
    # tamper with the stored skill: immutability check must fire
    store.upsert_skill(config_skill().model_copy(update={"instructions": "changed!"}))
    store.db.execute("UPDATE current_states SET status='running' WHERE run_id=?", (out.run_id,))
    with pytest.raises(RuntimeError, match="hash changed"):
        await rt.resume(out.run_id)


async def test_pause_and_resume_with_step_budget(config, store):
    rt = Runtime(config, store, reasoner=ScriptedReasoner(three_step_script), tools=default_registry())
    out = await rt.start(config_skill(), "go", max_steps=1)
    assert out.status == RunStatus.PAUSED and out.reason == "max_steps_this_invocation"
    assert store.get_state(out.run_id).status == RunStatus.PAUSED
    out2 = await rt.resume(out.run_id)
    assert out2.status == RunStatus.COMPLETED


async def test_concurrent_writer_pauses_run_cleanly(config, store):
    """A second process advanced the state between our read and our commit: no lost update, run pauses."""
    other = Store(Database(config.db_path))

    def script(ctx):
        kind, ev, text = obs_info(ctx)
        if kind == "task_input":
            # emulate a concurrent writer bumping the version right before our commit
            st = other.get_state(ctx.run_id)
            other.commit_state(st.model_copy(update={"state_version": st.state_version + 1, "current_phase": "other-process"}),
                               expected_version=st.state_version, patch_id=None, step=0)
            return tool(ctx, "list_directory", path=".")
        return complete(ctx)

    out = await Runtime(config, store, reasoner=ScriptedReasoner(script), tools=default_registry()).start(config_skill(), "go")
    assert out.status == RunStatus.PAUSED and out.reason == "stale_write"
    assert store.get_state(out.run_id).current_phase == "other-process"
    patches = store.list_patches(out.run_id)
    assert [p["status"] for p in patches] == ["rejected"] and patches[0]["error_code"] == "stale_write"
    assert store.list_state_versions(out.run_id)[-1]["patch_id"] is None


def config_skill():
    from skillstate.models.skill import SkillSpecification

    return SkillSpecification(skill_id="crash-skill", name="Crash skill", version="1", description="d",
                              required_tools=["list_directory", "read_text_file"], instructions="i", default_max_steps=50)
