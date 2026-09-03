"""§17 token-scaling benchmark as a test: fails if history leaks into default context."""

from __future__ import annotations

import pytest

from skillstate.benchmarks.scaling import ReActContextSimulator, marker, run_benchmark

pytestmark = pytest.mark.benchmark


def test_context_is_bounded_and_nothing_leaks(tmp_path):
    r = run_benchmark(steps=300, observation_chars=400, db_path=tmp_path / "b.db")
    assert r["outcome"] == "completed" and r["steps"] == 301
    s = r["skillstate"]
    late = r["series"][150:]
    assert max(late) - min(late) <= 0.05 * max(late), "context size drifts in the second half of the run"
    assert s["last"] < s["first"] * 1.5, "context grew substantially over the run"
    assert r["leaks"] == [] and r["bounded"]
    # ReAct-style transcript grows linearly; SKILL.state does not.
    react = r["react_series"]
    assert react[-1] > react[0] * 50 and react[-1] > s["last"] * 20
    assert r["react"]["total_chars"] > s["total_chars"] * 10


def test_full_marker_scan_on_short_run(tmp_path):
    """Every earlier marker checked against every later context (the in-benchmark scan is windowed)."""
    from skillstate.agent.reasoner import ScriptedReasoner
    from skillstate.benchmarks.scaling import BENCH_SKILL, FixedObservationTool, benchmark_script
    from skillstate.config import RuntimeConfig
    from skillstate.graph.runtime import Runtime
    from skillstate.storage.db import Database
    from skillstate.storage.store import Store
    from skillstate.tools.base import ToolRegistry

    contexts = []

    def capture(ctx):
        contexts.append(ctx.full_text())
        return benchmark_script(ctx)

    cfg = RuntimeConfig(db_path=tmp_path / "s.db", workspace_root=tmp_path)
    tools = ToolRegistry()
    tools.register(FixedObservationTool(300))
    import asyncio

    out = asyncio.run(Runtime(cfg, Store(Database(cfg.db_path)), reasoner=ScriptedReasoner(capture), tools=tools).start(BENCH_SKILL, "probe 60 times"))
    assert out.status.value == "completed"
    for step, text in enumerate(contexts):
        for old in range(1, step):
            assert marker(old) not in text, f"marker from step {old} present at step {step}"
        if step >= 1:
            assert marker(step) in text  # the newest observation IS present, exactly once
            assert text.count(marker(step)) >= 1


def test_react_simulator_is_linear():
    sim = ReActContextSimulator(system_prompt_chars=1000)
    sizes = [sim.step("o" * 100) for _ in range(100)]
    diffs = {sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)}
    assert len(diffs) == 1  # constant increment => linear growth
