"""Context-scaling benchmark: SKILL.state bounded context vs. a ReAct-style accumulating transcript.

Both simulations feed observations of fixed size at every step. We measure the serialized
model-context size (characters, and an approximate token estimate) per step.

This is an *empirical* measurement of this implementation with a scripted reasoner and a
mock tool. It does not by itself prove asymptotic O(1)/O(T) properties; see docs/BENCHMARK_REPORT.md
for the distinction between the architectural argument and the measured numbers.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from mnestic.agent.reasoner import ScriptedReasoner
from mnestic.benchmarks.scripts import observation_attr, observation_text, parse_state
from mnestic.config import RuntimeConfig
from mnestic.context.builder import ModelContext
from mnestic.graph.runtime import Runtime
from mnestic.models.decision import AgentDecision
from mnestic.models.skill import SkillSpecification
from mnestic.storage.db import Database
from mnestic.storage.store import Store
from mnestic.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

MARKER_PREFIX = "OBSMARK"


def marker(step: int) -> str:
    """A unique token per step, used to prove earlier observations do not leak into later contexts."""
    return f"{MARKER_PREFIX}-{step:06d}-{(step * 7919) % 100003:05d}"


class FixedObservationTool(Tool):
    """Mock tool returning an observation of fixed size containing a unique step marker."""

    name: ClassVar[str] = "probe"
    description: ClassVar[str] = "Return a fixed-size probe observation for benchmark step N."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        step: int = Field(ge=0)

    def __init__(self, observation_chars: int):
        self.observation_chars = observation_chars

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        body = f"{marker(args.step)} probe reading value={args.step % 97} status=ok "
        body = (body * (self.observation_chars // len(body) + 1))[: self.observation_chars]
        return ToolResult(ok=True, output=body, data={"step": args.step})


def benchmark_script(context: ModelContext) -> AgentDecision:
    """Each step: record a one-line summary (replace, not append), then call the probe tool."""
    state = parse_state(context)
    v = state["state_version"]
    n = int(state["environment"]["properties"].get("n", 0))
    target = int(state["environment"]["properties"].get("target", 0))
    kind = observation_attr(context, "kind")
    ops: list[dict[str, Any]] = []
    if kind == "task_input":
        import re

        target = int(re.search(r"(\d+)", observation_text(context)).group(1))  # type: ignore[union-attr]
        ops += [{"op": "set_environment", "key": "target", "value": target}, {"op": "set_environment", "key": "n", "value": 0}]
    else:
        n += 1
        ops += [{"op": "set_environment", "key": "n", "value": n},
                {"op": "set_observation_summary", "summary": f"probe {n} ok"}]
    if n >= target:
        return AgentDecision(rationale_summary="done", state_patch={"expected_state_version": v, "ops": ops},
                             completion={"outcome": "success", "summary": f"{n} probes"})
    return AgentDecision(rationale_summary="probe", state_patch={"expected_state_version": v, "ops": ops},
                         action={"kind": "tool", "tool_name": "probe", "arguments": {"step": n + 1}})


BENCH_SKILL = SkillSpecification(
    skill_id="benchmark-probe", name="Benchmark probe", version="1.0.0",
    description="Calls the probe tool N times, keeping only a one-line summary in state.",
    required_tools=["probe"], instructions="Call probe repeatedly until n == target, then complete.",
    completion_criteria=["n == target"], default_max_steps=100_000,
    allowed_ops=["set_environment", "set_observation_summary"],
)


class ReActContextSimulator:
    """Baseline: an append-only transcript. Every step re-sends everything that happened before.

    Sizes mirror what a chat-history agent would send: system prompt + every prior
    (thought, action, observation) triple + the newest observation.
    """

    def __init__(self, system_prompt_chars: int, thought_chars: int = 120, action_chars: int = 60):
        self.system_prompt_chars = system_prompt_chars
        self.transcript: list[str] = []
        self.thought_chars = thought_chars
        self.action_chars = action_chars

    def step(self, observation: str) -> int:
        self.transcript.append(observation)
        context_chars = self.system_prompt_chars + sum(len(t) for t in self.transcript)
        # the model's reply (thought + action) becomes part of the next context
        self.transcript.append("T" * self.thought_chars + "A" * self.action_chars)
        return context_chars


def run_benchmark(*, steps: int = 1000, observation_chars: int = 400, checkpoints: list[int] | None = None,
                  db_path: Path | None = None) -> dict[str, Any]:
    tmp = None
    if db_path is None:
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "bench.db"
    try:
        return asyncio.run(_run(steps, observation_chars, checkpoints, db_path))
    finally:
        if tmp:
            tmp.cleanup()


async def _run(steps: int, observation_chars: int, checkpoints: list[int] | None, db_path: Path) -> dict[str, Any]:
    cfg = RuntimeConfig(db_path=db_path, workspace_root=db_path.parent, max_observation_chars=max(observation_chars + 200, 1000))
    store = Store(Database(db_path))
    tools = ToolRegistry()
    tools.register(FixedObservationTool(observation_chars))
    sizes: list[int] = []
    tokens: list[int] = []
    leaks: list[tuple[int, str]] = []
    state_bytes: list[int] = []

    def capture(context: ModelContext) -> AgentDecision:
        sizes.append(context.char_count)
        tokens.append(context.approx_tokens)
        state_bytes.append(len(context.sections["execution_state"].encode("utf-8")))
        full = context.full_text()
        # Leak detection: any marker from a step older than the newest observation must be absent.
        step = context.step
        for old in range(max(0, step - 50), step):  # windowed scan keeps the benchmark fast; the full check is in tests
            if old >= 1 and marker(old) in full:
                leaks.append((step, marker(old)))
        return benchmark_script(context)

    rt = Runtime(cfg, store, reasoner=ScriptedReasoner(capture), tools=tools)
    outcome = await rt.start(BENCH_SKILL, f"probe {steps} times")
    react = ReActContextSimulator(system_prompt_chars=len(_skill_prompt_chars()))
    react_sizes = [react.step("X" * observation_chars) for _ in range(len(sizes))]
    checkpoints = checkpoints or [c for c in (1, 10, 50, 100, 250, 500, 750, 1000) if c <= len(sizes)]
    rows = [{"step": c, "mnestic_chars": sizes[c - 1], "mnestic_tokens_est": tokens[c - 1], "state_section_bytes": state_bytes[c - 1],
             "react_chars": react_sizes[c - 1]} for c in checkpoints]
    late = sizes[len(sizes) // 2 :] or sizes
    bounded = (max(late) - min(late)) <= max(200, int(0.05 * max(late))) and not leaks
    from mnestic.models.decision import AgentDecision, decision_type_for

    schema_chars = len(json.dumps(decision_type_for(BENCH_SKILL.allowed_ops).model_json_schema(), separators=(",", ":")))
    full_chars = len(json.dumps(AgentDecision.model_json_schema(), separators=(",", ":")))
    return {
        "output_schema_chars": schema_chars, "full_schema_chars": full_chars,
        "steps": len(sizes), "observation_chars": observation_chars, "outcome": outcome.status.value,
        "mnestic": {"first": sizes[0], "min": min(sizes), "max": max(sizes), "mean": round(statistics.mean(sizes), 1),
                       "last": sizes[-1], "total_chars": sum(sizes)},
        "react": {"first": react_sizes[0], "last": react_sizes[-1], "total_chars": sum(react_sizes)},
        "leaks": leaks[:10], "bounded": bounded, "rows": rows, "series": sizes, "react_series": react_sizes,
        "archive_events": store.count_events(outcome.run_id), "run_metrics": store.run_metrics(outcome.run_id),
    }


def _skill_prompt_chars() -> str:
    from mnestic.context.builder import OUTPUT_CONTRACT, ContextBuilder

    return ContextBuilder().render_skill(BENCH_SKILL) + OUTPUT_CONTRACT


def render_report(r: dict[str, Any]) -> str:
    lines = [
        "# Context-scaling benchmark report", "",
        "*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, "
        "[arXiv:2608.26263](https://arxiv.org/abs/2608.26263).*", "",
        f"Simulated task: {r['steps']} steps, each producing a tool observation of {r['observation_chars']} chars. "
        "Scripted reasoner (no model), mock `probe` tool, real runtime/graph/SQLite path.", "",
        "## Measured model-context size per step (characters)", "",
        "| step | SKILL.state context | ≈tokens | state section | ReAct-style transcript |", "|---:|---:|---:|---:|---:|",
    ]
    for row in r["rows"]:
        lines.append(f"| {row['step']} | {row['mnestic_chars']:,} | {row['mnestic_tokens_est']:,} | {row['state_section_bytes']:,} | {row['react_chars']:,} |")
    s, b = r["mnestic"], r["react"]
    lines += [
        "", "## What the provider actually receives", "",
        "The context above is *our* text. Every call also carries the structured-output schema (`AgentDecision`), sent as a "
        f"tool definition: **{r.get('output_schema_chars', 0):,} chars** with this skill's `allowed_ops` "
        f"(full 30-op schema: {r.get('full_schema_chars', 0):,} chars). Some proxy routes add their own system prompt on top. "
        "Measure per-call tokens at the provider, not at the context builder — see docs/WAREHOUSE_BENCHMARK.md.",
        "", "## Summary", "",
        f"- SKILL.state: first={s['first']:,} min={s['min']:,} max={s['max']:,} mean={s['mean']:,} last={s['last']:,} chars; "
        f"cumulative={s['total_chars']:,} chars over {r['steps']} steps.",
        f"- ReAct-style baseline: first={b['first']:,} last={b['last']:,} chars; cumulative={b['total_chars']:,} chars "
        f"({b['total_chars'] / max(1, s['total_chars']):.1f}× SKILL.state).",
        f"- Leak check (marker from an older observation found in a later context): {'NONE' if not r['leaks'] else r['leaks']}.",
        f"- Bounded (late-run context variation ≤5% and no leaks): **{'yes' if r['bounded'] else 'NO'}**.",
        f"- Archive events written: {r['archive_events']:,} (the archive grows linearly; the prompt does not).", "",
        "## Interpretation", "",
        "The architecture argues that per-step context is `|P| + |Σ_t| + |O_t|`, independent of t, so cumulative tokens are O(T) "
        "while an append-only transcript is O(T²). The table above is an empirical measurement of this implementation under a "
        "fixed-size-observation workload; it shows the per-step context stays flat only because the scripted policy keeps Σ_t "
        "bounded (it replaces `last_observation_summary` instead of appending). A policy that appends to the state at every "
        "step would grow Σ_t linearly — which is exactly what `StateLimits` reject. The ReAct numbers come from a simulator of "
        "transcript growth, not from a second live agent; they illustrate the contrast, not a head-to-head accuracy comparison.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({k: v for k, v in run_benchmark(steps=100).items() if k not in {"series", "react_series"}}, indent=2))
