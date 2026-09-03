"""Warehouse long-horizon benchmark: mnestic (SKILL.state) vs a live ReAct baseline on the same model.

usage: uv run python scripts/warehouse_bench.py --model openai-chat:cc/claude-sonnet-5 --orders 60 [--mode both|skillstate|react|mock]
Set OPENAI_BASE_URL / OPENAI_API_KEY for proxies. Results are appended to docs/WAREHOUSE_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from mnestic.benchmarks.warehouse import render, run_react, run_skillstate, warehouse_script

p = argparse.ArgumentParser()
p.add_argument("--model", default="mock")
p.add_argument("--orders", type=int, default=60)
p.add_argument("--shelves", type=int, default=12)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--mode", default="both", choices=["both", "skillstate", "react", "mock"])
p.add_argument("--timeout", type=float, default=float(os.environ.get("MNESTIC_MODEL_TIMEOUT", "300")))
p.add_argument("--out", default="docs/WAREHOUSE_BENCHMARK.md")
a = p.parse_args()


async def main() -> list:
    results = []
    if a.model == "mock" or a.mode == "mock":
        from mnestic.agent.reasoner import ScriptedReasoner

        results.append(await run_skillstate(ScriptedReasoner(warehouse_script), orders=a.orders, shelves=a.shelves, seed=a.seed))
    else:
        from mnestic.agent.pydantic_ai_reasoner import PydanticAIReasoner, build_model

        if a.mode in ("both", "skillstate"):
            from mnestic.benchmarks.warehouse import WAREHOUSE_SKILL

            r = PydanticAIReasoner(a.model, retries=2, model_settings={"timeout": a.timeout}, wall_clock_timeout=a.timeout,
                                   allowed_ops=WAREHOUSE_SKILL.allowed_ops)
            print(f"output schema: {r.output_schema_chars:,} chars", flush=True)
            results.append(await run_skillstate(r, orders=a.orders, shelves=a.shelves, seed=a.seed))
            print(render(results[-1:]), flush=True)
        if a.mode in ("both", "react"):
            results.append(await run_react(build_model(a.model), orders=a.orders, shelves=a.shelves, seed=a.seed, model_settings={"timeout": a.timeout}))
    return results


results = asyncio.run(main())
table = render(results)
print(table)
out = Path(a.out)
header = ("# Warehouse long-horizon benchmark\n\n*Based on the paper: Badhe, Tiwari, Chung — SKILL.state, "
          "[arXiv:2608.26263](https://arxiv.org/abs/2608.26263) (SkillExecBench Warehouse).*\n\n"
          "One order per step; the environment never restates inventory; score = correct orders / orders.\n\n")
existing = out.read_text() if out.exists() else header
out.write_text(existing + f"\n## {a.model} — {a.orders} orders, {a.shelves} shelves, seed {a.seed}\n\n{table}\n")
for r in results:
    wrong = [x for x in r.per_order if not x["correct"]]
    print(f"{r.mode}: first wrong orders: {json.dumps(wrong[:5])}")
