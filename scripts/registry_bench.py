"""Service-registry long-horizon benchmark: mnestic vs a live ReAct baseline on the same model.

usage: uv run python scripts/registry_bench.py --model openai-chat:cc/claude-sonnet-5 --orders 200 [--mode both|skillstate|react|mock]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from mnestic.benchmarks.registry import REGISTRY_SKILL, registry_script, run_react, run_skillstate
from mnestic.benchmarks.warehouse import render

p = argparse.ArgumentParser()
p.add_argument("--model", default="mock")
p.add_argument("--orders", type=int, default=200)
p.add_argument("--services", type=int, default=12)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--mode", default="both", choices=["both", "skillstate", "react", "mock"])
p.add_argument("--timeout", type=float, default=float(os.environ.get("MNESTIC_MODEL_TIMEOUT", "300")))
p.add_argument("--out", default="docs/REGISTRY_BENCHMARK.md")
a = p.parse_args()


async def main() -> list:
    results = []
    if a.model == "mock" or a.mode == "mock":
        from mnestic.agent.reasoner import ScriptedReasoner

        results.append(await run_skillstate(ScriptedReasoner(registry_script), orders=a.orders, services=a.services, seed=a.seed))
    else:
        from mnestic.agent.pydantic_ai_reasoner import PydanticAIReasoner, build_model

        extra = json.loads(os.environ.get("MNESTIC_MODEL_SETTINGS", "{}"))
        if a.mode in ("both", "skillstate"):
            r = PydanticAIReasoner(a.model, output_mode=os.environ.get("MNESTIC_OUTPUT_MODE", "tool"), retries=2, model_settings={"timeout": a.timeout, **extra},
                                   wall_clock_timeout=a.timeout, allowed_ops=REGISTRY_SKILL.allowed_ops, allowed_actions=REGISTRY_SKILL.allowed_actions)
            results.append(await run_skillstate(r, orders=a.orders, services=a.services, seed=a.seed))
            print(render(results[-1:]), flush=True)
        if a.mode in ("both", "react"):
            results.append(await run_react(build_model(a.model), orders=a.orders, services=a.services, seed=a.seed, model_settings={"timeout": a.timeout, **extra}))
    return results


results = asyncio.run(main())
table = render(results)
print(table)
out = Path(a.out)
header = "# Service-registry long-horizon benchmark\n\nMutations + graph queries over a service registry; the runtime keeps the latest record per service from what the tool returns (generic ledger, no task-specific code).\n\n"
existing = out.read_text() if out.exists() else header
out.write_text(existing + f"\n## {a.model} — {a.orders} orders, {a.services} services, seed {a.seed}\n\n{table}\n")
for r in results:
    wrong = [x for x in r.per_order if not x["correct"]]
    print(f"{r.mode}: wrong={len(wrong)} first: {json.dumps(wrong[:4])[:400]}")
