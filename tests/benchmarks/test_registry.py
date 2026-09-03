from __future__ import annotations

import asyncio

from mnestic.agent.reasoner import ScriptedReasoner
from mnestic.benchmarks.registry import RegistryEnv, registry_script, run_skillstate


def test_registry_env_and_scripted_policy(tmp_path):
    env = RegistryEnv(orders=20, seed=3)
    svc = env.order_list[0].service
    assert env.dependents(svc) == sorted(n for n, r in env.records.items() if svc in r["deps"])
    r = asyncio.run(run_skillstate(ScriptedReasoner(registry_script), orders=200, services=12, seed=11, db_path=tmp_path / "r.db"))
    assert r.status == "completed" and r.score == 1.0 and r.max_context_chars < 12000
