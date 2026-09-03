from __future__ import annotations

import asyncio

from mnestic.agent.reasoner import ScriptedReasoner
from mnestic.benchmarks.warehouse import WarehouseEnv, run_skillstate, warehouse_script


def test_env_is_deterministic_and_scores():
    a, b = WarehouseEnv(orders=30, seed=3), WarehouseEnv(orders=30, seed=3)
    assert [o.text() for o in a.order_list] == [o.text() for o in b.order_list] and a.inventory == b.inventory
    ok, msg = a.act("count", None, None, None, -1)  # wrong for every order kind
    assert not ok and msg.startswith("WRONG") and a.cursor == 1 and "ORDER #2" in msg
    o2 = a.current_order()
    if o2.kind == "count":
        truth = sum(s.get(o2.item, 0) for s in a.inventory.values())
        assert a.act("count", None, None, None, truth)[0]
    elif o2.kind == "store":
        shelf = next(s for s in a.inventory if a.used(s) + o2.qty <= a.capacity)
        assert a.act("store", shelf, o2.item, o2.qty, None)[0]
    else:
        shelf = next(s for s in a.inventory if a.inventory[s].get(o2.item, 0) >= o2.qty)
        assert a.act("ship", shelf, o2.item, o2.qty, None)[0]
    assert a.score == 0.5


def test_scripted_policy_scores_perfectly_over_200_orders(tmp_path):
    r = asyncio.run(run_skillstate(ScriptedReasoner(warehouse_script), orders=200, shelves=12, seed=11, db_path=tmp_path / "wh.db"))
    assert r.status == "completed" and r.score == 1.0 and r.correct == 200
    assert r.max_context_chars < 9000, r.max_context_chars  # bounded regardless of horizon
