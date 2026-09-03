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


def test_react_baseline_runs_with_a_function_model():
    """The ReAct runner must at least drive the tool loop (regression: tool type hints failed to evaluate)."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from mnestic.benchmarks.warehouse import run_react

    calls = {"n": 0}

    def fn(messages, info):
        calls["n"] += 1
        if calls["n"] <= 3:
            return ModelResponse(parts=[ToolCallPart(tool_name="warehouse", args={"action": "count", "answer": 0})])
        return ModelResponse(parts=[TextPart(content="DONE")])

    r = asyncio.run(run_react(FunctionModel(fn), orders=10, seed=5))
    assert r.mode == "react" and r.steps == 3 and r.model_calls == 4 and r.status == "completed" and r.max_context_chars > 0


def test_inspect_verifies_without_consuming_the_order():
    env = WarehouseEnv(orders=5, seed=2)
    before = env.cursor
    ok, msg = env.act("inspect", "S01", None, None, None)
    assert ok and msg.startswith("INSPECT S01: holds") and "free)" in msg and "ORDER #1" in msg
    assert env.cursor == before and env.log == [] and env.inspections == 1
    assert not env.act("inspect", "S99", None, None, None)[0]


def test_books_are_kept_by_the_runtime_not_the_model(tmp_path):
    """After a store, domain.shelves/free/totals reflect it without any model-authored op."""
    from mnestic.storage.db import Database
    from mnestic.storage.store import Store

    r = asyncio.run(run_skillstate(ScriptedReasoner(warehouse_script), orders=40, shelves=12, seed=3, db_path=tmp_path / "b.db"))
    assert r.score == 1.0
    st = Store(Database(tmp_path / "b.db")).get_state(r.notes.split("run_id=")[1].split()[0])
    ledger = st.domain["warehouse"]
    shelves = {k: v for k, v in ledger.items() if not k.startswith("total_")}
    assert len(shelves) == 12 and all(v["free"] == 12 - sum(v["holds"].values()) for v in shelves.values())
    assert all("_event" in v for v in shelves.values() if "_step" in v)  # provenance on every runtime-written fact
