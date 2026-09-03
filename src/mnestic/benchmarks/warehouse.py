"""Long-horizon warehouse benchmark (after SkillExecBench "Warehouse" in the SKILL.state paper).

A deterministic environment hands the agent one order per step (store / ship / count). The agent must
track the shelf inventory itself — the environment never restates it — and answer with a valid action.
Ground truth comes from the simulation, so every action is scored objectively.

Two runners share the same environment and tool:
- ``run_skillstate``: the mnestic runtime (bounded context, state patches).
- ``run_react``: a plain PydanticAI agent with the same tool and an accumulating message history
  (one long ``agent.run`` whose internal loop is exactly a ReAct transcript).
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.settings import ModelSettings

from mnestic.models.skill import SkillSpecification
from mnestic.tools.base import Tool, ToolContext, ToolResult

ITEMS = ["bolt", "gear", "valve", "wire", "lens", "pump", "seal", "fuse", "clamp", "hinge"]


@dataclass
class Order:
    index: int
    kind: Literal["store", "ship", "count"]
    item: str
    qty: int = 0

    def text(self) -> str:
        if self.kind == "store":
            return f"ORDER #{self.index}: store {self.qty} {self.item} — choose a shelf with enough free capacity"
        if self.kind == "ship":
            return f"ORDER #{self.index}: ship {self.qty} {self.item} — choose a shelf that holds at least that many"
        return f"ORDER #{self.index}: count — how many {self.item} are in the warehouse in total?"


@dataclass
class WarehouseEnv:
    """Deterministic simulator. ``seed`` fixes shelves, capacities and the order stream."""

    shelves: int = 12
    capacity: int = 12
    orders: int = 60
    seed: int = 7
    inventory: dict[str, dict[str, int]] = field(default_factory=dict)
    order_list: list[Order] = field(default_factory=list)
    cursor: int = 0
    inspections: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)  # noqa: S311 - reproducible benchmark, not security
        self.inventory = {f"S{i + 1:02d}": {} for i in range(self.shelves)}
        # pre-stock a few shelves so early ship orders are satisfiable
        for _ in range(self.shelves):
            shelf = rng.choice(list(self.inventory))
            item = rng.choice(ITEMS[:6])
            qty = rng.randint(1, 4)
            if self.used(shelf) + qty <= self.capacity:
                self.inventory[shelf][item] = self.inventory[shelf].get(item, 0) + qty
        totals = {item: sum(s.get(item, 0) for s in self.inventory.values()) for item in ITEMS}
        room = self.shelves * self.capacity
        for i in range(1, self.orders + 1):
            r = rng.random()
            # store only while the warehouse is comfortably below capacity (fragmentation must never make an order infeasible);
            # ship exactly 1 so any shelf holding the item can fulfil it (the generator cannot know shelf placement).
            if r < 0.45 and sum(totals.values()) + 4 <= 0.6 * room:
                item, qty = rng.choice(ITEMS), rng.randint(1, 4)
                self.order_list.append(Order(i, "store", item, qty))
                totals[item] += qty
            elif r < 0.85 and any(totals[x] > 0 for x in ITEMS):
                item = rng.choice([x for x in ITEMS if totals[x] > 0])
                self.order_list.append(Order(i, "ship", item, 1))
                totals[item] -= 1
            else:
                self.order_list.append(Order(i, "count", rng.choice(ITEMS)))
        self.initial_inventory = json.loads(json.dumps(self.inventory))

    def used(self, shelf: str) -> int:
        return sum(self.inventory[shelf].values())

    def current_order(self) -> Order | None:
        return self.order_list[self.cursor] if self.cursor < len(self.order_list) else None

    def initial_ops(self) -> list[dict[str, Any]]:
        """Σ₀: the known starting inventory, typed (domain.shelves.<id>.<item> = qty) — the runtime keeps these books."""
        return [{"op": "set_path", "path": "capacity", "value": self.capacity},
                {"op": "set_path", "path": "shelves", "value": {shelf: dict(items) for shelf, items in self.inventory.items()}}]

    def initial_observation(self) -> str:
        lines = [f"Warehouse: {self.shelves} shelves ({', '.join(self.inventory)}), capacity {self.capacity} units each.", "Initial contents:"]
        for shelf, items in self.inventory.items():
            lines.append(f"  {shelf}: " + (", ".join(f"{k}={v}" for k, v in sorted(items.items())) or "empty"))
        o = self.current_order()
        lines += ["", "Your state's `domain.shelves` holds this inventory and the runtime updates it after every successful store/ship.",
                  o.text() if o else "no orders"]
        return "\n".join(lines)

    def act(self, action: str, shelf: str | None, item: str | None, qty: int | None, answer: int | None) -> tuple[bool, str]:
        """Apply the agent's response to the current order. Returns (correct, message)."""
        o = self.current_order()
        if o is None:
            return False, "no pending order"
        if action == "inspect":
            # Verification: look at one shelf's true contents. Does not consume the order and is not scored.
            if shelf not in self.inventory:
                return False, f"INSPECT: unknown shelf {shelf}\n{o.text()}"
            self.inspections += 1
            return True, f"INSPECT {shelf}: holds {self._desc(shelf)} ({self.capacity - self.used(shelf)} free)\n{o.text()}"
        ok, msg = False, ""
        if o.kind == "store":
            if action != "store" or item != o.item or qty != o.qty:
                msg = f"WRONG: expected store {o.qty} {o.item}, got {action} {qty} {item}"
            elif shelf not in self.inventory:
                msg = f"WRONG: unknown shelf {shelf}"
            elif self.used(shelf) + o.qty > self.capacity:
                msg = f"WRONG: {shelf} has only {self.capacity - self.used(shelf)} free (holds {self._desc(shelf)})"
            else:
                self.inventory[shelf][o.item] = self.inventory[shelf].get(o.item, 0) + o.qty
                ok, msg = True, f"OK: stored {o.qty} {o.item} on {shelf}"
        elif o.kind == "ship":
            if action != "ship" or item != o.item or qty != o.qty:
                msg = f"WRONG: expected ship {o.qty} {o.item}, got {action} {qty} {item}"
            elif shelf not in self.inventory:
                msg = f"WRONG: unknown shelf {shelf}"
            elif self.inventory[shelf].get(o.item, 0) < o.qty:
                msg = f"WRONG: {shelf} holds only {self.inventory[shelf].get(o.item, 0)} {o.item} (holds {self._desc(shelf)})"
            else:
                self.inventory[shelf][o.item] -= o.qty
                if self.inventory[shelf][o.item] == 0:
                    del self.inventory[shelf][o.item]
                ok, msg = True, f"OK: shipped {o.qty} {o.item} from {shelf}"
        else:
            truth = sum(s.get(o.item, 0) for s in self.inventory.values())
            if action != "count" or answer is None:
                msg = f"WRONG: expected count with an answer, got {action}"
            elif answer == truth:
                ok, msg = True, f"OK: {o.item} total is {truth}"
            else:
                msg = f"WRONG: {o.item} total is {truth}, you said {answer}"
        # A wrong store/ship does not change the world; the order is still consumed (as in the paper: scored per order).
        self.log.append({"order": o.index, "kind": o.kind, "correct": ok, "message": msg})
        self.cursor += 1
        nxt = self.current_order()
        return ok, msg + ("\n" + nxt.text() if nxt else "\nALL ORDERS DONE — submit completion.")

    def _desc(self, shelf: str) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.inventory[shelf].items())) or "empty"

    @property
    def score(self) -> float:
        return sum(1 for r in self.log if r["correct"]) / max(1, len(self.log))


class WarehouseTool(Tool):
    name: ClassVar[str] = "warehouse"
    description: ClassVar[str] = (
        "Respond to the current order. store/ship need shelf+item+qty; count needs answer. Returns the outcome and the NEXT order. "
        "action=inspect with a shelf returns that shelf's true contents WITHOUT consuming the order (use it to verify)."
    )

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Literal["store", "ship", "count", "inspect"]
        shelf: str | None = Field(default=None, description="e.g. S03")
        item: str | None = None
        qty: int | None = Field(default=None, ge=1)
        answer: int | None = Field(default=None, ge=0, description="for count orders")

    def __init__(self, env: WarehouseEnv):
        self.env = env

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        ok, msg = self.env.act(args.action, args.shelf, args.item, args.qty, args.answer)
        effects: list[dict[str, Any]] = []
        if ok and args.action in ("store", "ship") and args.shelf and args.item and args.qty:
            delta = args.qty if args.action == "store" else -args.qty
            effects.append({"op": "adjust_path", "path": f"shelves.{args.shelf}.{args.item}", "delta": delta})
        return ToolResult(ok=True, output=msg, data={"correct": ok, "orders_done": self.env.cursor, "orders_total": len(self.env.order_list)},
                          state_effects=effects)


WAREHOUSE_SKILL = SkillSpecification(
    skill_id="warehouse-ops", name="Warehouse operations", version="1.0.0",
    description="Fulfil a stream of warehouse orders while tracking shelf inventory yourself.",
    required_tools=["warehouse"],
    instructions=(
        "You operate a warehouse. Each observation gives you the outcome of your last action and the NEXT order.\n"
        "`domain.shelves` in your state is the inventory (shelf -> item -> quantity) and `domain.capacity` the per-shelf limit. "
        "The runtime updates `domain.shelves` itself after every successful store/ship — you never edit it. Your job is to choose: "
        "for store, a shelf whose total quantity + qty <= capacity; for ship, a shelf whose quantity of the item >= qty; for count, "
        "the sum of that item across all shelves, answered with `answer`.\n"
        "Respond to every order with exactly one `warehouse` tool action; an empty state_patch is fine. `inspect` a shelf (does not "
        "consume the order) if a store/ship was rejected. When the observation says ALL ORDERS DONE, submit completion with outcome=success."
    ),
    completion_criteria=["all orders processed"], phases=["operating", "done"], initial_phase="operating", default_max_steps=5000,
    allowed_ops=["set_observation_summary", "set_phase"],
    allowed_actions=["tool"],
    domain_schema={
        "type": "object", "additionalProperties": False,
        "properties": {
            "capacity": {"type": "integer", "minimum": 1},
            "shelves": {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}}},
        },
        "required": ["capacity", "shelves"],
    },
)


def warehouse_script(context: Any) -> dict[str, Any]:
    """Deterministic optimal policy for the scripted reasoner (validates the harness; no model). Reads the runtime-kept books."""
    from mnestic.benchmarks.scripts import observation_text, parse_state

    state = parse_state(context)
    v = state["state_version"]
    obs = observation_text(context)
    shelves: dict[str, dict[str, int]] = state["domain"]["shelves"]
    cap = int(state["domain"]["capacity"])
    if "ALL ORDERS DONE" in obs:
        return {"rationale_summary": "done", "state_patch": {"expected_state_version": v, "ops": []}, "completion": {"outcome": "success", "summary": "orders complete"}}
    if "ORDER #" not in obs:  # runtime feedback: the pending order is whatever we recorded last time
        obs = state.get("last_observation_summary") or ""
    m = re.search(r"ORDER #\d+: (store|ship|count)(?: (\d+) (\w+)| — how many (\w+))", obs)
    assert m is not None, f"no order in observation: {obs[:120]}"
    ops: list[dict[str, Any]] = [{"op": "set_observation_summary", "summary": m.group(0)}]
    kind = m.group(1)
    action: dict[str, Any]
    if kind == "store":
        qty, item = int(m.group(2)), m.group(3)
        shelf = next(sh for sh, inv in sorted(shelves.items()) if sum(inv.values()) + qty <= cap)
        action = {"kind": "tool", "tool_name": "warehouse", "arguments": {"action": "store", "shelf": shelf, "item": item, "qty": qty}}
    elif kind == "ship":
        qty, item = int(m.group(2)), m.group(3)
        shelf = next(sh for sh, inv in sorted(shelves.items()) if inv.get(item, 0) >= qty)
        action = {"kind": "tool", "tool_name": "warehouse", "arguments": {"action": "ship", "shelf": shelf, "item": item, "qty": qty}}
    else:
        item = m.group(4)
        action = {"kind": "tool", "tool_name": "warehouse", "arguments": {"action": "count", "answer": sum(inv.get(item, 0) for inv in shelves.values())}}
    return {"rationale_summary": kind, "state_patch": {"expected_state_version": v, "ops": ops}, "action": action}


# ---- runners -------------------------------------------------------------------------------------


@dataclass
class BenchResult:
    mode: str
    model: str
    orders: int
    score: float
    correct: int
    steps: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    max_context_chars: int
    wall_seconds: float
    status: str
    per_order: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    inspections: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


async def run_skillstate(reasoner: Any, *, orders: int = 60, shelves: int = 12, seed: int = 7, db_path: Any = None, max_steps: int | None = None) -> BenchResult:
    import tempfile
    from pathlib import Path

    from mnestic.config import RuntimeConfig
    from mnestic.graph.runtime import Runtime
    from mnestic.storage.db import Database
    from mnestic.storage.store import Store
    from mnestic.tools.base import ToolRegistry

    env = WarehouseEnv(shelves=shelves, orders=orders, seed=seed)
    tmp = tempfile.mkdtemp()
    db_path = Path(db_path or Path(tmp) / "wh.db")
    cfg = RuntimeConfig(db_path=db_path, workspace_root=Path(tmp), max_consecutive_continues=2)
    store = Store(Database(db_path))
    tools = ToolRegistry()
    tools.register(WarehouseTool(env))
    t0 = time.time()
    out = await Runtime(cfg, store, reasoner=reasoner, tools=tools).start(WAREHOUSE_SKILL, env.initial_observation(), max_steps=max_steps or orders * 3,
                                                                          initial_ops=env.initial_ops())
    m = store.run_metrics(out.run_id)
    return BenchResult("skillstate", getattr(reasoner, "model_name", "?"), orders, env.score, sum(1 for r in env.log if r["correct"]), out.steps,
                       m["model_calls"] or 0, m["input_tokens"] or 0, m["output_tokens"] or 0, m["max_context_chars"] or 0, time.time() - t0,
                       out.status.value, env.log, f"run_id={out.run_id} db={db_path}", env.inspections, m["cache_read_tokens"] or 0, m["cache_write_tokens"] or 0)


async def run_react(model: Any, *, orders: int = 60, shelves: int = 12, seed: int = 7, max_tool_calls: int | None = None, model_settings: dict[str, Any] | None = None) -> BenchResult:
    """Baseline: one PydanticAI agent run; the tool loop accumulates every call/result in the transcript."""
    env = WarehouseEnv(shelves=shelves, orders=orders, seed=seed)

    agent: Agent[None, str] = Agent(
        model, instructions=WAREHOUSE_SKILL.instructions.replace("in `important_entities`", "in your working notes")
        .replace("(set_entity replaces it)", "").replace("Do not add facts or hypotheses; the entities ARE the state. ", "")
        + "\nWhen ALL ORDERS DONE, reply with the single word DONE.", retries=2,
        model_settings=ModelSettings(**(model_settings or {})) if model_settings else None,  # type: ignore[typeddict-item]
    )

    @agent.tool_plain
    def warehouse(action: Literal["store", "ship", "count", "inspect"], shelf: str | None = None, item: str | None = None,
                  qty: int | None = None, answer: int | None = None) -> str:
        """Respond to the current order. store/ship need shelf+item+qty; count needs answer. Returns the outcome and the NEXT order.
        action=inspect with a shelf returns that shelf's true contents WITHOUT consuming the order (use it to verify)."""
        return env.act(action, shelf, item, qty, answer)[1]

    t0 = time.time()
    status = "completed"
    usage = None
    max_ctx = 0
    try:
        result = await agent.run(env.initial_observation(), usage_limits=UsageLimits(request_limit=(max_tool_calls or orders * 3) + 5, tool_calls_limit=max_tool_calls or orders * 3))
        usage = result.usage
        # transcript size at the last request = everything the model was shown on its final call
        from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest

        msgs = result.all_messages()
        last_req = max((i for i, m in enumerate(msgs) if isinstance(m, ModelRequest)), default=-1)
        max_ctx = len(ModelMessagesTypeAdapter.dump_json(msgs[: last_req + 1])) if last_req >= 0 else 0
    except Exception as exc:  # usage limit or model failure: score what was done
        status = f"stopped: {type(exc).__name__}: {str(exc)[:120]}"
    return BenchResult("react", getattr(model, "model_name", str(model)), orders, env.score, sum(1 for r in env.log if r["correct"]), len(env.log),
                       usage.requests if usage else 0, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0,
                       max_ctx, time.time() - t0, status, env.log, "", env.inspections, usage.cache_read_tokens if usage else 0, usage.cache_write_tokens if usage else 0)


def render(results: list[BenchResult]) -> str:
    lines = ["| mode | model | orders | score | correct | inspections | steps | model calls | input tok (uncached) | cache read | cache write | total shown | output tok | max ctx chars | wall s | status |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        total = r.input_tokens + r.cache_read_tokens + r.cache_write_tokens
        lines.append(f"| {r.mode} | {r.model} | {r.orders} | {r.score:.2f} | {r.correct} | {r.inspections} | {r.steps} | {r.model_calls} | {r.input_tokens:,} | {r.cache_read_tokens:,} | {r.cache_write_tokens:,} | {total:,} | {r.output_tokens:,} | {r.max_context_chars:,} | {r.wall_seconds:.0f} | {r.status} |")
    return "\n".join(lines)
