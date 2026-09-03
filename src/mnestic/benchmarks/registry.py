"""Service-registry long-horizon benchmark — a second, different task for the generic tool-fact ledger.

A registry of services (port, status, dependencies) receives sequential orders: mutations (set_port, set_status,
add_dep, remove_dep) and queries (port, status, deps, *dependents* — a reverse-graph question that needs aggregation).
The ``registry`` tool behaves like any real API: it returns the resource it touched as ``facts``; the runtime keeps the
latest record per service under ``domain.registry.<service>``. There is no registry-specific code in the runtime.
"""

from __future__ import annotations

import json
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.settings import ModelSettings

from mnestic.benchmarks.warehouse import BenchResult
from mnestic.models.skill import SkillSpecification
from mnestic.tools.base import Tool, ToolContext, ToolResult

NAMES = ["auth", "billing", "cart", "catalog", "gateway", "inventory", "ledger", "mailer", "notify", "orders", "payments",
         "pricing", "reports", "search", "sessions", "shipping", "users", "webhooks"]


@dataclass
class Order:
    index: int
    kind: str
    service: str
    arg: str | int | None = None

    def text(self) -> str:
        k = self.kind
        if k == "set_port":
            return f"ORDER #{self.index}: set the port of {self.service} to {self.arg}"
        if k == "set_status":
            return f"ORDER #{self.index}: mark {self.service} as {self.arg}"
        if k == "add_dep":
            return f"ORDER #{self.index}: make {self.service} depend on {self.arg}"
        if k == "remove_dep":
            return f"ORDER #{self.index}: remove the dependency of {self.service} on {self.arg}"
        if k == "port":
            return f"ORDER #{self.index}: query — what port is {self.service} on?"
        if k == "status":
            return f"ORDER #{self.index}: query — what is the status of {self.service}?"
        if k == "deps":
            return f"ORDER #{self.index}: query — which services does {self.service} depend on?"
        return f"ORDER #{self.index}: query — which services depend on {self.service}?"


@dataclass
class RegistryEnv:
    services: int = 12
    orders: int = 200
    seed: int = 7
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    order_list: list[Order] = field(default_factory=list)
    cursor: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)  # noqa: S311 - reproducible benchmark
        names = NAMES[: self.services]
        for i, n in enumerate(names):
            deps = rng.sample([x for x in names if x != n], k=rng.randint(0, 2))
            self.records[n] = {"port": 8000 + i, "status": "up", "deps": sorted(deps)}
        for i in range(1, self.orders + 1):
            r = rng.random()
            svc = rng.choice(names)
            if r < 0.15:
                self.order_list.append(Order(i, "set_port", svc, rng.randint(7000, 9999)))
            elif r < 0.27:
                self.order_list.append(Order(i, "set_status", svc, rng.choice(["up", "down", "degraded"])))
            elif r < 0.40:
                self.order_list.append(Order(i, "add_dep", svc, rng.choice([x for x in names if x != svc])))
            elif r < 0.50:
                self.order_list.append(Order(i, "remove_dep", svc, rng.choice([x for x in names if x != svc])))
            elif r < 0.65:
                self.order_list.append(Order(i, "port", svc))
            elif r < 0.75:
                self.order_list.append(Order(i, "status", svc))
            elif r < 0.87:
                self.order_list.append(Order(i, "deps", svc))
            else:
                self.order_list.append(Order(i, "dependents", svc))

    def current_order(self) -> Order | None:
        return self.order_list[self.cursor] if self.cursor < len(self.order_list) else None

    def dependents(self, svc: str) -> list[str]:
        return sorted(n for n, r in self.records.items() if svc in r["deps"])

    def initial_observation(self) -> str:
        lines = [f"Service registry with {len(self.records)} services. Initial records:"]
        for n, r in self.records.items():
            lines.append(f"  {n}: port={r['port']} status={r['status']} deps={','.join(r['deps']) or '-'}")
        o = self.current_order()
        lines += ["", "Your state's domain.registry holds the latest record per service (kept by the runtime from tool results).", o.text() if o else ""]
        return "\n".join(lines)

    def initial_ops(self) -> list[dict[str, Any]]:
        return [{"op": "set_path", "path": f"registry.{n}", "value": dict(r)} for n, r in self.records.items()]

    def act(self, action: str, service: str | None, value: Any, answer: Any) -> tuple[bool, str, dict[str, dict[str, Any]]]:
        """Apply a tool call to the pending order. Returns (correct, message, facts)."""
        o = self.current_order()
        if o is None:
            return False, "no pending order", {}
        facts: dict[str, dict[str, Any]] = {}
        if action == "get":
            if service in self.records:
                facts[service] = dict(self.records[service])
                return True, f"GET {service}: {json.dumps(self.records[service])}\n{o.text()}", facts  # does not consume the order
            return False, f"GET: unknown service {service}\n{o.text()}", {}
        ok, msg = False, ""
        mutation = o.kind in ("set_port", "set_status", "add_dep", "remove_dep")
        if mutation:
            if action != o.kind or service != o.service or str(value) != str(o.arg):
                msg = f"WRONG: expected {o.kind} {o.service} {o.arg}, got {action} {service} {value}"
            else:
                rec = self.records[o.service]
                if o.kind == "set_port":
                    rec["port"] = int(o.arg)  # type: ignore[arg-type]
                elif o.kind == "set_status":
                    rec["status"] = str(o.arg)
                elif o.kind == "add_dep":
                    if o.arg not in rec["deps"]:
                        rec["deps"] = sorted([*rec["deps"], str(o.arg)])
                else:
                    rec["deps"] = [d for d in rec["deps"] if d != o.arg]
                ok, msg = True, f"OK: {o.kind} {o.service} -> {json.dumps(rec)}"
                facts[o.service] = dict(rec)
        else:
            if action != "answer":
                msg = f"WRONG: expected an answer to the query, got {action}"
            else:
                truth: Any
                if o.kind == "port":
                    truth = self.records[o.service]["port"]
                    ok = str(answer).strip() == str(truth)
                elif o.kind == "status":
                    truth = self.records[o.service]["status"]
                    ok = str(answer).strip().lower() == truth
                elif o.kind == "deps":
                    truth = self.records[o.service]["deps"]
                    ok = sorted(_split(answer)) == truth
                else:
                    truth = self.dependents(o.service)
                    ok = sorted(_split(answer)) == truth
                msg = f"OK: {o.kind} of {o.service} is {truth}" if ok else f"WRONG: {o.kind} of {o.service} is {truth}, you said {answer}"
        self.log.append({"order": o.index, "kind": o.kind, "correct": ok, "message": msg})
        self.cursor += 1
        nxt = self.current_order()
        return ok, msg + ("\n" + nxt.text() if nxt else "\nALL ORDERS DONE — submit completion."), facts

    @property
    def score(self) -> float:
        return sum(1 for r in self.log if r["correct"]) / max(1, len(self.log))


def _split(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(a).strip() for a in answer]
    text = str(answer or "").strip()
    if text in ("", "-", "none", "None", "[]"):
        return []
    return [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]


class RegistryTool(Tool):
    name: ClassVar[str] = "registry"
    description: ClassVar[str] = (
        "Service registry API. Mutations: set_port(service, value) / set_status(service, value) / add_dep(service, value) / "
        "remove_dep(service, value) — each returns the updated record. Queries are answered with action=answer and `answer` "
        "(a number, a status, or a comma-separated list; empty list = '-'). action=get(service) returns a record without consuming the order."
    )

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Literal["set_port", "set_status", "add_dep", "remove_dep", "answer", "get"]
        service: str | None = None
        value: str | int | None = None
        answer: str | None = Field(default=None, description="for queries: number, status, or comma-separated list ('-' for none)")

    def __init__(self, env: RegistryEnv):
        self.env = env

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        ok, msg, facts = self.env.act(args.action, args.service, args.value, args.answer)
        return ToolResult(ok=True, output=msg, data={"correct": ok, "orders_done": self.env.cursor}, facts=facts)


REGISTRY_SKILL = SkillSpecification(
    skill_id="service-registry", name="Service registry operations", version="1.0.0",
    description="Apply a stream of registry mutations and answer queries about ports, statuses and the dependency graph.",
    required_tools=["registry"],
    instructions=(
        "You operate a service registry. Each observation gives the outcome of your last action and the NEXT order.\n"
        "`domain.registry.<service>` in your state is the latest record for each service — {port, status, deps} — kept by the runtime "
        "from what the registry tool returns; you never edit it.\n"
        "Mutation orders: call the matching registry action with service and value. Query orders: call action=answer with `answer`: "
        "port -> the number; status -> the word; deps -> that service's deps as a comma-separated list ('-' if none); "
        "dependents -> every service whose deps include it, comma-separated ('-' if none) — scan all records.\n"
        "One registry call per order; an empty state_patch is fine. When the observation says ALL ORDERS DONE, submit completion."
    ),
    completion_criteria=["all orders processed"], phases=["operating", "done"], initial_phase="operating", default_max_steps=3000,
    allowed_ops=["set_observation_summary", "set_phase"], allowed_actions=["tool"],
    domain_schema={"type": "object", "properties": {"registry": {"type": "object"}}, "required": ["registry"]},
)


def registry_script(context: Any) -> dict[str, Any]:
    """Optimal scripted policy reading the runtime-kept ledger (validates the harness)."""
    from mnestic.benchmarks.scripts import observation_text, parse_state

    state = parse_state(context)
    v = state["state_version"]
    obs = observation_text(context)
    ledger = state["domain"]["registry"]
    if "ALL ORDERS DONE" in obs:
        return {"rationale_summary": "done", "state_patch": {"expected_state_version": v, "ops": []}, "completion": {"outcome": "success", "summary": "done"}}
    if "ORDER #" not in obs:
        obs = state.get("last_observation_summary") or ""
    m = re.search(r"ORDER #\d+: (.*)$", obs, re.M)
    assert m is not None, obs[:100]
    line = m.group(1)
    ops = [{"op": "set_observation_summary", "summary": m.group(0)}]
    args: dict[str, Any]
    if line.startswith("set the port of"):
        svc, port = re.match(r"set the port of (\w+) to (\d+)", line).groups()  # type: ignore[union-attr]
        args = {"action": "set_port", "service": svc, "value": int(port)}
    elif line.startswith("mark"):
        svc, st = re.match(r"mark (\w+) as (\w+)", line).groups()  # type: ignore[union-attr]
        args = {"action": "set_status", "service": svc, "value": st}
    elif line.startswith("make"):
        svc, dep = re.match(r"make (\w+) depend on (\w+)", line).groups()  # type: ignore[union-attr]
        args = {"action": "add_dep", "service": svc, "value": dep}
    elif line.startswith("remove the dependency"):
        svc, dep = re.match(r"remove the dependency of (\w+) on (\w+)", line).groups()  # type: ignore[union-attr]
        args = {"action": "remove_dep", "service": svc, "value": dep}
    elif "what port is" in line:
        svc = re.search(r"what port is (\w+)", line).group(1)  # type: ignore[union-attr]
        args = {"action": "answer", "answer": str(ledger[svc]["port"])}
    elif "status of" in line:
        svc = re.search(r"status of (\w+)", line).group(1)  # type: ignore[union-attr]
        args = {"action": "answer", "answer": ledger[svc]["status"]}
    elif "does" in line and "depend on" in line:
        svc = re.search(r"which services does (\w+) depend on", line).group(1)  # type: ignore[union-attr]
        args = {"action": "answer", "answer": ",".join(ledger[svc]["deps"]) or "-"}
    else:
        svc = re.search(r"which services depend on (\w+)", line).group(1)  # type: ignore[union-attr]
        args = {"action": "answer", "answer": ",".join(sorted(n for n, r in ledger.items() if svc in r["deps"])) or "-"}
    return {"rationale_summary": "op", "state_patch": {"expected_state_version": v, "ops": ops}, "action": {"kind": "tool", "tool_name": "registry", "arguments": args}}


async def run_skillstate(reasoner: Any, *, orders: int = 200, services: int = 12, seed: int = 7, db_path: Any = None) -> BenchResult:
    from mnestic.config import RuntimeConfig
    from mnestic.graph.runtime import Runtime
    from mnestic.storage.db import Database
    from mnestic.storage.store import Store
    from mnestic.tools.base import ToolRegistry

    env = RegistryEnv(services=services, orders=orders, seed=seed)
    tmp = tempfile.mkdtemp()
    db_path = Path(db_path or Path(tmp) / "reg.db")
    cfg = RuntimeConfig(db_path=db_path, workspace_root=Path(tmp), max_consecutive_continues=2)
    store = Store(Database(db_path))
    tools = ToolRegistry()
    tools.register(RegistryTool(env))
    t0 = time.time()
    out = await Runtime(cfg, store, reasoner=reasoner, tools=tools).start(REGISTRY_SKILL, env.initial_observation(), max_steps=orders * 3, initial_ops=env.initial_ops())
    m = store.run_metrics(out.run_id)
    return BenchResult("skillstate", getattr(reasoner, "model_name", "?"), orders, env.score, sum(1 for r in env.log if r["correct"]), out.steps,
                       m["model_calls"] or 0, m["input_tokens"] or 0, m["output_tokens"] or 0, m["max_context_chars"] or 0, time.time() - t0,
                       out.status.value, env.log, f"run_id={out.run_id} db={db_path}", 0, m["cache_read_tokens"] or 0, m["cache_write_tokens"] or 0)


async def run_react(model: Any, *, orders: int = 200, services: int = 12, seed: int = 7, model_settings: dict[str, Any] | None = None) -> BenchResult:
    env = RegistryEnv(services=services, orders=orders, seed=seed)
    instructions = REGISTRY_SKILL.instructions.replace(
        "`domain.registry.<service>` in your state is the latest record for each service — {port, status, deps} — kept by the runtime "
        "from what the registry tool returns; you never edit it.", "Track each service's record {port, status, deps} yourself from the tool results."
    ).replace("an empty state_patch is fine. ", "") + "\nWhen ALL ORDERS DONE, reply with the single word DONE."
    agent: Agent[None, str] = Agent(model, instructions=instructions, retries=2, model_settings=ModelSettings(**(model_settings or {})) if model_settings else None)  # type: ignore[typeddict-item]

    @agent.tool_plain
    def registry(action: Literal["set_port", "set_status", "add_dep", "remove_dep", "answer", "get"], service: str | None = None,
                 value: str | int | None = None, answer: str | None = None) -> str:
        """Service registry API: mutations return the updated record; queries are answered with action=answer and `answer`; get(service) returns a record without consuming the order."""
        return env.act(action, service, value, answer)[1]

    t0 = time.time()
    status, usage, max_ctx = "completed", None, 0
    try:
        result = await agent.run(env.initial_observation(), usage_limits=UsageLimits(request_limit=orders * 3 + 5, tool_calls_limit=orders * 3))
        usage = result.usage
        from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest

        msgs = result.all_messages()
        last_req = max((i for i, m in enumerate(msgs) if isinstance(m, ModelRequest)), default=-1)
        max_ctx = len(ModelMessagesTypeAdapter.dump_json(msgs[: last_req + 1])) if last_req >= 0 else 0
    except Exception as exc:
        status = f"stopped: {type(exc).__name__}: {str(exc)[:120]}"
    return BenchResult("react", getattr(model, "model_name", str(model)), orders, env.score, sum(1 for r in env.log if r["correct"]), len(env.log),
                       usage.requests if usage else 0, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0, max_ctx, time.time() - t0,
                       status, env.log, "", 0, usage.cache_read_tokens if usage else 0, usage.cache_write_tokens if usage else 0)
