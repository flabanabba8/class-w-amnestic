"""``mnestic`` CLI — inspect exactly what the agent believes and exactly what the model sees."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mnestic import __version__
from mnestic.agent.reasoner import Reasoner, ScriptedReasoner
from mnestic.config import RuntimeConfig
from mnestic.context.builder import ContextBuilder
from mnestic.memory.semantic import SemanticMemoryStore
from mnestic.models.archive import MemoryQuery
from mnestic.models.skill import SkillSpecification
from mnestic.observability.logging import configure_logging, maybe_configure_logfire
from mnestic.skills.loader import SkillRegistry
from mnestic.storage.db import Database
from mnestic.storage.store import Store
from mnestic.tools import default_registry


def _print(obj: Any, as_json: bool) -> None:
    if as_json or isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    else:
        print(obj)


def _open(cfg: RuntimeConfig) -> Store:
    return Store(Database(cfg.db_path))


def _reasoner(cfg: RuntimeConfig, skill: SkillSpecification | None, model: str | None) -> Reasoner:
    name = model or cfg.model
    if name == "mock":
        if skill is None or not skill.reasoner_script:
            raise SystemExit("--model mock requires a skill with a `reasoner_script` (deterministic skills only)")
        from mnestic.benchmarks.scripts import load_script

        return ScriptedReasoner(load_script(skill.reasoner_script))
    from mnestic.agent.pydantic_ai_reasoner import PydanticAIReasoner

    settings = {"timeout": cfg.model_timeout_seconds, **cfg.model_settings}
    return PydanticAIReasoner(name, output_mode=cfg.output_mode, retries=cfg.model_retries, model_settings=settings,
                              wall_clock_timeout=cfg.model_timeout_seconds)


def _runtime(cfg: RuntimeConfig, store: Store, reasoner: Reasoner):
    from mnestic.graph.runtime import Runtime

    return Runtime(cfg, store, reasoner=reasoner, tools=default_registry())


# ---- commands ------------------------------------------------------------------------------


def cmd_init(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    ws = cfg.resolved_workspace()
    skills_dir = cfg.skills_dirs[0] if cfg.skills_dirs else Path("skills")
    skills_dir.mkdir(parents=True, exist_ok=True)
    print(f"database: {cfg.db_path.resolve()} (fts5={'yes' if store.db.fts_enabled else 'no'})")
    print(f"workspace: {ws}")
    print(f"skills dir: {skills_dir.resolve()}")
    return 0


def cmd_skills_list(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    reg = SkillRegistry(cfg.skills_dirs)
    rows: list[dict[str, Any]] = [{"skill_id": s.skill_id, "version": s.version, "name": s.name, "tools": s.required_tools,
             "mock": bool(s.reasoner_script), "hash": s.content_hash[:12], "path": str(reg.path_of(s))} for s in reg.list()]
    if args.json:
        _print(rows, True)
    else:
        for r in rows:
            print(f"{r['skill_id']}@{r['version']:<8} {r['name']:<32} tools={','.join(r['tools']) or '-'} mock={'yes' if r['mock'] else 'no'}  {r['path']}")
        for e in reg.errors:
            print(f"ERROR {e}", file=sys.stderr)
    return 0


def cmd_run(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    reg = SkillRegistry(cfg.skills_dirs)
    skill = reg.get(args.skill)
    store = _open(cfg)
    reasoner = _reasoner(cfg, skill, args.model)
    rt = _runtime(cfg, store, reasoner)
    task = args.task if args.task is not None else sys.stdin.read()
    outcome = asyncio.run(rt.start(skill, task, max_steps=args.max_steps))
    _print(outcome.model_dump(mode="json"), args.json)
    return 0 if outcome.status.value in {"completed", "waiting_for_human", "paused"} else 1


def cmd_resume(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    meta = store.get_run(args.run_id)
    if meta is None:
        raise SystemExit(f"run {args.run_id} not found")
    skill = store.get_skill(meta.skill_id, meta.skill_version)
    reasoner = _reasoner(cfg, skill, args.model or (meta.model_name if meta.model_name == "mock" else None))
    rt = _runtime(cfg, store, reasoner)
    outcome = asyncio.run(rt.resume(args.run_id, human_input=args.input, max_steps=args.max_steps))
    _print(outcome.model_dump(mode="json"), args.json)
    return 0 if outcome.status.value in {"completed", "waiting_for_human", "paused"} else 1


def cmd_status(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    if args.run_id:
        meta = store.get_run(args.run_id)
        if meta is None:
            raise SystemExit(f"run {args.run_id} not found")
        latest = store.latest_step(args.run_id)
        info = meta.model_dump(mode="json")
        info["latest_step_phase"] = latest.phase if latest else None
        info["metrics"] = store.run_metrics(args.run_id)
        _print(info, args.json)
    else:
        runs = store.list_runs()
        if args.json:
            _print([r.model_dump(mode="json") for r in runs], True)
        else:
            for r in runs:
                print(f"{r.run_id}  {r.status:<18} {r.skill_id}@{r.skill_version:<8} step={r.last_step:<4} v={r.state_version:<4} {r.created_at:%Y-%m-%d %H:%M:%S}  {r.task_input[:50]!r}")
    return 0


def cmd_state(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    state = store.get_state_at_version(args.run_id, args.version) if args.version is not None else store.get_state(args.run_id)
    if state is None:
        raise SystemExit("state version not found")
    _print(state.model_view() if args.model_view else state.model_dump(mode="json"), True)
    return 0


def cmd_history(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    versions = store.list_state_versions(args.run_id)
    patches = {p["resulting_version"]: p for p in store.list_patches(args.run_id) if p["status"] == "applied"}
    rows = []
    for v in versions:
        p = patches.get(v["version"])
        rows.append({"version": v["version"], "step": v["step"], "bytes": v["state_bytes"], "created_at": v["created_at"],
                     "changes": p["changes"] if p else ["(runtime bookkeeping)" if v["version"] else "(initial)"]})
    if args.json:
        _print(rows, True)
    else:
        for r in rows:
            print(f"v{r['version']:<4} step={r['step']:<4} {r['bytes']:>6}B  {'; '.join(r['changes'])[:120]}")
    rejected = [p for p in store.list_patches(args.run_id) if p["status"] == "rejected"]
    if rejected and not args.json:
        print(f"\n{len(rejected)} rejected patch(es):")
        for p in rejected:
            print(f"  step={p['step']} {p['error_code']}: {p['error']}")
    return 0


def cmd_events(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    events = store.list_events(args.run_id, limit=args.limit, offset=args.offset, event_type=args.type, step=args.step)
    if args.json:
        _print([e.model_dump(mode="json") for e in events], True)
    else:
        for e in events:
            print(f"{e.seq:>5} step={e.step:<4} {e.event_type.value:<22} {e.event_id}  {e.summary[:100]}")
            if args.payload:
                print("      " + json.dumps(e.payload, default=str, ensure_ascii=False)[:2000])
    return 0


def _diff(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a:
                out.append({"path": p, "change": "added", "value": b[k]})
            elif k not in b:
                out.append({"path": p, "change": "removed", "value": a[k]})
            else:
                out += _diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(x, dict) and "id" in x for x in a + b):
        am, bm = {x["id"]: x for x in a}, {x["id"]: x for x in b}
        for k in sorted(set(am) | set(bm)):
            p = f"{path}[{k}]"
            if k not in am:
                out.append({"path": p, "change": "added", "value": bm[k]})
            elif k not in bm:
                out.append({"path": p, "change": "removed", "value": am[k]})
            else:
                out += _diff(am[k], bm[k], p)
    elif a != b:
        out.append({"path": path, "change": "changed", "from": a, "to": b})
    return out


def cmd_diff(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    a = store.get_state_at_version(args.run_id, args.version_a)
    b = store.get_state_at_version(args.run_id, args.version_b)
    if a is None or b is None:
        raise SystemExit("one of the versions does not exist")
    changes = _diff(a.model_view(), b.model_view())
    if args.json:
        _print(changes, True)
    else:
        for c in changes:
            if c["change"] == "changed":
                print(f"~ {c['path']}: {json.dumps(c['from'], default=str)[:80]} -> {json.dumps(c['to'], default=str)[:80]}")
            else:
                sign = "+" if c["change"] == "added" else "-"
                print(f"{sign} {c['path']}: {json.dumps(c['value'], default=str)[:120]}")
    return 0


def cmd_memory_search(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    from mnestic.memory.retrieval import ArchiveRetriever

    store = _open(cfg)
    q = MemoryQuery(query_type=args.type, text=args.query, limit=args.limit, event_type=args.event_type, event_id=args.event_id,
                    version=args.version)
    result = ArchiveRetriever(store).retrieve(args.run_id, q)
    if args.json:
        _print(result.model_dump(mode="json"), True)
    else:
        print(f"{len(result.events)} of {result.total_matches} matches {('(' + result.note + ')') if result.note else ''}")
        for e in result.events:
            print(f"[{e.event_id}] step={e.step} {e.event_type} {e.summary}\n    {e.excerpt[:400]}")
    return 0


def cmd_semantic(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    sem = SemanticMemoryStore(store)
    if args.semantic_cmd == "list":
        _print([m.model_dump(mode="json") for m in sem.list_all(args.category)], True)
    elif args.semantic_cmd == "promote":
        m = sem.promote(key=args.key, content=args.content, category=args.category or "general", source_run_id=args.run_id,
                        source_event_ids=args.event_id or [], promoted_by="cli")
        _print(m.model_dump(mode="json"), True)
    elif args.semantic_cmd == "forget":
        print("deleted" if sem.forget(args.key) else "not found")
    elif args.semantic_cmd == "search":
        _print([m.model_dump(mode="json") for m in sem.search(args.query)], True)
    return 0


def cmd_inspect_context(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    store = _open(cfg)
    meta = store.get_run(args.run_id)
    if meta is None:
        raise SystemExit(f"run {args.run_id} not found")
    if args.next:
        # What WOULD be sent next: current state + the pending observation of the latest open step.
        latest = store.latest_step(args.run_id)
        if latest is None or latest.observation_id is None:
            raise SystemExit("no pending step")
        skill = store.get_skill(meta.skill_id, meta.skill_version)
        obs = store.get_observation(latest.observation_id)
        assert skill is not None and obs is not None
        builder = ContextBuilder(tool_specs=default_registry().specs(skill.required_tools), max_retrieved_chars=cfg.max_retrieved_chars,
                                 max_retrieved_excerpt_chars=cfg.max_retrieved_excerpt_chars)
        ctx = builder.build(skill, store.get_state(args.run_id), obs, latest.retrieved)
        payload = ctx.model_dump(mode="json")
        label = f"(rebuilt for step {latest.step}, phase {latest.phase})"
    else:
        calls = store.get_model_calls(args.run_id, args.step)
        if not calls:
            raise SystemExit("no model calls recorded" + (f" for step {args.step}" if args.step is not None else ""))
        call = calls[-1]
        payload = json.loads(call["context_json"])
        payload["_model_call"] = {k: call[k] for k in ("call_id", "step", "attempt", "model_name", "input_tokens", "output_tokens", "requests", "status", "duration_ms", "created_at")}
        label = f"(as sent at step {call['step']}, attempt {call['attempt']}, model {call['model_name']})"
    if args.json:
        _print(payload, True)
        return 0
    print(f"# Model context {label}")
    print(f"# chars={payload['char_count']} approx_tokens={payload['approx_tokens']} state_version={payload['state_version']} "
          f"observation={payload['observation_id']} retrieved={len(payload['retrieved_event_ids'])}")
    if args.sections:
        for name, text in payload["sections"].items():
            print(f"\n# --- section {name} ({len(text)} chars) ---")
            print(text)
    else:
        print("\n# --- instructions ---")
        print(payload["instructions"])
        print("\n# --- prompt ---")
        print(payload["prompt"])
    return 0


def cmd_graph(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    from mnestic.graph.runtime import build_graph

    print(build_graph().render(direction="TB"))
    return 0


def cmd_doctor(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    import pydantic
    import pydantic_ai

    ok = True
    print(f"mnestic {__version__}")
    print(f"python {sys.version.split()[0]}")
    from importlib.metadata import version as _v

    print(f"pydantic {pydantic.VERSION}  pydantic-ai {pydantic_ai.__version__}  pydantic-graph {_v('pydantic-graph')}")
    print(f"sqlite {sqlite3.sqlite_version}")
    try:
        db = Database(cfg.db_path)
        from mnestic.storage.migrations import current_schema_version

        print(f"database ok: {cfg.db_path.resolve()} schema v{current_schema_version(db.conn)} fts5={'yes' if db.fts_enabled else 'no (LIKE fallback)'}")
    except Exception as exc:
        ok = False
        print(f"database ERROR: {exc}")
    ws = cfg.resolved_workspace()
    print(f"workspace: {ws} {'(exists)' if ws.is_dir() else '(MISSING)'}")
    reg = SkillRegistry(cfg.skills_dirs)
    print(f"skills: {len(reg.list())} found in {[str(p) for p in cfg.skills_dirs]}")
    for e in reg.errors:
        ok = False
        print(f"  skill ERROR: {e}")
    print(f"tools: {', '.join(default_registry().names())}")
    print(f"shell policy: {cfg.shell.mode}")
    print(f"model: {cfg.model} (output_mode={cfg.output_mode})")
    if cfg.model != "mock":
        try:
            from mnestic.agent.pydantic_ai_reasoner import build_model

            m = build_model(cfg.model)
            print(f"model resolves: {m.model_name} via {m.system}")
        except Exception as exc:
            print(f"model NOT usable: {exc}")
    print("logfire: " + ("enabled" if maybe_configure_logfire(cfg.logfire) else "off"))
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def cmd_benchmark(cfg: RuntimeConfig, args: argparse.Namespace) -> int:
    from mnestic.benchmarks.scaling import render_report, run_benchmark

    report = run_benchmark(steps=args.steps, observation_chars=args.observation_chars, checkpoints=None)
    text = render_report(report)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0 if report["bounded"] else 1


# ---- parser --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=argparse.SUPPRESS, help="SQLite path (default .mnestic/mnestic.db or MNESTIC_DB_PATH)")
    common.add_argument("--workspace", type=Path, default=argparse.SUPPRESS, help="workspace root (default cwd or MNESTIC_WORKSPACE)")
    common.add_argument("--skills-dir", action="append", type=Path, default=argparse.SUPPRESS, help="skills directory (repeatable)")
    common.add_argument("--log-level", default=argparse.SUPPRESS)
    common.add_argument("--log-json", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")
    p = argparse.ArgumentParser(prog="mnestic", description="SKILL.state agent runtime", parents=[common])

    def subparsers(parent: argparse.ArgumentParser, **kw: Any) -> Any:
        """add_subparsers whose every subcommand also accepts the global flags."""
        sp = parent.add_subparsers(**kw)
        orig = sp.add_parser

        def add_parser(name: str, **akw: Any) -> argparse.ArgumentParser:
            return orig(name, parents=[common], **akw)

        sp.add_parser = add_parser  # type: ignore[method-assign]
        return sp

    sub = subparsers(p, dest="cmd", required=True)

    s = sub.add_parser("init", help="create database and skills directory"); s.set_defaults(fn=cmd_init)
    s = sub.add_parser("run", help="start a run from a skill"); s.add_argument("skill"); s.add_argument("--task", help="task input (stdin if omitted)")
    s.add_argument("--model", help="PydanticAI model string or 'mock'"); s.add_argument("--max-steps", type=int); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("resume", help="resume a run from SQLite"); s.add_argument("run_id"); s.add_argument("--input", help="answer to a pending human-input request")
    s.add_argument("--model"); s.add_argument("--max-steps", type=int); s.set_defaults(fn=cmd_resume)
    s = sub.add_parser("status", help="list runs or show one"); s.add_argument("run_id", nargs="?"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("state", help="show the canonical execution state"); s.add_argument("run_id"); s.add_argument("--version", type=int)
    s.add_argument("--model-view", action="store_true", help="exactly as rendered for the model"); s.set_defaults(fn=cmd_state)
    s = sub.add_parser("history", help="state versions and patches"); s.add_argument("run_id"); s.set_defaults(fn=cmd_history)
    s = sub.add_parser("events", help="archive events"); s.add_argument("run_id"); s.add_argument("--type"); s.add_argument("--step", type=int)
    s.add_argument("--limit", type=int, default=100); s.add_argument("--offset", type=int, default=0); s.add_argument("--payload", action="store_true"); s.set_defaults(fn=cmd_events)
    s = sub.add_parser("diff", help="diff two state versions"); s.add_argument("run_id"); s.add_argument("version_a", type=int); s.add_argument("version_b", type=int); s.set_defaults(fn=cmd_diff)
    m = sub.add_parser("memory", help="archival memory"); ms = subparsers(m, dest="memory_cmd", required=True)
    s = ms.add_parser("search"); s.add_argument("run_id"); s.add_argument("query", nargs="?"); s.add_argument("--type", default="search",
        choices=["recent", "event", "events_by_type", "search", "state_at_version", "state_history", "observations", "tool_executions", "artifacts", "semantic"])
    s.add_argument("--limit", type=int, default=10); s.add_argument("--event-type"); s.add_argument("--event-id"); s.add_argument("--version", type=int); s.set_defaults(fn=cmd_memory_search)
    sm = sub.add_parser("semantic", help="durable semantic memory"); sms = subparsers(sm, dest="semantic_cmd", required=True)
    s = sms.add_parser("list"); s.add_argument("--category"); s.set_defaults(fn=cmd_semantic)
    s = sms.add_parser("promote"); s.add_argument("key"); s.add_argument("content"); s.add_argument("--category"); s.add_argument("--run-id"); s.add_argument("--event-id", action="append"); s.set_defaults(fn=cmd_semantic)
    s = sms.add_parser("forget"); s.add_argument("key"); s.set_defaults(fn=cmd_semantic)
    s = sms.add_parser("search"); s.add_argument("query"); s.set_defaults(fn=cmd_semantic)
    s = sub.add_parser("skills", help="skills"); ss = subparsers(s, dest="skills_cmd", required=True); ss.add_parser("list").set_defaults(fn=cmd_skills_list)
    s = sub.add_parser("inspect-context", help="show the exact model context (as sent, or as it would be sent next)")
    s.add_argument("run_id"); s.add_argument("--step", type=int); s.add_argument("--next", action="store_true"); s.add_argument("--sections", action="store_true"); s.set_defaults(fn=cmd_inspect_context)
    sub.add_parser("graph", help="render the lifecycle graph as Mermaid").set_defaults(fn=cmd_graph)
    sub.add_parser("doctor", help="environment check").set_defaults(fn=cmd_doctor)
    s = sub.add_parser("benchmark", help="context-scaling benchmark"); s.add_argument("--steps", type=int, default=1000); s.add_argument("--observation-chars", type=int, default=400)
    s.add_argument("--output"); s.set_defaults(fn=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, default in (("db", None), ("workspace", None), ("skills_dir", None), ("log_level", None), ("log_json", False), ("json", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    cfg = RuntimeConfig.from_env(db_path=args.db, workspace_root=args.workspace, skills_dirs=args.skills_dir, log_level=args.log_level,
                                 log_json=True if args.log_json else None)
    configure_logging(cfg.log_level, json_output=cfg.log_json)
    try:
        return int(args.fn(cfg, args))
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
