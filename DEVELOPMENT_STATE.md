# DEVELOPMENT_STATE.md

Current development state only. Git history is the diary.

## Objective

Deliver a working v1 SKILL.state runtime (`skillstate`) in Python with
PydanticAI + pydantic-graph + SQLite, validated by tests and a token-scaling
benchmark.

## Verified architectural decisions

- Upstream versions: pydantic-ai-slim 2.37.0, pydantic-graph 2.37.0, pydantic 2.13.5, Python 3.12 (uv-managed). SQLite 3.53 with FTS5.
- pydantic-graph 2.x has NO built-in state persistence; the lifecycle graph is built with `GraphBuilder` + `BaseNode` classes and all durability is our own SQLite layer.
- Each reasoning step = fresh `Agent.run(prompt, instructions=...)` with no `message_history`. Verified with `FunctionModel` that a fresh run sends exactly one `ModelRequest`.
- `Reasoner` protocol decouples the graph from PydanticAI; `ScriptedReasoner` serves tests/benchmarks.
- Structured output: PydanticAI `output_type=AgentDecision` (tool mode default, configurable native/prompted); `expected_state_version` mismatch raises `ModelRetry` inside the step.
- State mutation only via discriminated-union `StatePatch` ops; commit via `UPDATE ... WHERE state_version = expected` (optimistic concurrency).
- Step lifecycle phases persisted in `steps` table drive crash resume.

## Completed components

- Research of paper + upstream APIs (spikes in scratchpad, findings recorded here and in docs/IMPLEMENTATION_NOTES.md)
- Project skeleton, AGENTS.md, invariants doc

## In progress

- Phase 2 vertical slice: models → patch apply → storage → context builder → scripted reasoner → graph loop.

## Current blockers

- None.

## Open design questions

- Whether `ContinueAction` (reason again without external action) should exist; currently allowed but bounded by `max_consecutive_continues`.

## Failing tests

- None yet (no tests written).

## Next actions

1. Domain models (`models/`).
2. Patch application + limits (`state/`).
3. SQLite schema + repositories (`storage/`).
4. ContextBuilder + scripted reasoner + graph loop; first end-to-end test.
