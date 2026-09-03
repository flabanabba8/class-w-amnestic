# AGENTS.md — guidance for coding agents working on Class-W Mnestic

## Purpose

Class-W Mnestic (Python package `mnestic`) is a long-horizon autonomous agent runtime implementing the
**SKILL.state** architecture (Badhe, Tiwari, Chung — "SKILL.state: Scalable
Long-Horizon Agent Skills", arXiv:2608.26263). The model reasons over a small,
explicit, validated **Execution State** rather than an ever-growing chat
transcript. It is built on Python, PydanticAI, pydantic-graph, Pydantic and
SQLite, managed with `uv`, tested with `pytest`.

The most important property of this codebase is its **information
architecture**, not its feature count. Read `docs/CONTEXT_INVARIANTS.md` before
touching anything in `context/`, `agent/`, or `graph/`.

## Architectural invariants (do not regress these)

1. **Passing an accumulated full conversation/message history into each
   reasoning step is an architectural regression.** Never call
   `agent.run(..., message_history=...)` with material from previous steps.
   Never feed `result.all_messages()` / `result.new_messages()` back into a
   later step. PydanticAI may produce messages *within* one step (e.g. output
   validation retries); those are archived after the step and then discarded.
2. Every ordinary reasoning step receives exactly four things, assembled by
   `mnestic.context.ContextBuilder`:
   - the immutable **Skill Specification**,
   - the canonical **ExecutionState** (current version),
   - the **newest Observation**,
   - **explicitly retrieved** archival evidence (only when the agent asked for it
     in the previous step).
   Nothing else. `ContextBuilder` has no access to storage by construction.
3. **The archive is NOT default model context.** Archival memory
   (`archive_events`, observations, tool executions, state versions) is
   append-only and only enters a prompt through an explicit `MemoryQuery` whose
   result is bounded, recorded as an archive event, and injected once.
4. **ExecutionState is the canonical operational memory.** It is Pydantic
   validated, versioned (`state_version`), mutated only through validated
   `StatePatch` operations, committed transactionally with optimistic
   concurrency (`expected_state_version` must equal the stored version).
   A model can never overwrite state with arbitrary JSON.
5. Facts vs hypotheses: `add_fact`/`promote_hypothesis` require evidence event
   ids that exist in the archive. A hypothesis cannot silently become a fact.
6. Working state is size-bounded (`StateLimits`). Overflow is handled by explicit,
   archived compaction/spill (`state.compaction` events), never by silent
   truncation. Large tool output goes to the archive + artifacts; the
   observation carries a bounded excerpt plus the event id to retrieve the rest.
7. Intermediate chain-of-thought is never requested, stored or re-injected.
   `AgentDecision.rationale_summary` is a short, externally safe justification.
8. Skills are immutable during a run (content hash checked on resume).
9. Every state change, observation, action, tool execution, model call and
   retrieval is archived with provenance. Old state versions are recoverable.
10. Runs must be resumable from SQLite alone (no transcript replay).

If a proposed feature makes the default prompt contain more *historical*
material, redesign it. Ask: "Does this preserve current-state-based reasoning,
or does it turn the agent back into a giant conversation?"

## Repository structure

```
src/mnestic/
  config.py          RuntimeConfig (env + CLI), StateLimits, ShellPolicy
  models/            Pydantic domain models (skill, state, patch, decision, observation, action, archive)
  state/             patch application, status transitions, size limits/compaction
  storage/           SQLite schema, migrations, repositories (transactional, optimistic concurrency)
  memory/            archival retrieval interface + SQLite/FTS5 implementation; semantic memory
  context/           ContextBuilder — THE only place model context is assembled
  agent/             Reasoner protocol; PydanticAI reasoner; scripted (mock) reasoner; model factory
  graph/             pydantic-graph lifecycle nodes + Runtime orchestrator (start/resume)
  tools/             typed tool abstraction, workspace-restricted filesystem tools, shell policy, memory tool
  skills/            skill loading (skill.yaml + SKILL.md), registry
  observability/     structured logging, metrics, optional logfire
  cli/               `mnestic` CLI
  benchmarks/        context-scaling benchmark + ReAct baseline simulator
skills/              example skills (versioned directories)
tests/unit|integration|benchmarks
docs/                architecture, state model, persistence, memory, invariants, crash recovery, security, notes
```

## Commands

```
uv sync                         # install
uv run pytest                   # full suite (no API keys needed)
uv run pytest tests/benchmarks  # scaling benchmark tests
uv run ruff check src tests     # lint
uv run mypy src                 # types
uv run mnestic doctor        # environment check
uv run mnestic run <skill> --task "..." --model mock
uv run mnestic inspect-context <run-id> [--step N]
uv run python scripts/benchmark.py   # writes docs/BENCHMARK_REPORT.md
```

## Testing requirements

- The full suite must pass without network access or API credentials.
  Set `MNESTIC_LIVE_TESTS=1` plus provider credentials to enable optional
  live tests (`tests/integration/test_live_model.py`).
- Any change to `context/` must keep `tests/unit/test_context_builder.py`
  and `tests/benchmarks/test_context_scaling.py` passing; these are the
  regression guards for invariants 1–3.
- Any new patch op needs: validation test, apply test, rejection test.
- Any new tool needs: archive test + failure-as-observation test + workspace
  boundary test if it touches the filesystem.
- Crash/resume tests (`tests/integration/test_crash_resume.py`) must cover any
  new lifecycle phase you add.

## Database migration policy

- Schema lives in `src/mnestic/storage/migrations.py` as an ordered list of
  numbered migrations. Never edit an applied migration; add a new one.
- Every migration is applied inside a transaction and recorded in
  `schema_migrations`.
- Keep important entities relational and queryable. JSON columns are for
  structured sub-documents (state snapshots, patches, payloads), not for the
  whole system.
- Foreign keys are ON; WAL mode is used.

## Documentation expectations

- `DEVELOPMENT_STATE.md` holds only *current* development state (objective,
  decisions, completed components, blockers, open questions, next actions).
  It is not a diary — git history is the diary.
- API discrepancies vs. the original spec go into
  `docs/IMPLEMENTATION_NOTES.md`.
- Security boundaries are documented in `docs/SECURITY.md`; do not overstate
  them.
- Keep `docs/CONTEXT_INVARIANTS.md` in sync with any change to context
  construction.
