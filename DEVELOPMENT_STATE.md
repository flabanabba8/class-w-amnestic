# DEVELOPMENT_STATE.md

Current development state only. Git history is the diary.

## Objective

v1 SKILL.state runtime (`mnestic`): validated, bounded-context long-horizon agent
loop on PydanticAI + pydantic-graph + SQLite. **Status: v0.1.0 complete; all acceptance
criteria in the original brief met (see docs/AUDIT.md).**

## Verified architectural decisions

- Upstream: pydantic-ai-slim 2.37.0, pydantic-graph 2.37.0, pydantic 2.13.5, Python 3.12, SQLite 3.53 (FTS5).
- pydantic-graph 2.x has no persistence; lifecycle = `GraphBuilder` + `BaseNode`s with a single `Enter` dispatcher; durability = our `steps` table.
- One fresh `Agent.run(prompt, instructions=skill+contract)` per step; never `message_history`; per-step messages archived to `model_calls.raw_messages_json`.
- `ContextBuilder(skill, state, observation, retrieved)` is the only prompt assembler; it has no storage import (test-enforced).
- `StatePatch` = strict discriminated-union ops; `apply_patch` pure + atomic; commit with `UPDATE … WHERE state_version=?` (stale → recorded rejection + pause).
- Facts need archive-verified evidence ids; hypotheses promote only via `promote_hypothesis`.
- Limits: reject with guidance for curated lists / bytes; automatic archived compaction for bookkeeping lists.
- Step close + next step open is one transaction; tool execution is at-least-once (documented).
- Runtime-owned counters committed as their own state versions.

## Completed components

models · state/apply · storage (schema v1, Store) · memory (ArchiveRetriever, SemanticMemoryStore) · context/builder ·
agent (PydanticAIReasoner, ScriptedReasoner) · graph (10 nodes, Runtime.start/resume) · tools (6, workspace-confined, shell policy) ·
skills loader + 2 example skills with mock scripts · CLI (15 commands) · observability (structured logging, optional logfire) ·
benchmark + ReAct simulator · 83 tests (unit/integration/benchmark, no credentials) · docs (README, ARCHITECTURE, STATE_MODEL,
PERSISTENCE, MEMORY, CONTEXT_INVARIANTS, CRASH_RECOVERY, SECURITY, IMPLEMENTATION_NOTES, BENCHMARK_REPORT, AUDIT) · AGENTS.md.

## Current blockers

- None.

## Open design questions

- Multi-process lease for resume (currently: optimistic concurrency only).
- Whether `ContinueAction` should be rate-limited by wall clock as well as count.
- Semantic memory is queried only on demand; should skills be able to declare "always inject these keys" (bounded)? Deferred — risk of prompt creep.

## Failing tests

- None (`uv run pytest`: 83 passed, 1 skipped — the optional live-model test).

## Next actions

1. Try a live model end-to-end (`MNESTIC_LIVE_TESTS=1`) and tune the output contract wording from real decisions.
2. Add `runs.lease_*` migration for cooperative multi-process resume.
3. Stage 3+: semantic retriever behind `Retriever`; richer tools; MCP; subagents (see docs/ARCHITECTURE.md roadmap).
