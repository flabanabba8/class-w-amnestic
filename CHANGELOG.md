# Changelog

## 0.1.0 — 2026-09-03

Initial release of the SKILL.state runtime.

- Typed `SkillSpecification`, `ExecutionState`, `StatePatch` (30 ops), `AgentDecision`,
  `Observation`, actions, archive events, memory queries.
- Validated, atomic patch application with evidence checks, status rules, size limits and
  archived compaction.
- SQLite persistence (WAL, FK, FTS5) with migrations, optimistic-concurrency state commits,
  append-only archive, per-step lifecycle phases, metrics.
- `ContextBuilder` producing an inspectable, bounded `ModelContext`.
- PydanticAI reasoner (tool/native/prompted output, in-step retries, usage capture) and a
  scripted reasoner for tests/benchmarks.
- pydantic-graph lifecycle with crash-safe resume at every phase, human-input pause,
  explicit archival retrieval, failure feedback loop.
- Workspace-confined tools (`read_text_file`, `list_directory`, `search_text`,
  `write_workspace_file`, `run_shell` with policy, `archival_memory_search`).
- Semantic memory with audited promotion.
- CLI: `init run resume status state history events diff memory semantic skills
  inspect-context graph doctor benchmark`.
- Context-scaling benchmark with ReAct-style baseline; 83 tests without credentials.
