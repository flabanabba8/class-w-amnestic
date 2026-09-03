# Changelog

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

## 0.1.0 — 2026-09-03

Initial release of Class-W Mnestic, the SKILL.state runtime.

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
