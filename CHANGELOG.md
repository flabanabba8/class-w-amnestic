# Changelog

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

## Unreleased

Found by running `codebase-research` on this repository with four models through 9Router
(Kimi K3, Sonnet 5, Haiku 4.5, Codex gpt-5.6-luna) and reading the archived traces:

- Tool-result observations now carry the request that produced them (`request={tool, arguments}`),
  rendered in the `<latest_observation>` header; `list_directory` and `search_text` outputs are
  labelled with their target/pattern/mode. A stateless step must be able to interpret an
  observation without a previous turn (a mislabelled listing caused a 20-step loop).
- `search_text` treats patterns as regex by default; invalid regex falls back to literal.
- Sliding-window repeated-action loop guard (`max_repeated_actions` within `action_window`);
  loop trips fail the run independently of the (now correctly resetting) failure streak.
- Output contract: one tool call per step; finish only via `completion`; clearer rejection
  message when a patch tries to set a terminal status.
- `docs/PAPER_ALIGNMENT.md`: paper-and-brief alignment review (what matches, what is stricter, gaps).
- `MNESTIC_MODEL_TIMEOUT` (default 300 s) per model request.
- `codebase-research` skill: pacing guidance.
- Result: zero wasted tool calls for all four models; Kimi 12→9 steps, 141K→87K tokens;
  Luna never-completed → 14 clean steps.
- Warehouse long-horizon benchmark (`benchmarks/warehouse.py`, `scripts/warehouse_bench.py`): deterministic
  order stream, `inspect` verification, live ReAct baseline; 300-order results for Sonnet 5, Kimi K3, Gemma 4.
- Output-schema slimming: per-skill `allowed_ops` (default core set of 16, `["*"]` for all) and `allowed_actions`;
  no titles/descriptions/string bounds in model-facing schemas; discriminators forced `required` (llama.cpp grammar).
- `Runtime.start(initial_ops=…)` seeds Σ₀; runtime feedback repeats the observation it corrects; idempotent
  promote/reject; wall-clock step timeout; prompt-cache token accounting (migration 2); `MNESTIC_MODEL_SETTINGS`.
- Typed domain state: skill `domain_schema` (JSON Schema, validated on every write), `set_path`/`adjust_path`/
  `delete_path` ops (runtime arithmetic), tool `state_effects` applied deterministically by the runtime.
- Lean state rendering (`progress` block instead of counters/budgets; identity fields and empties omitted).

## 0.1.0 — 2026-09-03

Initial release of Class W Amnestic, the SKILL.state runtime.

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
