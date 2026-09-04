# DEVELOPMENT_STATE.md

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263).*

Current development state only. Git history is the diary. Last updated 2026-09-04.

## Objective

Class W Amnestic (`mnestic`): a SKILL.state runtime — bounded per-step context, validated execution state, append-only
archive with explicit retrieval, durable resume — that beats or matches transcript agents on long tasks at a fraction of
the tokens. **Status: v0.1 complete; long-horizon claim validated on two tasks with three models; repo at `0d3b008` on
GitHub, Codeberg, GitLab; 113 tests, ruff + mypy clean.**

## What is true now (verified)

- **Renamed 2026-09-04: Class-W Mnestic → Class W Amnestic.** Forge repos are now `flabanabba8/class-w-amnestic` (GitHub),
  `antimemetics-division/class-w-amnestic` (Codeberg, public URL), `flabanabba8/class-w-amnestic` (GitLab); old URLs redirect.
  Distribution name is `class-w-amnestic`; the Python package, `mnestic` CLI and `MNESTIC_*` env vars are unchanged.
  README tagline reworded for the amnestic framing (per-step transcript wipe, archive keeps everything); CHANGELOG has the
  rename entry under Unreleased. Local git remotes and the session memory file point at the new URLs.
- Context per step is flat regardless of horizon (mock 1000 steps; live 300 orders: 5–8K chars at order 300 = order 1).
- **Runtime keeps the books (generic):** `ToolResult.facts` → `domain.<tool>.<key>` with provenance, bounded per tool
  (`ledger_entries_per_tool`). Built-in tools report facts. Typed `domain_schema` + `set_path`/`adjust_path`/`delete_path`
  for skill-owned state. No task-specific bookkeeping code exists in `src/mnestic/`.
- **Decision schema is shaped per skill:** `allowed_ops` (default core set of 19, `["*"]` = all), `allowed_actions`, one typed
  `ToolAction` variant per required tool (`Reasoner.bind(skill, tools)` before start/resume), no titles/descriptions/string
  bounds in model-facing schemas, discriminators forced required (llama.cpp grammar), cached one-line op reference in the
  contract.
- **Small models work in `native` (grammar) mode** with `MNESTIC_MODEL_SETTINGS='{"openai_reasoning_effort":"low","max_tokens":4000}'`;
  budget exhaustion retries with 2× budget then counts as a stall (pause), not a decision failure.
- Runtime feedback repeats the observation it corrects; promote/reject are idempotent; loop guard keys on (action, observation)
  in a sliding window; provider timeouts have their own budget; `Runtime.start(initial_ops=…)` seeds Σ₀.
- Results (`docs/WAREHOUSE_BENCHMARK.md`, `docs/REGISTRY_BENCHMARK.md`): warehouse 300 — Sonnet 1.00 vs ReAct 0.96 (1.0M vs
  8.5M tokens, 15 vs 97 min); Gemma 4 12B 1.00 vs ReAct 0.83. Registry 200 — Sonnet 0.97 vs 0.98 (2.6× fewer tokens, not
  faster); Gemma 1.00 vs 0.99. Every "small model can't" number before 2026-09-04 was measured with reasoning off.
- Measured-at-the-provider rule (`docs/CONTEXT_INVARIANTS.md` I8): per-call cost = context + output schema + route prefix;
  9Router's Claude Code route adds ~4.4K tokens/call that is not ours.

## Known limitations / honest gaps

- The ledger helps where tools return records. Where tools return prose (research, code review), facts are the model's
  judgments and the built-in facts (hashes, match counts) are bookkeeping only — untested whether that changes accuracy.
- `dependents`-style aggregation over many ledger entries is where remaining misses live (all models, both modes);
  a generic "index facts by field" option would remove that reasoning step — not built.
- Cache-hit accounting through 9Router reads 0 (non-standard `cached_tokens` field); percentages in docs come from 9Router's log.
- No multi-process lease (optimistic concurrency only); tool execution is at-least-once; no skill migration events.
- ReAct baselines for Sonnet/Kimi on the warehouse predate `inspect`; Gemma's baselines had it.
- Kimi K3 300-order SKILL.state on the current build not rerun (NVIDIA stalls; 0.94 over 153 on an earlier build).
- Usability pass not done: `mnestic run` should need no flags (model from config, no step numbers surfaced); a config file
  (`.mnestic/config.toml`) does not exist yet; env vars and CLI flags only.

## Next actions (in order)

1. Usability: `mnestic init` writes a config file; `run` defaults to it; hide step budgets; friendly errors when no model.
2. A prose-tool task (multi-file coding change with a test suite; facts = test results + file hashes) with Gemma and Sonnet, both
   modes — the test that says whether the ledger matters beyond record-returning tools.
3. Generic fact indexing (`index_by` on a tool's facts) for reverse lookups.
4. Cache-token mapping for 9Router; Kimi rerun on the current build when NVIDIA is stable.

## How to run the benchmarks

```
uv run python scripts/warehouse_bench.py --model mock --orders 300
uv run python scripts/registry_bench.py  --model mock --orders 200
# local Gemma via llama.cpp (systemctl --user start llama-server):
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_API_KEY=local MNESTIC_OUTPUT_MODE=native \
MNESTIC_MODEL_SETTINGS='{"openai_reasoning_effort":"low","temperature":0.2,"max_tokens":4000}' \
uv run python scripts/registry_bench.py --model openai-chat:gemma-4-12b-uncensored --orders 200 --mode both --timeout 300
```
