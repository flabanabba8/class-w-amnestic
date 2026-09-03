# Architectural audit (v0.1.0)

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

Performed after implementation, as a reviewer of "someone else's" code, looking
specifically for accidental history leakage.

## Static search

| pattern | hits in `src/` | verdict |
|---|---|---|
| `message_history=` | 0 | never passed |
| `all_messages()` | 1 — `agent/pydantic_ai_reasoner.py:96`, dumps the step's own messages into `model_calls.raw_messages_json` for audit | archived only; no reader feeds it to a prompt |
| `new_messages()` | 0 | — |
| `list[ModelMessage]` / lists of `Observation` in state or graph state | 0 | no message lists exist in `ExecutionState` or `RuntimeGraphState` |
| `ContextBuilder.build` inputs | `(skill, state, observation, retrieved)` | no store/connection; `context/` imports no storage module (test-enforced) |
| observation appended to any state list without a patch | none; `last_observation_summary` is set by an explicit `set_observation_summary` op and replaces | — |
| tool output accumulation | tool output → `tool_executions.output` (full) + one bounded `Observation`; never concatenated | — |
| growing state fields | all lists are count-limited; total bytes limited; overflow rejects or archives | — |
| hidden framework persistence | pydantic-graph 2.x has none; PydanticAI `Agent.run` without `message_history` sends one `ModelRequest` (verified with `FunctionModel`, see `tests/integration/test_pydantic_ai_reasoner.py::test_fresh_request_per_step_no_history`) | — |

## Trace: 25-step simulated run — what bytes from step 1 are in the model input at step 20, and why?

Script: `scratchpad/audit_trace.py` (reproduced by `tests/benchmarks/test_context_scaling.py::test_full_marker_scan_on_short_run`).

- `instructions` at step 20 are byte-identical to step 1: the immutable skill
  specification + output contract. Present because **P is constant by design**.
- Of the prompt, the lines shared with step 1 are exactly: JSON structural delimiters and
  state fields whose *current* value is unchanged (`run_id`, `skill_id`, `objective`,
  `status`, empty lists, `budgets`, `environment.target`). Present because **they are
  current state**, not because they occurred earlier; if a patch changed them they would
  differ (e.g. `environment.n`, `last_observation_summary`, `counters.steps_completed` differ).
- The step-1 observation marker (`OBSMARK-000001-…`) is **absent** at step 20. The
  step-20 marker is present (newest observation only).
- Nothing from the step-1 decision, rationale, model messages or tool result is present.

Conclusion: no byte is present *merely because it happened earlier*.

## Trace: deliberate recovery

`MemoryQuery(search, text=OBSMARK-000001-…)` returns the archived
`observation.captured` event of step 1 with the marker in its excerpt (2 matches: the
observation event and the tool.finished event). It would enter exactly one later context,
inside `<retrieved_archival_evidence>`, and disappear again the step after
(`tests/integration/test_memory_and_poisoning.py`).

## Result

All 83 tests pass after the audit (`uv run pytest`), ruff and mypy are clean, and the
1000-step benchmark reports a flat context (2,878 → 3,364 chars) with zero leaks against
a 581,972-char simulated transcript (`docs/BENCHMARK_REPORT.md`).

**Invariant satisfied: at each ordinary reasoning step the model receives current state
rather than an accumulating execution transcript.** The one caveat is intrinsic to the
design, not a leak: bounded retries *inside* a single step do accumulate that step's
retry messages (PydanticAI's `ModelRetry` loop), which are archived and never re-sent.
