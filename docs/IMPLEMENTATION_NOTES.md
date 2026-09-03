# Implementation notes and API discrepancies

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

Research was done against the installed packages, not from memory:
pydantic-ai-slim 2.37.0, pydantic-graph 2.37.0, pydantic 2.13.5, Python 3.12.14,
SQLite 3.53.1 (FTS5 available). Spike scripts verified each API before use.

## Discrepancies vs. the original specification

| spec assumption | reality (verified) | what we did |
|---|---|---|
| pydantic-graph offers state persistence (`FileStatePersistence`, `load_next`, snapshots) | pydantic-graph 2.x removed the `persistence` module entirely; graphs are built with `GraphBuilder` (+ `BaseNode` classes) and only `Graph.run/iter` exist | all durability is our own `steps` table; `Runtime.resume` re-enters the graph through a single `Enter` node whose target is chosen from the persisted phase |
| `Graph(nodes=[...])` constructor | `GraphBuilder(...).add(g.node(...)) .build()`; edges inferred from `run()` return annotations; multiple start edges become a *broadcast fork* (parallel!) | one start edge → `Enter` dispatcher node |
| `result.usage()` is a method | `AgentRunResult.usage` is a property returning `RunUsage(input_tokens, output_tokens, requests, tool_calls, …)` | used as property |
| `agent.run(..., retries=)` per call | supported (`retries: int \| AgentRetries`); output validators raise `ModelRetry`; retries stay within one run | version check implemented as an output validator |
| structured output via `result_type` | `output_type` with `ToolOutput` / `NativeOutput` / `PromptedOutput` wrappers | configurable `output_mode` |
| `pydantic_ai.models.ALLOW_MODEL_REQUESTS` | exists; set `False` in tests | done |
| docs at `ai.pydantic.dev` | redirected to `pydantic.dev/docs/ai/...`; several pages 404 | source inspection instead |
| `openai:<model>` works with any OpenAI-compatible server | in pydantic-ai 2.x `openai:` → `OpenAIResponsesModel` (Responses API); chat-completions-only proxies (9Router, LiteLLM, llama-server) need `openai-chat:<model>` + `OPENAI_BASE_URL` | documented in README; `doctor` prints the resolved model class |
| `sqlite3.executescript` inside a transaction | it issues an implicit COMMIT first | custom statement splitter (`run_script`) |

## Design decisions worth knowing

- **Instructions vs. system prompt**: the skill specification is passed as PydanticAI
  `instructions` per run (not `system_prompt`), because instructions are explicitly "not
  part of message history" in PydanticAI 2.x. We never pass `message_history` anyway.
- **Messages from a step are archived, not reused**: `result.all_messages()` is dumped to
  `model_calls.raw_messages_json` for audit. Nothing reads it back into a prompt.
- **`ContinueAction`** exists so the model can digest retrieved evidence without an external
  action; a continue with an empty patch counts toward `max_consecutive_continues`.
- **Counters are runtime-owned**: `steps_completed`, `model_calls`, `tool_calls`,
  `retrievals` are updated by the runtime and committed as their own state version at step
  close; the model cannot fake them.
- **Evidence ids are real archive ids**: the observation's `event_id` is shown in
  `<latest_observation>`; `add_fact` must cite ids that exist in the run's archive.
- **Rationale is bounded**: `rationale_summary ≤ 600` chars, archived with the decision,
  never re-injected. No chain-of-thought is requested.
- **Token estimate**: `approx_tokens = ceil(chars/4)`; real provider counts are stored per
  call from `RunUsage` when a real model is used.
- **`--model mock`** uses the skill's `reasoner_script` (a `module:function` path) with
  `ScriptedReasoner`; scripts read the *rendered context*, so they double as tests that the
  context carries what a policy needs.
- **Dependencies**: `pydantic`, `pydantic-ai-slim[openai,anthropic]`, `pydantic-graph`,
  `pyyaml`; dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`. No vector DB, no Redis, no
  Postgres, no web framework. The CLI is argparse.

## Live verification

`openai-chat:` + `OPENAI_BASE_URL` against a local 9Router: Claude Haiku 4.5 completes `codebase-research` in tool
mode; Groq `gpt-oss-120b` needs `prompted` mode (Groq validates tool calls server-side and rejects the large nested
schema). `model_calls` recorded real token usage from the proxy in both cases.

## Things intentionally left for later

- leases for multi-process resume; exactly-once tools; vector retrieval; MCP; subagents;
  network tools; a `runs.parent_run_id` column.
