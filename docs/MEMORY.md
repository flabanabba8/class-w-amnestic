# Memory

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

**THE ARCHIVE IS NOT DEFAULT MODEL CONTEXT.**

## Archival (episodic) memory

Everything needed to reconstruct what happened is written to `archive_events` (plus the
relational records it references): raw task input, every observation (bounded excerpt in
the observation, full content in `observations.full_content` / `tool_executions.output`),
every tool request and result, every model request/response (`model_calls` keeps the exact
`ModelContext`), every accepted or rejected decision and patch, every state commit and
compaction, every retrieval, human requests/responses, errors, timing and usage.

Event types: `run.created|resumed|completed|failed|paused`, `task.input`,
`observation.captured`, `context.built`, `model.request|response|error`,
`decision.accepted|rejected`, `patch.applied|rejected`, `state.committed|compaction`,
`action.requested`, `tool.started|finished|failed|interrupted`, `memory.retrieval`,
`human.request|response`, `semantic.promoted`, `error`.

## Retrieval

`MemoryQuery { query_type, text?, event_id?, event_type?, version?, limit ≤ 20, reason }`
→ `MemoryResult { events[] (id, step, type, summary, bounded excerpt), total_matches, truncated }`.

| query_type | source |
|---|---|
| `recent` | newest events (optionally by type) |
| `event` | one event by id |
| `events_by_type` | newest events of a type |
| `search` | FTS5 keyword search over summaries + flattened payloads (AND, then OR fallback; LIKE if FTS5 is unavailable) |
| `observations` | search restricted to observations / task input / human responses |
| `tool_executions` | search restricted to tool events |
| `artifacts` | artifact references by locator/description |
| `state_at_version` | a historical `ExecutionState` (model view) |
| `state_history` | version list with the changes each patch made |
| `semantic` | durable semantic memories |

The result is injected exactly once, into the next step's
`<retrieved_archival_evidence>` section (budgeted by `max_retrieved_chars`), together
with the *same* observation the agent was looking at. The retrieval itself is recorded as a
`memory.retrieval` event holding the query and the returned event ids, so audits can see
what the model was shown. The `archival_memory_search` tool wraps the same retriever for
skills that prefer a tool call.

Future semantic/vector retrievers implement the same `Retriever` protocol
(`memory/base.py`) and are still bounded by `limit` — retrieval quality may improve, prompt
size may not.

## Semantic (durable) memory

`semantic_memories` holds curated cross-run knowledge (machine properties, project
conventions, user preferences). Nothing is promoted automatically: promotion happens via
`SemanticMemoryStore.promote()` / `mnestic semantic promote`, records who promoted it
and which archive events support it, and writes a `semantic.promoted` event into the source
run. Agents reach it through `MemoryQuery(query_type="semantic")`; it is not injected by
default either.

## Working-memory hygiene

The state is what the model *chooses* to keep, under `StateLimits`. Good policies replace
rather than append (`set_observation_summary`, `set_plan`, `supersede_fact`) and reference
big things through artifacts and event ids. Bad policies hit limits and get an explicit
rejection telling them how to consolidate; what they drop is always archived.
