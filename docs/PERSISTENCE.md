# Persistence

SQLite is a first-class component (`src/mnestic/storage/`). Settings: WAL journal,
`synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=30s`, explicit reentrant
transactions (`BEGIN IMMEDIATE … COMMIT/ROLLBACK`).

## Schema (migration 1)

| table | purpose | notable columns |
|---|---|---|
| `schema_migrations` | applied migrations | version, name, applied_at |
| `skills` | procedural memory | (skill_id, version) PK, content_hash, spec_json |
| `runs` | run metadata | status, skill_content_hash, workspace_root, model_name, task_input, last_step, outcome_json |
| `current_states` | one row per run: canonical state | state_version, status, state_json, state_bytes |
| `state_versions` | every version ever committed | (run_id, version) PK, step, patch_id, state_json |
| `state_patches` | every proposed patch | status applied\|rejected, error_code, patch_json, changes_json |
| `archive_events` | append-only event log | (run_id, seq) UNIQUE, step, event_type, summary, payload_json, payload_text, ref_table/ref_id |
| `archive_events_fts` | FTS5 external-content index over summary + payload_text | maintained by triggers |
| `observations` | bounded content + full content | kind, source, content, full_content, truncated, event_id |
| `actions` | model-requested actions | kind, action_json, status |
| `tool_executions` | every tool run | status started\|succeeded\|failed\|interrupted, output, error, duration_ms |
| `artifacts` | artifact references | locator, description, originating_event_id |
| `model_calls` | every model invocation | context_json (exact), context_chars, approx_tokens, input/output tokens, requests, decision_json, raw_messages_json |
| `memory_retrievals` | every explicit retrieval | query_json, result_json, result_count |
| `semantic_memories` | durable cross-run memory | key UNIQUE, category, content, source_event_ids, promoted_by |
| `steps` | lifecycle phase per step (resume anchor) | phase, observation_id, retrieved_json, decision_json, patch_id, action_id |
| `step_metrics` | per-step metrics | context_chars, tokens, state_bytes, tool_calls, retrievals, elapsed_ms |
| `errors` | runtime errors | kind, message, traceback |

Migrations are an ordered list in `migrations.py`; applied inside a transaction and
recorded. Never edit an applied migration; append a new one. Statements are split with
`sqlite3.complete_statement` so trigger bodies survive and the whole migration commits
atomically (`executescript` would auto-commit).

## Transaction boundaries (what is atomic)

| lifecycle moment | one transaction containing |
|---|---|
| run start | skill upsert, run row, state v0, `run.created`, `task.input`, observation, step 0 |
| context built | `context.built` event, step phase |
| model responded | `model_calls` row, `model.request/response` events, step phase + decision |
| decision applied | patch record, `current_states` update (version check), `state_versions` insert, archived-item events, `patch.applied`, `state.committed`, step phase |
| action requested | `action.requested`, action row, step phase |
| tool started | tool_execution row, `tool.started`, action status, step phase |
| tool finished → next step | `tool.finished/failed`, tool_execution update, artifact, observation row + event, step `done`, metrics, counters commit (new version), **next step row** |
| retrieval → next step | `memory.retrieval`, retrieval row, step `done`, metrics, counters commit, next step row |
| finalize | status commit (new version), `run.*` event, run row update, metrics |

## Consistency guarantees

- **Durable**: every phase transition is committed before the next node runs; a crash
  never loses a committed step. WAL + `synchronous=NORMAL` means a power loss (not a
  process crash) could lose the last transactions; use `synchronous=FULL` if that matters.
- **Atomic state commits**: state, version history, patch record and derived archive
  events succeed or fail together.
- **No lost updates**: optimistic version check on `current_states`.
- **Append-only archive**: `seq` is contiguous per run; nothing is updated except
  through the documented `tool_executions`/`actions` status columns (which are records,
  not archive events).
- **Recoverable history**: `state_versions` holds every version; `patch_json` and
  `changes_json` explain each transition.

What is *not* guaranteed: exactly-once tool execution (see `CRASH_RECOVERY.md`), and
multi-process leases (two processes may both *try* to resume; the second one's first
commit fails cleanly and it pauses).

## Sizes

Full tool output is stored in `tool_executions.output` and `observations.full_content`
(event payloads keep the first 20 000 chars). Only a bounded excerpt
(`max_observation_chars`, default 6000) reaches the model, with the event id for retrieval.
