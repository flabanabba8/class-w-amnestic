# State model

`ExecutionState` (`src/mnestic/models/state.py`) is the canonical operational memory.

```
ExecutionState
├─ run_id, skill_id, skill_version, state_schema_version
├─ objective: Objective {statement, success_criteria[]}
├─ status: RunStatus  (pending|running|waiting_for_human|blocked|paused|completed|failed|cancelled)
├─ current_phase: str
├─ verified_facts[]: VerifiedFact {id, statement, confidence, evidence_event_ids[≥1], created_at, updated_at}
├─ active_hypotheses[]: Hypothesis {id, statement, confidence, evidence_event_ids[]}
├─ rejected_hypotheses[]: RejectedHypothesis {id, statement, reason, evidence_event_ids[], rejected_at}
├─ environment: Environment {properties: dict[str, scalar]}
├─ constraints[]: str
├─ artifacts[]: ArtifactReference {id, kind, locator, description, originating_event_id}
├─ current_plan[]: PlanStep {id, description, status}
├─ pending_actions[]: PendingAction {id, description, tool_name}
├─ blockers[]: Blocker {id, description}
├─ unresolved_questions[]: OpenQuestion {id, question}
├─ important_entities: dict[name, description]
├─ counters: Counters {steps_completed, model_calls, tool_calls, retrievals, patches_applied, patches_rejected, errors, consecutive_failures}
├─ budgets: Budgets {max_steps, max_tool_calls, max_consecutive_failures}
├─ last_observation_summary: str | None      (replaced, never appended)
├─ metadata: dict[str, scalar]
├─ state_version: int
└─ created_at, updated_at
```

`ExecutionState.model_view()` is the exact dict rendered for the model: it drops
timestamps and `metadata` and keeps a stable key order.

## Facts vs hypotheses vs rejected hypotheses

- A **fact** requires ≥1 `evidence_event_ids` that exist in this run's archive
  (`add_fact`, `supersede_fact`, `promote_hypothesis`). Unknown ids → patch rejected.
- A **hypothesis** may have no evidence. It becomes a fact only via
  `promote_hypothesis` with evidence. `add_fact` with a statement equal (normalised) to
  an active hypothesis is rejected with `hypothesis_promotion_required`.
- A **rejected hypothesis** records the reason and evidence. The list is a bounded
  ring: overflow beyond `max_rejected_hypotheses` is spilled to the archive as
  `state.compaction` events.

Model validators also reject duplicate ids across lists, ids shared between facts and
hypotheses, and unknown fields (`extra="forbid"`).

## StatePatch

`StatePatch { expected_state_version, ops[≤50] }` where each op is a strict discriminated
union member (`op` field). Available ops:

| group | ops |
|---|---|
| control | `set_phase`, `set_status` (only `running`/`blocked`; terminal statuses are runtime-owned), `set_objective` |
| facts | `add_fact`, `supersede_fact`, `remove_fact`, `archive_facts` |
| hypotheses | `add_hypothesis`, `update_hypothesis`, `reject_hypothesis`, `promote_hypothesis` |
| artifacts | `add_artifact`, `remove_artifact` |
| plan | `set_plan` (canonical replacement), `update_plan_step` |
| bookkeeping | `add_pending_action`, `complete_pending_action`, `add_blocker`, `remove_blocker`, `add_question`, `resolve_question` |
| environment | `set_environment`, `clear_environment`, `set_entity`, `remove_entity`, `add_constraint`, `remove_constraint` |
| misc | `set_observation_summary` (replace), `set_metadata` |

`apply_patch(state, patch, *, limits, evidence_checker)`:

1. `expected_state_version` must equal `state.state_version` → else `StaleStateError`.
2. Evidence ids are checked against the archive before any mutation.
3. Ops apply in order to a deep copy; the first failure rejects the *whole* patch with
   `code` and `op_index` (`not_found`, `duplicate_id`, `forbidden_status`,
   `bad_transition`, `missing_evidence`, `unknown_evidence`,
   `hypothesis_promotion_required`, `limit_exceeded`, `state_too_large`, `invalid`, …).
4. Automatic compaction (rejected hypotheses, finished plan steps) → archived items.
5. Count limits, whole-state re-validation, byte-size limit.
6. `state_version += 1`.

Everything removed from working state (superseded facts, removed artifacts, replaced
plans, resolved questions, compacted items) is returned as `ArchivedItem`s and written to
the archive by the runtime (`patch.applied` / `state.compaction` events) before the commit
transaction completes.

## Size control (`StateLimits`)

Defaults: 48 KB serialized state, 150 facts, 40 hypotheses, 40 rejected, 100 artifacts,
50 plan steps, 30 pending actions, 20 blockers, 30 questions, 60 entities, 60 environment
keys, 50 constraints, 40 metadata keys; 2000-char statements.

- Overflow on *curated* lists (facts, hypotheses, artifacts, …) **rejects** the patch with
  a message naming the op to use (`archive_facts`, `reject_hypothesis`, …). The model
  sees the rejection as the next observation and must consolidate.
- Overflow on *bookkeeping* lists (rejected hypotheses, finished plan steps) is
  **compacted automatically**, archived, and reported in the patch's change log.
- Nothing is silently truncated: every removal is an archive event that retrieval can
  find (`memory search <run> "<statement>"`).

## Status transitions

```
pending → running → {waiting_for_human, blocked, paused, completed, failed, cancelled}
waiting_for_human | blocked | paused → running
completed | failed | cancelled → (terminal)
```

The model may only request `running` (unblock) or `blocked`. `completed`/`failed` come
from `CompletionResult`; `waiting_for_human` from `RequestHumanInput`; `paused` from
step budgets or a stale write.

## Versioning and concurrency

Every commit is `UPDATE current_states SET … WHERE run_id=? AND state_version=?`
inside a transaction that also inserts `state_versions` and the patch record. A
concurrent writer that advanced the version makes the update affect 0 rows →
`StaleWriteError`, full rollback, the rejection is recorded, and the run pauses. The
runtime's own bookkeeping (counters at step close, status changes) is committed the same
way as its own version, so `history` shows them as `(runtime bookkeeping)`.
