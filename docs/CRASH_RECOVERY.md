# Crash recovery and resume

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

`mnestic resume <run-id>` rebuilds the runtime from SQLite alone: the current
`ExecutionState`, the latest `steps` row (phase + pointers) and the pending observation.
No transcript exists to replay.

## Resume matrix

| latest step phase | meaning | resume action |
|---|---|---|
| `observation_ready` / `context_built` / `failed` | observation persisted, no decision yet | rebuild context, call the model (a model call may repeat — the earlier one, if any, was never recorded) |
| `decision_recorded` | decision persisted, patch not committed | re-apply the persisted decision (`ApplyDecision`) — no new model call |
| `state_committed` / `action_pending` | patch committed, action not started | dispatch the persisted action / retrieval |
| `action_dispatched` | tool started, no result | mark the execution `interrupted`, produce a `TOOL FAILED … interrupted … side effects unknown` observation, continue |
| `awaiting_human` | waiting for an answer | `--input` required; it becomes a `human_input` observation |
| `done` (run `paused`/`blocked`) | step closed normally | continue from the next open step |
| terminal status | completed/failed/cancelled | refuse |

Skill immutability is enforced: if the stored skill's content hash differs from the one
recorded on the run, resume refuses.

## Idempotency

- Context building and model calls are side-effect free; repeating them is safe.
- Patch application is idempotent through the version check: the persisted decision's
  `expected_state_version` either matches (apply once) or is stale (rejected, run pauses).
- Closing a step and opening the next (observation, metrics, counters commit, next `steps`
  row) is **one transaction**, so there is never a closed step without a successor.
- **Tool execution is at-least-once, not exactly-once.** If the process dies after
  `tool.started` but before `tool.finished`, we cannot know whether the side effect
  happened. The runtime records `tool.interrupted` and tells the model, which must verify
  before retrying (e.g. list the directory before re-writing a file). Read-only tools are
  safe to repeat. Exactly-once would require tool-level idempotency keys and transactional
  tools, which v1 does not have.

## Concurrency

Two processes may open the same run. Every state commit checks `state_version`; the loser
gets `StaleWriteError`, records a `patch.rejected (stale_write)` event, reloads the state and
pauses (`status=paused`, `reason=stale_write`). No update is lost. There is no lease/lock;
adding one (`runs.lease_owner`, `lease_expires_at`) is a small future migration.

## Tests

`tests/integration/test_crash_resume.py` injects a crash right after the commit of each
phase (`context_built`, `decision_recorded`, `state_committed`, `action_pending`, `done`),
during a tool call, after a human-input request, on a step budget, and with a concurrent
writer; every case resumes to completion with contiguous archive sequence numbers and no
duplicated patch application.
