# Context Invariants

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

**THE ARCHIVE IS NOT DEFAULT MODEL CONTEXT.**

These invariants define what the model may see at an ordinary reasoning step.
They are enforced by construction (`ContextBuilder` has no storage access),
by tests (`tests/unit/test_context_builder.py`,
`tests/benchmarks/test_context_scaling.py`) and by review (`AGENTS.md`).

## I1 — Bounded default context

At step *t* the model input is exactly:

```
A_t = (P, Σ_t, O_t, [E_t])
```

- `P`  — the immutable Skill Specification (plus the runtime's fixed output contract and the specs of tools the skill requires)
- `Σ_t` — the canonical ExecutionState at its current `state_version`
- `O_t` — the newest Observation (bounded excerpt; full content is archived)
- `E_t` — optional: the result of an *explicit* `MemoryQuery` issued at step *t-1*, bounded by `MemoryQuery.limit` and `max_retrieved_chars`

Nothing from steps `< t-1` appears unless it is (a) in `Σ_t` because the model
put it there through a validated patch, or (b) in `E_t` because the model asked
for it.

## I2 — No transcript accumulation

- `agent.run()` is called without `message_history` at every step.
- Messages produced inside one step (validation retries) are archived after the
  step and never re-sent.
- `ContextBuilder.build()` takes `(SkillSpecification, ExecutionState, Observation, list[MemoryResult])` and nothing else. It cannot query the database.

## I3 — Reasoning is not persisted as context

`AgentDecision.rationale_summary` is a short (≤ 600 chars) externally safe
justification. It is archived as part of the decision event but is **not**
placed into `ExecutionState` and is never re-injected. Chain-of-thought is not
requested.

## I4 — Observations do not accumulate

An observation is presented once (at the step where it is newest). After the
model has folded whatever matters into `Σ` through a patch, the observation only
exists in the archive. `ExecutionState.last_observation_summary` is a single
bounded string the model *chooses* to write; it is replaced, not appended.

## I5 — Retrieval is explicit, bounded, and audited

Archive retrieval only happens through `MemoryQuery`. The runtime records the
query and the returned event ids as a `memory.retrieval` archive event. The
retrieved subset is injected once into the next context (`E_{t+1}`) and then
dropped.

## I6 — State growth is bounded by policy, not by the model's goodwill

`StateLimits` cap counts and byte sizes. Overflow rejects the patch with
actionable feedback or triggers explicit archived compaction. See
`docs/STATE_MODEL.md`.

## I7 — Context is inspectable

Every model call stores the exact rendered context (`model_calls.context_json`).
`mnestic inspect-context <run>` shows it. The rendered text has
unambiguous section delimiters:

```
<skill_specification> … </skill_specification>
<execution_state version="N"> … </execution_state>
<latest_observation id="…" event_id="…"> … </latest_observation>
<retrieved_archival_evidence> … </retrieved_archival_evidence>   (only if present)
```

## What "leakage" looks like (grep for these in review)

- `message_history=` with anything other than `None`
- `all_messages()` / `new_messages()` flowing into a later `run()`
- lists of `ModelMessage` stored in graph state or ExecutionState
- observation content appended into any state list without a patch
- ContextBuilder importing `storage` or receiving a store/connection
- tool outputs concatenated across steps
