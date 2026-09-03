# Is this what we set out to build? — paper and brief alignment review

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263).*

Re-read of the paper (method, experiments, limitations) and of the original engineering brief, checked
against the code as of the loop-guard/observation-labelling changes. Verdict first, evidence after.

**Verdict.** The runtime implements the paper's core mechanism faithfully — per-step input is exactly
`(P, Σ_t, O_t)`, the model's output is a validated state patch plus an action, and reasoning traces
never re-enter a prompt — and it goes beyond the paper in the two places the authors flag as open
problems (retrospective relevance, silent state deletion). It deviates from the paper in three
deliberate ways (no free-text reasoning trace, typed patch ops instead of dict-merge, a general
schema instead of per-domain schemas), and it has not yet reproduced the paper's *accuracy*
comparison against history-based agents, only the token-scaling one.

## 1. Paper mechanism → implementation

| paper | this implementation | status |
|---|---|---|
| Algorithm 1: receive `O_t`, build prompt `(P, Σ_t, O_t)`, generate `(R_t, ΔΣ_t, a_t)`, validate `ΔΣ_t`, `Σ_{t+1} = Σ_t ⊕ ΔΣ_t`, execute `a_t` | `BuildContext → Reason → ApplyDecision → ExecuteAction → CaptureObservation` in `graph/nodes.py`; `ContextBuilder.build(skill, state, observation, retrieved)` | **matches** (+ optional `E_t` from explicit retrieval) |
| Prompt template: *Instructions* / *Skill Execution State* (JSON) / *Latest Observation* / *Required Output Format* | `<skill_specification>` (+ tool specs) / `<execution_state version=N>` JSON / `<latest_observation request=…>` / `<output_contract>` | **matches**; ours also labels the observation with the request that produced it (needed — see Luna trace) |
| `R_t`: "step-by-step reasoning (will be discarded after execution)" emitted in the same output | not requested; `rationale_summary ≤ 600 chars`, archived, never re-injected. Reasoning models think natively; non-reasoning models get no scratchpad | **deviates, per the brief** ("do NOT request or store hidden chain-of-thought"). Trade-off noted in §4 |
| `ΔΣ_t`: JSON dict merged with null-deletion; "an invalid patch triggers a rollback-retry cycle" | 30 typed ops (`add_fact`, `supersede_fact`, `promote_hypothesis`, `set_plan`, …), atomic per patch, evidence checks, status rules, count/byte limits; rejection → bounded feedback observation → retry | **stricter**. Directly targets the paper's dominant small-model failure ("Premature State Overwrite/Deletion", 68% of Gemma-4 failures): nothing can be deleted without an op that names it, and every removal is archived |
| Schema "authored once per domain" (e.g. CTF: `discovered_flags, tested_hypotheses, active_files, working_dir, cmd_summary`) | one general agent schema (`ExecutionState`: objective, phase, facts, hypotheses, rejected, environment, plan, artifacts, …). Skills can only extend it through `environment.properties`, `important_entities`, `metadata` | **gap**: no skill-authored typed state fields. The general schema *contains* the CTF example (hypotheses, artifacts, environment, summary) but a skill cannot declare "`discovered_flags: list[str]`" and have it validated |
| Prompt footprint `O(|P|+|Σ|+|O|)`, cumulative `O(T)` | measured: 2,878 → 3,364 chars over 1,000 steps (benchmark workload); 7–19K chars in real research runs, capped by `StateLimits` (48 KB) | **matches** |
| Validation and schema ownership "reside in the deterministic runtime rather than the model" | `apply_patch` is pure and deterministic; commit is an optimistic `UPDATE … WHERE state_version=?` | **matches** |
| State drift experiment: obsolete facts in history "overpower contradictory new observations"; SKILL.state recovers in zero steps | `test_stale_fact_leaves_context_but_stays_in_archive` (port 8000 → 9000): the next prompt contains only 9000; 8000 stays in the archive | **matches** |
| Grammar-constrained decoding recommended for small models | `output_mode = native` → llama.cpp GBNF grammar; verified with Gemma 4 12B (local): zero malformed decisions, research task completed in 7 steps | **matches, verified** (schema had to drop string length bounds and force discriminators required) |
| Multi-agent: "concurrent writes require deterministic conflict-resolution semantics" | optimistic concurrency: stale writer's commit fails, is recorded, run pauses | **partial** (single-agent as in the paper; conflicts are detected, not merged) |

## 2. Where the implementation goes beyond the paper

The paper explicitly does not discuss archival memory or retrieval; it discards reasoning and relies on
the state being a *sufficient statistic*. Its own limitations section names three cases where that
fails. The brief asked us to address them, and we did:

| paper limitation | what we do |
|---|---|
| retrospective relevance — "an earlier observation whose relevance was not recognized when first observed … was never committed to state" | append-only archive + explicit `MemoryQuery` (`search`, `observations`, `tool_executions`, `event`), injected once as `E_t`; `test_forgotten_information_recovered_via_explicit_retrieval` |
| historical-trajectory objectives (auditing, provenance) | `state_history`, `state_at_version`, every patch and its changes recorded; CLI `history`/`diff`/`events` |
| dynamic schema discovery | partially: hypotheses, entities, environment and metadata are open-ended; but see the schema gap above |
| silent state deletion by small models | removal only through named ops with reasons; archived; limits reject instead of truncating |

Also beyond the paper: durable crash/resume at every lifecycle phase, provenance (`evidence_event_ids`
must exist in the archive), human-input as a first-class observation, semantic memory with audited
promotion, and per-call context inspection.

## 3. The original brief — acceptance criteria

All 23 items in §35 of the brief are met (see `docs/AUDIT.md` for the leakage audit and the 25-step
byte trace). Items worth re-stating because they were checked again after the live runs:

- *"At least one optional live-model example works"* — now verified with Kimi K3, Claude Haiku 4.5,
  Claude Sonnet 5, Codex gpt-5.6-luna and Groq gpt-oss-120b through 9Router, on the repo itself.
- *"Tests run without external API credentials"* — 90 tests, no network.
- *"Token/context-scaling benchmark demonstrates bounded-history behaviour"* — `docs/BENCHMARK_REPORT.md`.
- *"A ReAct-style baseline demonstrates the architectural contrast"* — a **simulator** of transcript growth,
  as the brief allowed ("baseline/context simulator"). It is not a live history-based agent.

Deviations from the brief, all recorded in `docs/IMPLEMENTATION_NOTES.md`: pydantic-graph 2.x has no
persistence layer (ours is SQLite); `ContinueAction` added; skill *migration* events are not
implemented (resume refuses a changed skill rather than migrating).

## 4. Gaps and what they would take

1. ~~**Skill-authored state schema**~~ — done: `domain_schema` on the skill, `ExecutionState.domain`,
   `set_path`/`adjust_path`/`delete_path`, and tool `state_effects` applied by the runtime. This closed the
   warehouse accuracy gap (Sonnet 0.95 → 1.00, Gemma 0.48 → 0.71+).
2. **Optional reasoning scratchpad** (paper's `R_t`). For non-reasoning models a bounded, archived,
   never-re-injected `scratchpad` field could raise decision quality. The brief forbids storing hidden
   chain-of-thought; this would be opt-in per config and archived only. Decision for the owner.
3. **Accuracy comparison vs. a history-based agent** — now measured on the warehouse task
   (`docs/WAREHOUSE_BENCHMARK.md`): at 60 orders SKILL.state ties or beats ReAct on accuracy (Sonnet 1.00 vs 0.98) once
   the fixed per-call overhead is trimmed, and costs fewer tokens; at 300 orders Kimi K3 holds 0.88 with a flat
   5.7K-token call. Still missing: a 300-order ReAct run (expensive), and the paper's other environments.
4. **Skill migration events** (brief §14) — a versioned `skill.migrated` event instead of refusal.
5. **Deterministic merge for concurrent writers** (paper §multi-agent) — leases or CRDT-style merges
   before subagents (stage 6).

## 5. Observed in practice (this repository, four models)

What the paper predicts, we saw: a model that fails to fold observations into `Σ` loops (Codex Luna,
before the loop guard); one that maintains `Σ` well converges with a flat context (Kimi, Sonnet). What
the paper does not cover, we hit: observations must be *self-describing* because there is no previous
turn to anchor them, and tool semantics (regex vs literal) must be explicit in the tool spec. Both are
now in the runtime and are, arguably, SKILL.state-specific requirements the paper's environments
(fixed action strings) never exposed.
