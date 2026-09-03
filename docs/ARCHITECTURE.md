# Architecture

*Based on the paper: Sanket Badhe, Priyanka Tiwari, Jonghyun Chung — **SKILL.state: Scalable Long-Horizon Agent Skills** (EMNLP 2026), [arXiv:2608.26263](https://arxiv.org/abs/2608.26263) · [PDF](https://arxiv.org/pdf/2608.26263).*

## The four memories

| memory | what | where | mutability |
|---|---|---|---|
| **Procedural** | `SkillSpecification` (`skills/<id>/skill.yaml` + `SKILL.md`) | `skills` table, content-hashed | immutable during a run |
| **Operational / working** | `ExecutionState` | `current_states` (+ every version in `state_versions`) | only via validated `StatePatch`, optimistic commit |
| **Episodic / archival** | `ArchiveEvent`s, observations, actions, tool executions, model calls, retrievals | `archive_events` (+FTS5), … | append-only |
| **Semantic / durable** | `SemanticMemory` (cross-run facts) | `semantic_memories` | explicit, audited promotion |

They are never collapsed into a message list. The model's input at step *t* is
`A_t = (P, Σ_t, O_t, [E_t])` (paper eq. 1 plus optional retrieved evidence).

## Components

```mermaid
flowchart TB
    CLI[cli/main.py] --> RT[graph/runtime.py<br/>Runtime.start / resume]
    RT --> G[graph/nodes.py<br/>pydantic-graph lifecycle]
    G --> CB[context/builder.py<br/>ContextBuilder]
    G --> RE[agent/*<br/>Reasoner: PydanticAI | Scripted]
    G --> AP[state/apply.py<br/>apply_patch + limits]
    G --> TL[tools/*<br/>ToolRegistry]
    G --> MR[memory/retrieval.py<br/>ArchiveRetriever]
    G --> ST[storage/store.py<br/>Store]
    ST --> DB[(SQLite<br/>storage/migrations.py)]
    MR --> ST
    SK[skills/loader.py] --> RT
```

- **ContextBuilder** is the only place model input is assembled. It has no storage
  dependency (enforced by `tests/unit/test_context_builder.py`), so it *cannot* pull
  history. Output is a `ModelContext` with delimited sections and measured size.
- **Reasoner** is a protocol: `decide(ModelContext) -> ReasonerResult`.
  `PydanticAIReasoner` performs one fresh `Agent.run(prompt, instructions=…)` per step
  with `output_type=AgentDecision` (tool / native / prompted modes) and an output
  validator that forces `expected_state_version` to match; retries stay inside the step.
  `ScriptedReasoner` drives tests, benchmarks and `--model mock`.
- **apply_patch** is pure: `(state, patch) -> new state`, never mutating the input,
  atomic per patch, with evidence checks, status-transition rules, count/byte limits and
  explicit compaction. The runtime commits the result with
  `UPDATE current_states … WHERE state_version = expected`.
- **Tools** are typed (`Args` model, `ToolResult`), workspace-confined, archived on
  every execution; failures become observations.
- **ArchiveRetriever** implements the `Retriever` protocol over SQLite (FTS5 when
  available). A vector implementation can be added behind the same protocol.

## One step, end to end

```mermaid
sequenceDiagram
    participant R as Runtime/graph
    participant S as SQLite
    participant C as ContextBuilder
    participant M as Model (PydanticAI)
    participant T as Tool
    R->>C: build(skill, state_t, obs_t, [retrieved])
    C-->>R: ModelContext (bounded)
    R->>S: context.built event, step phase=context_built
    R->>M: agent.run(prompt, instructions)  — no message_history
    M-->>R: AgentDecision (validated)
    R->>S: model_calls row (exact context), model.response event, phase=decision_recorded
    R->>R: apply_patch(state_t, patch)
    R->>S: BEGIN; UPDATE current_states WHERE version=t; INSERT state_versions; events; phase=state_committed; COMMIT
    alt action = tool
        R->>S: tool.started, phase=action_dispatched
        R->>T: execute(args)
        T-->>R: ToolResult
        R->>S: BEGIN; tool.finished/failed; observation row+event; phase=done; create step t+1; COMMIT
    else memory_query
        R->>S: memory.retrieval event (query + returned ids); step t+1 with retrieved
    else human_input
        R->>S: human.request event; phase=awaiting_human; run status waiting_for_human
    else completion
        R->>S: run.completed / run.failed
    end
```

## Graph state vs execution state

`RuntimeGraphState` (`graph/state.py`) is the pydantic-graph `ctx.state`: which step we
are on, the current observation, the decision just produced, the tool result in flight,
per-step counters. It exists only for the duration of one process invocation and is
rebuilt by `Runtime.resume` from the `steps` table. `ExecutionState` is the durable,
model-facing semantic state. Only `ExecutionState` is ever rendered into a prompt, and
only by `ContextBuilder`.

## Why pydantic-graph

The lifecycle really is a state machine with data-dependent transitions
(`ApplyDecision` fans out to four successors; failures loop back with bounded feedback).
Modelling it as `BaseNode` classes gives typed transitions that pydantic-graph validates
from the `run()` return annotations, a rendered Mermaid diagram, and a single `Enter`
node that lets `Runtime.resume` re-enter at any persisted phase. pydantic-graph 2.x has
no persistence layer of its own; ours is the SQLite `steps` table.

## Roadmap

1. **Core runtime** (this v1): immutable skills, typed state, bounded context,
   validated patches, SQLite durability, crash/resume, inspectable context. ✔
2. **SQLite archival retrieval**: ids, types, FTS5 keyword search, state history,
   tool executions, artifacts. ✔ (v1 uses this as the only retrieval mechanism)
3. **Semantic / vector retrieval**: a second `Retriever` (embeddings, hybrid,
   reranking). Must remain *retrieval into a bounded `E_t`*, never a bigger prompt.
4. **Richer tools**: robust shell/terminal, git, HTTP, code execution. Large outputs stay
   in the archive/artifacts; observations carry bounded excerpts.
5. **MCP**: MCP tool result → `Observation`/`ArchiveEvent`, never a transcript; tool
   metadata loaded selectively.
6. **Subagents**: own `ExecutionState`, objective and skill; parent receives structured
   `CompletionResult` + artifacts, not the child's transcript.
7. **Browser/terminal autonomy** and realistic long-horizon benchmarks vs history agents.
8. **Self-improving skills**: experience → archived evidence → proposed skill → validation
   → new *version*; a running skill never rewrites its own instructions.
