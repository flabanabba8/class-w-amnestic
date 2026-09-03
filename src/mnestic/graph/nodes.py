"""Lifecycle nodes for the pydantic-graph state machine.

    BuildContext -> Reason -> ApplyDecision -> { RetrieveMemory | ExecuteAction | AwaitHuman | Finalize }
    RetrieveMemory -> BuildContext            ExecuteAction -> CaptureObservation -> BuildContext
    HandleFailure -> BuildContext | Finalize

Each node persists its phase transition transactionally so a crash between nodes is
recoverable by ``Runtime.resume`` (see docs/CRASH_RECOVERY.md).

Two kinds of state exist here and must not be confused:
- ``RuntimeGraphState`` (``ctx.state``): ephemeral controller bookkeeping for this process.
- ``ExecutionState`` (``ctx.state.execution_state``): the canonical semantic state the model sees,
  durably versioned in SQLite.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any

from pydantic_graph import BaseNode, End, GraphRunContext

from mnestic.graph.state import RunOutcome, RuntimeDeps, RuntimeGraphState
from mnestic.models.action import ContinueAction, RequestHumanInput, ToolAction
from mnestic.models.archive import EventType, MemoryResult
from mnestic.models.common import utcnow
from mnestic.models.observation import Observation, ObservationKind
from mnestic.models.state import RunStatus
from mnestic.state.apply import PatchRejected, apply_patch
from mnestic.storage.store import StaleWriteError
from mnestic.tools.base import ToolContext

Ctx = GraphRunContext[RuntimeGraphState, RuntimeDeps]


@dataclass
class BuildContext(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Assemble A_t = (P, Σ_t, O_t, [E_t]) via ContextBuilder and archive a bounded record of it."""

    async def run(self, ctx: Ctx) -> Reason | Finalize:
        s, d = ctx.state, ctx.deps
        if s.steps_this_invocation >= s.max_steps_this_invocation:
            return Finalize(reason="max_steps_this_invocation", status=RunStatus.PAUSED)
        if s.execution_state.counters.steps_completed >= s.execution_state.budgets.max_steps:
            return Finalize(reason="budget_exhausted", status=RunStatus.FAILED, summary="max_steps budget exhausted")
        assert s.observation is not None, "BuildContext requires an observation"
        s.step_started = time.monotonic()
        s.context = d.context_builder.build(d.skill, s.execution_state, s.observation, s.retrieved)
        with d.store.transaction():
            d.store.append_event(
                s.run_id, s.step, EventType.CONTEXT_BUILT,
                f"context built: {s.context.char_count} chars (~{s.context.approx_tokens} tokens), state v{s.context.state_version}",
                {
                    "char_count": s.context.char_count, "approx_tokens": s.context.approx_tokens,
                    "state_version": s.context.state_version, "observation_id": s.context.observation_id,
                    "retrieved_event_ids": s.context.retrieved_event_ids,
                    "section_chars": {k: len(v) for k, v in s.context.sections.items()},
                },
            )
            d.store.update_step(s.run_id, s.step, phase="context_built")
        d.log.info("context built", step=s.step, chars=s.context.char_count, state_version=s.context.state_version)
        return Reason()


@dataclass
class Reason(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """One fresh bounded model invocation. No message history is passed."""

    async def run(self, ctx: Ctx) -> ApplyDecision | HandleFailure:
        s, d = ctx.state, ctx.deps
        assert s.context is not None
        result = await d.reasoner.decide(s.context)
        s.model_calls_this_step += 1
        s.last_usage = result.usage
        with d.store.transaction():
            d.store.save_model_call(
                s.run_id, s.step, attempt=s.model_calls_this_step, model_name=result.model_name,
                context_json=s.context.model_dump_json(), context_chars=s.context.char_count,
                approx_tokens=s.context.approx_tokens, input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens, requests=result.usage.requests,
                status="ok" if result.ok else "error", error=result.error,
                decision_json=result.decision.model_dump_json() if result.decision else None,
                raw_messages_json=json.dumps(result.raw_messages) if result.raw_messages else None,
                duration_ms=result.duration_ms,
            )
            d.store.append_event(
                s.run_id, s.step, EventType.MODEL_REQUEST,
                f"model request to {result.model_name}: {s.context.char_count} chars",
                {"model": result.model_name, "context_chars": s.context.char_count, "approx_tokens": s.context.approx_tokens},
            )
            if result.ok:
                assert result.decision is not None
                d.store.append_event(
                    s.run_id, s.step, EventType.MODEL_RESPONSE,
                    f"decision: {result.decision.rationale_summary[:200]}",
                    {"decision": result.decision.model_dump(mode="json"), "usage": result.usage.model_dump(),
                     "duration_ms": result.duration_ms},
                )
                d.store.update_step(s.run_id, s.step, phase="decision_recorded", decision=result.decision.model_dump(mode="json"))
            else:
                d.store.append_event(
                    s.run_id, s.step, EventType.MODEL_ERROR, f"model error: {(result.error or '')[:200]}",
                    {"error": result.error, "kind": result.error_kind, "usage": result.usage.model_dump()},
                )
        if not result.ok:
            return HandleFailure(reason=result.error or "model returned no decision", kind=result.error_kind or "model")
        s.decision = result.decision
        return ApplyDecision()


@dataclass
class ApplyDecision(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Validate the patch against the current state and commit atomically (optimistic concurrency)."""

    async def run(self, ctx: Ctx) -> RetrieveMemory | ExecuteAction | AwaitHuman | Finalize | HandleFailure | BuildContext:
        s, d = ctx.state, ctx.deps
        decision = s.decision
        assert decision is not None
        state = s.execution_state
        patch = decision.state_patch

        def missing_evidence(ids: Any) -> set[str]:
            ids = set(ids)
            return ids - d.store.existing_event_ids(s.run_id, ids)

        try:
            applied = apply_patch(state, patch, limits=d.config.state_limits, evidence_checker=missing_evidence)
            new_state = applied.state
            new_state.counters.patches_applied += 1
            new_state.counters.model_calls += s.model_calls_this_step
            new_state.counters.consecutive_failures = 0
            if decision.completion is not None:
                new_state.status = RunStatus.COMPLETED if decision.completion.outcome == "success" else RunStatus.FAILED
                if decision.completion.outcome == "partial":
                    new_state.status = RunStatus.COMPLETED
            elif isinstance(decision.action, RequestHumanInput):
                new_state.status = RunStatus.WAITING_FOR_HUMAN
            with d.store.transaction():
                patch_id = d.store.record_patch(
                    s.run_id, s.step, patch, status="applied", resulting_version=new_state.state_version, changes=applied.changes
                )
                d.store.commit_state(new_state, expected_version=state.state_version, patch_id=patch_id, step=s.step)
                for item in applied.archived:
                    d.store.append_event(
                        s.run_id, s.step, EventType.STATE_COMPACTION if item.automatic else EventType.PATCH_APPLIED,
                        f"{'compaction' if item.automatic else 'removed from state'}: {item.kind} — {item.reason[:120]}",
                        {"kind": item.kind, "item": item.item, "reason": item.reason, "automatic": item.automatic},
                    )
                d.store.append_event(
                    s.run_id, s.step, EventType.PATCH_APPLIED,
                    f"patch applied v{state.state_version}->v{new_state.state_version}: {'; '.join(applied.changes)[:300]}",
                    {"patch_id": patch_id, "ops": len(patch.ops), "changes": applied.changes,
                     "from_version": state.state_version, "to_version": new_state.state_version},
                    ref_table="state_patches", ref_id=patch_id,
                )
                d.store.append_event(
                    s.run_id, s.step, EventType.STATE_COMMITTED, f"state committed v{new_state.state_version}",
                    {"version": new_state.state_version, "status": new_state.status.value, "phase": new_state.current_phase},
                    ref_table="state_versions", ref_id=str(new_state.state_version),
                )
                d.store.update_step(s.run_id, s.step, phase="state_committed", patch_id=patch_id, state_version_after=new_state.state_version)
        except StaleWriteError as exc:
            # The whole commit transaction rolled back; record the rejection separately, then pause (no lost update).
            with d.store.transaction():
                d.store.record_patch(s.run_id, s.step, patch, status="rejected", error_code="stale_write", error=str(exc))
                d.store.append_event(s.run_id, s.step, EventType.PATCH_REJECTED, f"patch rejected (stale_write): {exc}",
                                     {"code": "stale_write", "reason": str(exc), "expected": exc.expected, "actual": exc.actual})
                d.store.save_error(s.run_id, s.step, "stale_write", str(exc))
            s.execution_state = d.store.get_state(s.run_id)
            return Finalize(reason="stale_write", status=RunStatus.PAUSED, summary=str(exc))
        except PatchRejected as exc:
            with d.store.transaction():
                d.store.record_patch(s.run_id, s.step, patch, status="rejected", error_code=exc.code, error=exc.reason)
                d.store.append_event(
                    s.run_id, s.step, EventType.PATCH_REJECTED, f"patch rejected ({exc.code}): {exc.reason[:200]}",
                    {"code": exc.code, "reason": exc.reason, "op_index": exc.op_index, "patch": patch.model_dump(mode="json")},
                )
            return HandleFailure(reason=f"state patch rejected ({exc.code}): {exc}", kind="patch")

        s.execution_state = new_state
        s.patches_applied_this_step += 1
        d.log.info("state committed", step=s.step, version=new_state.state_version, changes=len(applied.changes))

        if decision.completion is not None:
            return Finalize(reason="completed", status=new_state.status, summary=decision.completion.summary,
                            completion=decision.completion.model_dump(mode="json"))
        if decision.memory_query is not None:
            return RetrieveMemory()
        action = decision.action
        assert action is not None
        with d.store.transaction():
            ev = d.store.append_event(
                s.run_id, s.step, EventType.ACTION_REQUESTED, f"action requested: {_action_summary(action)}",
                {"action": action.model_dump(mode="json")},
            )
            action_id = d.store.save_action(s.run_id, s.step, action.model_dump(mode="json"), status="pending", event_id=ev.event_id)
            d.store.update_step(s.run_id, s.step, phase="action_pending", action_id=action_id)
        s.action_id = action_id
        if isinstance(action, RequestHumanInput):
            return AwaitHuman()
        if isinstance(action, ContinueAction):
            # A 'continue' that changed nothing is spinning; one that mutated state is progress.
            s.consecutive_continues = s.consecutive_continues + 1 if patch.is_empty else 0
            if s.consecutive_continues >= d.config.max_consecutive_continues:
                return HandleFailure(reason=f"more than {d.config.max_consecutive_continues} consecutive 'continue' actions", kind="loop")
            obs = _make_observation(
                s, ObservationKind.CONTINUE, "runtime", f"No external action was taken. Note: {action.note or '(none)'}",
            )
            return await _advance(ctx, obs, EventType.OBSERVATION, action_id=action_id)
        return ExecuteAction()


@dataclass
class RetrieveMemory(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Explicit archival retrieval. Result is injected exactly once into the next context."""

    async def run(self, ctx: Ctx) -> BuildContext | Finalize:
        s, d = ctx.state, ctx.deps
        assert s.decision is not None and s.decision.memory_query is not None
        query = s.decision.memory_query
        result: MemoryResult = d.retriever.retrieve(s.run_id, query)
        s.retrievals_this_step += 1
        assert s.observation is not None
        with d.store.transaction():
            ev = d.store.append_event(
                s.run_id, s.step, EventType.MEMORY_RETRIEVAL,
                f"memory retrieval {query.query_type}: {len(result.events)}/{result.total_matches} events",
                {"query": query.model_dump(mode="json"), "event_ids": result.event_ids, "total_matches": result.total_matches,
                 "note": result.note},
            )
            d.store.save_retrieval(s.run_id, s.step, query, result, ev.event_id)
            d.store.update_step(s.run_id, s.step, phase="done")
            s.execution_state.counters.retrievals += 1
            # The same observation is re-presented together with the retrieved evidence; nothing else carries over.
            return await _next_step(ctx, observation=s.observation, retrieved=[result])


@dataclass
class ExecuteAction(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Run a tool under policy; every execution is archived; failures become observations."""

    async def run(self, ctx: Ctx) -> CaptureObservation:
        s, d = ctx.state, ctx.deps
        assert s.decision is not None and isinstance(s.decision.action, ToolAction)
        action = s.decision.action
        with d.store.transaction():
            execution_id = d.store.start_tool_execution(s.run_id, s.step, s.action_id, action.tool_name, action.arguments)
            d.store.update_action(s.action_id or "", status="executing")
            d.store.append_event(
                s.run_id, s.step, EventType.TOOL_STARTED, f"tool started: {action.tool_name}",
                {"tool": action.tool_name, "arguments": action.arguments, "execution_id": execution_id},
                ref_table="tool_executions", ref_id=execution_id,
            )
            d.store.update_step(s.run_id, s.step, phase="action_dispatched")
        started = time.monotonic()
        tool_ctx = ToolContext(workspace_root=d.workspace_root, config=d.config, run_id=s.run_id, step=s.step, retriever=d.retriever)
        result = await d.tools.execute(action.tool_name, action.arguments, tool_ctx)
        duration_ms = int((time.monotonic() - started) * 1000)
        s.tool_calls_this_step += 1
        s.tool_result = (execution_id, result, duration_ms)
        return CaptureObservation()


@dataclass
class CaptureObservation(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Persist the tool result (full) and produce the bounded observation for the next step."""

    async def run(self, ctx: Ctx) -> BuildContext | Finalize:
        s, d = ctx.state, ctx.deps
        assert s.tool_result is not None and s.decision is not None and isinstance(s.decision.action, ToolAction)
        execution_id, result, duration_ms = s.tool_result
        action = s.decision.action
        full = result.output if result.ok else f"TOOL FAILED: {result.error}\n{result.output}".strip()
        etype = EventType.TOOL_FINISHED if result.ok else EventType.TOOL_FAILED
        existing = d.store.get_tool_execution(execution_id)
        already_closed = existing is not None and existing.status != "started"  # e.g. marked 'interrupted' by resume
        with d.store.transaction():
            if already_closed and existing is not None and existing.event_id:
                ev_id = existing.event_id
            else:
                ev = d.store.append_event(
                    s.run_id, s.step, etype,
                    f"tool {'finished' if result.ok else 'failed'}: {action.tool_name} ({len(full)} chars)",
                    {"tool": action.tool_name, "arguments": action.arguments, "ok": result.ok, "output": full[:20_000],
                     "error": result.error, "data": result.data, "duration_ms": duration_ms, "execution_id": execution_id},
                    ref_table="tool_executions", ref_id=execution_id,
                )
                ev_id = ev.event_id
                d.store.finish_tool_execution(
                    execution_id, status="succeeded" if result.ok else "failed", output=result.output, error=result.error,
                    data=result.data, event_id=ev_id, duration_ms=duration_ms,
                )
                d.store.update_action(s.action_id or "", status="executed" if result.ok else "failed")
            if result.artifact is not None:
                art = result.artifact.model_copy(update={"originating_event_id": ev_id})
                d.store.save_artifact(s.run_id, art)
            s.execution_state.counters.tool_calls += 1
            if not result.ok:
                s.execution_state.counters.errors += 1
        data = dict(result.data or {})
        data["ok"] = result.ok
        if result.artifact is not None:
            data["artifact"] = {"locator": result.artifact.locator, "kind": result.artifact.kind}
        obs = _make_observation(s, ObservationKind.TOOL_RESULT, action.tool_name, full, data=data, ref_event_id=ev_id)
        return await _advance(ctx, obs, EventType.OBSERVATION, action_id=s.action_id, full_content=full)


@dataclass
class AwaitHuman(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Pause the run; the human's answer arrives as the next observation via ``Runtime.resume``."""

    async def run(self, ctx: Ctx) -> Finalize:
        s, d = ctx.state, ctx.deps
        assert s.decision is not None and isinstance(s.decision.action, RequestHumanInput)
        q = s.decision.action
        with d.store.transaction():
            d.store.append_event(
                s.run_id, s.step, EventType.HUMAN_REQUEST, f"human input requested: {q.question[:200]}",
                {"question": q.question, "options": q.options},
            )
            d.store.update_step(s.run_id, s.step, phase="awaiting_human")
        return Finalize(reason="waiting_for_human", status=RunStatus.WAITING_FOR_HUMAN, summary=q.question,
                        completion={"question": q.question, "options": q.options})


@dataclass
class HandleFailure(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Model/patch failures become bounded runtime observations; repeated failures fail the run."""

    reason: str
    kind: str = "model"

    async def run(self, ctx: Ctx) -> BuildContext | Finalize:
        s, d = ctx.state, ctx.deps
        s.execution_state.counters.errors += 1
        s.execution_state.counters.consecutive_failures += 1
        if self.kind == "patch":
            s.execution_state.counters.patches_rejected += 1
        s.consecutive_failures += 1
        d.log.warning("step failure", step=s.step, kind=self.kind, reason=self.reason[:200])
        with d.store.transaction():
            d.store.save_error(s.run_id, s.step, self.kind, self.reason)
            d.store.append_event(s.run_id, s.step, EventType.ERROR, f"{self.kind} failure: {self.reason[:200]}", {"kind": self.kind, "reason": self.reason})
            d.store.update_step(s.run_id, s.step, phase="failed")
        if s.consecutive_failures >= d.config.max_decision_failures:
            return Finalize(reason="too_many_failures", status=RunStatus.FAILED, summary=self.reason)
        # Bounded feedback: the model sees ONLY this reason plus current state next step (no transcript).
        obs = _make_observation(
            s, ObservationKind.RUNTIME, "runtime",
            f"Your previous decision was not applied. Reason: {self.reason}\n"
            f"The execution state is unchanged (version {s.execution_state.state_version}). Produce a corrected decision.",
            data={"failure_kind": self.kind, "consecutive_failures": s.consecutive_failures},
        )
        return await _advance(ctx, obs, EventType.OBSERVATION, action_id=None)


@dataclass
class Finalize(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    reason: str
    status: RunStatus
    summary: str = ""
    completion: dict[str, Any] | None = None

    async def run(self, ctx: Ctx) -> End[RunOutcome]:
        s, d = ctx.state, ctx.deps
        es = s.execution_state
        final_status = self.status
        if es.status != final_status:
            es = es.model_copy(update={"status": final_status, "updated_at": utcnow(), "state_version": es.state_version + 1})
            with d.store.transaction():
                try:
                    d.store.commit_state(es, expected_version=s.execution_state.state_version, patch_id=None, step=s.step)
                    s.execution_state = es
                except StaleWriteError:
                    pass
        terminal = final_status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
        etype = {RunStatus.COMPLETED: EventType.RUN_COMPLETED, RunStatus.FAILED: EventType.RUN_FAILED}.get(final_status, EventType.RUN_PAUSED)
        outcome = RunOutcome(
            run_id=s.run_id, status=final_status, reason=self.reason, summary=self.summary, steps=s.step,
            state_version=s.execution_state.state_version, completion=self.completion,
        )
        with d.store.transaction():
            d.store.append_event(s.run_id, s.step, etype, f"run {final_status.value}: {self.reason}", outcome.model_dump(mode="json"))
            d.store.update_run(s.run_id, status=final_status.value, last_step=s.step,
                               finished_at=utcnow() if terminal else None, outcome=outcome.model_dump(mode="json"))
            _flush_step_metrics(ctx)
        d.log.info("run finalized", step=s.step, status=final_status.value, reason=self.reason)
        return End(outcome)


@dataclass
class Enter(BaseNode[RuntimeGraphState, RuntimeDeps, RunOutcome]):
    """Single graph entry point. ``Runtime`` chooses where to (re)enter the lifecycle."""

    target: BuildContext | ApplyDecision | ExecuteAction | CaptureObservation | Finalize

    async def run(self, ctx: Ctx) -> BuildContext | ApplyDecision | ExecuteAction | CaptureObservation | Finalize:
        return self.target


# ---- shared helpers ------------------------------------------------------------------------


def _action_summary(action: Any) -> str:
    if isinstance(action, ToolAction):
        return f"tool {action.tool_name} {json.dumps(action.arguments, ensure_ascii=False)[:150]}"
    if isinstance(action, RequestHumanInput):
        return f"human_input: {action.question[:120]}"
    return f"continue: {getattr(action, 'note', '')[:120]}"


def _make_observation(
    s: RuntimeGraphState, kind: ObservationKind, source: str, full: str, *, data: dict[str, Any] | None = None,
    ref_event_id: str | None = None,
) -> Observation:
    limit = s.max_observation_chars
    content, truncated = full, False
    if len(full) > limit:
        head = limit * 2 // 3
        content = full[:head] + f"\n… [{len(full) - limit} chars omitted; retrieve event {ref_event_id or '(archived)'} for the full content] …\n" + full[-(limit - head):]
        truncated = True
    return Observation(run_id=s.run_id, step=s.step + 1, kind=kind, source=source, content=content, truncated=truncated,
                       full_length=len(full), event_id=ref_event_id, data=data)


async def _advance(ctx: Ctx, obs: Observation, etype: EventType, *, action_id: str | None, full_content: str | None = None) -> BuildContext | Finalize:
    """Archive the new observation, close the current step and open the next one (one transaction)."""
    s, d = ctx.state, ctx.deps
    with d.store.transaction():  # closing this step and opening the next is ONE transaction (crash-safe)
        ev = d.store.append_event(
            s.run_id, obs.step, etype, f"observation ({obs.kind.value} from {obs.source}): {obs.content[:200]}",
            {"observation_id": obs.id, "kind": obs.kind.value, "source": obs.source, "content": (full_content or obs.content)[:20_000],
             "truncated": obs.truncated, "data": obs.data, "tool_event_id": obs.event_id},
            ref_table="observations", ref_id=obs.id,
        )
        if obs.event_id is None:
            obs.event_id = ev.event_id
        d.store.save_observation(obs, full_content or obs.content)
        d.store.update_step(s.run_id, s.step, phase="done")
        return await _next_step(ctx, observation=obs, retrieved=None)


async def _next_step(ctx: Ctx, *, observation: Observation, retrieved: list[MemoryResult] | None) -> BuildContext | Finalize:
    s, d = ctx.state, ctx.deps
    with d.store.transaction():
        _flush_step_metrics(ctx)
        s.execution_state.counters.steps_completed += 1
        # Counter bookkeeping is a runtime-owned mutation, committed as its own state version for auditability.
        es = s.execution_state.model_copy(update={"state_version": s.execution_state.state_version + 1, "updated_at": utcnow()})
        d.store.commit_state(es, expected_version=s.execution_state.state_version, patch_id=None, step=s.step)
        s.execution_state = es
        next_step = s.step + 1
        obs = observation.model_copy(update={"step": next_step}) if observation.step != next_step else observation
        d.store.create_step(s.run_id, next_step, observation_id=obs.id, retrieved=retrieved, state_version_before=es.state_version)
    s.begin_step(next_step, obs, retrieved or [])
    return BuildContext()


def _flush_step_metrics(ctx: Ctx) -> None:
    s, d = ctx.state, ctx.deps
    from mnestic.state.apply import state_size_bytes

    elapsed = int((time.monotonic() - s.step_started) * 1000) if s.step_started else None
    d.store.save_step_metrics(
        s.run_id, s.step,
        context_chars=s.context.char_count if s.context else None,
        context_tokens_est=s.context.approx_tokens if s.context else None,
        input_tokens=s.last_usage.input_tokens if s.last_usage else None,
        output_tokens=s.last_usage.output_tokens if s.last_usage else None,
        state_bytes=state_size_bytes(s.execution_state),
        model_calls=s.model_calls_this_step, tool_calls=s.tool_calls_this_step, retrievals=s.retrievals_this_step,
        patch_applied=s.patches_applied_this_step, elapsed_ms=elapsed,
    )


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))
