"""Runtime: builds the lifecycle graph, starts runs and resumes them from SQLite alone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_graph import GraphBuilder

from mnestic.agent.reasoner import Reasoner
from mnestic.config import RuntimeConfig
from mnestic.context.builder import ContextBuilder
from mnestic.graph import nodes
from mnestic.graph.nodes import (
    ApplyDecision,
    AwaitHuman,
    BuildContext,
    CaptureObservation,
    Enter,
    ExecuteAction,
    Finalize,
    HandleFailure,
    Reason,
    RetrieveMemory,
)
from mnestic.graph.state import RunOutcome, RuntimeDeps, RuntimeGraphState
from mnestic.memory.base import Retriever
from mnestic.memory.retrieval import ArchiveRetriever
from mnestic.models.action import ToolAction
from mnestic.models.archive import EventType, MemoryResult, RunMetadata
from mnestic.models.common import new_id, utcnow
from mnestic.models.decision import AgentDecision
from mnestic.models.observation import Observation, ObservationKind
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import TERMINAL_STATUSES, Budgets, ExecutionState, Objective, RunStatus
from mnestic.observability.logging import get_logger
from mnestic.storage.store import Store
from mnestic.tools.base import ToolRegistry, ToolResult


def build_graph() -> Any:
    g: Any = GraphBuilder(
        name="mnestic_lifecycle", state_type=RuntimeGraphState, deps_type=RuntimeDeps, input_type=Enter, output_type=RunOutcome,
        auto_instrument=False,
    )
    g.add(
        g.node(Enter), g.node(BuildContext), g.node(Reason), g.node(ApplyDecision), g.node(RetrieveMemory),
        g.node(ExecuteAction), g.node(CaptureObservation), g.node(AwaitHuman), g.node(HandleFailure), g.node(Finalize),
    )
    # Single start edge; ``Enter`` dispatches to the resume point chosen by ``Runtime``.
    g.add(g.edge_from(g.start_node).to(Enter))
    return g.build()


LIFECYCLE_GRAPH = build_graph()


class Runtime:
    def __init__(
        self,
        config: RuntimeConfig,
        store: Store,
        *,
        reasoner: Reasoner,
        tools: ToolRegistry,
        retriever: Retriever | None = None,
        graph: Any = None,
    ):
        self.config = config
        self.store = store
        self.reasoner = reasoner
        self.tools = tools
        self.retriever = retriever or ArchiveRetriever(store, excerpt_chars=config.max_retrieved_excerpt_chars)
        self.graph = graph or LIFECYCLE_GRAPH
        self.log = get_logger("runtime")

    # ---- public API ----------------------------------------------------------------------

    async def start(
        self, skill: SkillSpecification, task_input: str, *, run_id: str | None = None, max_steps: int | None = None,
        workspace_root: Path | None = None,
    ) -> RunOutcome:
        run_id = run_id or new_id("run")
        workspace = (workspace_root or self.config.resolved_workspace()).resolve()
        missing = self.tools.missing(skill.required_tools)
        if missing:
            raise ValueError(f"skill {skill.key} requires unavailable tools: {missing}")
        state = ExecutionState(
            run_id=run_id, skill_id=skill.skill_id, skill_version=skill.version, state_schema_version=skill.state_schema_version,
            objective=Objective(statement=task_input[:2000] or skill.description[:2000], success_criteria=list(skill.completion_criteria[:20])),
            status=RunStatus.RUNNING, current_phase=skill.initial_phase, constraints=list(skill.constraints[:50]),
            budgets=Budgets(max_steps=skill.default_max_steps),
            environment=_env(workspace),
        )
        with self.store.transaction():
            self.store.upsert_skill(skill)
            self.store.create_run(RunMetadata(
                run_id=run_id, skill_id=skill.skill_id, skill_version=skill.version, skill_content_hash=skill.content_hash,
                status=RunStatus.RUNNING.value, created_at=utcnow(), updated_at=utcnow(), workspace_root=str(workspace),
                model_name=self.reasoner.model_name, task_input=task_input,
            ))
            self.store.init_state(state)
            self.store.append_event(run_id, 0, EventType.RUN_CREATED, f"run created for skill {skill.key}",
                                    {"skill_id": skill.skill_id, "skill_version": skill.version, "content_hash": skill.content_hash,
                                     "model": self.reasoner.model_name, "workspace_root": str(workspace)})
            ev = self.store.append_event(run_id, 0, EventType.TASK_INPUT, f"task input: {task_input[:200]}", {"content": task_input})
            obs = Observation(run_id=run_id, step=0, kind=ObservationKind.TASK_INPUT, source="task", content=task_input[: self.config.max_observation_chars],
                              truncated=len(task_input) > self.config.max_observation_chars, full_length=len(task_input), event_id=ev.event_id)
            self.store.save_observation(obs, task_input)
            self.store.create_step(run_id, 0, observation_id=obs.id, retrieved=None, state_version_before=0)
        self.log.info("run created", run_id=run_id, skill=skill.key)
        gstate = self._graph_state(run_id, state, 0, obs, [], max_steps)
        return await self._run_graph(gstate, skill, workspace, BuildContext())

    async def resume(self, run_id: str, *, human_input: str | None = None, max_steps: int | None = None) -> RunOutcome:
        """Reconstruct the runtime from SQLite (no transcript replay) and continue."""
        meta = self.store.get_run(run_id)
        if meta is None:
            raise KeyError(f"run {run_id} not found")
        skill = self.store.get_skill(meta.skill_id, meta.skill_version)
        if skill is None:
            raise RuntimeError(f"skill {meta.skill_id}@{meta.skill_version} missing from database")
        if skill.content_hash != meta.skill_content_hash:
            raise RuntimeError("skill content hash changed since the run started; skills are immutable during a run")
        state = self.store.get_state(run_id)
        if state.status in TERMINAL_STATUSES:
            raise RuntimeError(f"run {run_id} is {state.status.value}; nothing to resume")
        step_rec = self.store.latest_step(run_id)
        if step_rec is None:
            raise RuntimeError("run has no steps")
        workspace = Path(meta.workspace_root)
        self.store.append_event(run_id, step_rec.step, EventType.RUN_RESUMED,
                                f"run resumed at step {step_rec.step} phase {step_rec.phase}", {"phase": step_rec.phase, "human_input": human_input is not None})
        step = step_rec.step
        gstate = self._graph_state(run_id, state, step, None, step_rec.retrieved, max_steps)
        phase = step_rec.phase

        if phase == "awaiting_human" or state.status == RunStatus.WAITING_FOR_HUMAN:
            if human_input is None:
                return RunOutcome(run_id=run_id, status=RunStatus.WAITING_FOR_HUMAN, reason="waiting_for_human",
                                  summary="provide --input to answer the agent's question", steps=step, state_version=state.state_version)
            with self.store.transaction():
                state = state.model_copy(update={"status": RunStatus.RUNNING, "state_version": state.state_version + 1, "updated_at": utcnow()})
                self.store.commit_state(state, expected_version=state.state_version - 1, patch_id=None, step=step)
                gstate.execution_state = state
                self.store.append_event(run_id, step, EventType.HUMAN_RESPONSE, f"human response: {human_input[:200]}", {"content": human_input})
                if step_rec.action_id:
                    self.store.update_action(step_rec.action_id, status="executed")
            obs = nodes._make_observation(gstate, ObservationKind.HUMAN_INPUT, "human", human_input)
            gstate.decision = None
            entry = await nodes._advance(_ctx(gstate, self._deps(skill, workspace)), obs, EventType.OBSERVATION, action_id=step_rec.action_id, full_content=human_input)
            return await self._run_graph(gstate, skill, workspace, entry)

        if state.status != RunStatus.RUNNING:
            with self.store.transaction():
                state = state.model_copy(update={"status": RunStatus.RUNNING, "state_version": state.state_version + 1, "updated_at": utcnow()})
                self.store.commit_state(state, expected_version=state.state_version - 1, patch_id=None, step=step)
            gstate.execution_state = state

        if phase in {"observation_ready", "context_built", "failed"}:
            obs = self._load_observation(step_rec.observation_id)
            gstate.begin_step(step, obs, step_rec.retrieved)
            return await self._run_graph(gstate, skill, workspace, BuildContext())

        if phase == "decision_recorded":
            obs = self._load_observation(step_rec.observation_id)
            gstate.begin_step(step, obs, step_rec.retrieved)
            gstate.decision = AgentDecision.model_validate(step_rec.decision)
            return await self._run_graph(gstate, skill, workspace, ApplyDecision())

        if phase in {"state_committed", "action_pending"}:
            obs = self._load_observation(step_rec.observation_id)
            gstate.begin_step(step, obs, step_rec.retrieved)
            gstate.decision = AgentDecision.model_validate(step_rec.decision)
            gstate.action_id = step_rec.action_id
            if gstate.decision.memory_query is not None:
                return await self._run_graph(gstate, skill, workspace, RetrieveMemory())
            if isinstance(gstate.decision.action, ToolAction):
                if gstate.action_id is None:
                    return await self._run_graph(gstate, skill, workspace, ApplyDecision())
                return await self._run_graph(gstate, skill, workspace, ExecuteAction())
            return await self._run_graph(gstate, skill, workspace, ApplyDecision())

        if phase == "action_dispatched":
            # The tool was started but never finished: result unknown. Not exactly-once — documented.
            obs = self._load_observation(step_rec.observation_id)
            gstate.begin_step(step, obs, step_rec.retrieved)
            gstate.decision = AgentDecision.model_validate(step_rec.decision)
            gstate.action_id = step_rec.action_id
            open_exec = self.store.find_open_tool_execution(run_id, step)
            action = gstate.decision.action
            assert isinstance(action, ToolAction)
            with self.store.transaction():
                ev = self.store.append_event(run_id, step, EventType.TOOL_INTERRUPTED,
                                             f"tool interrupted: {action.tool_name} (process crashed during execution; result unknown)",
                                             {"tool": action.tool_name, "arguments": action.arguments, "execution_id": open_exec.execution_id if open_exec else None})
                if open_exec:
                    self.store.finish_tool_execution(open_exec.execution_id, status="interrupted", output=None,
                                                     error="interrupted by process crash", data=None, event_id=ev.event_id, duration_ms=None)
                if gstate.action_id:
                    self.store.update_action(gstate.action_id, status="interrupted")
            result = ToolResult(ok=False, error="tool execution was interrupted by a process restart; its side effects are unknown. "
                                                  "Verify before retrying.")
            gstate.tool_result = (open_exec.execution_id if open_exec else "unknown", result, 0)
            return await self._run_graph(gstate, skill, workspace, CaptureObservation())

        if phase == "done":
            # Crash after the step closed but before the next step row was created cannot happen (same transaction);
            # a 'done' latest step means the run finished or paused normally.
            return RunOutcome(run_id=run_id, status=state.status, reason="nothing_to_resume", steps=step, state_version=state.state_version)

        raise RuntimeError(f"unknown step phase {phase!r}")

    # ---- internals -------------------------------------------------------------------------

    def _graph_state(self, run_id: str, state: ExecutionState, step: int, obs: Observation | None,
                     retrieved: list[MemoryResult], max_steps: int | None) -> RuntimeGraphState:
        return RuntimeGraphState(
            run_id=run_id, execution_state=state, step=step, observation=obs, retrieved=retrieved,
            max_steps_this_invocation=max_steps or 10_000, max_observation_chars=self.config.max_observation_chars,
        )

    def _deps(self, skill: SkillSpecification, workspace: Path) -> RuntimeDeps:
        builder = ContextBuilder(
            tool_specs=self.tools.specs(skill.required_tools), max_retrieved_chars=self.config.max_retrieved_chars,
            max_retrieved_excerpt_chars=self.config.max_retrieved_excerpt_chars, allowed_ops=skill.allowed_ops,
        )
        return RuntimeDeps(config=self.config, store=self.store, skill=skill, reasoner=self.reasoner, tools=self.tools,
                           context_builder=builder, retriever=self.retriever, workspace_root=workspace, log=self.log)

    def _load_observation(self, observation_id: str | None) -> Observation:
        if observation_id is None:
            raise RuntimeError("step has no observation to resume from")
        obs = self.store.get_observation(observation_id)
        if obs is None:
            raise RuntimeError(f"observation {observation_id} missing")
        return obs

    async def _run_graph(self, gstate: RuntimeGraphState, skill: SkillSpecification, workspace: Path, entry: Any) -> RunOutcome:
        deps = self._deps(skill, workspace)
        try:
            return await self.graph.run(state=gstate, deps=deps, inputs=Enter(target=entry))
        except Exception as exc:
            self.store.save_error(gstate.run_id, gstate.step, "runtime_crash", str(exc), nodes.format_exception(exc))
            raise


def _ctx(gstate: RuntimeGraphState, deps: RuntimeDeps) -> Any:
    from pydantic_graph import GraphRunContext

    return GraphRunContext(state=gstate, deps=deps)


def _env(workspace: Path) -> Any:
    from mnestic.models.state import Environment

    return Environment(properties={"workspace_root": str(workspace)})
