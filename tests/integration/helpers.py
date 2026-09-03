"""Helpers for integration tests: decision builders and fault injection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from mnestic.benchmarks.scripts import observation_attr, observation_text, parse_state
from mnestic.context.builder import ModelContext
from mnestic.storage.db import Database
from mnestic.storage.store import Store
from mnestic.tools.base import Tool, ToolContext, ToolResult


class SimulatedCrash(BaseException):
    """Raised to emulate a process dying. BaseException so nothing 'handles' it."""


def decision(ctx: ModelContext, ops: list[dict[str, Any]] | None = None, **control: Any) -> dict[str, Any]:
    state = parse_state(ctx)
    return {"rationale_summary": "test", "state_patch": {"expected_state_version": state["state_version"], "ops": ops or []}, **control}


def complete(ctx: ModelContext, ops: list[dict[str, Any]] | None = None, summary: str = "done", answer: str | None = None) -> dict[str, Any]:
    return decision(ctx, ops, completion={"outcome": "success", "summary": summary, "final_answer": answer})


def tool(ctx: ModelContext, name: str, ops: list[dict[str, Any]] | None = None, **arguments: Any) -> dict[str, Any]:
    return decision(ctx, ops, action={"kind": "tool", "tool_name": name, "arguments": arguments})


def obs_info(ctx: ModelContext) -> tuple[str, str, str]:
    """(kind, event_id, text) of the newest observation."""
    return observation_attr(ctx, "kind"), observation_attr(ctx, "event_id"), observation_text(ctx)


class StepCounter:
    """Scripts that need a 'which call is this' counter without touching state."""

    def __init__(self) -> None:
        self.n = 0

    def tick(self) -> int:
        self.n += 1
        return self.n


class CrashingDatabase(Database):
    """Commits durably, then 'crashes' right after the commit that satisfies ``should_crash``."""

    def __init__(self, path: Path, should_crash: Callable[[], bool]):
        self.should_crash = should_crash
        super().__init__(path)

    def transaction(self):  # type: ignore[override]
        from contextlib import contextmanager

        parent = super().transaction

        @contextmanager
        def _tx():
            outermost = self._depth == 0
            with parent():
                yield self.conn
            if outermost and self.should_crash():
                raise SimulatedCrash()

        return _tx()


class PhaseCrashStore(Store):
    """Crash immediately after the transaction that persisted ``crash_after_phase``."""

    def __init__(self, path: Path, crash_after_phase: str | None, *, on_step: int | None = None):
        self.crash_after_phase = crash_after_phase
        self.on_step = on_step
        self.armed = False
        self.crashed = False
        super().__init__(CrashingDatabase(path, self._should_crash))

    def _should_crash(self) -> bool:
        if self.armed and not self.crashed:
            self.crashed = True
            return True
        return False

    def update_step(self, run_id: str, step: int, *, phase: str, **kw: Any) -> None:
        super().update_step(run_id, step, phase=phase, **kw)
        if phase == self.crash_after_phase and (self.on_step is None or step == self.on_step) and not self.crashed:
            self.armed = True


class CrashingTool(Tool):
    name: ClassVar[str] = "flaky"
    description: ClassVar[str] = "Crashes the process on first call, succeeds afterwards."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int = 0

    calls: int = 0

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        CrashingTool.calls += 1
        if CrashingTool.calls == 1:
            raise SimulatedCrash()
        return ToolResult(ok=True, output=f"flaky ok {args.value}", data={"value": args.value})
