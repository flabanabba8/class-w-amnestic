"""Reasoner protocol and a deterministic scripted implementation for tests/benchmarks."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from mnestic.context.builder import ModelContext
from mnestic.models.decision import AgentDecision


class UsageRecord(BaseModel):
    input_tokens: int | None = None
    """Uncached input tokens as reported by the provider (what is typically billed at full price)."""
    output_tokens: int | None = None
    cache_read_tokens: int = 0
    """Prompt-cache hits: tokens the model still processed but the provider read from cache."""
    cache_write_tokens: int = 0
    requests: int = 1
    retries: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Everything the model was shown: uncached + cached."""
        return (self.input_tokens or 0) + self.cache_read_tokens + self.cache_write_tokens


class ReasonerResult(BaseModel):
    decision: AgentDecision | None = None
    error: str | None = None
    error_kind: str | None = None  # validation | model | timeout | unknown
    model_name: str = "unknown"
    usage: UsageRecord = Field(default_factory=UsageRecord)
    raw_messages: list[dict[str, Any]] = Field(default_factory=list, description="Archived, never re-sent")
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.decision is not None


class Reasoner(Protocol):
    model_name: str

    async def decide(self, context: ModelContext) -> ReasonerResult: ...


ScriptFn = Callable[[ModelContext], AgentDecision | dict[str, Any] | Awaitable[AgentDecision | dict[str, Any]]]


class ScriptedReasoner:
    """Deterministic reasoner driven by a Python callable. Used by tests, benchmarks and ``--model mock``.

    The callable receives the *exact* ModelContext the real model would receive, so
    scripted reasoners are also a test that the context contains what they need.
    """

    def __init__(self, script: ScriptFn, *, model_name: str = "mock"):
        self.script = script
        self.model_name = model_name
        self.calls = 0

    async def decide(self, context: ModelContext) -> ReasonerResult:
        self.calls += 1
        started = time.monotonic()
        usage = UsageRecord(input_tokens=context.approx_tokens, output_tokens=None, requests=1)
        try:
            out = self.script(context)
            if hasattr(out, "__await__"):
                out = await out  # type: ignore[misc]
            decision = out if isinstance(out, AgentDecision) else AgentDecision.model_validate(out)
        except ValidationError as exc:
            return ReasonerResult(error=f"invalid decision: {exc.errors(include_url=False)}", error_kind="validation",
                                  model_name=self.model_name, usage=usage, duration_ms=_ms(started))
        except Exception as exc:
            return ReasonerResult(error=f"{type(exc).__name__}: {exc}", error_kind="model", model_name=self.model_name,
                                  usage=usage, duration_ms=_ms(started))
        usage.output_tokens = len(decision.model_dump_json()) // 4
        return ReasonerResult(decision=decision, model_name=self.model_name, usage=usage, duration_ms=_ms(started))


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
