"""PydanticAI-backed reasoner. One fresh, bounded ``Agent.run`` per step — no message_history.

Verified against pydantic-ai-slim 2.37: ``Agent(model, output_type=..., retries=...)``,
``agent.run(prompt, instructions=..., model_settings=...)``, ``result.output``,
``result.usage`` (RunUsage), ``result.all_messages()``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model, infer_model
from pydantic_ai.settings import ModelSettings

from mnestic.agent.reasoner import ReasonerResult, UsageRecord
from mnestic.context.builder import ModelContext
from mnestic.models.decision import AgentDecision


def build_model(model: str | Model) -> Model:
    """Resolve a PydanticAI model from a string like ``openai:gpt-4o-mini`` or ``anthropic:claude-...``.

    OpenAI-compatible proxies/local servers (9Router, LiteLLM, llama-server, vLLM): set ``OPENAI_BASE_URL``
    (and ``OPENAI_API_KEY``) and use ``openai-chat:<model-name>`` — plain ``openai:`` selects the Responses API,
    which chat-completions-only servers do not implement. Ollama: ``ollama:<model>`` with ``OLLAMA_BASE_URL``.
    """
    if isinstance(model, Model):
        return model
    return infer_model(model)


class PydanticAIReasoner:
    def __init__(
        self,
        model: str | Model,
        *,
        output_mode: str = "tool",
        retries: int = 2,
        model_settings: dict[str, Any] | None = None,
        wall_clock_timeout: float | None = None,
    ):
        """``wall_clock_timeout`` bounds the *whole* step (all in-step retries) in seconds. Provider/httpx timeouts
        are per-read and do not fire when a proxy keeps the connection alive, so this is enforced with
        ``asyncio.wait_for`` regardless of what the transport does."""
        self.wall_clock_timeout = wall_clock_timeout
        self.model = build_model(model)
        self.model_name = getattr(self.model, "model_name", str(model))
        self.retries = retries
        self.model_settings = ModelSettings(**model_settings) if model_settings else None  # type: ignore[typeddict-item]
        output_type: Any
        if output_mode == "native":
            output_type = NativeOutput(AgentDecision, name="AgentDecision")
        elif output_mode == "prompted":
            output_type = PromptedOutput(AgentDecision, name="AgentDecision")
        else:
            output_type = ToolOutput(AgentDecision, name="agent_decision", description="Submit the decision for this step")
        # Instructions are supplied per run (they contain the skill spec); deps carry the expected state version.
        self.agent: Agent[int, AgentDecision] = Agent(
            self.model,
            output_type=output_type,
            deps_type=int,
            retries=retries,
            name="mnestic-step",
        )

        @self.agent.output_validator
        async def _check_version(ctx: Any, decision: AgentDecision) -> AgentDecision:
            expected = ctx.deps
            if decision.state_patch.expected_state_version != expected:
                raise ModelRetry(
                    f"state_patch.expected_state_version must be {expected} (the version shown in <execution_state>), "
                    f"got {decision.state_patch.expected_state_version}. Resubmit the full decision."
                )
            return decision

    async def decide(self, context: ModelContext) -> ReasonerResult:
        started = time.monotonic()
        try:
            run = self.agent.run(
                context.prompt,
                instructions=context.instructions,
                deps=context.state_version,
                model_settings=self.model_settings,
            )
            result = await (asyncio.wait_for(run, self.wall_clock_timeout) if self.wall_clock_timeout else run)
        except TimeoutError:
            return ReasonerResult(
                error=f"model call exceeded the wall-clock timeout of {self.wall_clock_timeout:.0f}s", error_kind="timeout",
                model_name=self.model_name, duration_ms=_ms(started),
            )
        except (UnexpectedModelBehavior, UsageLimitExceeded, ValidationError) as exc:
            return ReasonerResult(
                error=f"{type(exc).__name__}: {exc}", error_kind="validation", model_name=self.model_name,
                duration_ms=_ms(started),
            )
        except Exception as exc:
            return ReasonerResult(
                error=f"{type(exc).__name__}: {exc}", error_kind="model", model_name=self.model_name, duration_ms=_ms(started)
            )
        usage = result.usage
        # Messages from *this* step are archived for audit and then discarded — never re-sent.
        raw = json.loads(ModelMessagesTypeAdapter.dump_json(result.all_messages()))
        return ReasonerResult(
            decision=result.output,
            model_name=self.model_name,
            usage=UsageRecord(
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests,
                cache_read_tokens=usage.cache_read_tokens or 0, cache_write_tokens=usage.cache_write_tokens or 0,
                retries=max(0, usage.requests - 1),
            ),
            raw_messages=raw,
            duration_ms=_ms(started),
        )


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
