"""PydanticAI adapter with a FunctionModel: structured output, retries, failures, and — above all —
proof that every step is a fresh bounded request with no message history."""

from __future__ import annotations

import json

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mnestic.agent.pydantic_ai_reasoner import PydanticAIReasoner
from mnestic.graph.runtime import Runtime
from mnestic.models.state import RunStatus
from mnestic.tools import default_registry

models.ALLOW_MODEL_REQUESTS = False


def _state_version(messages: list[ModelMessage]) -> int:
    prompt = next(p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart))
    assert isinstance(prompt, str)
    return int(prompt.split('<execution_state version="', 1)[1].split('"', 1)[0])


def _decision(v: int, **control) -> dict:
    return {"rationale_summary": "fn", "state_patch": {"expected_state_version": v, "ops": [{"op": "set_observation_summary", "summary": f"v{v}"}]}, **control}


async def test_fresh_request_per_step_no_history(config, store, simple_skill):
    requests: list[list[ModelMessage]] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(messages)
        v = _state_version(messages)
        n = len(requests)
        if n >= 4:
            control = {"completion": {"outcome": "success", "summary": "done"}}
        elif n == 1:
            control = {"action": {"kind": "tool", "tool_name": "read_text_file", "arguments": {"path": "README.md"}}}
        else:
            control = {"action": {"kind": "tool", "tool_name": "list_directory", "arguments": {"path": "."}}}
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=_decision(v, **control))])

    reasoner = PydanticAIReasoner(FunctionModel(fn), retries=1)
    out = await Runtime(config, store, reasoner=reasoner, tools=default_registry()).start(simple_skill, "list things")
    assert out.status == RunStatus.COMPLETED and len(requests) == 4
    for msgs in requests:
        # exactly one request message, containing exactly one user prompt: nothing accumulated
        assert [type(m).__name__ for m in msgs] == ["ModelRequest"]
        assert [type(p).__name__ for p in msgs[0].parts] == ["UserPromptPart"]
    prompts = [m[0].parts[0].content for m in requests]
    assert prompts[3].count("<latest_observation") == 1
    # the README content observed at step 1 must not be in later prompts (only the state carries forward)
    assert "ZETA-42" in prompts[1] and "ZETA-42" not in prompts[2] and "ZETA-42" not in prompts[3]
    calls = store.get_model_calls(out.run_id)
    assert all(c["input_tokens"] and c["output_tokens"] and c["requests"] == 1 for c in calls)
    assert json.loads(calls[0]["raw_messages_json"])  # archived, never re-sent


async def test_malformed_output_then_retry(config, store, simple_skill):
    attempts = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts.append(len(messages))
        v = _state_version(messages)
        name = info.output_tools[0].name
        if len(attempts) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args={"state_patch": {"expected_state_version": "nope"}, "action": {"kind": "teleport"}})])
        if len(attempts) == 2:  # wrong version -> output validator -> ModelRetry
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=_decision(v + 5, completion={"outcome": "success", "summary": "s"}))])
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=_decision(v, completion={"outcome": "success", "summary": "s"}))])

    reasoner = PydanticAIReasoner(FunctionModel(fn), retries=3)
    out = await Runtime(config, store, reasoner=reasoner, tools=default_registry()).start(simple_skill, "go")
    assert out.status == RunStatus.COMPLETED
    assert attempts == [1, 3, 5]  # retries happen *within* the step (message list grows only inside one run)
    call = store.get_model_calls(out.run_id)[0]
    assert call["requests"] == 3 and call["status"] == "ok"


async def test_model_failure_becomes_feedback_then_run_fails(config, store, simple_skill):
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="I refuse to use the tool")])  # never a structured decision

    cfg = config.model_copy(update={"max_decision_failures": 2})
    reasoner = PydanticAIReasoner(FunctionModel(fn), retries=0)
    out = await Runtime(cfg, store, reasoner=reasoner, tools=default_registry()).start(simple_skill, "go")
    assert out.status == RunStatus.FAILED and out.reason == "too_many_failures"
    types = [e.event_type.value for e in store.list_events(out.run_id, limit=500)]
    assert types.count("model.error") == 2 and "run.failed" in types
    obs = store.list_observations(out.run_id)
    assert obs[1].kind.value == "runtime" and "not applied" in obs[1].content


async def test_provider_exception_is_captured(config, store, simple_skill):
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ConnectionError("provider down")

    reasoner = PydanticAIReasoner(FunctionModel(fn), retries=0)
    from mnestic.context.builder import ContextBuilder
    from mnestic.models.observation import Observation, ObservationKind
    from mnestic.models.state import ExecutionState, Objective

    ctx = ContextBuilder().build(simple_skill, ExecutionState(run_id="r", skill_id="s", skill_version="1", objective=Objective(statement="o")),
                                 Observation(run_id="r", step=0, kind=ObservationKind.TASK_INPUT, source="task", content="x"))
    result = await reasoner.decide(ctx)
    assert not result.ok and result.error_kind == "model" and "provider down" in result.error


@pytest.mark.parametrize("mode", ["native", "prompted"])
async def test_other_output_modes_construct(mode):
    def fn(messages, info):
        return ModelResponse(parts=[TextPart(content="{}")])

    PydanticAIReasoner(FunctionModel(fn), output_mode=mode)
