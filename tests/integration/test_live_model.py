"""Optional live-model check. Enable with SKILLSTATE_LIVE_TESTS=1 and SKILLSTATE_MODEL=<provider:model> + credentials."""

from __future__ import annotations

import os

import pytest

from skillstate.agent.pydantic_ai_reasoner import PydanticAIReasoner
from skillstate.graph.runtime import Runtime
from skillstate.models.state import RunStatus
from skillstate.skills.loader import SkillRegistry
from skillstate.tools import default_registry

pytestmark = pytest.mark.live


@pytest.mark.skipif(os.environ.get("SKILLSTATE_LIVE_TESTS") != "1", reason="live model tests disabled")
async def test_live_codebase_research(config, store, registry: SkillRegistry, workspace):
    from pydantic_ai import models

    models.ALLOW_MODEL_REQUESTS = True
    model = os.environ.get("SKILLSTATE_MODEL", "openai:gpt-4o-mini")
    reasoner = PydanticAIReasoner(model, output_mode=os.environ.get("SKILLSTATE_OUTPUT_MODE", "tool"), retries=3)
    out = await Runtime(config, store, reasoner=reasoner, tools=default_registry()).start(
        registry.get("codebase-research"), "Where is `PORT` configured and what is its value?", max_steps=25)
    assert out.status in {RunStatus.COMPLETED, RunStatus.PAUSED}
    calls = store.get_model_calls(out.run_id)
    assert calls and all(c["input_tokens"] for c in calls)
