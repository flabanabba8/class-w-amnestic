"""Optional live-model check. Enable with MNESTIC_LIVE_TESTS=1 and MNESTIC_MODEL=<provider:model> + credentials."""

from __future__ import annotations

import os

import pytest

from mnestic.agent.pydantic_ai_reasoner import PydanticAIReasoner
from mnestic.graph.runtime import Runtime
from mnestic.models.state import RunStatus
from mnestic.skills.loader import SkillRegistry
from mnestic.tools import default_registry

pytestmark = pytest.mark.live


@pytest.mark.skipif(os.environ.get("MNESTIC_LIVE_TESTS") != "1", reason="live model tests disabled")
async def test_live_codebase_research(config, store, registry: SkillRegistry, workspace):
    from pydantic_ai import models

    models.ALLOW_MODEL_REQUESTS = True
    model = os.environ.get("MNESTIC_MODEL", "openai:gpt-4o-mini")
    reasoner = PydanticAIReasoner(model, output_mode=os.environ.get("MNESTIC_OUTPUT_MODE", "tool"), retries=3)
    out = await Runtime(config, store, reasoner=reasoner, tools=default_registry()).start(
        registry.get("codebase-research"), "Where is `PORT` configured and what is its value?", max_steps=25)
    assert out.status in {RunStatus.COMPLETED, RunStatus.PAUSED}
    calls = store.get_model_calls(out.run_id)
    assert calls and all(c["input_tokens"] for c in calls)
