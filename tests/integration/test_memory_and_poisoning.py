"""§18 stale-information (context poisoning) and §19 forgotten-information recovery."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from skillstate.models.archive import EventType
from skillstate.models.state import RunStatus
from skillstate.tools import default_registry
from skillstate.tools.base import Tool, ToolContext, ToolResult
from tests.integration.helpers import complete, decision, obs_info, tool


class PortProbe(Tool):
    """Step 1 reports port 8000; later an authoritative reading says 9000."""

    name: ClassVar[str] = "port_probe"
    description: ClassVar[str] = "probe"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        n: int

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        port = 8000 if args.n == 1 else 9000
        return ToolResult(ok=True, output=f"config reading #{args.n}: server_port = {port} (authoritative={'yes' if args.n > 1 else 'no'})")


async def test_stale_fact_leaves_context_but_stays_in_archive(make_runtime, store, simple_skill):
    contexts = []
    skill = simple_skill.model_copy(update={"required_tools": ["port_probe"]})

    def script(ctx):
        contexts.append(ctx)
        kind, ev, text = obs_info(ctx)
        if kind == "task_input":
            return tool(ctx, "port_probe", n=1)
        if "reading #1" in text:
            return tool(ctx, "port_probe", [{"op": "add_fact", "id": "fact_port", "statement": "server_port = 8000", "evidence_event_ids": [ev]}], n=2)
        if "reading #2" in text:
            return tool(ctx, "port_probe", [{"op": "supersede_fact", "fact_id": "fact_port", "statement": "server_port = 9000", "evidence_event_ids": [ev]}], n=3)
        return complete(ctx, [{"op": "set_observation_summary", "summary": "third reading consistent"}])

    reg = default_registry()
    reg.register(PortProbe())
    out = await make_runtime(script, tools=reg).start(skill, "find the port")
    assert out.status == RunStatus.COMPLETED
    state = store.get_state(out.run_id)
    assert [f.statement for f in state.verified_facts] == ["server_port = 9000"]
    # Step 3's context (after the supersede) must not mention 8000 anywhere.
    ctx3 = contexts[3]
    assert "9000" in ctx3.sections["execution_state"] and "8000" not in ctx3.full_text()
    # ...but the archive still has it, recoverable on demand.
    hits, total = store.search_events(out.run_id, "8000")
    assert total >= 1 and any("server_port = 8000" in (e.payload.get("output") or e.payload.get("content") or "") for e in hits)
    superseded = [e for e in store.list_events(out.run_id, limit=500, event_type=EventType.PATCH_APPLIED.value) if e.payload.get("kind") == "fact"]
    assert superseded and superseded[0].payload["item"]["statement"] == "server_port = 8000"
    assert store.get_state_at_version(out.run_id, superseded[0].step * 0 + 3).verified_facts[0].statement == "server_port = 8000"


async def test_forgotten_information_recovered_via_explicit_retrieval(make_runtime, store, simple_skill):
    """The README mentions a license key at step 1; the policy does not keep it. At step 4 it needs it."""
    contexts = []

    def script(ctx):
        contexts.append(ctx)
        kind, ev, text = obs_info(ctx)
        state_json = ctx.sections["execution_state"]
        if kind == "task_input":
            return tool(ctx, "read_text_file", path="README.md")
        if "ZETA-42" in text and "retrieved_archival_evidence" not in ctx.sections:
            # deliberately record only that the README exists — the key is 'forgotten' from working state
            return tool(ctx, "list_directory", [{"op": "add_fact", "id": "readme", "statement": "README.md was read", "evidence_event_ids": [ev]}], path=".")
        if "retrieved_archival_evidence" in ctx.sections:
            evidence = ctx.sections["retrieved_archival_evidence"]
            assert "ZETA-42" in evidence
            evt = evidence.split('<event id="', 1)[1].split('"', 1)[0]
            return complete(ctx, [{"op": "add_fact", "id": "key", "statement": "license key is ZETA-42", "evidence_event_ids": [evt]}], answer="ZETA-42")
        assert "ZETA-42" not in state_json and "ZETA-42" not in ctx.full_text()  # genuinely forgotten
        return decision(ctx, memory_query={"query_type": "search", "text": "license key", "limit": 3, "reason": "need the key"})

    out = await make_runtime(script).start(simple_skill, "what is the license key?")
    assert out.status == RunStatus.COMPLETED and out.completion["final_answer"] == "ZETA-42"
    # exactly one context contained retrieved evidence, and it was bounded (3 events max)
    with_evidence = [c for c in contexts if "retrieved_archival_evidence" in c.sections]
    assert len(with_evidence) == 1 and len(with_evidence[0].retrieved_event_ids) <= 3
    assert "ZETA-42" not in contexts[2].full_text()  # the step before retrieval had forgotten it
    retrievals = store.list_events(out.run_id, limit=500, event_type=EventType.MEMORY_RETRIEVAL.value)
    assert len(retrievals) == 1 and retrievals[0].payload["query"]["text"] == "license key"
    assert store.get_state(out.run_id).counters.retrievals == 1
