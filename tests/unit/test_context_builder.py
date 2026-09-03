"""Regression guards for docs/CONTEXT_INVARIANTS.md (I1, I2, I5, I7)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

from mnestic.context.builder import SECTION_EVIDENCE, SECTION_OBSERVATION, SECTION_SKILL, SECTION_STATE, ContextBuilder, ToolSpec
from mnestic.models.archive import MemoryQuery, MemoryResult, RetrievedEvent
from mnestic.models.observation import Observation, ObservationKind
from mnestic.models.patch import StatePatch
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ExecutionState
from mnestic.state.apply import apply_patch

SRC = Path(__file__).resolve().parents[2] / "src" / "mnestic" / "context"


def _obs(step: int, content: str, kind: ObservationKind = ObservationKind.TOOL_RESULT) -> Observation:
    return Observation(id=f"obs_{step}", run_id="run_test", step=step, kind=kind, source="probe", content=content, event_id=f"evt_{step}")


def test_context_module_has_no_storage_dependency():
    """ContextBuilder cannot secretly pull history: it has no import path to storage."""
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not n.startswith(("mnestic.storage", "mnestic.memory", "mnestic.graph", "sqlite3")), f"{py}: imports {n}"


def test_sections_are_delimited_and_ordered(simple_skill: SkillSpecification, base_state: ExecutionState):
    ctx = ContextBuilder(tool_specs=[ToolSpec(name="read_text_file", description="read", parameters_schema={"properties": {"path": {"type": "string"}}, "required": ["path"]})]).build(
        simple_skill, base_state, _obs(3, "hello world"))
    text = ctx.full_text()
    for name in (SECTION_SKILL, SECTION_STATE, SECTION_OBSERVATION):
        assert f"<{name}" in text and f"</{name}>" in text
    assert SECTION_EVIDENCE not in ctx.sections and "<retrieved_archival_evidence>" not in ctx.prompt
    assert text.index(f"<{SECTION_SKILL}") < text.index(f"<{SECTION_STATE}") < text.index(f"<{SECTION_OBSERVATION}")
    assert 'version="0"' in ctx.sections[SECTION_STATE] and ctx.state_version == 0
    assert "read_text_file" in ctx.sections[SECTION_SKILL] and "path" in ctx.sections[SECTION_SKILL]
    assert ctx.char_count == len(ctx.instructions) + len(ctx.prompt) and ctx.approx_tokens == -(-ctx.char_count // 4)


def test_only_required_tools_are_rendered(simple_skill: SkillSpecification, base_state: ExecutionState):
    specs = [ToolSpec(name=n, description=f"{n} desc") for n in ("read_text_file", "run_shell")]
    ctx = ContextBuilder(tool_specs=specs).build(simple_skill, base_state, _obs(0, "x"))
    assert "read_text_file" in ctx.sections[SECTION_SKILL] and "run_shell" not in ctx.sections[SECTION_SKILL]


def test_context_does_not_depend_on_number_of_prior_observations(simple_skill: SkillSpecification, base_state: ExecutionState):
    """Given the same state and the same newest observation, step 1000's context equals step 1's (modulo the step attribute)."""
    b = ContextBuilder()
    early = b.build(simple_skill, base_state, _obs(1, "probe reading"))
    late = b.build(simple_skill, base_state.model_copy(), _obs(1000, "probe reading"))
    assert early.sections[SECTION_SKILL] == late.sections[SECTION_SKILL]
    assert early.sections[SECTION_STATE] == late.sections[SECTION_STATE]
    assert early.sections[SECTION_OBSERVATION].replace('step="1"', 'step="1000"').replace("obs_1\"", "obs_1000\"").replace("evt_1\"", "evt_1000\"") == late.sections[SECTION_OBSERVATION]
    assert abs(early.char_count - late.char_count) <= 12


def test_observation_appears_once_and_older_observations_never(simple_skill: SkillSpecification, base_state: ExecutionState):
    b = ContextBuilder()
    state = base_state
    seen = []
    for step in range(1, 30):
        marker = f"MARK-{step:04d}-{step * 31337}"
        ctx = b.build(simple_skill, state, _obs(step, f"reading {marker}"))
        text = ctx.full_text()
        assert text.count(marker) == 1
        for old in seen:
            assert old not in text, f"observation from earlier step leaked into step {step}"
        seen.append(marker)
        # the policy folds a *summary* into state, replacing the previous one
        state = apply_patch(state, StatePatch(expected_state_version=state.state_version,
                                              ops=[{"op": "set_observation_summary", "summary": f"step {step} ok"}])).state


def test_retrieved_evidence_is_rendered_only_when_provided(simple_skill: SkillSpecification, base_state: ExecutionState):
    b = ContextBuilder(max_retrieved_chars=400, max_retrieved_excerpt_chars=100)
    now = datetime.now(UTC)
    events = [RetrievedEvent(event_id=f"evt_r{i}", step=i, event_type="observation.captured", created_at=now, summary=f"sum {i}", excerpt="e" * 300) for i in range(6)]
    result = MemoryResult(query=MemoryQuery(query_type="search", text="needle"), events=events, total_matches=6)
    ctx = b.build(simple_skill, base_state, _obs(9, "x"), [result])
    ev = ctx.sections[SECTION_EVIDENCE]
    assert 'text="needle"' in ev and ctx.retrieved_event_ids == [e.event_id for e in events]
    assert ev.count("<event ") < 6 and "<truncated" in ev  # budget enforced
    assert len(ev) < 400 + 300
    again = b.build(simple_skill, base_state, _obs(10, "y"))  # next step: evidence is gone unless requested again
    assert SECTION_EVIDENCE not in again.sections and "evt_r0" not in again.full_text()


def test_truncated_observation_is_flagged(simple_skill: SkillSpecification, base_state: ExecutionState):
    obs = Observation(run_id="run_test", step=1, kind=ObservationKind.TOOL_RESULT, source="t", content="short", truncated=True, full_length=99999, event_id="evt_big")
    ctx = ContextBuilder().build(simple_skill, base_state, obs)
    assert 'truncated="true"' in ctx.sections[SECTION_OBSERVATION] and 'full_length="99999"' in ctx.sections[SECTION_OBSERVATION]


def test_tool_observation_header_carries_the_request(simple_skill: SkillSpecification, base_state: ExecutionState):
    """A stateless step cannot know what `d 0 mnestic` is a listing *of* unless the observation says so."""
    obs = Observation(run_id="run_test", step=2, kind=ObservationKind.TOOL_RESULT, source="list_directory", content="d 0 mnestic",
                      data={"count": 1, "ok": True, "request": {"tool": "list_directory", "arguments": {"path": "src"}}})
    section = ContextBuilder().build(simple_skill, base_state, obs).sections[SECTION_OBSERVATION]
    header = section.split("\n", 1)[0]
    assert 'request=' in header and '\\"path\\":\\"src\\"' in header
    assert "<data>" in section and "request" not in section.split("<data>", 1)[1]  # not duplicated in <data>
