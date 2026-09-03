"""ContextBuilder: (SkillSpecification, ExecutionState, Observation, [MemoryResult]) -> ModelContext.

THE ARCHIVE IS NOT DEFAULT MODEL CONTEXT.

This module deliberately has no dependency on ``mnestic.storage`` or any database
handle. It can only render what it is handed, which is exactly the bounded set
A_t = (P, Σ_t, O_t, [E_t]) described in docs/CONTEXT_INVARIANTS.md.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mnestic.models.archive import MemoryResult
from mnestic.models.observation import Observation
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ExecutionState

SECTION_SKILL = "skill_specification"
SECTION_STATE = "execution_state"
SECTION_OBSERVATION = "latest_observation"
SECTION_EVIDENCE = "retrieved_archival_evidence"
SECTION_CONTRACT = "output_contract"


class ToolSpec(BaseModel):
    """Bounded description of a tool the skill may use (part of P, not history)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)


class ModelContext(BaseModel):
    """The exact, inspectable input for one reasoning step."""

    run_id: str
    step: int
    skill_id: str
    skill_version: str
    skill_content_hash: str
    state_version: int
    observation_id: str
    observation_event_id: str | None
    retrieved_event_ids: list[str] = Field(default_factory=list)
    instructions: str = Field(description="System-level instructions: skill specification + output contract")
    prompt: str = Field(description="User-level prompt: execution state + latest observation + retrieved evidence")
    sections: dict[str, str] = Field(description="Each delimited section, for inspection/tests")
    char_count: int
    approx_tokens: int

    def full_text(self) -> str:
        return f"{self.instructions}\n\n{self.prompt}"


OUTPUT_CONTRACT = """\
You are one reasoning step of a long-horizon agent. You do NOT have a conversation history.
Everything you know is in <execution_state>; the newest input is in <latest_observation>.
Older observations, tool outputs and your previous reasoning are NOT visible to you; they are archived and can
be recovered only with an explicit memory_query (results appear once in <retrieved_archival_evidence>).

Respond with a single structured AgentDecision:
- state_patch: ops that fold what matters from the observation into the state (facts need evidence_event_ids that
  reference real archive events such as the observation's event_id; hypotheses are for unverified beliefs; use
  supersede_fact/remove_fact when something you believed is no longer true; replace last_observation_summary,
  never append). expected_state_version must equal the version shown in <execution_state>.
- exactly one of: action (tool | human_input | continue), memory_query (recover archived information), or
  completion (when the completion criteria are met or the task is impossible).
- rationale_summary: one or two short externally-safe sentences. Do not include step-by-step reasoning.
Keep the state small: reference large content via artifacts and event ids instead of copying it in.
"""


class ContextBuilder:
    """Assembles a bounded ModelContext. Holds only rendering options and tool specs."""

    def __init__(
        self,
        *,
        tool_specs: list[ToolSpec] | None = None,
        max_retrieved_chars: int = 8000,
        max_retrieved_excerpt_chars: int = 1200,
        chars_per_token: float = 4.0,
    ):
        self.tool_specs = list(tool_specs or [])
        self.max_retrieved_chars = max_retrieved_chars
        self.max_retrieved_excerpt_chars = max_retrieved_excerpt_chars
        self.chars_per_token = chars_per_token

    def build(
        self,
        skill: SkillSpecification,
        state: ExecutionState,
        observation: Observation,
        retrieved: list[MemoryResult] | None = None,
    ) -> ModelContext:
        skill_section = self.render_skill(skill)
        state_section = self.render_state(state)
        obs_section = self.render_observation(observation)
        evidence_section = self.render_evidence(retrieved or [])
        contract_section = f"<{SECTION_CONTRACT}>\n{OUTPUT_CONTRACT}</{SECTION_CONTRACT}>"

        instructions = f"{skill_section}\n\n{contract_section}"
        prompt = state_section + "\n\n" + obs_section
        if evidence_section:
            prompt += "\n\n" + evidence_section
        sections = {
            SECTION_SKILL: skill_section,
            SECTION_CONTRACT: contract_section,
            SECTION_STATE: state_section,
            SECTION_OBSERVATION: obs_section,
        }
        if evidence_section:
            sections[SECTION_EVIDENCE] = evidence_section
        char_count = len(instructions) + len(prompt)
        return ModelContext(
            run_id=state.run_id,
            step=observation.step,
            skill_id=skill.skill_id,
            skill_version=skill.version,
            skill_content_hash=skill.content_hash,
            state_version=state.state_version,
            observation_id=observation.id,
            observation_event_id=observation.event_id,
            retrieved_event_ids=[e.event_id for r in (retrieved or []) for e in r.events],
            instructions=instructions,
            prompt=prompt,
            sections=sections,
            char_count=char_count,
            approx_tokens=math.ceil(char_count / self.chars_per_token),
        )

    # ---- section renderers (stable, deterministic) ---------------------------------------

    def render_skill(self, skill: SkillSpecification) -> str:
        lines = [
            f'<{SECTION_SKILL} id="{skill.skill_id}" version="{skill.version}">',
            f"# {skill.name}",
            skill.description.strip(),
            "",
            "## Instructions",
            skill.instructions.strip(),
        ]
        if skill.phases:
            lines += ["", "## Phases", ", ".join(skill.phases)]
        if skill.constraints:
            lines += ["", "## Constraints"] + [f"- {c}" for c in skill.constraints]
        if skill.completion_criteria:
            lines += ["", "## Completion criteria"] + [f"- {c}" for c in skill.completion_criteria]
        specs = [t for t in self.tool_specs if t.name in set(skill.required_tools)] if skill.required_tools else []
        lines += ["", "## Available tools"]
        if specs:
            for t in specs:
                schema = json.dumps(_compact_schema(t.parameters_schema), separators=(",", ":"), ensure_ascii=False)
                lines.append(f"- {t.name}: {t.description} arguments={schema}")
        else:
            lines.append("- (none)")
        lines.append(f"</{SECTION_SKILL}>")
        return "\n".join(lines)

    @staticmethod
    def render_state(state: ExecutionState) -> str:
        body = json.dumps(state.model_view(), indent=1, ensure_ascii=False, sort_keys=False)
        return f'<{SECTION_STATE} version="{state.state_version}">\n{body}\n</{SECTION_STATE}>'

    @staticmethod
    def render_observation(obs: Observation) -> str:
        attrs = f'id="{obs.id}" event_id="{obs.event_id or ""}" step="{obs.step}" kind="{obs.kind.value}" source="{obs.source}"'
        if obs.truncated:
            attrs += f' truncated="true" full_length="{obs.full_length}"'
        data = f"\n<data>{json.dumps(obs.data, separators=(',', ':'), ensure_ascii=False)}</data>" if obs.data else ""
        return f"<{SECTION_OBSERVATION} {attrs}>\n{obs.content}{data}\n</{SECTION_OBSERVATION}>"

    def render_evidence(self, results: list[MemoryResult]) -> str:
        if not results:
            return ""
        lines = [f"<{SECTION_EVIDENCE}>"]
        budget = self.max_retrieved_chars
        for r in results:
            q = r.query
            header = f'<retrieval type="{q.query_type}" matches="{r.total_matches}" returned="{len(r.events)}"'
            if q.text:
                header += f' text="{_attr(q.text)}"'
            if r.note:
                header += f' note="{_attr(r.note)}"'
            lines.append(header + ">")
            for e in r.events:
                excerpt = e.excerpt[: self.max_retrieved_excerpt_chars]
                block = f'<event id="{e.event_id}" step="{e.step}" type="{e.event_type}">\n{e.summary}\n{excerpt}\n</event>'
                if len(block) > budget:
                    lines.append("<truncated reason=\"retrieved evidence budget exhausted\"/>")
                    break
                budget -= len(block)
                lines.append(block)
            lines.append("</retrieval>")
        lines.append(f"</{SECTION_EVIDENCE}>")
        return "\n".join(lines)


def _attr(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:200]


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep only what a model needs to call the tool: property names, types, descriptions, required."""
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for name, p in props.items():
        entry: dict[str, Any] = {}
        if "type" in p:
            entry["type"] = p["type"]
        elif "anyOf" in p:
            entry["type"] = "|".join(str(a.get("type", "?")) for a in p["anyOf"])
        if "description" in p:
            entry["description"] = p["description"][:200]
        if "default" in p:
            entry["default"] = p["default"]
        out[name] = entry
    if req := schema.get("required"):
        out["_required"] = req
    return out
