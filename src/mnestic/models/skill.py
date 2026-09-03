"""Skill Specification — procedural memory. Immutable during a run."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SkillSpecification(BaseModel):
    """A versioned, immutable description of what the agent must do.

    Loaded from ``skills/<name>/skill.yaml`` + ``SKILL.md``. The ``content_hash``
    is stored with the run so a resumed run can detect a changed skill.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(max_length=2000)
    required_tools: list[str] = Field(default_factory=list)
    state_schema_version: str = "1"
    instructions: str = Field(min_length=1, max_length=60_000)
    completion_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    initial_phase: str = "start"
    default_max_steps: int = Field(default=200, ge=1, le=100_000)
    allowed_ops: list[str] | None = Field(
        default=None,
        description="Patch ops this skill uses (e.g. ['set_entity','set_environment','set_observation_summary']). "
        "Prunes the output schema sent to the model; None = the default core set, ['*'] = every op.",
    )
    allowed_actions: list[str] | None = Field(
        default=None,
        description="Action kinds the model may request: subset of ['tool','human_input','continue']. None = all. "
        "Unattended tasks should omit 'human_input'.",
    )
    reasoner_script: str | None = Field(
        default=None,
        description="Optional dotted path to a scripted reasoner used with --model mock (deterministic skills).",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def key(self) -> str:
        return f"{self.skill_id}@{self.version}"
