"""AgentDecision — the structured output of one reasoning step."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from skillstate.models.action import Action
from skillstate.models.archive import MemoryQuery
from skillstate.models.common import StrictModel
from skillstate.models.patch import StatePatch


class CompletionResult(StrictModel):
    outcome: Literal["success", "failure", "partial"]
    summary: str = Field(min_length=1, max_length=4000)
    final_answer: str | None = Field(default=None, max_length=20_000)
    artifact_ids: list[str] = Field(default_factory=list, max_length=50)


class AgentDecision(StrictModel):
    """Exactly one of ``action``, ``memory_query``, ``completion`` must be set.

    ``rationale_summary`` is a short externally-safe justification, NOT chain of thought.
    It is archived with the decision and never re-injected into a later prompt.
    """

    rationale_summary: str = Field(default="", max_length=600)
    state_patch: StatePatch
    action: Action | None = None
    memory_query: MemoryQuery | None = None
    completion: CompletionResult | None = None

    @model_validator(mode="after")
    def _exactly_one_control(self) -> AgentDecision:
        set_fields = [
            name
            for name, value in (
                ("action", self.action),
                ("memory_query", self.memory_query),
                ("completion", self.completion),
            )
            if value is not None
        ]
        if len(set_fields) != 1:
            raise ValueError(
                "exactly one of action, memory_query or completion must be provided "
                f"(got {set_fields or 'none'})"
            )
        return self
