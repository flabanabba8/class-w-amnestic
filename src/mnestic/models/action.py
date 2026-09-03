"""Actions the model can request. Executed by the runtime, never by the model."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from mnestic.models.common import StrictModel


class ToolAction(StrictModel):
    kind: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(default="", max_length=500, description="Short statement of why (externally safe)")


class RequestHumanInput(StrictModel):
    kind: Literal["human_input"] = "human_input"
    question: str = Field(min_length=1, max_length=2000)
    options: list[str] | None = Field(default=None, max_length=10)


class ContinueAction(StrictModel):
    """Reason again next step with no external action (e.g. after digesting retrieved evidence)."""

    kind: Literal["continue"] = "continue"
    note: str = Field(default="", max_length=500)


Action = Annotated[ToolAction | RequestHumanInput | ContinueAction, Field(discriminator="kind")]
