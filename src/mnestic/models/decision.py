"""AgentDecision — the structured output of one reasoning step."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mnestic.models.action import Action
from mnestic.models.archive import MemoryQuery
from mnestic.models.common import StrictModel
from mnestic.models.patch import StatePatch


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


def decision_type_for(allowed_ops: list[str] | None) -> type[AgentDecision]:
    """Build an AgentDecision subclass whose StatePatch accepts only ``allowed_ops``.

    The runtime still validates and applies with the full ``StatePatch``; this only shrinks the output schema the model
    is shown (the 30-op union is ~18K chars; a three-op skill needs a fraction of that).
    """
    if not allowed_ops:
        return AgentDecision
    from typing import Annotated, Any, Union, get_args

    from pydantic import Field as _Field
    from pydantic import create_model

    from mnestic.models.patch import PatchOp, StatePatch

    members = [m for m in get_args(get_args(PatchOp)[0]) if m.model_fields["op"].default in set(allowed_ops)]
    unknown = set(allowed_ops) - {m.model_fields["op"].default for m in members}
    if unknown:
        raise ValueError(f"unknown patch ops in allowed_ops: {sorted(unknown)}")
    op_union: Any = Annotated[Union[tuple(members)], _Field(discriminator="op")]  # noqa: UP007 - runtime union construction  # type: ignore[valid-type]
    patch_cls = create_model("StatePatch", __base__=StatePatch, ops=(list[op_union], _Field(default_factory=list, max_length=50)))  # type: ignore[valid-type]
    return create_model("AgentDecision", __base__=AgentDecision, state_patch=(patch_cls, ...))  # type: ignore[call-overload,no-any-return]
