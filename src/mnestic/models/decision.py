"""AgentDecision — the structured output of one reasoning step."""

from __future__ import annotations

from typing import Any, Literal

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


DEFAULT_ALLOWED_OPS: list[str] = [
    "set_phase", "set_plan", "update_plan_step",
    "add_fact", "supersede_fact", "remove_fact",
    "add_hypothesis", "reject_hypothesis", "promote_hypothesis",
    "add_artifact", "add_blocker", "remove_blocker",
    "set_environment", "set_entity", "remove_entity",
    "set_observation_summary",
    "set_path", "adjust_path", "delete_path",
]
"""The op subset a skill gets when it declares nothing: enough for facts/hypotheses/plan/phase/entities. Everything
else (questions, pending actions, constraints, metadata, status, objective, archive_facts, update_hypothesis,
remove_artifact, clear_environment) is opt-in through ``allowed_ops``; ``["*"]`` means all ops."""


def resolve_allowed_ops(allowed_ops: list[str] | None) -> list[str]:
    from typing import get_args

    from mnestic.models.patch import PatchOp

    all_ops = [m.model_fields["op"].default for m in get_args(get_args(PatchOp)[0])]
    if allowed_ops is None:
        return list(DEFAULT_ALLOWED_OPS)
    if allowed_ops == ["*"]:
        return all_ops
    unknown = set(allowed_ops) - set(all_ops)
    if unknown:
        raise ValueError(f"unknown patch ops in allowed_ops: {sorted(unknown)}")
    return list(allowed_ops)


ACTION_KINDS = ["tool", "human_input", "continue"]


def decision_type_for(
    allowed_ops: list[str] | None,
    allowed_actions: list[str] | None = None,
    tool_args: dict[str, type[Any]] | None = None,
) -> type[AgentDecision]:
    """Build an AgentDecision subclass whose StatePatch accepts only the resolved ``allowed_ops``, whose ``action``
    accepts only ``allowed_actions`` kinds (e.g. no ``human_input`` for unattended skills), and — when ``tool_args``
    maps tool names to their argument models — whose tool action is a typed variant per tool, so the output schema
    (and a grammar-constrained decoder) enforces each tool's parameters exactly like native tool calling.

    The runtime still validates and applies with the full ``StatePatch``; this only shapes the schema the model is shown.
    """
    allowed_ops = resolve_allowed_ops(allowed_ops)
    from typing import Annotated, Union, get_args

    from pydantic import Field as _Field
    from pydantic import create_model

    from mnestic.models.action import Action
    from mnestic.models.patch import PatchOp, StatePatch

    members = [m for m in get_args(get_args(PatchOp)[0]) if m.model_fields["op"].default in set(allowed_ops)]
    op_union: Any = Annotated[Union[tuple(members)], _Field(discriminator="op")]  # noqa: UP007 - runtime union construction  # type: ignore[valid-type]
    patch_cls = create_model("StatePatch", __base__=StatePatch, ops=(list[op_union], _Field(default_factory=list, max_length=50)))  # type: ignore[valid-type]
    fields: dict[str, Any] = {"state_patch": (patch_cls, ...)}
    if allowed_actions is not None or tool_args:
        wanted = set(allowed_actions) if allowed_actions is not None else set(ACTION_KINDS)
        unknown = wanted - set(ACTION_KINDS)
        if unknown:
            raise ValueError(f"unknown action kinds in allowed_actions: {sorted(unknown)}")
        from typing import Literal

        from mnestic.models.action import ToolAction

        kinds: list[Any] = []
        for m in get_args(get_args(Action)[0]):
            k = m.model_fields["kind"].default
            if k not in wanted:
                continue
            if k == "tool" and tool_args:
                # One typed variant per tool: {"kind": "tool", "tool_name": "<name>", "arguments": <Args>}.
                for name, args_cls in tool_args.items():
                    kinds.append(create_model(f"ToolAction_{name}", __base__=ToolAction,
                                              tool_name=(Literal[name], ...), arguments=(args_cls, ...)))  # type: ignore[valid-type]
            else:
                kinds.append(m)
        # tool variants share kind="tool", so discriminate on kind only when every variant has a distinct kind
        if len(kinds) == 1:
            action_union: Any = kinds[0]
        elif len({k.model_fields["kind"].default for k in kinds}) == len(kinds):
            action_union = Annotated[Union[tuple(kinds)], _Field(discriminator="kind")]  # noqa: UP007  # type: ignore[valid-type]
        else:
            action_union = Union[tuple(kinds)]  # noqa: UP007  # type: ignore[valid-type]
        fields["action"] = (action_union | None, None)
    return create_model("AgentDecision", __base__=AgentDecision, **fields)  # type: ignore[call-overload,no-any-return]


def normalize_decision(decision: AgentDecision) -> AgentDecision:
    """Convert a decision produced by a skill-shaped subclass (typed tool arguments) into the plain runtime type
    (``arguments`` as a dict), so the rest of the runtime never sees per-skill classes."""
    if type(decision) is AgentDecision:
        return decision
    return AgentDecision.model_validate(decision.model_dump(mode="json"))
