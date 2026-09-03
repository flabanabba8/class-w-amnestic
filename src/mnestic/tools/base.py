"""Tool protocol, typed request/result records, registry and workspace path confinement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from mnestic.config import RuntimeConfig
from mnestic.context.builder import ToolSpec
from mnestic.memory.base import Retriever
from mnestic.models.state import ArtifactReference


class ToolError(Exception):
    """A tool refused or failed. Becomes a first-class failure observation, never an exception to the model."""


@dataclass
class ToolContext:
    workspace_root: Path
    config: RuntimeConfig
    run_id: str
    step: int
    retriever: Retriever | None = None

    def resolve_path(self, raw: str) -> Path:
        """Resolve ``raw`` inside the workspace; reject traversal/symlink escapes unless configured otherwise."""
        root = self.workspace_root.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if self.config.allow_workspace_escape:
            return resolved
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ToolError(f"path {raw!r} resolves outside the workspace root {root}") from None
        return resolved


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    output: str = ""
    error: str | None = None
    data: dict[str, Any] | None = None
    artifact: ArtifactReference | None = None


class Tool(ABC):
    """A typed tool. Arguments are validated against ``Args`` before ``run`` is invoked."""

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters_schema=self.Args.model_json_schema())

    def parse_args(self, arguments: dict[str, Any]) -> BaseModel:
        try:
            return self.Args.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {self.name}: {exc.errors(include_url=False)}") from None

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(self._tools)}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        wanted = set(names) if names is not None else set(self._tools)
        return [t.spec() for n, t in sorted(self._tools.items()) if n in wanted]

    def missing(self, names: list[str]) -> list[str]:
        return [n for n in names if n not in self._tools]

    async def execute(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate and run. Any exception becomes a failed ToolResult (tool failures are observations)."""
        try:
            tool = self.get(name)
            args = tool.parse_args(arguments)
            return await tool.run(args, ctx)
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


def truncate_output(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n… [{len(text) - limit} chars omitted; full output archived] …\n" + text[-tail:], True


