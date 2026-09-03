"""Workspace-confined filesystem tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from skillstate.tools.base import Tool, ToolContext, ToolError, ToolResult, truncate_output

MAX_READ_CHARS = 200_000
MAX_WRITE_CHARS = 1_000_000


class ReadTextFileTool(Tool):
    name: ClassVar[str] = "read_text_file"
    description: ClassVar[str] = "Read a UTF-8 text file inside the workspace (optionally a line range)."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str = Field(description="Path relative to the workspace root")
        start_line: int = Field(default=1, ge=1)
        max_lines: int = Field(default=400, ge=1, le=5000)

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve_path(args.path)
        if not path.is_file():
            raise ToolError(f"not a file: {args.path}")
        if path.stat().st_size > MAX_READ_CHARS * 4:
            raise ToolError(f"file too large to read directly ({path.stat().st_size} bytes); use search_text or a line range")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise ToolError(f"cannot read {args.path}: {exc}") from None
        selected = lines[args.start_line - 1 : args.start_line - 1 + args.max_lines]
        numbered = "\n".join(f"{args.start_line + i:>6}: {line}" for i, line in enumerate(selected))
        output, truncated = truncate_output(numbered, MAX_READ_CHARS)
        return ToolResult(
            ok=True, output=output,
            data={"path": str(path.relative_to(ctx.workspace_root.resolve())) if not ctx.config.allow_workspace_escape else str(path),
                  "total_lines": len(lines), "returned_lines": len(selected), "truncated": truncated},
        )


class ListDirectoryTool(Tool):
    name: ClassVar[str] = "list_directory"
    description: ClassVar[str] = "List entries of a directory inside the workspace."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str = Field(default=".", description="Directory path relative to the workspace root")
        max_entries: int = Field(default=200, ge=1, le=2000)

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve_path(args.path)
        if not path.is_dir():
            raise ToolError(f"not a directory: {args.path}")
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = []
        for p in entries[: args.max_entries]:
            kind = "d" if p.is_dir() else ("l" if p.is_symlink() else "f")
            size = p.stat().st_size if p.is_file() else 0
            lines.append(f"{kind} {size:>10} {p.name}")
        note = f"\n… {len(entries) - args.max_entries} more entries" if len(entries) > args.max_entries else ""
        return ToolResult(ok=True, output="\n".join(lines) + note, data={"count": len(entries)})


class SearchTextTool(Tool):
    name: ClassVar[str] = "search_text"
    description: ClassVar[str] = "Search files under a workspace directory for a regex/plain pattern; returns file:line matches."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pattern: str = Field(min_length=1, max_length=500)
        path: str = Field(default=".")
        regex: bool = False
        glob: str = Field(default="*", description="Filename glob filter, e.g. '*.py'")
        max_results: int = Field(default=100, ge=1, le=1000)

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        root = ctx.resolve_path(args.path)
        if not root.exists():
            raise ToolError(f"path does not exist: {args.path}")
        try:
            rx = re.compile(args.pattern if args.regex else re.escape(args.pattern))
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from None
        results: list[str] = []
        files = [root] if root.is_file() else sorted(p for p in root.rglob(args.glob) if p.is_file())
        scanned = 0
        explicit_hidden = any(part.startswith(".") for part in Path(args.path).parts if part not in {".", ".."})
        for f in files:
            rel_parts = f.relative_to(root).parts if f != root else ()
            if not explicit_hidden and any(part.startswith(".") for part in rel_parts):
                continue  # skip .git, .venv, .skillstate, … unless the caller pointed at them explicitly
            if f.stat().st_size > 5_000_000 or _looks_binary(f):
                continue
            scanned += 1
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = f.relative_to(ctx.workspace_root.resolve()) if not ctx.config.allow_workspace_escape else f
                        results.append(f"{rel}:{i}: {line.strip()[:300]}")
                        if len(results) >= args.max_results:
                            break
            except OSError:
                continue
            if len(results) >= args.max_results:
                break
        output, truncated = truncate_output("\n".join(results) or "(no matches)", 50_000)
        return ToolResult(ok=True, output=output, data={"matches": len(results), "files_scanned": scanned, "capped": len(results) >= args.max_results})


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(1024)
    except OSError:
        return True


class WriteWorkspaceFileTool(Tool):
    name: ClassVar[str] = "write_workspace_file"
    description: ClassVar[str] = "Create or overwrite a UTF-8 text file inside the workspace. Returns an artifact reference."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str
        content: str = Field(max_length=MAX_WRITE_CHARS)
        append: bool = False

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve_path(args.path)
        if path.is_dir():
            raise ToolError(f"{args.path} is a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if args.append else "w", encoding="utf-8") as fh:
            fh.write(args.content)
        from skillstate.models.state import ArtifactReference

        rel = str(path.relative_to(ctx.workspace_root.resolve())) if not ctx.config.allow_workspace_escape else str(path)
        artifact = ArtifactReference(kind="file", locator=rel, description=f"written by write_workspace_file at step {ctx.step}")
        return ToolResult(ok=True, output=f"wrote {len(args.content)} chars to {rel}", data={"path": rel, "chars": len(args.content)}, artifact=artifact)
