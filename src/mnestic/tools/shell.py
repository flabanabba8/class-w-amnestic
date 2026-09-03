"""Shell execution under an explicit policy. Default: allowlisted argv, no shell interpretation."""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from mnestic.tools.base import Tool, ToolContext, ToolError, ToolResult, truncate_output

SHELL_METACHARACTERS = (";", "|", "&", ">", "<", "`", "$(", "\n")
MINIMAL_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TZ")


class ShellTool(Tool):
    name: ClassVar[str] = "run_shell"
    description: ClassVar[str] = "Run a command in the workspace under the configured shell policy (allowlist by default)."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str = Field(min_length=1, max_length=4000)
        timeout_seconds: float | None = Field(default=None, gt=0, le=600)

    async def run(self, args: Args, ctx: ToolContext) -> ToolResult:
        policy = ctx.config.shell
        if policy.mode == "disabled":
            raise ToolError("shell execution is disabled by policy (MNESTIC_SHELL_MODE)")
        timeout = min(args.timeout_seconds or policy.timeout_seconds, policy.timeout_seconds)
        env = os.environ.copy() if policy.pass_environment else {k: v for k, v in os.environ.items() if k in MINIMAL_ENV_KEYS}
        cwd = str(ctx.workspace_root.resolve())

        if policy.mode == "allowlist":
            if any(m in args.command for m in SHELL_METACHARACTERS):
                raise ToolError("shell metacharacters are not allowed in allowlist mode")
            try:
                argv = shlex.split(args.command)
            except ValueError as exc:
                raise ToolError(f"cannot parse command: {exc}") from None
            if not argv or not _allowed(argv, policy.allowed_commands):
                raise ToolError(f"command not in allowlist: {argv[0] if argv else ''!r}")
            for a in argv[1:]:
                if a.startswith("/") or ".." in a.split("/"):
                    ctx.resolve_path(a)  # raises if it escapes the workspace
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        else:  # unrestricted
            proc = await asyncio.create_subprocess_shell(
                args.command, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        started = time.monotonic()
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolError(f"command timed out after {timeout}s") from None
        duration_ms = int((time.monotonic() - started) * 1000)
        out = out_b.decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        combined = out + (f"\n[stderr]\n{err}" if err.strip() else "")
        text, truncated = truncate_output(combined, policy.max_output_chars)
        return ToolResult(
            ok=proc.returncode == 0, output=text, error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            data={"exit_code": proc.returncode, "duration_ms": duration_ms, "truncated": truncated},
            facts={args.command[:80]: {"exit_code": proc.returncode, "duration_ms": duration_ms, "output_chars": len(combined)}},
        )


def _allowed(argv: list[str], allowed: list[str]) -> bool:
    joined = " ".join(argv)
    for entry in allowed:
        parts = entry.split()
        if argv[: len(parts)] == parts:
            return True
        if joined == entry:
            return True
    return False
