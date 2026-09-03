from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillstate.config import RuntimeConfig, ShellPolicy
from skillstate.tools import default_registry
from skillstate.tools.base import ToolContext, ToolError


@pytest.fixture
def ctx(config: RuntimeConfig, workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, config=config, run_id="r", step=1)


def test_path_traversal_rejected(ctx: ToolContext, tmp_path: Path):
    with pytest.raises(ToolError, match="outside the workspace"):
        ctx.resolve_path("../outside.txt")
    with pytest.raises(ToolError):
        ctx.resolve_path(str(tmp_path / "elsewhere"))
    assert ctx.resolve_path("src/app.py").name == "app.py"


def test_symlink_escape_rejected(ctx: ToolContext, workspace: Path, tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    (workspace / "link").symlink_to(secret)
    with pytest.raises(ToolError, match="outside the workspace"):
        ctx.resolve_path("link")


def test_escape_allowed_when_configured(config: RuntimeConfig, workspace: Path, tmp_path: Path):
    cfg = config.model_copy(update={"allow_workspace_escape": True})
    c = ToolContext(workspace_root=workspace, config=cfg, run_id="r", step=1)
    assert c.resolve_path(str(tmp_path)) == tmp_path.resolve()


async def test_filesystem_tools(ctx: ToolContext, workspace: Path):
    reg = default_registry()
    r = await reg.execute("list_directory", {"path": "."}, ctx)
    assert r.ok and "src" in r.output and "README.md" in r.output
    r = await reg.execute("read_text_file", {"path": "src/app.py", "max_lines": 1}, ctx)
    assert r.ok and "PORT = 8000" in r.output and r.data["total_lines"] == 3
    r = await reg.execute("search_text", {"pattern": "server_port", "path": "."}, ctx)
    assert r.ok and "src/app.py:2" in r.output
    r = await reg.execute("write_workspace_file", {"path": "out/notes.md", "content": "hi"}, ctx)
    assert r.ok and (workspace / "out" / "notes.md").read_text() == "hi" and r.artifact.locator == "out/notes.md"


async def test_tool_failures_are_results_not_exceptions(ctx: ToolContext):
    reg = default_registry()
    r = await reg.execute("read_text_file", {"path": "../../etc/passwd"}, ctx)
    assert not r.ok and "outside the workspace" in r.error
    r = await reg.execute("read_text_file", {"path": "missing.txt"}, ctx)
    assert not r.ok and "not a file" in r.error
    r = await reg.execute("read_text_file", {"nope": 1}, ctx)
    assert not r.ok and "invalid arguments" in r.error
    r = await reg.execute("no_such_tool", {}, ctx)
    assert not r.ok and "unknown tool" in r.error


async def test_search_skips_hidden_and_binary(ctx: ToolContext, workspace: Path):
    (workspace / ".skillstate").mkdir()
    (workspace / ".skillstate" / "db.sqlite").write_bytes(b"PORT\0\0\0binary")
    (workspace / "blob.bin").write_bytes(b"\0PORT\0")
    r = await default_registry().execute("search_text", {"pattern": "PORT", "path": "."}, ctx)
    assert r.ok and ".skillstate" not in r.output and "blob.bin" not in r.output and "src/app.py" in r.output


async def test_shell_allowlist(ctx: ToolContext):
    reg = default_registry()
    r = await reg.execute("run_shell", {"command": "echo hello"}, ctx)
    assert r.ok and r.output.strip() == "hello" and r.data["exit_code"] == 0
    r = await reg.execute("run_shell", {"command": "echo hi; rm -rf /"}, ctx)
    assert not r.ok and "metacharacters" in r.error
    r = await reg.execute("run_shell", {"command": "rm -rf ."}, ctx)
    assert not r.ok and "allowlist" in r.error
    r = await reg.execute("run_shell", {"command": "cat /etc/hostname"}, ctx)
    assert not r.ok and "outside the workspace" in r.error
    r = await reg.execute("run_shell", {"command": "ls nonexistent-dir"}, ctx)
    assert not r.ok and r.data["exit_code"] != 0


async def test_shell_disabled_and_env_isolation(config: RuntimeConfig, workspace: Path, monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "hunter2")
    cfg = config.model_copy(update={"shell": ShellPolicy(mode="disabled")})
    r = await default_registry().execute("run_shell", {"command": "echo x"}, ToolContext(workspace_root=workspace, config=cfg, run_id="r", step=1))
    assert not r.ok and "disabled" in r.error
    cfg = config.model_copy(update={"shell": ShellPolicy(mode="unrestricted", pass_environment=False)})
    r = await default_registry().execute("run_shell", {"command": "env"}, ToolContext(workspace_root=workspace, config=cfg, run_id="r", step=1))
    assert r.ok and "SUPER_SECRET_TOKEN" not in r.output and "hunter2" not in r.output
    assert os.environ["SUPER_SECRET_TOKEN"] == "hunter2"


async def test_shell_timeout(config: RuntimeConfig, workspace: Path):
    cfg = config.model_copy(update={"shell": ShellPolicy(mode="unrestricted", timeout_seconds=0.3)})
    r = await default_registry().execute("run_shell", {"command": "sleep 5"}, ToolContext(workspace_root=workspace, config=cfg, run_id="r", step=1))
    assert not r.ok and "timed out" in r.error
