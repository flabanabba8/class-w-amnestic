"""Shared fixtures. No test here needs network access or API credentials."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from skillstate.agent.reasoner import ScriptedReasoner
from skillstate.config import RuntimeConfig, ShellPolicy, StateLimits
from skillstate.context.builder import ModelContext
from skillstate.graph.runtime import Runtime
from skillstate.memory.retrieval import ArchiveRetriever
from skillstate.models.decision import AgentDecision
from skillstate.models.skill import SkillSpecification
from skillstate.models.state import ExecutionState, Objective, RunStatus
from skillstate.skills.loader import SkillRegistry
from skillstate.storage.db import Database
from skillstate.storage.store import Store
from skillstate.tools import default_registry
from skillstate.tools.base import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text("PORT = 8000\n# server_port setting\nDEBUG = False\n", encoding="utf-8")
    (ws / "README.md").write_text("# demo\nThe license key is ZETA-42.\n", encoding="utf-8")
    return ws


@pytest.fixture
def config(tmp_path: Path, workspace: Path) -> RuntimeConfig:
    return RuntimeConfig(
        db_path=tmp_path / "test.db", workspace_root=workspace, skills_dirs=[SKILLS_DIR],
        shell=ShellPolicy(mode="allowlist"), state_limits=StateLimits(),
    )


@pytest.fixture
def db(config: RuntimeConfig) -> Database:
    database = Database(config.db_path)
    yield database  # type: ignore[misc]
    database.close()


@pytest.fixture
def store(db: Database) -> Store:
    return Store(db)


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry([SKILLS_DIR])


@pytest.fixture
def simple_skill() -> SkillSpecification:
    return SkillSpecification(
        skill_id="test-skill", name="Test skill", version="1.0.0", description="A skill used by tests.",
        required_tools=["read_text_file", "list_directory", "search_text", "write_workspace_file", "archival_memory_search"],
        instructions="Do what the scripted reasoner says.", completion_criteria=["the script completes"],
        phases=["start", "work", "done"], default_max_steps=500,
    )


@pytest.fixture
def base_state() -> ExecutionState:
    return ExecutionState(run_id="run_test", skill_id="test-skill", skill_version="1.0.0", objective=Objective(statement="test objective"),
                         status=RunStatus.RUNNING)


ScriptT = Callable[[ModelContext], AgentDecision | dict[str, Any]]


@pytest.fixture
def make_runtime(config: RuntimeConfig, store: Store) -> Callable[..., Runtime]:
    def _make(script: ScriptT, *, tools: ToolRegistry | None = None, cfg: RuntimeConfig | None = None, store_: Store | None = None) -> Runtime:
        st = store_ or store
        return Runtime(cfg or config, st, reasoner=ScriptedReasoner(script), tools=tools or default_registry(), retriever=ArchiveRetriever(st))

    return _make


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)
