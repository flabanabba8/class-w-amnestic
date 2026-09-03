from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnestic.cli.main import main
from tests.conftest import SKILLS_DIR


@pytest.fixture
def cli(tmp_path: Path, workspace: Path, capsys):
    db = tmp_path / "cli.db"

    def run(*args: str) -> tuple[int, str]:
        code = main([*args, "--db", str(db), "--workspace", str(workspace), "--skills-dir", str(SKILLS_DIR), "--log-level", "ERROR"])
        return code, capsys.readouterr().out

    return run


def test_cli_full_workflow(cli, workspace):
    assert cli("doctor")[0] == 0
    code, out = cli("skills", "list")
    assert code == 0 and "codebase-research" in out
    code, out = cli("run", "codebase-research", "--task", "Where is `PORT` configured?", "--model", "mock", "--json")
    assert code == 0
    run_id = json.loads(out)["run_id"]
    assert cli("status")[1].strip().startswith(run_id)
    code, out = cli("status", run_id, "--json")
    assert json.loads(out)["status"] == "completed"
    code, out = cli("state", run_id, "--model-view")
    assert json.loads(out)["current_phase"] == "done"
    code, out = cli("history", run_id)
    assert "v0" in out and "fact added" in out
    code, out = cli("events", run_id, "--type", "tool.finished")
    assert out.count("tool.finished") == 4
    code, out = cli("diff", run_id, "0", "3")
    assert "current_phase" in out
    code, out = cli("memory", "search", run_id, "PORT 8000", "--json")
    assert json.loads(out)["total_matches"] >= 1
    code, out = cli("inspect-context", run_id, "--step", "2", "--sections")
    assert "<execution_state" in out and "<latest_observation" in out and "search_text" in out
    code, out = cli("inspect-context", run_id, "--json")
    assert json.loads(out)["char_count"] > 0
    code, out = cli("graph")
    assert "stateDiagram" in out and "Reason --> " in out
    code, out = cli("semantic", "promote", "proj.report", "reports go in RESEARCH_REPORT.md", "--run-id", run_id)
    assert json.loads(out)["key"] == "proj.report"
    assert "proj.report" in cli("semantic", "list")[1]
    assert cli("resume", run_id)[0] == 2  # completed runs cannot be resumed


def test_cli_human_input_resume(cli):
    code, out = cli("run", "deterministic-counter", "--task", "count to 2", "--model", "mock", "--json")
    assert code == 0 and json.loads(out)["status"] == "completed"
