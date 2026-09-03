"""Runtime configuration. Environment variables use the ``MNESTIC_`` prefix."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StateLimits(BaseModel):
    """Caps that keep the working state from becoming a transcript.

    Count overflows on *model-curated* lists (facts, hypotheses, artifacts, …) reject the
    patch with actionable feedback. ``rejected_hypotheses`` overflow is compacted
    automatically: oldest entries are spilled into the archive as a ``state.compaction``
    event (recoverable via retrieval).
    """

    model_config = ConfigDict(extra="forbid")

    max_state_bytes: int = Field(default=48_000, ge=1000)
    max_facts: int = Field(default=150, ge=1)
    max_hypotheses: int = Field(default=40, ge=1)
    max_rejected_hypotheses: int = Field(default=40, ge=1)
    max_artifacts: int = Field(default=100, ge=1)
    max_plan_steps: int = Field(default=50, ge=1)
    max_pending_actions: int = Field(default=30, ge=1)
    max_blockers: int = Field(default=20, ge=1)
    max_questions: int = Field(default=30, ge=1)
    max_entities: int = Field(default=60, ge=1)
    max_environment_keys: int = Field(default=60, ge=1)
    max_constraints: int = Field(default=50, ge=1)
    max_metadata_keys: int = Field(default=40, ge=1)


DEFAULT_SHELL_ALLOWLIST = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo", "pwd", "stat", "file",
    "sort", "uniq", "cut", "tr", "diff", "python3 --version", "python --version",
    "git status", "git log", "git diff", "git show", "git branch", "uname",
]


class ShellPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["disabled", "allowlist", "unrestricted"] = "allowlist"
    allowed_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_ALLOWLIST))
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_output_chars: int = Field(default=20_000, ge=100)
    pass_environment: bool = Field(
        default=False, description="If False, the subprocess gets a minimal environment (no secrets leak)."
    )


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_path: Path = Field(default=Path(".mnestic/mnestic.db"))
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())
    skills_dirs: list[Path] = Field(default_factory=lambda: [Path("skills")])
    model: str = Field(default="mock", description="PydanticAI model string (e.g. openai:gpt-4o-mini) or 'mock'")
    output_mode: Literal["tool", "native", "prompted"] = "tool"
    model_retries: int = Field(default=2, ge=0, le=10)
    model_settings: dict[str, Any] = Field(default_factory=dict, description="Passed to PydanticAI ModelSettings")
    model_timeout_seconds: float = Field(default=300.0, gt=0, description="Per model request timeout; a hung provider becomes a bounded model-error observation")
    max_observation_chars: int = Field(default=6000, ge=200)
    max_retrieved_chars: int = Field(default=8000, ge=200)
    max_retrieved_excerpt_chars: int = Field(default=1200, ge=100)
    max_consecutive_continues: int = Field(default=3, ge=1)
    max_repeated_actions: int = Field(default=3, ge=2, description="Identical tool action (same tool + arguments) seen this many times within action_window is treated as a loop")
    action_window: int = Field(default=12, ge=2, description="Sliding window (in tool actions) for the repeated-action loop guard")
    max_decision_failures: int = Field(default=3, ge=1, description="Consecutive rejected/failed decisions before the run fails")
    max_timeout_failures: int = Field(default=8, ge=1, description="Consecutive provider timeouts before the run fails (stalls are not model mistakes)")
    allow_workspace_escape: bool = False
    state_limits: StateLimits = Field(default_factory=StateLimits)
    shell: ShellPolicy = Field(default_factory=ShellPolicy)
    log_level: str = "INFO"
    log_json: bool = False
    logfire: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> RuntimeConfig:
        env = os.environ
        values: dict[str, Any] = {}
        if v := env.get("MNESTIC_DB_PATH"):
            values["db_path"] = Path(v)
        if v := env.get("MNESTIC_WORKSPACE"):
            values["workspace_root"] = Path(v)
        if v := env.get("MNESTIC_SKILLS_DIRS"):
            values["skills_dirs"] = [Path(p) for p in v.split(os.pathsep) if p]
        if v := env.get("MNESTIC_MODEL"):
            values["model"] = v
        if v := env.get("MNESTIC_OUTPUT_MODE"):
            values["output_mode"] = v
        if v := env.get("MNESTIC_MODEL_SETTINGS"):
            import json

            values["model_settings"] = json.loads(v)
        if v := env.get("MNESTIC_MAX_TIMEOUT_FAILURES"):
            values["max_timeout_failures"] = int(v)
        if v := env.get("MNESTIC_MODEL_TIMEOUT"):
            values["model_timeout_seconds"] = float(v)
        if v := env.get("MNESTIC_MODEL_RETRIES"):
            values["model_retries"] = int(v)
        if v := env.get("MNESTIC_SHELL_MODE"):
            values["shell"] = ShellPolicy(mode=v)  # type: ignore[arg-type]
        if v := env.get("MNESTIC_LOG_LEVEL"):
            values["log_level"] = v
        if env.get("MNESTIC_LOG_JSON") in {"1", "true", "yes"}:
            values["log_json"] = True
        if env.get("MNESTIC_LOGFIRE") in {"1", "true", "yes"}:
            values["logfire"] = True
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def resolved_workspace(self) -> Path:
        return self.workspace_root.expanduser().resolve()
