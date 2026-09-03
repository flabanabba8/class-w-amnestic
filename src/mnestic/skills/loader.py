"""Load SkillSpecification objects from ``<dir>/skill.yaml`` + ``<dir>/SKILL.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mnestic.models.skill import SkillSpecification


class SkillLoadError(Exception):
    pass


def load_skill_dir(path: Path) -> SkillSpecification:
    meta_path = path / "skill.yaml"
    md_path = path / "SKILL.md"
    if not meta_path.exists():
        raise SkillLoadError(f"{path}: missing skill.yaml")
    if not md_path.exists():
        raise SkillLoadError(f"{path}: missing SKILL.md")
    try:
        meta: dict[str, Any] = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SkillLoadError(f"{meta_path}: invalid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillLoadError(f"{meta_path}: top level must be a mapping")
    meta.setdefault("skill_id", path.name.split("@")[0])
    meta.setdefault("name", meta["skill_id"])
    meta["instructions"] = md_path.read_text(encoding="utf-8")
    try:
        return SkillSpecification.model_validate(meta)
    except ValidationError as exc:
        raise SkillLoadError(f"{path}: {exc}") from exc


def _version_key(version: str) -> tuple[int | str, ...]:
    return tuple(int(p) if p.isdigit() else p for p in version.replace("-", ".").split("."))


class SkillRegistry:
    """Indexes every skill directory under the configured roots."""

    def __init__(self, roots: list[Path]):
        self.roots = [Path(r) for r in roots]
        self._skills: dict[tuple[str, str], SkillSpecification] = {}
        self._paths: dict[tuple[str, str], Path] = {}
        self.errors: list[str] = []
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        self._paths.clear()
        self.errors.clear()
        for root in self.roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "skill.yaml").exists():
                    try:
                        spec = load_skill_dir(child)
                    except SkillLoadError as exc:
                        self.errors.append(str(exc))
                        continue
                    self._skills[(spec.skill_id, spec.version)] = spec
                    self._paths[(spec.skill_id, spec.version)] = child

    def list(self) -> list[SkillSpecification]:
        return sorted(self._skills.values(), key=lambda s: (s.skill_id, _version_key(s.version)))

    def get(self, skill_id: str, version: str | None = None) -> SkillSpecification:
        if "@" in skill_id and version is None:
            skill_id, version = skill_id.split("@", 1)
        candidates = [s for (sid, _), s in self._skills.items() if sid == skill_id]
        if not candidates:
            raise KeyError(f"skill {skill_id!r} not found in {[str(r) for r in self.roots]}")
        if version is not None:
            for s in candidates:
                if s.version == version:
                    return s
            raise KeyError(f"skill {skill_id!r} version {version!r} not found")
        return max(candidates, key=lambda s: _version_key(s.version))

    def path_of(self, spec: SkillSpecification) -> Path | None:
        return self._paths.get((spec.skill_id, spec.version))
