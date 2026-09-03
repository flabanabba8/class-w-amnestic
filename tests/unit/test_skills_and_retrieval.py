from __future__ import annotations

from pathlib import Path

import pytest

from mnestic.memory.retrieval import ArchiveRetriever
from mnestic.models.archive import EventType, MemoryQuery, RunMetadata
from mnestic.models.common import utcnow
from mnestic.models.patch import StatePatch
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ArtifactReference, ExecutionState
from mnestic.skills.loader import SkillLoadError, SkillRegistry, load_skill_dir
from mnestic.state.apply import apply_patch
from mnestic.storage.store import Store


def test_example_skills_load(registry: SkillRegistry):
    ids = {s.skill_id for s in registry.list()}
    assert {"deterministic-counter", "codebase-research"} <= ids
    assert registry.get("codebase-research").required_tools
    assert registry.get("deterministic-counter@1.0.0").reasoner_script
    assert not registry.errors


def test_skill_versions_and_errors(tmp_path: Path):
    for v in ("1.0.0", "1.10.0", "1.2.0"):
        d = tmp_path / f"s@{v}"
        d.mkdir()
        (d / "skill.yaml").write_text(f"skill_id: s\nversion: '{v}'\ndescription: d\n")
        (d / "SKILL.md").write_text("instructions")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "skill.yaml").write_text("skill_id: bad\n")  # no SKILL.md
    reg = SkillRegistry([tmp_path])
    assert reg.get("s").version == "1.10.0" and reg.get("s", "1.2.0").version == "1.2.0"
    assert len(reg.errors) == 1 and "SKILL.md" in reg.errors[0]
    with pytest.raises(SkillLoadError):
        load_skill_dir(bad)
    with pytest.raises(KeyError):
        reg.get("nope")


def test_retriever_query_types(store: Store, simple_skill: SkillSpecification, base_state: ExecutionState):
    rid = base_state.run_id
    with store.transaction():
        store.upsert_skill(simple_skill)
        store.create_run(RunMetadata(run_id=rid, skill_id=simple_skill.skill_id, skill_version=simple_skill.version, skill_content_hash=simple_skill.content_hash,
                                     status="running", created_at=utcnow(), updated_at=utcnow(), workspace_root="/tmp", model_name="mock", task_input="t"))
        store.init_state(base_state)
        e1 = store.append_event(rid, 0, EventType.OBSERVATION, "obs one", {"content": "the license key is ZETA-42"})
        store.append_event(rid, 1, EventType.TOOL_FINISHED, "tool done", {"tool": "read_text_file", "output": "PORT = 8000"})
        store.append_event(rid, 2, EventType.OBSERVATION, "obs two", {"content": "unrelated"})
        store.save_artifact(rid, ArtifactReference(id="a1", locator="report.md", description="the final report", originating_event_id=e1.event_id))
        v1 = apply_patch(base_state, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "p1"}])).state
        pid = store.record_patch(rid, 0, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "p1"}]), status="applied", resulting_version=1, changes=["phase -> p1"])
        store.commit_state(v1, expected_version=0, patch_id=pid, step=0)
        store.upsert_semantic(key="user.pref", category="preference", content="prefers terse output", confidence=1, source_run_id=None, source_event_ids=[], promoted_by="t")
    r = ArchiveRetriever(store)
    assert [e.step for e in r.retrieve(rid, MemoryQuery(query_type="recent", limit=2)).events] == [2, 1]
    assert r.retrieve(rid, MemoryQuery(query_type="event", event_id=e1.event_id)).events[0].excerpt.startswith("the license")
    assert "not found" in r.retrieve(rid, MemoryQuery(query_type="event", event_id="evt_x")).note
    assert r.retrieve(rid, MemoryQuery(query_type="events_by_type", event_type="tool.finished")).total_matches == 1
    res = r.retrieve(rid, MemoryQuery(query_type="search", text="ZETA-42", limit=5))
    assert res.total_matches == 1 and res.events[0].event_id == e1.event_id
    assert r.retrieve(rid, MemoryQuery(query_type="observations", text="license")).total_matches == 1
    assert r.retrieve(rid, MemoryQuery(query_type="tool_executions", text="PORT")).total_matches == 1
    assert r.retrieve(rid, MemoryQuery(query_type="artifacts", text="report")).events[0].summary.startswith("artifact a1")
    assert '"current_phase":"start"' in r.retrieve(rid, MemoryQuery(query_type="state_at_version", version=0)).events[0].excerpt
    hist = r.retrieve(rid, MemoryQuery(query_type="state_history", limit=5))
    assert [e.summary for e in hist.events][-1].startswith("version 1") and "phase -> p1" in hist.events[-1].excerpt
    assert r.retrieve(rid, MemoryQuery(query_type="semantic", text="terse")).events[0].summary == "[preference] user.pref"
    assert r.retrieve(rid, MemoryQuery(query_type="search", text="nothing-here-xyz")).events == []
    capped = r.retrieve(rid, MemoryQuery(query_type="recent", limit=1))
    assert len(capped.events) == 1 and capped.truncated
