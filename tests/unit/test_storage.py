from __future__ import annotations

from pathlib import Path

import pytest

from mnestic.models.archive import EventType, RunMetadata
from mnestic.models.common import utcnow
from mnestic.models.observation import Observation, ObservationKind
from mnestic.models.patch import StatePatch
from mnestic.models.skill import SkillSpecification
from mnestic.models.state import ExecutionState
from mnestic.state.apply import apply_patch
from mnestic.storage.db import Database
from mnestic.storage.migrations import current_schema_version
from mnestic.storage.store import StaleWriteError, Store


def _seed(store: Store, skill: SkillSpecification, state: ExecutionState) -> None:
    with store.transaction():
        store.upsert_skill(skill)
        store.create_run(RunMetadata(run_id=state.run_id, skill_id=skill.skill_id, skill_version=skill.version, skill_content_hash=skill.content_hash,
                                     status="running", created_at=utcnow(), updated_at=utcnow(), workspace_root="/tmp", model_name="mock", task_input="t"))
        store.init_state(state)


def test_migrations_are_idempotent_and_pragmas_set(tmp_path: Path):
    p = tmp_path / "m.db"
    db1 = Database(p)
    assert current_schema_version(db1.conn) == 2
    assert db1.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db1.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db1.close()
    db2 = Database(p)  # reopening applies nothing new
    assert current_schema_version(db2.conn) == 2
    db2.close()


def test_transaction_is_reentrant_and_rolls_back(db: Database):
    with pytest.raises(RuntimeError), db.transaction():
        db.execute("INSERT INTO skills(skill_id, version, name, description, content_hash, spec_json, loaded_at) VALUES ('a','1','n','d','h','{}','t')")
        with db.transaction():  # nested: joins the outer
            db.execute("INSERT INTO skills(skill_id, version, name, description, content_hash, spec_json, loaded_at) VALUES ('b','1','n','d','h','{}','t')")
        raise RuntimeError("boom")
    assert db.query("SELECT COUNT(*) AS n FROM skills")[0]["n"] == 0


def test_commit_state_optimistic_concurrency(store: Store, simple_skill: SkillSpecification, base_state: ExecutionState):
    _seed(store, simple_skill, base_state)
    v1 = apply_patch(base_state, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "a"}])).state
    v1b = apply_patch(base_state, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "b"}])).state
    store.commit_state(v1, expected_version=0, patch_id=None, step=0)
    with pytest.raises(StaleWriteError) as exc:
        store.commit_state(v1b, expected_version=0, patch_id=None, step=0)
    assert exc.value.expected == 0 and exc.value.actual == 1
    assert store.get_state(base_state.run_id).current_phase == "a"
    assert store.get_state_at_version(base_state.run_id, 0).current_phase == "start"
    assert [v["version"] for v in store.list_state_versions(base_state.run_id)] == [0, 1]


def test_two_connections_stale_write(tmp_path: Path, simple_skill: SkillSpecification, base_state: ExecutionState):
    """Two processes resuming the same run: the second writer loses cleanly."""
    p = tmp_path / "c.db"
    a, b = Store(Database(p)), Store(Database(p))
    _seed(a, simple_skill, base_state)
    sa = a.get_state(base_state.run_id)
    sb = b.get_state(base_state.run_id)
    a.commit_state(apply_patch(sa, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "from-a"}])).state, expected_version=0, patch_id=None, step=0)
    with pytest.raises(StaleWriteError):
        b.commit_state(apply_patch(sb, StatePatch(expected_state_version=0, ops=[{"op": "set_phase", "phase": "from-b"}])).state, expected_version=0, patch_id=None, step=0)
    assert b.get_state(base_state.run_id).current_phase == "from-a"


def test_archive_events_append_only_sequence_and_search(store: Store, simple_skill: SkillSpecification, base_state: ExecutionState):
    _seed(store, simple_skill, base_state)
    rid = base_state.run_id
    e1 = store.append_event(rid, 0, EventType.OBSERVATION, "saw the port", {"content": "server_port = 8000 on alpha"})
    e2 = store.append_event(rid, 1, EventType.OBSERVATION, "saw the port again", {"content": "server_port = 9000 on alpha"})
    assert (e1.seq, e2.seq) == (0, 1)
    assert store.count_events(rid) == 2
    hits, total = store.search_events(rid, "8000")
    assert total == 1 and hits[0].event_id == e1.event_id
    hits, total = store.search_events(rid, "server_port alpha")
    assert total == 2
    assert store.existing_event_ids(rid, {e1.event_id, "evt_nope"}) == {e1.event_id}
    assert store.search_events(rid, "!!!")[1] == 0
    assert [e.event_id for e in store.list_events(rid, newest_first=True, limit=1)] == [e2.event_id]


def test_observation_and_step_records(store: Store, simple_skill: SkillSpecification, base_state: ExecutionState):
    _seed(store, simple_skill, base_state)
    rid = base_state.run_id
    obs = Observation(run_id=rid, step=0, kind=ObservationKind.TASK_INPUT, source="task", content="short", truncated=True, full_length=10)
    store.save_observation(obs, "full text!")
    assert store.get_observation(obs.id).content == "short" and store.get_observation_full_content(obs.id) == "full text!"
    rec = store.create_step(rid, 0, observation_id=obs.id, retrieved=None, state_version_before=0)
    assert rec.phase == "observation_ready"
    store.update_step(rid, 0, phase="decision_recorded", decision={"x": 1})
    assert store.latest_step(rid).decision == {"x": 1}
    store.save_step_metrics(rid, 0, context_chars=100, model_calls=1)
    store.save_step_metrics(rid, 0, input_tokens=50)
    m = store.list_step_metrics(rid)[0]
    assert m["context_chars"] == 100 and m["input_tokens"] == 50 and m["model_calls"] == 1
    assert store.run_metrics(rid)["steps"] == 1


def test_semantic_memory_crud(store: Store):
    store.upsert_semantic(key="machine.os", category="environment", content="Ubuntu 24.04", confidence=1.0, source_run_id=None, source_event_ids=[], promoted_by="test")
    store.upsert_semantic(key="machine.os", category="environment", content="Ubuntu 26.04", confidence=1.0, source_run_id=None, source_event_ids=[], promoted_by="test")
    assert store.get_semantic("machine.os").content == "Ubuntu 26.04"
    assert [m.key for m in store.search_semantic("ubuntu")] == ["machine.os"]
    assert store.delete_semantic("machine.os") and store.get_semantic("machine.os") is None
