# Should Class W Amnestic use a database (and which)?

*Based on the paper: Badhe, Tiwari, Chung — SKILL.state: Scalable Long-Horizon Agent Skills, [arXiv:2608.26263](https://arxiv.org/abs/2608.26263).*

**Decision: yes — keep SQLite as the one durable store for v1–v3. Do not introduce a database server now.
Revisit when a concrete trigger below fires.** The `Store` facade is the seam where a second backend would plug in.

## What the store has to do

| need | why it matters here |
|---|---|
| atomic multi-table commits | a state commit = version check + `current_states` + `state_versions` + patch record + events, all-or-nothing |
| optimistic concurrency | `UPDATE … WHERE state_version=?` must be atomic against a second resumer |
| append-only, queryable archive | retrieval by id/type/step, keyword search (FTS), state history |
| durable phase log for crash/resume | `steps` row updated at every lifecycle transition |
| zero-ops for a CLI tool | `uv run mnestic run …` must work on a laptop with no services |
| inspectability | `history`, `diff`, `events`, `inspect-context` are plain SQL |

A flat-file/JSONL design fails the first two (no atomic multi-record commits, no compare-and-set without
locking games) and the third (search). So *some* database is required; the question is only which.

## Measured on this implementation (1000-step benchmark, `scripts/benchmark.py` workload)

- Throughput: **~2 ms per full step** including graph, patch validation and every SQLite write (WAL, `synchronous=NORMAL`). The model call dominates real runs by 4–5 orders of magnitude (Kimi K3 via 9Router: 18–169 s/call).
- Growth: **23.9 MB per 1000 steps ≈ 23 KB/step.** By object: `model_calls` 36 % (exact context per call), `archive_events` 26 %, `state_versions` 12 %, FTS index 5 %, observations 4 %.
- Latency at 9,008 events: `get_state` and `get_state_at_version` ≈ 0.01 ms; recent events 0.1 ms; FTS exact token 0.03 ms; FTS common-word 8.5 ms; `state_history` 3 ms; `append_event` 0.07 ms.
- Extrapolation: a 100k-step run ≈ 2.4 GB; SQLite's limits (281 TB file, 2^64 rows) are not the constraint — disk and backup hygiene are.

## Options considered

| option | verdict |
|---|---|
| **SQLite (current)** | Zero-ops, transactional, FTS5 built in, one file per deployment that can be copied/inspected/attached. Single-writer-at-a-time per file is fine because a run has one active process by design and the version check makes a second one fail safely. **Keep.** |
| PostgreSQL | Wins only for: many concurrent writer *processes* on many machines, network access from workers/subagents on other hosts, cross-run analytics at scale, row-level permissions. Costs: a service to run, migrations in two dialects, slower per-step round trips over the network, and loss of the "copy the file" workflow. **Not now.** |
| DuckDB | Great for analytics over many runs, poor fit for high-frequency small transactional writes and no FTS parity. Could be a *read* side later (`ATTACH` the SQLite file). **No.** |
| Files / JSONL | No atomic multi-record commits, no CAS, no search. **No.** |
| Vector DB | Not a replacement — a possible *additional* retriever behind the `Retriever` protocol (roadmap stage 3). **Not a database decision.** |

## Triggers that would justify adding PostgreSQL (as a second `Store` implementation)

1. Subagents or workers on **different hosts** need to write to the same run archive.
2. More than one resumer per run must *cooperate* (leases/queues) rather than lose-and-pause.
3. A fleet of runs needs central analytics/dashboards (thousands of runs, queries across them).
4. Compliance needs (row-level access, audit users) beyond file permissions.

None applies to v1–v3 (single-agent, single-host, CLI-first).

## Cheap improvements inside SQLite (do these first)

- **Cut `model_calls.context_json` (36 % of bytes):** the context is reconstructible from `state_versions` + the observation + retrieval record; store a content hash by default and the raw text only when `MNESTIC_ARCHIVE_CONTEXT=full` (or for the last N steps). Keeps `inspect-context --next` exact and `--step N` reproducible via rebuild.
- **Checkpoint/vacuum policy:** `PRAGMA wal_checkpoint(TRUNCATE)` at run end; optional `VACUUM` command in the CLI.
- **Per-run or per-project DB files:** already supported via `--db`; recommend one DB per project workspace.
- **Retention:** an `archive prune` command that drops `model_calls.raw_messages_json` older than N days (never events or state versions).
- **Backups:** `sqlite3 .backup` or copying the file while no process holds a write lock; document in README.

## Summary

SQLite is not an "afterthought" here; it is the right tool for a single-host, one-process-per-run agent runtime and
it is 3–5 orders of magnitude faster than the model calls it records. Keep it, trim what we store per call, and add a
PostgreSQL `Store` only when a multi-host trigger actually appears.
