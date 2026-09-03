"""SQLite connection management with explicit, reentrant transactions."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mnestic.storage.migrations import apply_migrations


class Database:
    """A single SQLite connection with WAL, foreign keys and explicit transactions.

    ``transaction()`` is reentrant: nested uses join the outer transaction so a
    multi-repository lifecycle phase commits atomically.
    """

    def __init__(self, path: Path | str, *, timeout: float = 30.0):
        self.path = Path(path)
        self.is_memory = str(path) == ":memory:"
        if not self.is_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(path),
            timeout=timeout,
            isolation_level=None,  # autocommit; we issue BEGIN/COMMIT ourselves
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        self.conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory:
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        self.fts_enabled = _fts5_available(self.conn)
        with self.transaction():
            apply_migrations(self.conn, fts_enabled=self.fts_enabled)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._depth == 0:
                self.conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.conn
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    self.conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if self._depth == 0:
                    self.conn.execute("COMMIT")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False
