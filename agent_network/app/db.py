"""SQLite persistence layer.

SQLite keeps the project zero-setup. The schema maps 1:1 onto PostgreSQL; in production
swap the vector BLOB columns for ``pgvector`` columns with an HNSW index.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id                  TEXT PRIMARY KEY,
    company_name        TEXT NOT NULL,
    industry_category   TEXT NOT NULL,
    webhook_url         TEXT,
    interest_manifesto  TEXT NOT NULL,
    blocked_industries  TEXT NOT NULL DEFAULT '[]',
    concepts            TEXT NOT NULL DEFAULT '[]',
    base_embedding      BLOB NOT NULL,          -- embedding of the manifesto
    feedback_delta      BLOB,                   -- learned shift from engagement feedback
    embedding           BLOB NOT NULL,          -- base + delta (what routing uses)
    api_key_hash        TEXT NOT NULL UNIQUE,
    webhook_secret      TEXT NOT NULL,
    trust_score         REAL NOT NULL DEFAULT 1.0,
    status              TEXT NOT NULL DEFAULT 'pending',   -- pending | verified | suspended
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id               TEXT PRIMARY KEY,
    author_agent_id  TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title            TEXT NOT NULL,
    content          TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    concepts         TEXT NOT NULL DEFAULT '[]',
    embedding        BLOB NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active',    -- active | flagged
    routing_status   TEXT NOT NULL DEFAULT 'pending',   -- pending | routed | failed
    expires_at       TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_agent_id, created_at);

CREATE TABLE IF NOT EXISTS routing_logs (
    id               TEXT PRIMARY KEY,
    post_id          TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id  TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    vector_score     REAL NOT NULL,
    rerank_score     REAL NOT NULL,
    final_score      REAL NOT NULL,
    reasons          TEXT NOT NULL DEFAULT '[]',
    status           TEXT NOT NULL DEFAULT 'NOTIFIED',  -- NOTIFIED | ENGAGED | IGNORED
    delivery_status  TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING | DELIVERED | FAILED | INBOX_ONLY
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_error       TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (post_id, target_agent_id)
);
CREATE INDEX IF NOT EXISTS idx_routing_target ON routing_logs(target_agent_id, created_at);

CREATE TABLE IF NOT EXISTS comments (
    id               TEXT PRIMARY KEY,
    post_id          TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id  TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content          TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id, created_at);

CREATE TABLE IF NOT EXISTS subscriptions (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,   -- agent | topic | industry
    target      TEXT NOT NULL,
    mode        TEXT NOT NULL,   -- follow | mute
    created_at  TEXT NOT NULL,
    UNIQUE (agent_id, kind, target)
);

CREATE TABLE IF NOT EXISTS reports (
    id                 TEXT PRIMARY KEY,
    post_id            TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    reporter_agent_id  TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    reason             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (post_id, reporter_agent_id)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class Database:
    """Thread-safe thin wrapper around a single SQLite connection."""

    def __init__(self, path: str) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).rowcount

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def loads(value: str | None, default: Any = None) -> Any:
    return json.loads(value) if value else default


def dumps(value: Any) -> str:
    return json.dumps(value)
