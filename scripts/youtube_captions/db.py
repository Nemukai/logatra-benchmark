"""SQLite state for the collector (data/private/youtube_captions/yt.sqlite)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    video_id     TEXT NOT NULL,
    lang         TEXT NOT NULL,          -- language whose query found it
    query        TEXT,
    filter       TEXT,
    title        TEXT,
    channel      TEXT,
    channel_id   TEXT,
    duration     REAL,
    discovered_at REAL,
    PRIMARY KEY (video_id, lang)
);
CREATE TABLE IF NOT EXISTS searches (
    lang TEXT, query TEXT, filter TEXT, n_results INTEGER, done_at REAL,
    PRIMARY KEY (lang, query, filter)
);
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    lang         TEXT,                   -- final manifest language
    status       TEXT,                   -- accepted | rejected | blocked | error
    reason       TEXT,                   -- rejection / error reason
    title        TEXT,
    channel      TEXT,
    channel_id   TEXT,
    url          TEXT,
    upload_date  TEXT,
    freshness    TEXT,                   -- fresh | older | stale
    license      TEXT,
    duration     REAL,
    category     TEXT,
    sub_lang     TEXT,                   -- manual subtitle track used
    sub_path     TEXT,
    quality      TEXT,                   -- JSON caption-quality metrics
    audio_status TEXT,                   -- NULL | done | blocked | error:<msg>
    probed_at    REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL,
    lang         TEXT NOT NULL,
    idx          INTEGER,
    start        REAL,
    end          REAL,
    duration     REAL,
    reference_text TEXT,
    num_speakers INTEGER,
    audio_path   TEXT                    -- relative to data/private/youtube_captions
);
CREATE INDEX IF NOT EXISTS chunks_video ON chunks(video_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, table: str, row: dict) -> None:
    row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
           for k, v in row.items()}
    cols = ",".join(row)
    qs = ",".join("?" for _ in row)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
    con.commit()


def now() -> float:
    return time.time()
