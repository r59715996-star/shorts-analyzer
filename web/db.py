"""SQLite persistence for per-clip feature data."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "insights.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS clips (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    channel_name TEXT,
    title TEXT,
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    engagement_rate REAL,
    days_since_publish INTEGER,
    published_at TEXT,
    -- quant features (12)
    duration_s REAL,
    word_count INTEGER,
    wpm REAL,
    hook_word_count INTEGER,
    hook_wpm REAL,
    filler_count INTEGER,
    filler_density REAL,
    question_start INTEGER,
    first_person_ratio REAL,
    second_person_ratio REAL,
    num_sentences INTEGER,
    reading_level REAL,
    -- qual features (8)
    hook_type TEXT,
    hook_emotion TEXT,
    topic_primary TEXT,
    technical_depth TEXT,
    has_payoff INTEGER,
    has_numbers INTEGER,
    has_examples INTEGER,
    insider_language INTEGER,
    -- performance indices
    vpi REAL,
    lpi REAL,
    cpi REAL,
    -- metadata
    created_at TEXT NOT NULL
);
"""

_QUANT_COLS = [
    "duration_s", "word_count", "wpm", "hook_word_count", "hook_wpm",
    "filler_count", "filler_density", "question_start", "first_person_ratio",
    "second_person_ratio", "num_sentences", "reading_level",
]

_QUAL_COLS = [
    "hook_type", "hook_emotion", "topic_primary", "technical_depth",
    "has_payoff", "has_numbers", "has_examples", "insider_language",
]

_PERF_COLS = ["vpi", "lpi", "cpi"]

_ALL_COLS = [
    "video_id", "channel_id", "channel_name", "title",
    "view_count", "like_count", "comment_count", "engagement_rate",
    "days_since_publish", "published_at",
    *_QUANT_COLS,
    *_QUAL_COLS,
    *_PERF_COLS,
    "created_at",
]


def init_db() -> None:
    """Create the clips table if it doesn't exist, and migrate schema."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    try:
        con.execute(_CREATE_TABLE)
        # Migrate existing DBs: add columns that may not exist yet
        for col in ("vpi REAL", "lpi REAL", "cpi REAL"):
            try:
                con.execute(f"ALTER TABLE clips ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        con.commit()
    finally:
        con.close()


def save_clips(
    clips: List[Dict[str, Any]], channel_id: str, channel_name: str
) -> None:
    """Persist processed clips to SQLite. Re-analyzing a channel updates existing rows."""
    placeholders = ", ".join("?" for _ in _ALL_COLS)
    col_names = ", ".join(_ALL_COLS)
    sql = f"INSERT OR REPLACE INTO clips ({col_names}) VALUES ({placeholders})"

    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(str(_DB_PATH))
    try:
        for clip in clips:
            quant = clip.get("quant_features", {})
            qual = clip.get("qual_features", {})
            row = (
                clip.get("video_id"),
                channel_id,
                channel_name,
                clip.get("title"),
                clip.get("view_count"),
                clip.get("like_count"),
                clip.get("comment_count"),
                clip.get("engagement_rate"),
                clip.get("days_since_publish"),
                clip.get("published_at"),
                *[quant.get(c) for c in _QUANT_COLS],
                *[qual.get(c) for c in _QUAL_COLS],
                *[clip.get(c) for c in _PERF_COLS],
                now,
            )
            con.execute(sql, row)
        con.commit()
    finally:
        con.close()


def load_clips_for_report(channel_id: str) -> List[Dict[str, Any]]:
    """Load clips from DB in the nested format that report_generator expects.

    Returns list of dicts with top-level metadata + nested quant_features/qual_features.
    """
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM clips WHERE channel_id = ?", (channel_id,)
        ).fetchall()
    finally:
        con.close()

    clips = []
    for row in rows:
        r = dict(row)
        clip: Dict[str, Any] = {
            "video_id": r.get("video_id"),
            "channel_name": r.get("channel_name"),
            "title": r.get("title"),
            "view_count": r.get("view_count"),
            "like_count": r.get("like_count"),
            "comment_count": r.get("comment_count"),
            "engagement_rate": r.get("engagement_rate"),
            "days_since_publish": r.get("days_since_publish"),
            "published_at": r.get("published_at"),
            "quant_features": {c: r.get(c) for c in _QUANT_COLS},
            "qual_features": {c: r.get(c) for c in _QUAL_COLS},
        }
        clips.append(clip)
    return clips
