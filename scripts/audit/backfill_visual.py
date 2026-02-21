"""
Backfill visual features for existing clips in the database.

Queries for clips where v_brightness IS NULL, downloads the first 3s of video,
computes visual features, and updates the row.

Usage:
    python3 -m scripts.audit.backfill_visual
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.data_pipeline.visual_v1 import (
    compute_visual_v1,
    download_short_video,
)

_DB_PATH = _PROJECT_ROOT / "data" / "insights.db"
_SEMAPHORE_LIMIT = 5

_VISUAL_COLS = [
    "v_brightness", "v_contrast", "v_edge_density", "v_colorfulness",
    "v_face_present", "v_face_area_frac", "v_text_area_frac",
    "v_motion_magnitude", "v_scene_cut",
]


def _get_missing_video_ids() -> list[str]:
    """Return video_ids where visual features have not been computed."""
    con = sqlite3.connect(str(_DB_PATH))
    try:
        rows = con.execute(
            "SELECT video_id FROM clips WHERE v_brightness IS NULL"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def _update_visual_features(video_id: str, features: dict) -> None:
    """Update a single clip row with visual features."""
    set_clause = ", ".join(f"{c} = ?" for c in _VISUAL_COLS)
    values = [features.get(c) for c in _VISUAL_COLS]
    values.append(video_id)

    con = sqlite3.connect(str(_DB_PATH))
    try:
        con.execute(
            f"UPDATE clips SET {set_clause} WHERE video_id = ?",
            values,
        )
        con.commit()
    finally:
        con.close()


def _process_one(video_id: str) -> bool:
    """Download video, compute features, update DB. Returns True on success."""
    with tempfile.TemporaryDirectory(prefix=f"backfill_{video_id}_") as tmp_dir:
        video_path = Path(tmp_dir) / f"{video_id}.mp4"
        try:
            download_short_video(video_id, video_path)
        except Exception as exc:
            print(f"  SKIP {video_id}: download failed — {exc}")
            return False

        features = compute_visual_v1(video_path)
        if features.get("v_brightness") is None:
            print(f"  SKIP {video_id}: feature extraction failed")
            return False

        _update_visual_features(video_id, features)
        return True


async def _run_backfill() -> None:
    video_ids = _get_missing_video_ids()
    total = len(video_ids)
    if total == 0:
        print("No clips need visual backfill.")
        return

    print(f"Backfilling visual features for {total} clips...\n")

    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    success = 0
    failed = 0

    async def process(vid: str, idx: int) -> None:
        nonlocal success, failed
        async with semaphore:
            ok = await asyncio.to_thread(_process_one, vid)
            if ok:
                success += 1
            else:
                failed += 1
            done = success + failed
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total} (success={success}, failed={failed})")

    tasks = [process(vid, i) for i, vid in enumerate(video_ids)]
    await asyncio.gather(*tasks)

    print(f"\nDone. {success} updated, {failed} failed out of {total} clips.")


def main() -> None:
    asyncio.run(_run_backfill())


if __name__ == "__main__":
    main()
