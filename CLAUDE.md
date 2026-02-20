# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
export YOUTUBE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
pip install -r requirements.txt
uvicorn web.app:app --port 8000
```

No test suite exists yet. To verify imports and report generation logic:
```bash
python3 -c "from web.app import app; print('OK')"
```

## Architecture

The app analyzes a YouTube channel's Shorts to find what separates top performers from bottom performers. A user submits a channel URL, the backend processes it as a background job, and the frontend polls for progress until a report is ready.

### Request Flow

```
POST /api/analyze → creates Job, spawns async run_pipeline()
GET  /api/status/{job_id} → poll progress (frontend polls every 2s)
GET  /api/report/{job_id} → returns ChannelReport JSON when done
```

### Pipeline Stages (in `web/worker.py`)

`worker.py` is the orchestrator — it chains all `scripts/` modules in sequence:

1. **Parse URL** → `web/channel_parser.py` extracts handle/channel_id from various URL formats
2. **Fetch Shorts** → `scripts/analysis_pipeline/youtube_performance_extractor.py` pulls up to 50 Shorts via YouTube Data API
3. **Enhance metrics** → `scripts/analysis_pipeline/enhance_shorts_performance.py` adds `engagement_rate = (likes + comments*3) / views`
4. **Process clips** (5 concurrent via `asyncio.Semaphore`):
   - Download audio via yt-dlp → `scripts/analysis_pipeline/transcribe_inbox_clips.py:download_short_audio()`
   - Transcribe via Groq Whisper → `transcribe_audio()`
   - Extract 12 quant features → `scripts/data_pipeline/quant_v1.py:compute_quant_v1()`
   - Extract 8 qual features via LLM → `scripts/data_pipeline/groq_qual_client.py:compute_qual_v1_from_transcript()`
5. **Persist clips** → `web/db.py:save_clips()` writes per-clip features to SQLite (`data/insights.db`) for future ML training
6. **Generate report** → `web/report_generator.py` splits top/bottom 25%, computes Cohen's d + Welch's t-test, generates recommendations

### Key Design Decisions

- **Dual storage** — Job state lives in-memory (`_jobs` dict in `worker.py`, lost on restart). Clip feature data persists to SQLite via `web/db.py` (`data/insights.db`), using `INSERT OR REPLACE` so re-analyzing a channel updates existing rows.
- **DB initialization** — `init_db()` is called at import time in `web/app.py`, creating the `clips` table if it doesn't exist.
- **Temp files** — audio downloads go to `tempfile.TemporaryDirectory`, auto-cleaned per clip.
- **Qual prompt** loaded from `data/config/qualv1.txt` — defines the 8 qualitative labels the LLM extracts.
- **Frontend** is a single `web/static/index.html` using Tailwind CDN + vanilla JS (no build step).

### Module Dependency Graph

All `scripts/` modules are standalone (no cross-imports). `web/worker.py` is the only file that imports from both `scripts/` and `web/`. The `scripts/__init__.py` files exist solely to enable these imports.

```
web/app.py → web/worker.py → scripts/analysis_pipeline/*
                            → scripts/data_pipeline/*
                            → web/report_generator.py
                            → web/channel_parser.py
                            → web/db.py (save_clips)
           → web/db.py (init_db at startup)
           → web/models.py (shared by app, worker, report_generator)
```
