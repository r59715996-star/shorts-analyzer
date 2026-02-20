# Shorts Analyzer

Analyze a YouTube channel's recent Shorts and generate a report that compares top vs bottom performers using transcript-derived quantitative and qualitative features.

## What It Does

- Accepts a YouTube channel URL/identifier from the web UI.
- Pulls up to 50 recent Shorts (10-180s) with YouTube Data API.
- Computes engagement and publish-time metadata.
- Downloads each Short's audio and transcribes with Groq Whisper.
- Extracts 12 quantitative features (`wpm`, filler density, hook pace, pronoun ratios, reading level, etc.).
- Extracts 8 qualitative tags (hook type, emotion, topic, payoff, examples, etc.) via Groq LLM.
- Splits clips into top 25% and bottom 25% by engagement rate.
- Produces statistical comparisons (Cohen's d + approximate significance), hook breakdowns, recommendations, and example performers.
- Persists per-clip features into SQLite (`data/insights.db`) for reuse/training.

## Architecture

- `web/app.py`: FastAPI app and HTTP routes.
- `web/worker.py`: Background job orchestration and pipeline execution.
- `web/report_generator.py`: Tier splitting, feature comparison, recommendations.
- `web/db.py`: SQLite schema and persistence.
- `web/static/index.html`: Single-page UI (form, polling, report rendering).
- `scripts/analysis_pipeline/*`: YouTube fetch, metric enhancement, transcription/audio.
- `scripts/data_pipeline/*`: Quantitative and qualitative feature extraction.
- `data/config/qualv1.txt`: System prompt for qualitative tagging.

## API

- `POST /api/analyze`
- Body: `{"channel_url":"...", "email":"..."}`
- Returns: `{"job_id":"..."}`
- `GET /api/status/{job_id}`
- Returns current stage/progress/message.
- `GET /api/report/{job_id}`
- Returns final `ChannelReport` when complete.

Job stages:
- `queued`
- `fetching_shorts`
- `enhancing_metrics`
- `processing_clips`
- `generating_report`
- `completed`
- `failed`

## Setup

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Install system tools

- `yt-dlp` must be available on your PATH.

### 3) Configure environment

```bash
export YOUTUBE_API_KEY="your-youtube-data-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

### 4) Run the app

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

## Notes and Current Limitations

- Jobs are stored in-memory (`web/worker.py`), so server restarts clear active job status.
- The `email` field is collected but not used for delivery yet.
- Qualitative tagging depends on `data/config/qualv1.txt` being present.
- Failures in individual clip processing are skipped; full job fails only if all clips fail.

## Data Output

- SQLite DB: `data/insights.db`
- Table: `clips` (video metadata + quant features + qual features + timestamp)
