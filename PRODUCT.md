# Channel Insights for Talking-Head Creators

## What It Does

A creator pastes their YouTube channel URL and gets a personalized report showing what structurally separates their top-performing Shorts from their worst.

## How It Works

1. User pastes channel URL + enters email
2. System pulls last 50 Shorts via YouTube Data API
3. Transcribes each clip using Groq Whisper
4. Extracts 12 quantitative features (WPM, hook speed, filler density, etc.)
5. Extracts 8 qualitative features via LLM (hook type, emotion, topic, etc.)
6. Splits clips into top 25% vs bottom 25% by engagement rate
7. Computes statistical comparisons (Cohen's d, significance testing)
8. Generates 3-5 actionable recommendations with specific numbers

## Report Includes

- **Feature comparisons** — each feature with means, difference, effect size, significance
- **Hook type breakdown** — performance ranking of each hook type
- **Actionable recommendations** — data-driven, with specific numbers from their channel
- **Top/bottom performer examples** — video IDs, titles, engagement rates
- **Common traits summary** — dominant patterns in each tier

## Cost

~$0.05 per analysis (50 clips x $0.001 transcription cost)

## Stack

- Python 3.13, FastAPI, Tailwind CSS
- YouTube Data API v3
- Groq Whisper large-v3 (transcription)
- Groq Llama 3.3-70b (qualitative tagging)
- yt-dlp (audio download)

## Running

```bash
export YOUTUBE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
pip install -r requirements.txt
uvicorn web.app:app --port 8000
```

Open http://localhost:8000
