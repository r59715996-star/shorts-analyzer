# Hey, I'm Raghav

I build AI systems that automate content workflows — from creation to distribution to measurement.

Currently focused on **podcast growth automation**: turning long-form episodes into viral short-form clips, predicting which clips will perform before posting, and closing the feedback loop with ML.

## What I'm building

**[ro-growth-engine](https://github.com/r59715996-star/ro-growth-engine)** — End-to-end podcast growth system. Automated clipping pipeline (hook scoring algorithm + LLM refinement) combined with a YouTube analytics pipeline that trains a CatBoost engagement model to predict clip virality. The model's feature importance feeds back into the hook scoring weights.

**[ro-clips-pipeline](https://github.com/r59715996-star/ro-clips-pipeline)** — The clipping engine. WhisperX transcription with speaker diarization → 3-tier algorithmic hook detection (scroll-stoppers, amplifiers, minor signals) → Gemini 2.5 Flash refinement using full-episode context (1M token window) → FFmpeg batch rendering with subtitles.

**[gen-engine-v2](https://github.com/r59715996-star/gen-engine-v2)** — AI video generation pipeline for TikTok UGC. Provider-agnostic abstraction layer (Kie AI, Replicate), Sora 2 prompting guidelines, creative spec generation (angles x personas), human approval gates, per-video cost tracking.

**[meta-qualifier](https://github.com/r59715996-star/meta-qualifier)** — Brand qualification at scale. Headless Chromium scrapes Meta Ad Library to tier brands by ad spend intensity, enriches with website/social data, discovers contact emails. Pydantic models, diskcache, full test suite.

## Stack

Python, FFmpeg, Playwright, WhisperX, Gemini, Groq, CatBoost, SQLite, YouTube Data API
