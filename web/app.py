"""FastAPI app — 3 API routes + static file serving."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.db import init_db
from web.insight_model import (
    fit_and_explain,
    generate_shap_recommendations,
    load_clips_from_db,
    prepare_features,
)
from web.models import (
    AnalyzeRequest,
    FeatureInsightModel,
    InteractionInsightModel,
    JobStage,
    JobStatusResponse,
    ModelInsightsResponse,
)
from web.worker import create_job, get_job, run_pipeline

app = FastAPI(title="Channel Insights", version="1.0.0")

init_db()

# Static files
_STATIC_DIR = Path(__file__).parent / "static"


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Start a new channel analysis job."""
    job_id = str(uuid.uuid4())[:8]
    job = create_job(job_id, request.channel_url, request.email)
    asyncio.create_task(run_pipeline(job))
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Poll job progress."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        channel_name=job.channel_name,
    )


@app.get("/api/report/{job_id}")
async def report(job_id: str):
    """Get completed report."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.stage == JobStage.failed:
        raise HTTPException(status_code=500, detail=job.error or "Analysis failed")
    if job.stage != JobStage.completed or job.report is None:
        raise HTTPException(status_code=202, detail="Report not ready yet")
    return job.report.model_dump()


@app.get("/api/insights/{channel_id}")
async def insights(channel_id: str):
    """Run SHAP-based analysis on persisted clips for a channel."""
    clips = load_clips_from_db(channel_id)
    if len(clips) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 clips for SHAP analysis, found {len(clips)}",
        )
    X, y, feature_names = prepare_features(clips)
    model_insights = fit_and_explain(X, y, feature_names)
    recommendations = generate_shap_recommendations(model_insights)
    return ModelInsightsResponse(
        loocv_r2=model_insights.loocv_r2,
        permutation_rank=model_insights.permutation_rank,
        signal_detected=model_insights.signal_detected,
        stability_threshold=model_insights.stability_threshold,
        n_samples=model_insights.n_samples,
        features=[
            FeatureInsightModel(
                name=f.name,
                display_name=f.display_name,
                importance=f.importance,
                direction=f.direction,
                stability=f.stability,
                passes_threshold=f.passes_threshold,
            )
            for f in model_insights.features
        ],
        interactions=[
            InteractionInsightModel(
                feature_a=i.feature_a,
                feature_b=i.feature_b,
                strength=i.strength,
                threshold=i.threshold,
                description=i.description,
            )
            for i in model_insights.interactions
        ],
        recommendations=recommendations,
    ).model_dump()


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(_STATIC_DIR / "index.html")


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
