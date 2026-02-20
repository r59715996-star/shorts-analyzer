"""FastAPI app — 3 API routes + static file serving."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.models import AnalyzeRequest, JobStage, JobStatusResponse
from web.worker import create_job, get_job, run_pipeline

app = FastAPI(title="Channel Insights", version="1.0.0")

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


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(_STATIC_DIR / "index.html")


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
