"""Background pipeline — chains existing modules, reports progress, manages temp files."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.analysis_pipeline.youtube_performance_extractor import (
    build_youtube_client,
    extract_shorts_performance,
)
from scripts.analysis_pipeline.enhance_shorts_performance import enhance_entries
from scripts.analysis_pipeline.transcribe_inbox_clips import (
    download_short_audio,
    transcribe_audio,
)
from scripts.data_pipeline.quant_v1 import compute_quant_v1
from scripts.data_pipeline.groq_qual_client import (
    compute_qual_v1_from_transcript,
    get_groq_client,
)
from web.channel_parser import parse_channel_url
from web.models import ChannelReport, JobStage
from web.report_generator import generate_report


# Concurrency limit for parallel clip processing
MAX_CONCURRENT_CLIPS = 5

# Load the qualitative prompt once
_QUAL_PROMPT_PATH = _PROJECT_ROOT / "data" / "config" / "qualv1.txt"


def _load_qual_prompt() -> str:
    """Load the qualitative tagging system prompt."""
    if _QUAL_PROMPT_PATH.exists():
        return _QUAL_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Qualitative prompt not found: {_QUAL_PROMPT_PATH}")


class Job:
    """Tracks the state of a single analysis job."""

    def __init__(self, job_id: str, channel_url: str, email: str):
        self.job_id = job_id
        self.channel_url = channel_url
        self.email = email
        self.stage: JobStage = JobStage.queued
        self.progress: int = 0
        self.message: str = "Queued"
        self.channel_name: Optional[str] = None
        self.report: Optional[ChannelReport] = None
        self.error: Optional[str] = None

    def update(self, stage: JobStage, progress: int, message: str) -> None:
        self.stage = stage
        self.progress = progress
        self.message = message


# In-memory job store
_jobs: Dict[str, Job] = {}


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def create_job(job_id: str, channel_url: str, email: str) -> Job:
    job = Job(job_id, channel_url, email)
    _jobs[job_id] = job
    return job


async def run_pipeline(job: Job) -> None:
    """Execute the full analysis pipeline for a job."""
    try:
        # Step 1: Parse channel URL
        job.update(JobStage.fetching_shorts, 5, "Parsing channel URL...")
        identifier = parse_channel_url(job.channel_url)

        # Step 2: Fetch Shorts from YouTube API
        job.update(JobStage.fetching_shorts, 10, "Connecting to YouTube API...")
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable not set")

        youtube = build_youtube_client(api_key)

        job.update(JobStage.fetching_shorts, 15, "Fetching channel Shorts...")
        shorts, channel_id, channel_name = await asyncio.to_thread(
            extract_shorts_performance,
            youtube,
            identifier.search_term,
            is_channel_id=identifier.is_channel_id,
            max_shorts=50,
        )

        if not shorts:
            raise ValueError("No Shorts found for this channel")
        if not channel_id:
            raise ValueError("Could not resolve channel ID")

        job.channel_name = channel_name or identifier.search_term
        job.update(
            JobStage.fetching_shorts,
            20,
            f"Found {len(shorts)} Shorts for {job.channel_name}",
        )

        # Step 3: Enhance metrics
        job.update(JobStage.enhancing_metrics, 25, "Computing engagement metrics...")
        enhanced = await asyncio.to_thread(enhance_entries, shorts)
        job.update(JobStage.enhancing_metrics, 30, "Metrics enhanced")

        # Step 4: Process clips (transcribe + extract features)
        job.update(JobStage.processing_clips, 35, "Starting clip processing...")
        qual_prompt = _load_qual_prompt()

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)
        processed_clips: List[Dict[str, Any]] = []
        total = len(enhanced)
        completed_count = 0

        async def process_single_clip(clip: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            nonlocal completed_count
            video_id = clip.get("video_id", "")
            async with semaphore:
                try:
                    result = await asyncio.to_thread(
                        _process_clip_sync, video_id, qual_prompt
                    )
                    clip["quant_features"] = result["quant"]
                    clip["qual_features"] = result["qual"]
                    completed_count += 1
                    pct = 35 + int((completed_count / total) * 50)
                    job.update(
                        JobStage.processing_clips,
                        pct,
                        f"Processed {completed_count}/{total} clips",
                    )
                    return clip
                except Exception as exc:
                    print(f"Failed to process clip {video_id}: {exc}")
                    completed_count += 1
                    pct = 35 + int((completed_count / total) * 50)
                    job.update(
                        JobStage.processing_clips,
                        pct,
                        f"Processed {completed_count}/{total} clips (1 failed)",
                    )
                    return None

        tasks = [process_single_clip(clip) for clip in enhanced]
        results = await asyncio.gather(*tasks)
        processed_clips = [r for r in results if r is not None]

        if not processed_clips:
            raise ValueError("All clip processing failed")

        # Step 5: Generate report
        job.update(JobStage.generating_report, 90, "Generating insights report...")
        report = await asyncio.to_thread(
            generate_report, processed_clips, job.channel_name, channel_id
        )

        job.report = report
        job.update(JobStage.completed, 100, "Report ready!")

    except Exception as exc:
        job.stage = JobStage.failed
        job.error = str(exc)
        job.message = f"Failed: {exc}"
        job.progress = 0
        traceback.print_exc()


def _process_clip_sync(video_id: str, qual_prompt: str) -> Dict[str, Any]:
    """Process a single clip synchronously — download, transcribe, extract features."""
    with tempfile.TemporaryDirectory(prefix=f"clip_{video_id}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_path = tmp_path / f"{video_id}.wav"

        # Download audio using yt-dlp
        download_short_audio(video_id, audio_path)

        # Transcribe using Groq Whisper
        groq_client = get_groq_client()
        transcript = transcribe_audio(
            client=groq_client,
            audio_path=audio_path,
            model_name="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
            timestamp_granularity="word",
        )

        # Extract quantitative features
        quant = compute_quant_v1(transcript)

        # Extract qualitative features
        qual = compute_qual_v1_from_transcript(transcript, qual_prompt)

    return {"quant": quant, "qual": qual}
