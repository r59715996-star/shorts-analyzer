#!/usr/bin/env python3
"""
Transcribe clips with Groq Whisper v3.

Provides extract_audio() and transcribe_audio() for use by the web worker.

Requirements:
    pip install groq
    ffmpeg (installed and available on PATH)
    export GROQ_API_KEY="your-key-here"
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, List

try:
    from groq import Groq
except ImportError:
    Groq = None


def extract_audio(
    ffmpeg_bin: str,
    clip_path: Path,
    audio_path: Path,
    sample_rate: int,
) -> None:
    """Use ffmpeg to convert clip video into mono WAV audio."""
    cmd: List[str] = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(clip_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg binary not found: {ffmpeg_bin}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed while extracting audio: {stderr}") from exc


def transcribe_audio(
    client: Any,
    audio_path: Path,
    model_name: str,
    response_format: str,
    temperature: float,
    timestamp_granularity: str,
) -> dict:
    """Send the WAV audio to Groq Whisper and return the parsed JSON."""
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=[timestamp_granularity],
        )

    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    if hasattr(transcription, "to_dict"):
        return transcription.to_dict()
    return json.loads(transcription)


def download_short_audio(video_id: str, output_path: Path) -> Path:
    """Download audio from a YouTube Short using yt-dlp."""
    url = f"https://www.youtube.com/shorts/{video_id}"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        "--no-playlist",
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"yt-dlp failed for {video_id}: {stderr}") from exc
    return output_path
