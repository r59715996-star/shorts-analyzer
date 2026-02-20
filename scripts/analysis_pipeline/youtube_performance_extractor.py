#!/usr/bin/env python3
"""
youtube_performance_extractor.py - Fetch recent Shorts stats into JSON.

Pulls the latest Shorts from a channel and records:
- video_id
- view_count
- like_count
- comment_count
- days_since_publish

Output: data/tagging/{niche}/{channel_name}/performance/shorts_performance.json

Usage:
    python youtube_performance_extractor.py --channel-name "Millennial Masters" --api-key YOUR_KEY
    python youtube_performance_extractor.py --channel-id UCxxxxxxxxxxxx --api-key YOUR_KEY

Environment:
    export YOUTUBE_API_KEY="your-key-here"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_NICHE = "entrepreneurship"
DEFAULT_CHANNEL_OUTPUT = "millennial_masters"

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError as exc:
    print("google-api-python-client is required. Install with:")
    print("   pip install google-api-python-client")
    raise SystemExit(1) from exc


DEFAULT_MAX_SHORTS = 50
OUTPUT_FILENAME = "shorts_performance.json"


def build_youtube_client(api_key: str):
    """Instantiate a YouTube Data API client."""
    return build("youtube", "v3", developerKey=api_key)


def find_channel_id(youtube, channel_name: str) -> Optional[str]:
    """Find a channel ID by name (first search hit)."""
    try:
        resp = (
            youtube.search()
            .list(part="snippet", q=channel_name, type="channel", maxResults=5)
            .execute()
        )
    except HttpError as exc:
        print(f"Channel search failed: {exc}")
        return None

    items = resp.get("items") or []
    if not items:
        print(f"No channels found for '{channel_name}'")
        return None

    best = items[0]
    channel_id = best["id"]["channelId"]
    title = best["snippet"]["title"]
    print(f"Selected channel: {title} ({channel_id})")
    return channel_id


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    """Return the uploads playlist for a channel."""
    try:
        resp = (
            youtube.channels()
            .list(part="contentDetails", id=channel_id)
            .execute()
        )
    except HttpError as exc:
        print(f"Failed to fetch uploads playlist: {exc}")
        return None

    items = resp.get("items") or []
    if not items:
        print("No channel items returned")
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_recent_videos(youtube, playlist_id: str, max_results: int = 100) -> List[Dict]:
    """List recent uploads (basic info) from a playlist."""
    videos: List[Dict] = []
    next_page_token: Optional[str] = None

    try:
        while len(videos) < max_results:
            resp = (
                youtube.playlistItems()
                .list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token,
                )
                .execute()
            )
            for item in resp.get("items", []):
                content_details = item.get("contentDetails", {}) or {}
                snippet = item.get("snippet", {}) or {}
                published_at = (
                    content_details.get("videoPublishedAt")
                    or snippet.get("publishedAt")
                    or ""
                )
                videos.append(
                    {
                        "video_id": content_details.get("videoId"),
                        "published_at": published_at,
                        "title": snippet.get("title", ""),
                    }
                )
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as exc:
        print(f"Error listing playlist items: {exc}")
    return videos


def fetch_video_details(youtube, video_ids: List[str]) -> List[Dict]:
    """Fetch detailed metadata for a list of video IDs."""
    details: List[Dict] = []
    batch_size = 50
    try:
        for start in range(0, len(video_ids), batch_size):
            batch = video_ids[start : start + batch_size]
            resp = (
                youtube.videos()
                .list(
                    part="snippet,contentDetails,statistics,topicDetails",
                    id=",".join(batch),
                )
                .execute()
            )
            for item in resp.get("items", []):
                snippet = item.get("snippet", {}) or {}
                item["channel_name"] = snippet.get("channelTitle", "")
                details.append(item)
            time.sleep(0.1)
    except HttpError as exc:
        print(f"Error fetching video details: {exc}")
    return details


MIN_SHORT_DURATION_S = 10
MAX_SHORT_DURATION_S = 180


def parse_duration_seconds(duration_iso8601: str) -> int | None:
    """Parse an ISO 8601 duration string and return total seconds, or None if unparseable."""
    import re

    if not duration_iso8601 or "H" in duration_iso8601:
        return None

    match = re.match(r"^PT(?:(\d+)M)?(?:(\d+)S)?$", duration_iso8601)
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2) or 0)
    return minutes * 60 + seconds


def is_short(duration_iso8601: str) -> bool:
    """Determine if a video duration corresponds to a Short (10-180s)."""
    total = parse_duration_seconds(duration_iso8601)
    if total is None:
        return False
    return MIN_SHORT_DURATION_S <= total <= MAX_SHORT_DURATION_S


def compute_days_since(published_at: str) -> int:
    """Return integer days since published timestamp."""
    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0
    now = datetime.now(published_dt.tzinfo)
    return (now - published_dt).days


def extract_shorts_performance(
    youtube,
    channel_identifier: str,
    *,
    is_channel_id: bool = False,
    max_shorts: int = DEFAULT_MAX_SHORTS,
) -> tuple[List[Dict], Optional[str], Optional[str]]:
    """Collect recent Shorts performance stats for a channel."""
    if is_channel_id:
        channel_id = channel_identifier
    else:
        channel_id = find_channel_id(youtube, channel_identifier)
        if not channel_id:
            return [], None, None

    uploads_playlist = get_uploads_playlist_id(youtube, channel_id)
    if not uploads_playlist:
        return [], channel_id, None

    print("Fetching recent uploads...")
    recent_videos = list_recent_videos(youtube, uploads_playlist, max_results=500)
    print(f"  Found {len(recent_videos)} recent uploads")

    if not recent_videos:
        return [], channel_id, None

    print("Fetching details and filtering for Shorts...")
    details = fetch_video_details(youtube, [v["video_id"] for v in recent_videos])
    shorts: List[Dict] = []
    channel_name: Optional[str] = None
    for detail in details:
        duration = detail.get("contentDetails", {}).get("duration", "")
        duration_s = parse_duration_seconds(duration)
        if duration_s is None or not (MIN_SHORT_DURATION_S <= duration_s <= MAX_SHORT_DURATION_S):
            continue

        stats = detail.get("statistics", {})
        snippet = detail.get("snippet", {})
        topic_details = detail.get("topicDetails", {}) or {}
        published_at = snippet.get("publishedAt", "")
        channel_title = detail.get("channel_name") or snippet.get("channelTitle", "")
        if channel_title:
            channel_name = channel_title

        shorts.append(
            {
                "video_id": detail.get("id"),
                "duration_s": duration_s,
                "view_count": int(stats.get("viewCount", 0) or 0),
                "like_count": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
                "days_since_publish": compute_days_since(published_at),
                "published_at": published_at,
                "title": snippet.get("title", ""),
                "channel_name": channel_title,
                "topic_categories_raw": topic_details.get("topicCategories", []),
            }
        )
        if len(shorts) >= max_shorts:
            break

    shorts.sort(key=lambda item: item.get("published_at", ""), reverse=True)
    return shorts[:max_shorts], channel_id, channel_name
