#!/usr/bin/env python3
"""
enhance_shorts_performance.py - Add engagement metrics to Shorts performance JSON.

Computes per-video:
  engagement_rate = (like_count + (comment_count * 3)) / view_count
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def compute_engagement_rate(entry: Dict[str, Any]) -> float:
    """Compute engagement rate safely."""
    likes = entry.get("like_count") or 0
    comments = entry.get("comment_count") or 0
    views = entry.get("view_count") or 0
    try:
        likes_val = float(likes)
        comments_val = float(comments)
        views_val = float(views)
    except (TypeError, ValueError):
        return 0.0
    if views_val <= 0:
        return 0.0
    return (likes_val + (comments_val * 3.0)) / views_val


def compute_time_features(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate entries with publish day-of-week and hour."""
    updated: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        published_at = entry.get("published_at")
        day_number = None
        hour_number = None
        if isinstance(published_at, str):
            try:
                dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                day_number = dt.weekday()
                hour_number = dt.hour
            except ValueError:
                pass
        new_entry = dict(entry)
        new_entry["day_published_number"] = day_number
        new_entry["hour_published_number"] = hour_number
        updated.append(new_entry)
    return updated


def add_topic_slugs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add normalized topic category slugs derived from topic_categories_raw URLs."""
    for entry in entries:
        urls = entry.get("topic_categories_raw") or []
        slugs = []
        for url in urls:
            slug = url.rsplit("/", 1)[-1].lower()
            slugs.append(slug)
        entry["topic_categories_slugs"] = slugs
    return entries


def enhance_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add engagement_rate to each entry."""
    entries = compute_time_features(entries)
    enhanced: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        new_entry["engagement_rate"] = compute_engagement_rate(entry)
        enhanced.append(new_entry)
    enhanced = add_topic_slugs(enhanced)
    return enhanced
