"""Pydantic models for Channel Insights API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Request / Response ---

class AnalyzeRequest(BaseModel):
    channel_url: str = Field(..., description="YouTube channel URL")
    email: str = Field(..., description="User email for report delivery")


class JobStage(str, Enum):
    queued = "queued"
    fetching_shorts = "fetching_shorts"
    enhancing_metrics = "enhancing_metrics"
    processing_clips = "processing_clips"
    generating_report = "generating_report"
    completed = "completed"
    failed = "failed"


class JobStatusResponse(BaseModel):
    job_id: str
    stage: JobStage
    progress: int = Field(ge=0, le=100)
    message: str = ""
    channel_name: Optional[str] = None


# --- Report models ---

class FeatureComparison(BaseModel):
    feature: str
    top_mean: float
    bottom_mean: float
    difference: float
    cohens_d: Optional[float] = None
    p_value: Optional[float] = None
    significant: bool = False
    direction: str = ""  # "higher_is_better" or "lower_is_better"


class HookTypeBreakdown(BaseModel):
    hook_type: str
    count: int
    avg_engagement: float
    top_tier_count: int = 0
    bottom_tier_count: int = 0


class Recommendation(BaseModel):
    title: str
    detail: str
    evidence: str
    priority: int = Field(ge=1, le=5)


class PerformerExample(BaseModel):
    video_id: str
    title: str
    engagement_rate: float
    view_count: int
    tier: str  # "top" or "bottom"


class ChannelReport(BaseModel):
    channel_name: str
    channel_id: str
    total_shorts_analyzed: int
    top_tier_count: int
    bottom_tier_count: int
    feature_comparisons: List[FeatureComparison] = []
    hook_type_breakdown: List[HookTypeBreakdown] = []
    recommendations: List[Recommendation] = []
    top_performers: List[PerformerExample] = []
    bottom_performers: List[PerformerExample] = []
    common_traits_top: List[str] = []
    common_traits_bottom: List[str] = []
    raw_features: Optional[Dict[str, Any]] = None
