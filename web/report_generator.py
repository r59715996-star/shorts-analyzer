"""Statistical comparison engine — top vs bottom 25%, Cohen's d, recommendations."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from web.models import (
    ChannelReport,
    FeatureComparison,
    HookTypeBreakdown,
    PerformerExample,
    Recommendation,
)


# Features to compare (computed quant features — no raw counts)
QUANT_FEATURES = [
    "duration_s",
    "wpm",
    "hook_wpm",
    "filler_density",
    "first_person_ratio",
    "second_person_ratio",
    "reading_level",
    "lexical_diversity",
    "avg_sentence_length",
    "hook_density",
    "repetition_score",
]

# Binary qualitative features
BINARY_QUAL_FEATURES = [
    "has_numbers",
    "has_examples",
    "has_cta",
    "has_social_proof",
]

# Feature display names
FEATURE_LABELS = {
    "duration_s": "Duration (seconds)",
    "wpm": "Words Per Minute",
    "hook_wpm": "Hook Words Per Minute",
    "filler_density": "Filler Density",
    "first_person_ratio": "First Person Ratio (I/me/we)",
    "second_person_ratio": "Second Person Ratio (you/your)",
    "reading_level": "Reading Level (Flesch-Kincaid)",
    "lexical_diversity": "Lexical Diversity",
    "avg_sentence_length": "Avg Sentence Length",
    "hook_density": "Hook Density",
    "repetition_score": "Repetition Score",
    "has_numbers": "Uses Numbers/Stats",
    "has_examples": "Has Concrete Examples",
    "has_cta": "Has Call to Action",
    "has_social_proof": "Has Social Proof",
}


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _cohens_d(group1: List[float], group2: List[float]) -> float | None:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return None
    m1, m2 = _mean(group1), _mean(group2)
    s1, s2 = _std(group1), _std(group2)
    n1, n2 = len(group1), len(group2)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def _welch_t_test(group1: List[float], group2: List[float]) -> float | None:
    """Compute approximate p-value using Welch's t-test."""
    if len(group1) < 2 or len(group2) < 2:
        return None
    m1, m2 = _mean(group1), _mean(group2)
    s1, s2 = _std(group1), _std(group2)
    n1, n2 = len(group1), len(group2)

    se1 = s1**2 / n1
    se2 = s2**2 / n2
    se_diff = math.sqrt(se1 + se2)

    if se_diff == 0:
        return 1.0

    t_stat = abs(m1 - m2) / se_diff

    # Approximate degrees of freedom (Welch-Satterthwaite)
    if se1 + se2 == 0:
        return 1.0
    df = (se1 + se2) ** 2 / (
        se1**2 / (n1 - 1) + se2**2 / (n2 - 1)
    ) if (se1**2 / (n1 - 1) + se2**2 / (n2 - 1)) > 0 else 1

    # Approximate p-value using normal distribution for simplicity
    # (good enough for df > 30, reasonable approximation otherwise)
    p_value = 2 * (1 - _normal_cdf(t_stat))
    return max(p_value, 1e-10)


def _normal_cdf(x: float) -> float:
    """Approximate CDF of the standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _extract_feature_values(
    clips: List[Dict[str, Any]], feature: str
) -> List[float]:
    """Extract numeric values for a feature across clips."""
    values = []
    for clip in clips:
        quant = clip.get("quant_features", {})
        qual = clip.get("qual_features", {})
        val = quant.get(feature, qual.get(feature))
        if val is None:
            continue
        if isinstance(val, bool):
            values.append(1.0 if val else 0.0)
        else:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
    return values


def split_tiers(
    clips: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split clips into top 25% and bottom 25% by engagement_rate."""
    sorted_clips = sorted(
        clips, key=lambda c: c.get("engagement_rate", 0), reverse=True
    )
    n = len(sorted_clips)
    cutoff = max(1, n // 4)
    return sorted_clips[:cutoff], sorted_clips[-cutoff:]


def compare_features(
    top_clips: List[Dict[str, Any]],
    bottom_clips: List[Dict[str, Any]],
) -> List[FeatureComparison]:
    """Compare all features between top and bottom tiers."""
    comparisons = []
    all_features = QUANT_FEATURES + BINARY_QUAL_FEATURES

    for feature in all_features:
        top_vals = _extract_feature_values(top_clips, feature)
        bot_vals = _extract_feature_values(bottom_clips, feature)

        if not top_vals or not bot_vals:
            continue

        top_mean = _mean(top_vals)
        bot_mean = _mean(bot_vals)
        diff = top_mean - bot_mean
        d = _cohens_d(top_vals, bot_vals)
        p = _welch_t_test(top_vals, bot_vals)

        significant = p is not None and p < 0.05
        direction = "higher_is_better" if diff > 0 else "lower_is_better"

        comparisons.append(
            FeatureComparison(
                feature=FEATURE_LABELS.get(feature, feature),
                top_mean=round(top_mean, 4),
                bottom_mean=round(bot_mean, 4),
                difference=round(diff, 4),
                cohens_d=round(d, 3) if d is not None else None,
                p_value=round(p, 4) if p is not None else None,
                significant=significant,
                direction=direction,
            )
        )

    # Sort by absolute effect size
    comparisons.sort(
        key=lambda c: abs(c.cohens_d) if c.cohens_d is not None else 0,
        reverse=True,
    )
    return comparisons


def compute_hook_breakdown(
    clips: List[Dict[str, Any]],
    top_clips: List[Dict[str, Any]],
    bottom_clips: List[Dict[str, Any]],
) -> List[HookTypeBreakdown]:
    """Compute performance breakdown by hook type."""
    hook_groups: Dict[str, List[float]] = {}
    for clip in clips:
        hook_type = (clip.get("qual_features") or {}).get("hook_type", "unknown")
        eng = clip.get("engagement_rate", 0)
        hook_groups.setdefault(hook_type, []).append(eng)

    top_hooks = Counter(
        (c.get("qual_features") or {}).get("hook_type", "unknown")
        for c in top_clips
    )
    bot_hooks = Counter(
        (c.get("qual_features") or {}).get("hook_type", "unknown")
        for c in bottom_clips
    )

    breakdowns = []
    for hook_type, engagements in hook_groups.items():
        breakdowns.append(
            HookTypeBreakdown(
                hook_type=hook_type,
                count=len(engagements),
                avg_engagement=round(_mean(engagements), 6),
                top_tier_count=top_hooks.get(hook_type, 0),
                bottom_tier_count=bot_hooks.get(hook_type, 0),
            )
        )

    breakdowns.sort(key=lambda b: b.avg_engagement, reverse=True)
    return breakdowns


def generate_recommendations(
    comparisons: List[FeatureComparison],
    hook_breakdown: List[HookTypeBreakdown],
    top_clips: List[Dict[str, Any]],
    bottom_clips: List[Dict[str, Any]],
) -> List[Recommendation]:
    """Generate 3-5 actionable recommendations from the analysis."""
    recs: List[Recommendation] = []
    priority = 1

    # Recommendation from top significant features
    sig_features = [c for c in comparisons if c.significant and c.cohens_d is not None and abs(c.cohens_d) > 0.3]

    for comp in sig_features[:3]:
        direction_text = "more" if comp.direction == "higher_is_better" else "less"
        recs.append(
            Recommendation(
                title=f"Optimize {comp.feature}",
                detail=(
                    f"Your top-performing Shorts average {comp.top_mean:.2f} for {comp.feature}, "
                    f"while your worst average {comp.bottom_mean:.2f}. "
                    f"Aim for {direction_text} of this in your content."
                ),
                evidence=(
                    f"Cohen's d = {comp.cohens_d:.2f}, p = {comp.p_value:.4f} "
                    f"({abs(comp.cohens_d):.1f}x standard deviations apart)"
                ),
                priority=priority,
            )
        )
        priority += 1

    # Hook type recommendation
    if hook_breakdown:
        best_hook = hook_breakdown[0]
        worst_hook = hook_breakdown[-1] if len(hook_breakdown) > 1 else None
        detail = (
            f"Your best-performing hook type is '{best_hook.hook_type}' "
            f"(avg engagement: {best_hook.avg_engagement:.4f}, "
            f"used {best_hook.count} times)."
        )
        if worst_hook and worst_hook.hook_type != best_hook.hook_type:
            detail += (
                f" Consider reducing '{worst_hook.hook_type}' hooks "
                f"(avg engagement: {worst_hook.avg_engagement:.4f})."
            )
        recs.append(
            Recommendation(
                title=f"Lead with '{best_hook.hook_type}' hooks",
                detail=detail,
                evidence=f"{best_hook.top_tier_count} of your top clips use this hook type vs {best_hook.bottom_tier_count} in bottom tier.",
                priority=priority,
            )
        )
        priority += 1

    # Ensure at least 3 recommendations
    if len(recs) < 3:
        # Add a pacing recommendation based on WPM
        wpm_comp = next((c for c in comparisons if "Words Per Minute" in c.feature), None)
        if wpm_comp:
            recs.append(
                Recommendation(
                    title="Adjust your speaking pace",
                    detail=(
                        f"Top Shorts average {wpm_comp.top_mean:.0f} WPM vs "
                        f"{wpm_comp.bottom_mean:.0f} WPM in bottom performers."
                    ),
                    evidence=f"Difference of {abs(wpm_comp.difference):.0f} WPM between tiers.",
                    priority=priority,
                )
            )
            priority += 1

    return recs[:5]


def extract_common_traits(clips: List[Dict[str, Any]], tier: str) -> List[str]:
    """Identify dominant patterns in a tier of clips."""
    traits = []

    # Most common hook type
    hooks = [
        (c.get("qual_features") or {}).get("hook_type", "unknown")
        for c in clips
    ]
    if hooks:
        most_common = Counter(hooks).most_common(1)[0]
        pct = most_common[1] / len(hooks) * 100
        traits.append(f"{pct:.0f}% use '{most_common[0]}' hooks")

    # Average WPM
    wpms = [
        (c.get("quant_features") or {}).get("wpm", 0)
        for c in clips
        if (c.get("quant_features") or {}).get("wpm")
    ]
    if wpms:
        traits.append(f"Average speaking pace: {_mean(wpms):.0f} WPM")

    # Most common structure type
    structures = [
        (c.get("qual_features") or {}).get("structure_type", "unknown")
        for c in clips
    ]
    if structures:
        most_common_struct = Counter(structures).most_common(1)[0]
        pct = most_common_struct[1] / len(structures) * 100
        traits.append(f"{pct:.0f}% use '{most_common_struct[0]}' structure")

    # Has CTA
    ctas = [
        (c.get("qual_features") or {}).get("has_cta", False)
        for c in clips
    ]
    cta_pct = sum(1 for c in ctas if c) / len(ctas) * 100 if ctas else 0
    traits.append(f"{cta_pct:.0f}% include a call to action")

    # Specificity level
    specs = [
        (c.get("qual_features") or {}).get("specificity_level", "unknown")
        for c in clips
    ]
    if specs:
        most_common_spec = Counter(specs).most_common(1)[0]
        pct = most_common_spec[1] / len(specs) * 100
        traits.append(f"{pct:.0f}% have '{most_common_spec[0]}' specificity")

    return traits


def generate_report(
    clips: List[Dict[str, Any]],
    channel_name: str,
    channel_id: str,
) -> ChannelReport:
    """Generate a full channel report from processed clips."""
    top_clips, bottom_clips = split_tiers(clips)

    comparisons = compare_features(top_clips, bottom_clips)
    hook_breakdown = compute_hook_breakdown(clips, top_clips, bottom_clips)
    recommendations = generate_recommendations(
        comparisons, hook_breakdown, top_clips, bottom_clips
    )

    top_performers = [
        PerformerExample(
            video_id=c.get("video_id", ""),
            title=c.get("title", ""),
            engagement_rate=round(c.get("engagement_rate", 0), 6),
            view_count=c.get("view_count", 0),
            tier="top",
        )
        for c in top_clips[:5]
    ]
    bottom_performers = [
        PerformerExample(
            video_id=c.get("video_id", ""),
            title=c.get("title", ""),
            engagement_rate=round(c.get("engagement_rate", 0), 6),
            view_count=c.get("view_count", 0),
            tier="bottom",
        )
        for c in bottom_clips[:5]
    ]

    return ChannelReport(
        channel_name=channel_name,
        channel_id=channel_id,
        total_shorts_analyzed=len(clips),
        top_tier_count=len(top_clips),
        bottom_tier_count=len(bottom_clips),
        feature_comparisons=comparisons,
        hook_type_breakdown=hook_breakdown,
        recommendations=recommendations,
        top_performers=top_performers,
        bottom_performers=bottom_performers,
        common_traits_top=extract_common_traits(top_clips, "top"),
        common_traits_bottom=extract_common_traits(bottom_clips, "bottom"),
    )
