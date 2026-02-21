"""
Qualitative Feature Audit Framework

Runs 6 tests across all channels in the DB to assess signal quality
of the 8 qualitative features extracted by the LLM qual prompt.

Usage:
    python3 -m scripts.audit.qual_feature_audit

Output:
    ~/vault/shorts-analyzer/qual_audit_v1_results.md
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from web.insight_model import (
    _BINARY_FEATURES,
    _CATEGORICAL_FEATURES,
    _NUMERIC_FEATURES,
    _make_model,
    _loocv_r2,
    prepare_features,
    fit_and_explain,
)

_DB_PATH = _PROJECT_ROOT / "data" / "insights.db"
_OUTPUT_PATH = Path.home() / "vault" / "shorts-analyzer" / "qual_audit_v2_results.md"

# Qual feature definitions for reference (v2)
_QUAL_CATEGORICAL = ["hook_type", "structure_type", "specificity_level"]
_QUAL_BINARY = ["has_numbers", "has_examples", "has_cta", "has_social_proof"]
_ALL_QUAL = _QUAL_CATEGORICAL + _QUAL_BINARY

# Known category sets from qualv2.txt
_CATEGORY_SETS = {
    "hook_type": ["question", "claim", "story", "challenge"],
    "structure_type": ["single_point", "list", "narrative", "comparison"],
    "specificity_level": ["vague", "moderate", "specific"],
}

MIN_CLIPS = 20


# --- Data Loading ---

def load_all_channels() -> Dict[str, pd.DataFrame]:
    """Load all clips grouped by channel_id. Returns {channel_id: DataFrame}."""
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM clips").fetchall()
    finally:
        con.close()

    df = pd.DataFrame([dict(r) for r in rows])
    channels = {}
    for cid, group in df.groupby("channel_id"):
        if len(group) >= MIN_CLIPS:
            channels[cid] = group.copy()
    return channels


def get_channel_name(df: pd.DataFrame) -> str:
    names = df["channel_name"].dropna().unique()
    return names[0] if len(names) > 0 else "Unknown"


# --- Test 1: Per-Feature Distribution ---

def shannon_entropy(counts: List[int]) -> float:
    """Compute Shannon entropy in bits."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def normalized_entropy(counts: List[int], k: int) -> float:
    """Entropy / log2(k). 0 = collapsed, 1 = uniform."""
    if k <= 1:
        return 0.0
    h = shannon_entropy(counts)
    return h / math.log2(k)


def test1_distribution(channels: Dict[str, pd.DataFrame]) -> str:
    """Per-feature distribution analysis."""
    lines = ["## Test 1: Per-Feature Distribution\n"]

    # Categorical features
    for feat in _QUAL_CATEGORICAL:
        cats = _CATEGORY_SETS[feat]
        k = len(cats)
        lines.append(f"### {feat} ({k} categories)\n")
        lines.append(f"| Channel | Dominant | % Dominant | Norm Entropy | Empty Cats | Cats < 3 |")
        lines.append(f"|---|---|---|---|---|---|")

        all_values = []
        for cid, df in channels.items():
            name = get_channel_name(df)
            values = df[feat].dropna().tolist()
            all_values.extend(values)
            counter = Counter(values)
            counts = [counter.get(c, 0) for c in cats]
            # Include any unexpected values
            for v in counter:
                if v not in cats:
                    counts.append(counter[v])

            total = sum(counts)
            if total == 0:
                lines.append(f"| {name} | — | — | — | {k} | {k} |")
                continue

            dominant = max(counter, key=counter.get)
            pct_dominant = counter[dominant] / total * 100
            ne = normalized_entropy(counts, k)
            empty = sum(1 for c in cats if counter.get(c, 0) == 0)
            under3 = sum(1 for c in cats if counter.get(c, 0) < 3)

            lines.append(f"| {name} | {dominant} | {pct_dominant:.0f}% | {ne:.2f} | {empty} | {under3} |")

        # Aggregate
        counter = Counter(all_values)
        counts = [counter.get(c, 0) for c in cats]
        total = sum(counts)
        if total > 0:
            dominant = max(counter, key=counter.get)
            pct_dominant = counter[dominant] / total * 100
            ne = normalized_entropy(counts, k)
            empty = sum(1 for c in cats if counter.get(c, 0) == 0)
            lines.append(f"| **AGGREGATE** | {dominant} | {pct_dominant:.0f}% | {ne:.2f} | {empty} | — |")
        lines.append("")

    # Binary features
    lines.append("### Binary Features\n")
    lines.append(f"| Feature | Channel | n | % True | Effectively Constant? |")
    lines.append(f"|---|---|---|---|---|")

    for feat in _QUAL_BINARY:
        for cid, df in channels.items():
            name = get_channel_name(df)
            values = pd.to_numeric(df[feat], errors="coerce").dropna()
            n = len(values)
            pct_true = values.mean() * 100 if n > 0 else 0
            constant = "YES" if pct_true > 90 or pct_true < 10 else "no"
            lines.append(f"| {feat} | {name} | {n} | {pct_true:.0f}% | {constant} |")

        # Aggregate
        all_vals = pd.concat([pd.to_numeric(df[feat], errors="coerce").dropna() for df in channels.values()])
        pct_true = all_vals.mean() * 100 if len(all_vals) > 0 else 0
        constant = "YES" if pct_true > 90 or pct_true < 10 else "no"
        lines.append(f"| {feat} | **AGGREGATE** | {len(all_vals)} | {pct_true:.0f}% | {constant} |")
    lines.append("")

    return "\n".join(lines)


# --- Test 2: Cross-Channel Consistency ---

def test2_consistency(channels: Dict[str, pd.DataFrame]) -> str:
    """Cross-channel consistency analysis."""
    lines = ["## Test 2: Cross-Channel Consistency\n"]
    n_channels = len(channels)

    # Categorical: how many channels show entropy > 0.5?
    lines.append("### Categorical Features: Channels with Meaningful Variance (Norm Entropy > 0.5)\n")
    lines.append("| Feature | Channels with H > 0.5 | Channels with H > 0.3 | Assessment |")
    lines.append("|---|---|---|---|")

    for feat in _QUAL_CATEGORICAL:
        cats = _CATEGORY_SETS[feat]
        k = len(cats)
        h_above_05 = 0
        h_above_03 = 0
        for cid, df in channels.items():
            values = df[feat].dropna().tolist()
            counter = Counter(values)
            counts = [counter.get(c, 0) for c in cats]
            ne = normalized_entropy(counts, k)
            if ne > 0.5:
                h_above_05 += 1
            if ne > 0.3:
                h_above_03 += 1
        assess = "Universal" if h_above_05 >= n_channels * 0.6 else (
            "Niche-sensitive" if h_above_05 >= 3 else "Broken"
        )
        lines.append(f"| {feat} | {h_above_05}/{n_channels} | {h_above_03}/{n_channels} | {assess} |")
    lines.append("")

    # Binary: cross-channel std of %True
    lines.append("### Binary Features: Cross-Channel Variation in % True\n")
    lines.append("| Feature | Mean % True | Std % True | Min | Max | Assessment |")
    lines.append("|---|---|---|---|---|---|")

    for feat in _QUAL_BINARY:
        pcts = []
        for cid, df in channels.items():
            values = pd.to_numeric(df[feat], errors="coerce").dropna()
            if len(values) > 0:
                pcts.append(values.mean() * 100)
        if pcts:
            mean_pct = np.mean(pcts)
            std_pct = np.std(pcts)
            min_pct = np.min(pcts)
            max_pct = np.max(pcts)
            assess = "No variance" if std_pct < 10 else (
                "Universal" if std_pct < 25 else "Niche-sensitive"
            )
            if mean_pct > 90 or mean_pct < 10:
                assess = "Effectively constant"
            lines.append(f"| {feat} | {mean_pct:.0f}% | {std_pct:.0f}% | {min_pct:.0f}% | {max_pct:.0f}% | {assess} |")
    lines.append("")

    return "\n".join(lines)


# --- Test 3: SHAP Importance + Stability ---

def test3_shap(channels: Dict[str, pd.DataFrame]) -> str:
    """SHAP importance and stability across channels."""
    lines = ["## Test 3: SHAP Importance + Stability Across Channels\n"]

    # Collect per-channel SHAP results
    all_features = _NUMERIC_FEATURES + _BINARY_FEATURES + _CATEGORICAL_FEATURES
    # Track: {feature_name: {channel_name: (importance, stability)}}
    results: Dict[str, Dict[str, Tuple[float, float]]] = {f: {} for f in all_features}
    channel_r2s = {}
    channel_signals = {}

    for cid, df in channels.items():
        name = get_channel_name(df)
        print(f"  SHAP: {name} ({len(df)} clips)...")
        try:
            clips = df.to_dict("records")
            X, y, feature_names = prepare_features(clips)
            if len(y) < MIN_CLIPS:
                print(f"    Skipped (only {len(y)} valid clips)")
                continue
            insights = fit_and_explain(X, y, feature_names, n_bootstraps=50)
            channel_r2s[name] = insights.loocv_r2
            channel_signals[name] = insights.signal_detected
            for fi in insights.features:
                if fi.name in results:
                    results[fi.name][name] = (fi.importance, fi.stability)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    # Summary table: per channel model quality
    lines.append("### Per-Channel Model Quality\n")
    lines.append("| Channel | LOOCV R² | Signal Detected |")
    lines.append("|---|---|---|")
    for name in sorted(channel_r2s.keys()):
        sig = "YES" if channel_signals[name] else "no"
        lines.append(f"| {name} | {channel_r2s[name]:.3f} | {sig} |")
    lines.append("")

    # Feature importance summary
    lines.append("### Feature Importance Summary (across channels)\n")
    lines.append("| Feature | Type | Max Importance | Channels imp>0.01 | Channels stab>0.3 | Channels stab>0.6 | Channels passes (≥0.75) |")
    lines.append("|---|---|---|---|---|---|---|")

    n_ch = len(channel_r2s)
    for feat in all_features:
        ftype = "numeric" if feat in _NUMERIC_FEATURES else ("binary" if feat in _BINARY_FEATURES else "categorical")
        ch_data = results[feat]
        if not ch_data:
            lines.append(f"| {feat} | {ftype} | — | 0/{n_ch} | 0/{n_ch} | 0/{n_ch} | 0/{n_ch} |")
            continue
        importances = [v[0] for v in ch_data.values()]
        stabilities = [v[1] for v in ch_data.values()]
        max_imp = max(importances)
        imp_above_001 = sum(1 for i in importances if i > 0.01)
        stab_above_03 = sum(1 for s in stabilities if s > 0.3)
        stab_above_06 = sum(1 for s in stabilities if s > 0.6)
        passes = sum(1 for s in stabilities if s >= 0.75)
        lines.append(f"| {feat} | {ftype} | {max_imp:.4f} | {imp_above_001}/{n_ch} | {stab_above_03}/{n_ch} | {stab_above_06}/{n_ch} | {passes}/{n_ch} |")
    lines.append("")

    return "\n".join(lines)


# --- Test 4: Correlation with Duration ---

def point_biserial(binary_col: pd.Series, continuous_col: pd.Series) -> Tuple[float, float]:
    """Point-biserial correlation. Returns (r, p)."""
    mask = binary_col.notna() & continuous_col.notna()
    b = binary_col[mask].astype(float)
    c = continuous_col[mask].astype(float)
    if len(b) < 5 or b.std() == 0 or c.std() == 0:
        return (0.0, 1.0)
    r, p = scipy_stats.pointbiserialr(b, c)
    return (float(r), float(p))


def partial_correlation(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Partial correlation of x and y, controlling for z."""
    mask = x.notna() & y.notna() & z.notna()
    x, y, z = x[mask].astype(float), y[mask].astype(float), z[mask].astype(float)
    if len(x) < 10 or x.std() == 0:
        return 0.0
    # Residualize x and y on z
    from numpy.polynomial.polynomial import polyfit
    z_arr = z.values.reshape(-1, 1)
    x_resid = x.values - np.polyfit(z.values, x.values, 1)[0] * z.values - np.polyfit(z.values, x.values, 1)[1]
    y_resid = y.values - np.polyfit(z.values, y.values, 1)[0] * z.values - np.polyfit(z.values, y.values, 1)[1]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return 0.0
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def test4_duration_correlation(channels: Dict[str, pd.DataFrame]) -> str:
    """Per-channel correlation with duration, then aggregate."""
    lines = ["## Test 4: Correlation with Duration (per-channel, then aggregated)\n"]

    # Binary features vs duration
    lines.append("### Binary Features vs Duration\n")
    lines.append("| Feature | Mean r(duration) | Std | Mean partial r(engagement|duration) | Std | Confounded? |")
    lines.append("|---|---|---|---|---|---|")

    for feat in _QUAL_BINARY:
        dur_corrs = []
        partial_corrs = []
        for cid, df in channels.items():
            binary_col = pd.to_numeric(df[feat], errors="coerce")
            duration_col = pd.to_numeric(df["duration_s"], errors="coerce")
            eng_col = pd.to_numeric(df["engagement_rate"], errors="coerce")

            r, _ = point_biserial(binary_col, duration_col)
            dur_corrs.append(r)

            # Partial correlation with log(engagement) controlling for duration
            log_eng = np.log(eng_col.clip(lower=1e-10))
            pc = partial_correlation(binary_col, log_eng, duration_col)
            partial_corrs.append(pc)

        mean_r = np.mean(dur_corrs)
        std_r = np.std(dur_corrs)
        mean_pc = np.mean(partial_corrs)
        std_pc = np.std(partial_corrs)
        confounded = "YES" if abs(mean_r) > 0.5 else ("borderline" if abs(mean_r) > 0.3 else "no")
        lines.append(f"| {feat} | {mean_r:+.3f} | {std_r:.3f} | {mean_pc:+.3f} | {std_pc:.3f} | {confounded} |")
    lines.append("")

    # Categorical features vs duration (eta-squared)
    lines.append("### Categorical Features vs Duration (eta-squared)\n")
    lines.append("| Feature | Mean eta² | Std | Interpretation |")
    lines.append("|---|---|---|---|")

    for feat in _QUAL_CATEGORICAL:
        etas = []
        for cid, df in channels.items():
            duration_col = pd.to_numeric(df["duration_s"], errors="coerce")
            cat_col = df[feat].dropna()
            mask = cat_col.index.intersection(duration_col.dropna().index)
            if len(mask) < 10:
                continue
            groups = [duration_col[cat_col == c].dropna().values for c in cat_col.unique() if len(duration_col[cat_col == c].dropna()) >= 2]
            if len(groups) < 2:
                etas.append(0.0)
                continue
            try:
                f_stat, _ = scipy_stats.f_oneway(*groups)
                # Compute eta-squared from F
                k = len(groups)
                n = sum(len(g) for g in groups)
                eta2 = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k)) if (f_stat * (k - 1) + (n - k)) > 0 else 0
                etas.append(max(0, eta2))
            except Exception:
                etas.append(0.0)

        if etas:
            mean_eta = np.mean(etas)
            std_eta = np.std(etas)
            interp = "strong" if mean_eta > 0.14 else ("moderate" if mean_eta > 0.06 else "weak")
            lines.append(f"| {feat} | {mean_eta:.3f} | {std_eta:.3f} | {interp} |")
    lines.append("")

    return "\n".join(lines)


# --- Test 5: Variance Decomposition ---

def _loocv_r2_for_features(df: pd.DataFrame, feature_list: List[str], cat_features: List[str]) -> float:
    """Fit LOOCV R² using a subset of features."""
    from sklearn.model_selection import LeaveOneOut

    clips = df.to_dict("records")
    full_df = pd.DataFrame(clips)
    full_df = full_df[full_df["engagement_rate"] > 0].copy()
    y = np.log(full_df["engagement_rate"].astype(float))

    X = pd.DataFrame(index=full_df.index)
    for feat in feature_list:
        if feat in cat_features:
            if feat in full_df.columns:
                X[feat] = full_df[feat].astype("category")
            else:
                X[feat] = pd.Categorical(["unknown"] * len(full_df))
        else:
            X[feat] = pd.to_numeric(full_df.get(feat, pd.Series(dtype=float)), errors="coerce")
            if X[feat].isna().any():
                X[feat] = X[feat].fillna(X[feat].median())

    if len(y) < MIN_CLIPS:
        return float("nan")

    cat_in_subset = [f for f in cat_features if f in feature_list]

    # Run LOOCV with only the categorical features actually in our subset
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))
    y_arr = y.values

    for train_idx, test_idx in loo.split(X):
        model = _make_model()
        model.fit(
            X.iloc[train_idx], y_arr[train_idx],
            categorical_feature=cat_in_subset if cat_in_subset else "auto",
        )
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    ss_res = np.sum((y_arr - predictions) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def test5_variance_decomposition(channels: Dict[str, pd.DataFrame]) -> str:
    """R² with numeric-only, qual-only, and combined features."""
    lines = ["## Test 5: Variance Decomposition\n"]
    lines.append("| Channel | n | Numeric R² | Qual R² | Combined R² | Qual Marginal | Qual Standalone |")
    lines.append("|---|---|---|---|---|---|---|")

    numeric_feats = list(_NUMERIC_FEATURES)
    qual_feats = list(_BINARY_FEATURES) + list(_CATEGORICAL_FEATURES)
    all_feats = numeric_feats + qual_feats
    cat_feats = list(_CATEGORICAL_FEATURES)

    numeric_r2s = []
    qual_r2s = []
    combined_r2s = []
    marginals = []

    for cid, df in channels.items():
        name = get_channel_name(df)
        n = len(df)
        print(f"  Variance decomp: {name} ({n} clips)...")

        try:
            r2_num = _loocv_r2_for_features(df, numeric_feats, [])
            r2_qual = _loocv_r2_for_features(df, qual_feats, cat_feats)
            r2_combined = _loocv_r2_for_features(df, all_feats, cat_feats)

            marginal = r2_combined - r2_num
            numeric_r2s.append(r2_num)
            qual_r2s.append(r2_qual)
            combined_r2s.append(r2_combined)
            marginals.append(marginal)

            lines.append(f"| {name} | {n} | {r2_num:.3f} | {r2_qual:.3f} | {r2_combined:.3f} | {marginal:+.3f} | {r2_qual:.3f} |")
        except Exception as e:
            lines.append(f"| {name} | {n} | ERROR | ERROR | ERROR | — | — |")
            print(f"    FAILED: {e}")

    # Aggregate
    if numeric_r2s:
        lines.append(f"| **MEAN** | — | {np.mean(numeric_r2s):.3f} | {np.mean(qual_r2s):.3f} | {np.mean(combined_r2s):.3f} | {np.mean(marginals):+.3f} | {np.mean(qual_r2s):.3f} |")
        lines.append(f"| **STD** | — | {np.std(numeric_r2s):.3f} | {np.std(qual_r2s):.3f} | {np.std(combined_r2s):.3f} | {np.std(marginals):.3f} | {np.std(qual_r2s):.3f} |")
    lines.append("")

    return "\n".join(lines)


# --- Test 6: Structure Type Distribution ---

def test6_structure_distribution(channels: Dict[str, pd.DataFrame]) -> str:
    """Check structure_type distribution per channel."""
    lines = ["## Test 6: structure_type Distribution\n"]

    cats = _CATEGORY_SETS["structure_type"]
    k = len(cats)

    lines.append("| Channel | n | " + " | ".join(cats) + " | Norm Entropy |")
    lines.append("|---|---|" + "|".join(["---"] * k) + "|---|")

    for cid, df in channels.items():
        name = get_channel_name(df)
        values = df["structure_type"].dropna().tolist()
        counter = Counter(values)
        n = len(values)
        counts = [counter.get(c, 0) for c in cats]
        ne = normalized_entropy(counts, k)
        pcts = [f"{counter.get(c, 0)}" for c in cats]
        lines.append(f"| {name} | {n} | " + " | ".join(pcts) + f" | {ne:.2f} |")

    # Aggregate
    all_values = []
    for df in channels.values():
        all_values.extend(df["structure_type"].dropna().tolist())
    counter = Counter(all_values)
    counts = [counter.get(c, 0) for c in cats]
    ne = normalized_entropy(counts, k)
    pcts = [f"{counter.get(c, 0)}" for c in cats]
    lines.append(f"| **AGGREGATE** | {len(all_values)} | " + " | ".join(pcts) + f" | {ne:.2f} |")
    lines.append("")

    return "\n".join(lines)


# --- Executive Summary ---

def executive_summary(
    channels: Dict[str, pd.DataFrame],
    test1_text: str,
    test2_text: str,
) -> str:
    """Generate executive summary based on test results."""
    lines = ["## Executive Summary\n"]
    n_channels = len(channels)
    total_clips = sum(len(df) for df in channels.values())

    lines.append(f"**Data:** {n_channels} channels, {total_clips} total clips\n")
    lines.append("**Channels:**\n")
    for cid, df in channels.items():
        name = get_channel_name(df)
        lines.append(f"- {name}: {len(df)} clips")
    lines.append("")

    lines.append("### Per-Feature Verdict\n")
    lines.append("| Feature | Type | Verdict | Reason |")
    lines.append("|---|---|---|---|")

    # Compute verdicts from data
    for feat in _ALL_QUAL:
        if feat in _QUAL_CATEGORICAL:
            ftype = "categorical"
            cats = _CATEGORY_SETS[feat]
            k = len(cats)
            entropies = []
            for cid, df in channels.items():
                values = df[feat].dropna().tolist()
                counter = Counter(values)
                counts = [counter.get(c, 0) for c in cats]
                entropies.append(normalized_entropy(counts, k))
            mean_h = np.mean(entropies)
            high_h = sum(1 for h in entropies if h > 0.5)

            if high_h == 0:
                verdict = "KILL"
                reason = f"Collapsed everywhere (mean H={mean_h:.2f}, 0/{n_channels} channels >0.5)"
            elif high_h < 3:
                verdict = "REDESIGN"
                reason = f"Variance in only {high_h}/{n_channels} channels (mean H={mean_h:.2f})"
            else:
                verdict = "KEEP/IMPROVE"
                reason = f"Meaningful variance in {high_h}/{n_channels} channels (mean H={mean_h:.2f})"
        else:
            ftype = "binary"
            pcts = []
            for cid, df in channels.items():
                values = pd.to_numeric(df[feat], errors="coerce").dropna()
                if len(values) > 0:
                    pcts.append(values.mean() * 100)
            mean_pct = np.mean(pcts)
            std_pct = np.std(pcts)

            if mean_pct > 90 or mean_pct < 10:
                verdict = "KILL"
                reason = f"Effectively constant ({mean_pct:.0f}% ± {std_pct:.0f}% across channels)"
            elif std_pct < 10:
                verdict = "REDESIGN"
                reason = f"Low cross-channel variance ({mean_pct:.0f}% ± {std_pct:.0f}%)"
            else:
                verdict = "KEEP/IMPROVE"
                reason = f"Good variance ({mean_pct:.0f}% ± {std_pct:.0f}%)"

        lines.append(f"| {feat} | {ftype} | **{verdict}** | {reason} |")
    lines.append("")

    return "\n".join(lines)


# --- Main ---

def main():
    print("Loading data from DB...")
    channels = load_all_channels()
    print(f"Found {len(channels)} channels with ≥{MIN_CLIPS} clips\n")

    for cid, df in channels.items():
        print(f"  {get_channel_name(df)}: {len(df)} clips")
    print()

    print("Running Test 1: Per-Feature Distribution...")
    t1 = test1_distribution(channels)

    print("Running Test 2: Cross-Channel Consistency...")
    t2 = test2_consistency(channels)

    print("Running Test 3: SHAP Importance + Stability...")
    t3 = test3_shap(channels)

    print("Running Test 4: Duration Correlation...")
    t4 = test4_duration_correlation(channels)

    print("Running Test 5: Variance Decomposition...")
    t5 = test5_variance_decomposition(channels)

    print("Running Test 6: Structure Type Distribution...")
    t6 = test6_structure_distribution(channels)

    print("Generating executive summary...")
    summary = executive_summary(channels, t1, t2)

    # Assemble report
    report = f"""# Qualitative Feature Audit Results (v2)

*Generated from {sum(len(df) for df in channels.values())} clips across {len(channels)} channels*

---

{summary}
---

{t1}
---

{t2}
---

{t3}
---

{t4}
---

{t5}
---

{t6}
"""

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
