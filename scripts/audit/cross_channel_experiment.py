"""
Cross-Channel Pooled Model Experiment

Tests whether transcript-derived features predict relative clip performance
across creators. Uses within-channel z-scored engagement as target and
Leave-One-Channel-Out (LOCO) cross-validation.

Usage:
    python3 -m scripts.audit.cross_channel_experiment

Output:
    ~/vault/shorts-analyzer/cross_channel_experiment_results.md
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DB_PATH = _PROJECT_ROOT / "data" / "insights.db"
_OUTPUT_PATH = Path.home() / "vault" / "shorts-analyzer" / "cross_channel_experiment_results.md"

# --- Feature configuration (matches insight_model.py v2) ---

_NUMERIC_FEATURES = [
    "duration_s", "wpm", "hook_wpm", "filler_density",
    "first_person_ratio", "second_person_ratio", "reading_level",
    "lexical_diversity", "avg_sentence_length", "hook_density", "repetition_score",
]

_BINARY_FEATURES = [
    "has_numbers", "has_examples", "has_cta", "has_social_proof",
]

_CATEGORICAL_FEATURES = [
    "hook_type", "structure_type", "specificity_level",
]

_VISUAL_NUMERIC = [
    "v_brightness", "v_contrast", "v_edge_density", "v_colorfulness",
    "v_face_area_frac", "v_text_area_frac", "v_motion_magnitude",
]

_VISUAL_BINARY = [
    "v_face_present", "v_scene_cut",
]

_ALL_FEATURES = _NUMERIC_FEATURES + _BINARY_FEATURES + _CATEGORICAL_FEATURES + _VISUAL_NUMERIC + _VISUAL_BINARY

_DISPLAY_NAMES = {
    "duration_s": "Duration (seconds)",
    "wpm": "Words Per Minute",
    "hook_wpm": "Hook Words Per Minute",
    "filler_density": "Filler Density",
    "first_person_ratio": "First Person Ratio",
    "second_person_ratio": "Second Person Ratio",
    "reading_level": "Reading Level",
    "lexical_diversity": "Lexical Diversity",
    "avg_sentence_length": "Avg Sentence Length",
    "hook_density": "Hook Density",
    "repetition_score": "Repetition Score",
    "has_numbers": "Uses Numbers/Stats",
    "has_examples": "Has Concrete Examples",
    "has_cta": "Has Call to Action",
    "has_social_proof": "Has Social Proof",
    "hook_type": "Hook Type",
    "structure_type": "Structure Type",
    "specificity_level": "Specificity Level",
    "v_brightness": "Brightness (frame 0)",
    "v_contrast": "Contrast (frame 0)",
    "v_edge_density": "Edge Density",
    "v_colorfulness": "Colorfulness",
    "v_face_present": "Face Present",
    "v_face_area_frac": "Face Area Fraction",
    "v_text_area_frac": "Text Area Fraction",
    "v_motion_magnitude": "Motion Magnitude",
    "v_scene_cut": "Scene Cut (0→1s)",
}

MIN_CLIPS = 20


# --- Model ---

def _make_model() -> LGBMRegressor:
    """LightGBM config for n~470 pooled."""
    return LGBMRegressor(
        max_depth=3,
        n_estimators=50,
        num_leaves=8,
        min_child_samples=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        verbose=-1,
    )


# --- Data loading + preparation ---

def load_data() -> pd.DataFrame:
    """Load all clips, filter channels with >= MIN_CLIPS, compute z-scored target."""
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM clips").fetchall()
    finally:
        con.close()

    df = pd.DataFrame([dict(r) for r in rows])

    # Filter channels with enough clips
    channel_counts = df.groupby("channel_id").size()
    valid_channels = channel_counts[channel_counts >= MIN_CLIPS].index
    df = df[df["channel_id"].isin(valid_channels)].copy()

    # Filter zero/null engagement
    df = df[df["engagement_rate"] > 0].copy()

    # Filter rows missing v2 features
    df = df[df["structure_type"].notna()].copy()

    # Filter rows missing visual features
    df = df[df["v_brightness"].notna()].copy()

    # Compute within-channel z-scored log(engagement_rate)
    df["log_eng"] = np.log(df["engagement_rate"].astype(float))
    channel_stats = df.groupby("channel_id")["log_eng"].agg(["mean", "std"])
    df = df.merge(channel_stats, on="channel_id", suffixes=("", "_ch"))
    # Avoid division by zero for channels with constant engagement
    df["z_engagement"] = np.where(
        df["std"] > 0,
        (df["log_eng"] - df["mean"]) / df["std"],
        0.0,
    )

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_list: List[str],
) -> pd.DataFrame:
    """Build feature matrix X from dataframe for a given feature subset."""
    X = pd.DataFrame(index=df.index)

    cat_features = [f for f in feature_list if f in _CATEGORICAL_FEATURES]
    num_features = [f for f in feature_list if f not in _CATEGORICAL_FEATURES]

    for feat in num_features:
        X[feat] = pd.to_numeric(df.get(feat, pd.Series(dtype=float)), errors="coerce")
        if X[feat].isna().any():
            X[feat] = X[feat].fillna(X[feat].median())

    for feat in cat_features:
        if feat in df.columns:
            X[feat] = df[feat].astype("category")
        else:
            X[feat] = pd.Categorical(["unknown"] * len(df))

    return X


# --- LOCO CV ---

def loco_cv(
    df: pd.DataFrame,
    feature_list: List[str],
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Leave-One-Channel-Out cross-validation.

    Returns (overall_r2, {channel_name: r2}, predictions_array).
    """
    X = build_feature_matrix(df, feature_list)
    y = df["z_engagement"].values
    channels = df["channel_id"].values
    channel_names = df["channel_name"].values

    cat_feats = [f for f in feature_list if f in _CATEGORICAL_FEATURES]
    unique_channels = df["channel_id"].unique()

    predictions = np.full(len(y), np.nan)
    per_channel_r2 = {}

    for held_out_cid in unique_channels:
        train_mask = channels != held_out_cid
        test_mask = channels == held_out_cid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = _make_model()
        model.fit(X_train, y_train, categorical_feature=cat_feats if cat_feats else "auto")
        preds = model.predict(X_test)
        predictions[test_mask] = preds

        # Per-channel R²
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ch_name = channel_names[test_mask][0]
        per_channel_r2[ch_name] = r2

    # Overall R²
    valid = ~np.isnan(predictions)
    ss_res = np.sum((y[valid] - predictions[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
    overall_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return overall_r2, per_channel_r2, predictions


# --- Permutation test ---

def permutation_test_loco(
    df: pd.DataFrame,
    feature_list: List[str],
    real_r2: float,
    n_perms: int = 200,
) -> Tuple[float, np.ndarray]:
    """Permutation test respecting channel groups.

    Shuffles z_engagement within each channel (preserving channel structure).
    Returns (permutation_rank, null_r2s).
    """
    rng = np.random.RandomState(42)
    null_r2s = np.empty(n_perms)

    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            print(f"    Permutation {i + 1}/{n_perms}...", flush=True)

        # Shuffle z_engagement within each channel
        df_perm = df.copy()
        for cid in df_perm["channel_id"].unique():
            mask = df_perm["channel_id"] == cid
            vals = df_perm.loc[mask, "z_engagement"].values.copy()
            rng.shuffle(vals)
            df_perm.loc[mask, "z_engagement"] = vals

        null_r2, _, _ = loco_cv(df_perm, feature_list)
        null_r2s[i] = null_r2

    rank = float(np.mean(null_r2s < real_r2))
    return rank, null_r2s


# --- SHAP analysis on full data ---

def shap_analysis(
    df: pd.DataFrame,
    feature_list: List[str],
    n_bootstraps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Fit on full data, compute SHAP importance + bootstrap stability.

    Returns (mean_abs_shap, stability_scores, directions).
    """
    import shap

    X = build_feature_matrix(df, feature_list)
    y = df["z_engagement"].values
    cat_feats = [f for f in feature_list if f in _CATEGORICAL_FEATURES]

    # Full model SHAP
    model = _make_model()
    model.fit(X, y, categorical_feature=cat_feats if cat_feats else "auto")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Directions
    directions = {}
    for i, feat in enumerate(feature_list):
        if feat in _CATEGORICAL_FEATURES:
            directions[feat] = "varies"
        else:
            col_vals = X[feat].values.astype(float)
            shap_col = shap_values[:, i]
            if np.std(col_vals) > 0 and np.std(shap_col) > 0:
                corr = np.corrcoef(col_vals, shap_col)[0, 1]
                directions[feat] = "higher_is_better" if corr > 0 else "lower_is_better"
            else:
                directions[feat] = "neutral"

    # Bootstrap stability
    rng = np.random.RandomState(42)
    top_k = 5
    n_features = len(feature_list)
    feature_top_counts = np.zeros(n_features)

    for b in range(n_bootstraps):
        idx = rng.choice(len(y), size=len(y), replace=True)
        X_boot = X.iloc[idx]
        y_boot = y[idx]
        boot_model = _make_model()
        boot_model.fit(X_boot, y_boot, categorical_feature=cat_feats if cat_feats else "auto")
        boot_explainer = shap.TreeExplainer(boot_model)
        boot_shap = boot_explainer.shap_values(X_boot)
        boot_importance = np.mean(np.abs(boot_shap), axis=0)
        top_indices = np.argsort(boot_importance)[-top_k:]
        feature_top_counts[top_indices] += 1

    stability_scores = feature_top_counts / n_bootstraps

    return mean_abs_shap, stability_scores, directions


# --- Report generation ---

def generate_report(
    df: pd.DataFrame,
    all_r2: float,
    all_per_ch: Dict[str, float],
    perm_rank: float,
    null_r2s: np.ndarray,
    shap_imp: np.ndarray,
    stability: np.ndarray,
    directions: Dict[str, str],
    quant_r2: float,
    quant_per_ch: Dict[str, float],
    qual_r2: float,
    qual_per_ch: Dict[str, float],
    visual_r2: float,
    visual_per_ch: Dict[str, float],
) -> str:
    """Generate markdown report."""
    lines = []

    n_clips = len(df)
    n_channels = df["channel_id"].nunique()

    lines.append("# Cross-Channel Pooled Model Experiment\n")
    lines.append(f"*{n_clips} clips across {n_channels} channels*\n")
    lines.append("**Question:** Do transcript-derived and visual features predict relative clip performance across creators?\n")
    lines.append("**Target:** Within-channel z-scored log(engagement_rate)\n")
    lines.append("**Validation:** Leave-One-Channel-Out (LOCO) cross-validation\n")
    lines.append("---\n")

    # --- Section 1: Overall results ---
    lines.append("## 1. Overall Signal Detection\n")
    signal = perm_rank > 0.95
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| LOCO R² (combined) | {all_r2:.4f} |")
    lines.append(f"| Permutation rank | {perm_rank:.4f} |")
    lines.append(f"| Signal detected (rank > 0.95) | {'**YES**' if signal else 'no'} |")
    lines.append(f"| Null R² mean ± std | {np.mean(null_r2s):.4f} ± {np.std(null_r2s):.4f} |")
    lines.append(f"| Null R² 95th percentile | {np.percentile(null_r2s, 95):.4f} |")
    lines.append("")

    # --- Section 2: Variance decomposition ---
    lines.append("## 2. Variance Decomposition (LOCO R²)\n")
    lines.append("| Feature Set | LOCO R² | vs Combined |")
    lines.append("|---|---|---|")
    lines.append(f"| Quant only (11 numeric) | {quant_r2:.4f} | {quant_r2 - all_r2:+.4f} |")
    lines.append(f"| Qual only (4 binary + 3 categorical) | {qual_r2:.4f} | {qual_r2 - all_r2:+.4f} |")
    n_visual = len(_VISUAL_NUMERIC) + len(_VISUAL_BINARY)
    lines.append(f"| Visual only ({n_visual} features) | {visual_r2:.4f} | {visual_r2 - all_r2:+.4f} |")
    lines.append(f"| Combined (all {len(_ALL_FEATURES)}) | {all_r2:.4f} | — |")
    lines.append(f"| Visual marginal (combined - transcript) | {all_r2 - quant_r2 - qual_r2 + min(quant_r2, qual_r2):+.4f} | — |")
    lines.append("")

    # --- Section 3: Per-channel LOCO R² ---
    lines.append("## 3. Per-Channel LOCO R²\n")
    lines.append("| Channel | n | Combined | Quant Only | Qual Only | Visual Only | Predictable? |")
    lines.append("|---|---|---|---|---|---|---|")

    for ch_name in sorted(all_per_ch.keys()):
        n = len(df[df["channel_name"] == ch_name])
        r2_all = all_per_ch[ch_name]
        r2_q = quant_per_ch.get(ch_name, float("nan"))
        r2_ql = qual_per_ch.get(ch_name, float("nan"))
        r2_v = visual_per_ch.get(ch_name, float("nan"))
        predictable = "YES" if r2_all > 0 else "no"
        lines.append(f"| {ch_name} | {n} | {r2_all:.3f} | {r2_q:.3f} | {r2_ql:.3f} | {r2_v:.3f} | {predictable} |")
    lines.append("")

    # --- Section 4: SHAP importance ---
    lines.append("## 4. SHAP Feature Importance (full-data model)\n")
    lines.append("| Rank | Feature | Display Name | Importance | Stability | Direction | Passes (≥0.75) |")
    lines.append("|---|---|---|---|---|---|---|")

    sorted_idx = np.argsort(shap_imp)[::-1]
    for rank, i in enumerate(sorted_idx, 1):
        feat = _ALL_FEATURES[i]
        display = _DISPLAY_NAMES.get(feat, feat)
        imp = shap_imp[i]
        stab = stability[i]
        dirn = directions.get(feat, "—")
        passes = "YES" if stab >= 0.75 else "no"
        lines.append(f"| {rank} | {feat} | {display} | {imp:.4f} | {stab:.2f} | {dirn} | {passes} |")
    lines.append("")

    # --- Section 5: Feature type breakdown ---
    lines.append("## 5. Importance by Feature Type\n")

    num_imp = sum(shap_imp[i] for i, f in enumerate(_ALL_FEATURES) if f in _NUMERIC_FEATURES)
    bin_imp = sum(shap_imp[i] for i, f in enumerate(_ALL_FEATURES) if f in _BINARY_FEATURES)
    cat_imp = sum(shap_imp[i] for i, f in enumerate(_ALL_FEATURES) if f in _CATEGORICAL_FEATURES)
    vis_imp = sum(shap_imp[i] for i, f in enumerate(_ALL_FEATURES) if f in _VISUAL_NUMERIC + _VISUAL_BINARY)
    total_imp = num_imp + bin_imp + cat_imp + vis_imp

    lines.append("| Type | Count | Sum SHAP | % of Total |")
    lines.append("|---|---|---|---|")
    if total_imp > 0:
        lines.append(f"| Numeric (quant) | {len(_NUMERIC_FEATURES)} | {num_imp:.4f} | {num_imp/total_imp*100:.1f}% |")
        lines.append(f"| Binary (qual) | {len(_BINARY_FEATURES)} | {bin_imp:.4f} | {bin_imp/total_imp*100:.1f}% |")
        lines.append(f"| Categorical (qual) | {len(_CATEGORICAL_FEATURES)} | {cat_imp:.4f} | {cat_imp/total_imp*100:.1f}% |")
        lines.append(f"| Visual | {n_visual} | {vis_imp:.4f} | {vis_imp/total_imp*100:.1f}% |")
    lines.append("")

    # --- Section 6: Interpretation ---
    lines.append("## 6. Interpretation\n")

    if signal:
        lines.append("**Signal detected.** The combined model predicts held-out channel performance ")
        lines.append(f"better than chance (LOCO R² = {all_r2:.4f}, permutation rank = {perm_rank:.4f}).\n")
    else:
        lines.append("**No signal detected.** The combined model does not predict held-out channel ")
        lines.append(f"performance better than chance (LOCO R² = {all_r2:.4f}, permutation rank = {perm_rank:.4f}).\n")

    qual_marginal = all_r2 - quant_r2
    if qual_marginal > 0.01:
        lines.append(f"Qual features add marginal value ({qual_marginal:+.4f} R²) beyond quant features alone.\n")
    elif qual_marginal < -0.01:
        lines.append(f"Qual features hurt when added to quant ({qual_marginal:+.4f} R²) — likely overfitting.\n")
    else:
        lines.append(f"Qual features are neutral ({qual_marginal:+.4f} R²) — neither help nor hurt.\n")

    transcript_r2 = max(quant_r2, qual_r2)
    visual_marginal = all_r2 - transcript_r2
    if visual_marginal > 0.01:
        lines.append(f"Visual features add marginal value ({visual_marginal:+.4f} R²) beyond transcript features.\n")
    elif visual_marginal < -0.01:
        lines.append(f"Visual features hurt when added ({visual_marginal:+.4f} R²) — likely overfitting.\n")
    else:
        lines.append(f"Visual features are neutral ({visual_marginal:+.4f} R²) — neither help nor hurt.\n")

    predictable_channels = [ch for ch, r2 in all_per_ch.items() if r2 > 0]
    if predictable_channels:
        lines.append(f"**Predictable channels ({len(predictable_channels)}/{n_channels}):** ")
        lines.append(", ".join(sorted(predictable_channels)))
        lines.append("\n")

    stable_features = [(i, _ALL_FEATURES[i]) for i in range(len(_ALL_FEATURES)) if stability[i] >= 0.75]
    if stable_features:
        lines.append("**Stable features (bootstrap ≥ 0.75):**\n")
        for i, feat in stable_features:
            lines.append(f"- {_DISPLAY_NAMES.get(feat, feat)} (importance={shap_imp[i]:.4f}, stability={stability[i]:.2f}, {directions.get(feat, '—')})")
        lines.append("")
    else:
        lines.append("**No features passed the 0.75 stability threshold.**\n")

    return "\n".join(lines)


# --- Main ---

def main():
    print("Loading data...", flush=True)
    df = load_data()
    n_clips = len(df)
    n_channels = df["channel_id"].nunique()
    print(f"  {n_clips} clips across {n_channels} channels\n", flush=True)

    for cid in df["channel_id"].unique():
        ch_df = df[df["channel_id"] == cid]
        print(f"  {ch_df['channel_name'].iloc[0]}: {len(ch_df)} clips", flush=True)
    print(flush=True)

    # 1. Combined LOCO
    print("Running LOCO CV (combined, 18 features)...", flush=True)
    all_r2, all_per_ch, _ = loco_cv(df, _ALL_FEATURES)
    print(f"  LOCO R² = {all_r2:.4f}\n", flush=True)

    # 2. Quant-only LOCO
    print("Running LOCO CV (quant only, 11 features)...", flush=True)
    quant_r2, quant_per_ch, _ = loco_cv(df, _NUMERIC_FEATURES)
    print(f"  LOCO R² = {quant_r2:.4f}\n", flush=True)

    # 3. Qual-only LOCO
    qual_features = _BINARY_FEATURES + _CATEGORICAL_FEATURES
    print("Running LOCO CV (qual only, 7 features)...", flush=True)
    qual_r2, qual_per_ch, _ = loco_cv(df, qual_features)
    print(f"  LOCO R² = {qual_r2:.4f}\n", flush=True)

    # 3b. Visual-only LOCO
    visual_features = _VISUAL_NUMERIC + _VISUAL_BINARY
    print(f"Running LOCO CV (visual only, {len(visual_features)} features)...", flush=True)
    visual_r2, visual_per_ch, _ = loco_cv(df, visual_features)
    print(f"  LOCO R² = {visual_r2:.4f}\n", flush=True)

    # 4. Permutation test on combined
    print("Running permutation test (200 permutations)...", flush=True)
    perm_rank, null_r2s = permutation_test_loco(df, _ALL_FEATURES, all_r2, n_perms=200)
    print(f"  Permutation rank = {perm_rank:.4f}\n", flush=True)

    # 5. SHAP on full data
    print("Running SHAP analysis (100 bootstraps)...", flush=True)
    shap_imp, stability, directions = shap_analysis(df, _ALL_FEATURES, n_bootstraps=100)
    print("  Done.\n", flush=True)

    # 6. Generate report
    print("Generating report...", flush=True)
    report = generate_report(
        df, all_r2, all_per_ch, perm_rank, null_r2s,
        shap_imp, stability, directions,
        quant_r2, quant_per_ch, qual_r2, qual_per_ch,
        visual_r2, visual_per_ch,
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved to: {_OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
