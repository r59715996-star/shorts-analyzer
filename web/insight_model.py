"""SHAP-based LightGBM insight engine for per-channel analysis."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneOut

from web.models import Recommendation

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "insights.db"

# --- Feature configuration ---

# Numeric features (11)
_NUMERIC_FEATURES = [
    "duration_s", "wpm", "hook_wpm", "filler_density",
    "first_person_ratio", "second_person_ratio", "reading_level",
    "lexical_diversity", "avg_sentence_length", "hook_density", "repetition_score",
]

# Binary qualitative features (4)
_BINARY_FEATURES = [
    "has_numbers", "has_examples", "has_cta", "has_social_proof",
]

# Categorical features (3, LightGBM native categoricals)
_CATEGORICAL_FEATURES = [
    "hook_type", "structure_type", "specificity_level",
]

_ALL_MODEL_FEATURES = _NUMERIC_FEATURES + _BINARY_FEATURES + _CATEGORICAL_FEATURES

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
}


# --- Dataclasses ---

@dataclass
class FeatureInsight:
    name: str
    display_name: str
    importance: float
    direction: str  # "higher_is_better" or "lower_is_better"
    stability: float  # 0-1
    passes_threshold: bool


@dataclass
class InteractionInsight:
    feature_a: str
    feature_b: str
    strength: float
    threshold: float
    description: str


@dataclass
class ModelInsights:
    loocv_r2: float
    permutation_rank: float
    signal_detected: bool
    features: List[FeatureInsight]
    interactions: List[InteractionInsight]
    n_samples: int
    stability_threshold: float = 0.75


# --- LightGBM config (fixed for n~50) ---

def _make_model() -> LGBMRegressor:
    return LGBMRegressor(
        max_depth=2,
        n_estimators=30,
        num_leaves=4,
        min_child_samples=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        verbose=-1,
    )


# --- Data loading ---

def load_clips_from_db(channel_id: str) -> List[Dict[str, Any]]:
    """Load clip rows from SQLite for a given channel."""
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM clips WHERE channel_id = ?", (channel_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


# --- Feature preparation ---

def prepare_features(
    clips: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build feature matrix X and target y from clip dicts (DB row format).

    Target: log(engagement_rate).
    Returns (X, y, feature_names).
    """
    df = pd.DataFrame(clips)

    # Target: log-transform engagement_rate (filter zeros)
    df = df[df["engagement_rate"] > 0].copy()
    y = np.log(df["engagement_rate"].astype(float))

    # Build feature frame
    X = pd.DataFrame(index=df.index)

    for feat in _NUMERIC_FEATURES:
        X[feat] = pd.to_numeric(df.get(feat, pd.Series(dtype=float)), errors="coerce")

    for feat in _BINARY_FEATURES:
        X[feat] = pd.to_numeric(df.get(feat, pd.Series(dtype=float)), errors="coerce")

    for feat in _CATEGORICAL_FEATURES:
        if feat in df.columns:
            X[feat] = df[feat].astype("category")
        else:
            X[feat] = pd.Categorical(["unknown"] * len(df))

    # Fill numeric NaNs with column medians
    numeric_cols = _NUMERIC_FEATURES + _BINARY_FEATURES
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    feature_names = list(X.columns)
    return X, y, feature_names


# --- LOOCV R² ---

def _loocv_r2(X: pd.DataFrame, y: pd.Series) -> float:
    """Compute Leave-One-Out Cross-Validated R²."""
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))
    y_arr = y.values

    for train_idx, test_idx in loo.split(X):
        model = _make_model()
        model.fit(
            X.iloc[train_idx], y_arr[train_idx],
            categorical_feature=_CATEGORICAL_FEATURES,
        )
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    ss_res = np.sum((y_arr - predictions) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# --- Core engine ---

def fit_and_explain(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_bootstraps: int = 80,
) -> ModelInsights:
    """Fit LightGBM, compute SHAP values, run permutation + bootstrap analysis."""
    import shap

    n_samples = len(y)
    y_arr = y.values

    # 1. LOOCV R²
    real_r2 = _loocv_r2(X, y)

    # 2. Permutation test for overall signal (200 permutations)
    rng = np.random.RandomState(42)
    null_r2s = np.empty(200)
    for i in range(200):
        y_perm = rng.permutation(y_arr)
        null_r2s[i] = _loocv_r2(X, pd.Series(y_perm, index=y.index))

    permutation_rank = float(np.mean(null_r2s < real_r2))
    signal_detected = permutation_rank > 0.95

    # 3. Fit on full data + SHAP
    model = _make_model()
    model.fit(X, y_arr, categorical_feature=_CATEGORICAL_FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    # Mean |SHAP| importance per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Direction: correlation of SHAP value with feature value (numeric only)
    directions = {}
    for i, feat in enumerate(feature_names):
        if feat in _CATEGORICAL_FEATURES:
            directions[feat] = "varies"
        else:
            col_vals = X[feat].values.astype(float)
            shap_col = shap_values[:, i]
            if np.std(col_vals) > 0 and np.std(shap_col) > 0:
                corr = np.corrcoef(col_vals, shap_col)[0, 1]
                directions[feat] = "higher_is_better" if corr > 0 else "lower_is_better"
            else:
                directions[feat] = "higher_is_better"

    # 4. SHAP interaction filtering with permutation null (100 permutations)
    interactions: List[InteractionInsight] = []
    try:
        shap_interaction = explainer.shap_interaction_values(X)
        # shap_interaction shape: (n_samples, n_features, n_features)
        n_feat = len(feature_names)

        # Real interaction strengths (off-diagonal)
        real_strengths = {}
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                strength = float(np.mean(np.abs(shap_interaction[:, i, j])))
                real_strengths[(i, j)] = strength

        # Permutation null for max interaction strength
        null_max_strengths = np.empty(100)
        for perm_idx in range(100):
            y_perm = rng.permutation(y_arr)
            perm_model = _make_model()
            perm_model.fit(X, y_perm, categorical_feature=_CATEGORICAL_FEATURES)
            perm_explainer = shap.TreeExplainer(perm_model)
            perm_interact = perm_explainer.shap_interaction_values(X)
            max_str = 0.0
            for i in range(n_feat):
                for j in range(i + 1, n_feat):
                    s = float(np.mean(np.abs(perm_interact[:, i, j])))
                    if s > max_str:
                        max_str = s
            null_max_strengths[perm_idx] = max_str

        interaction_threshold = float(np.percentile(null_max_strengths, 95))

        # Surface interactions exceeding threshold
        for (i, j), strength in real_strengths.items():
            if strength > interaction_threshold:
                fa, fb = feature_names[i], feature_names[j]
                desc = _describe_interaction(fa, fb, X, shap_interaction[:, i, j])
                interactions.append(InteractionInsight(
                    feature_a=fa,
                    feature_b=fb,
                    strength=round(strength, 6),
                    threshold=round(interaction_threshold, 6),
                    description=desc,
                ))
    except Exception:
        # Some SHAP versions may not support interaction values for all configs
        pass

    # 5. Bootstrap stability (80 resamples)
    top_k = 5
    feature_top_counts = np.zeros(len(feature_names))
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y_arr[idx]
        boot_model = _make_model()
        boot_model.fit(X_boot, y_boot, categorical_feature=_CATEGORICAL_FEATURES)
        boot_explainer = shap.TreeExplainer(boot_model)
        boot_shap = boot_explainer.shap_values(X_boot)
        boot_importance = np.mean(np.abs(boot_shap), axis=0)
        top_indices = np.argsort(boot_importance)[-top_k:]
        feature_top_counts[top_indices] += 1

    stability_scores = feature_top_counts / n_bootstraps
    stability_threshold = 0.75

    # Build FeatureInsight list
    features = []
    for i, feat in enumerate(feature_names):
        features.append(FeatureInsight(
            name=feat,
            display_name=_DISPLAY_NAMES.get(feat, feat),
            importance=round(float(mean_abs_shap[i]), 6),
            direction=directions[feat],
            stability=round(float(stability_scores[i]), 3),
            passes_threshold=bool(stability_scores[i] >= stability_threshold),
        ))

    # Sort by importance descending
    features.sort(key=lambda f: f.importance, reverse=True)

    return ModelInsights(
        loocv_r2=round(real_r2, 4),
        permutation_rank=round(permutation_rank, 4),
        signal_detected=signal_detected,
        features=features,
        interactions=interactions,
        n_samples=n_samples,
        stability_threshold=stability_threshold,
    )


def _describe_interaction(
    feat_a: str, feat_b: str, X: pd.DataFrame, interaction_values: np.ndarray
) -> str:
    """Generate a natural-language description of an interaction effect."""
    name_a = _DISPLAY_NAMES.get(feat_a, feat_a)
    name_b = _DISPLAY_NAMES.get(feat_b, feat_b)

    # For binary features, describe the conditional effect
    if feat_b in _BINARY_FEATURES:
        mask = X[feat_b].values == 1
        if mask.any() and (~mask).any():
            mean_when_true = float(np.mean(np.abs(interaction_values[mask])))
            mean_when_false = float(np.mean(np.abs(interaction_values[~mask])))
            if mean_when_true > mean_when_false:
                return f"{name_a} matters more when {name_b} is true"
            else:
                return f"{name_a} matters more when {name_b} is false"

    if feat_a in _BINARY_FEATURES:
        mask = X[feat_a].values == 1
        if mask.any() and (~mask).any():
            mean_when_true = float(np.mean(np.abs(interaction_values[mask])))
            mean_when_false = float(np.mean(np.abs(interaction_values[~mask])))
            if mean_when_true > mean_when_false:
                return f"{name_b} matters more when {name_a} is true"
            else:
                return f"{name_b} matters more when {name_a} is false"

    return f"{name_a} and {name_b} have a combined effect on performance"


# --- Recommendations ---

def generate_shap_recommendations(insights: ModelInsights) -> List[Recommendation]:
    """Generate 3-5 recommendations from SHAP insights.

    Only runs if signal_detected is True.
    Prioritizes interaction-based recommendations, then stable main effects.
    """
    if not insights.signal_detected:
        return []

    recs: List[Recommendation] = []
    priority = 1

    # Interaction-based recommendations first
    for inter in insights.interactions:
        if priority > 5:
            break
        name_a = _DISPLAY_NAMES.get(inter.feature_a, inter.feature_a)
        name_b = _DISPLAY_NAMES.get(inter.feature_b, inter.feature_b)
        recs.append(Recommendation(
            title=f"Combine {name_a} with {name_b}",
            detail=inter.description,
            evidence=(
                f"SHAP interaction strength {inter.strength:.4f} "
                f"(threshold: {inter.threshold:.4f})"
            ),
            priority=priority,
        ))
        priority += 1

    # Stable main-effect features
    stable_features = [
        f for f in insights.features
        if f.passes_threshold and f.name not in _CATEGORICAL_FEATURES
    ]
    for feat in stable_features:
        if priority > 5:
            break
        direction_text = "more" if feat.direction == "higher_is_better" else "less"
        recs.append(Recommendation(
            title=f"Optimize {feat.display_name}",
            detail=(
                f"Aim for {direction_text} {feat.display_name.lower()}. "
                f"This feature consistently influences engagement."
            ),
            evidence=(
                f"SHAP importance: {feat.importance:.4f}, "
                f"bootstrap stability: {feat.stability:.0%}"
            ),
            priority=priority,
        ))
        priority += 1

    return recs[:5]
