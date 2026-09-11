"""
src/predict.py
==============
BioSentinel 2.0 — Inference Interface for the Regime Classifier.

Loads the trained LightGBM model from ``models/`` and exposes a clean
callable API for single-batch and row-level prediction.

Public API
----------
    load_model()                         -> (model, feature_cols, classes)
    predict_batch(df, model, ...)        -> dict with class + probabilities
    predict_row(row_series, model, ...)  -> dict with class + probabilities

No Streamlit / UI / API / database imports.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Ensure src/ is importable when used as a library
import sys as _sys
import os as _os
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from src.features import build_features, compute_healthy_baseline, get_feature_cols

__all__ = [
    "load_model",
    "predict_batch",
    "predict_row",
    "CLASSES",
]

CLASSES = ["Healthy", "kLa_Limitation", "Substrate_Overfeeding", "Contamination"]

# ---------------------------------------------------------------------------
# Default model artefact paths
# ---------------------------------------------------------------------------

_MODELS_DIR        = _ROOT / "models"
_MODEL_PATH        = _MODELS_DIR / "regime_classifier.pkl"
_FEATURE_COLS_PATH = _MODELS_DIR / "feature_cols.json"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


class LightGBMBoosterWrapper:
    """Lightweight wrapper around native lightgbm.Booster for sklearn-like API."""
    def __init__(self, booster: Any):
        self.booster = booster
        self.classes_ = CLASSES

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.booster.feature_importance()


def load_model(
    model_path:  Union[str, Path] = _MODEL_PATH,
    cols_path:   Union[str, Path] = _FEATURE_COLS_PATH,
) -> Tuple[Any, List[str], List[str]]:
    """Load the trained regime classifier from disk.

    Parameters
    ----------
    model_path : path-like
        Path to the joblib-serialised or text-serialised LightGBM model.
    cols_path : path-like
        Path to the JSON file listing expected feature column names.

    Returns
    -------
    (model, feature_cols, classes) : tuple
        model        — fitted LGBMClassifier or LightGBMBoosterWrapper
        feature_cols — ordered list of feature column names
        classes      — ordered list of class label strings
    """
    model_path = Path(model_path)
    cols_path  = Path(cols_path)
    txt_path   = model_path.with_suffix(".txt")

    if not cols_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {cols_path}\n"
            "Run `python src/train_model.py` first."
        )

    # Prefer lightweight native Booster (.txt) if present to avoid scikit-learn bundle bloat
    if txt_path.exists():
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(txt_path))
        model = LightGBMBoosterWrapper(booster)
    elif model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception:
            raise FileNotFoundError(f"Failed to load model from {model_path}")
    else:
        raise FileNotFoundError(
            f"Model file not found at {txt_path} or {model_path}\n"
            "Run `python src/train_model.py` first to train and save the model."
        )

    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols, CLASSES


# ---------------------------------------------------------------------------
# Internal feature alignment
# ---------------------------------------------------------------------------


def _align_features(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Align a feature DataFrame to the model's expected column order.

    Missing columns are filled with 0.  Extra columns are dropped.
    """
    col_data = {}
    for col in feature_cols:
        if col in feat_df.columns:
            col_data[col] = feat_df[col].values
        else:
            warnings.warn(
                f"Feature column '{col}' missing from input; filling with 0.",
                RuntimeWarning,
                stacklevel=3,
            )
            col_data[col] = np.zeros(len(feat_df), dtype=np.float32)
    aligned = pd.DataFrame(col_data, index=feat_df.index)
    return aligned.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch-level prediction
# ---------------------------------------------------------------------------


def predict_batch(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    classes: List[str] = CLASSES,
    *,
    do_mean: float = 0.006,
    do_std:  float = 0.001,
    aggregate: str = "majority",
) -> Dict[str, Any]:
    """Predict the regime for an entire batch DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``src.simulator.simulate_batch()`` (or real sensor data
        with the same stable-schema columns).
    model : fitted LGBMClassifier
        From ``load_model()``.
    feature_cols : list of str
        From ``load_model()``.
    classes : list of str
        Class label strings in model order.
    do_mean, do_std : float
        Healthy-baseline DO statistics for CUSUM feature.  If omitted, the
        module defaults are used (nominal healthy values).
    aggregate : {"majority", "mean_proba"}
        How to combine per-row predictions into a single batch-level decision.
        ``"majority"`` — majority vote on per-row class predictions.
        ``"mean_proba"`` — average per-row probabilities then argmax.

    Returns
    -------
    dict with keys:
        ``predicted_class``  — string class label
        ``class_index``      — integer index
        ``probabilities``    — dict of {class_label: mean_probability}
        ``row_predictions``  — list of per-row predicted class strings
        ``n_rows``           — number of rows processed
    """
    feat_df = build_features(df, do_mean=do_mean, do_std=do_std, drop_nan_rows=False)
    X = _align_features(feat_df, feature_cols)

    row_pred_idx = model.predict(X).astype(int)
    row_proba    = model.predict_proba(X)  # shape (n_rows, n_classes)

    mean_proba = row_proba.mean(axis=0)

    if aggregate == "mean_proba":
        batch_idx = int(np.argmax(mean_proba))
    else:  # majority
        counts = np.bincount(row_pred_idx, minlength=len(classes))
        batch_idx = int(np.argmax(counts))

    return {
        "predicted_class":  classes[batch_idx],
        "class_index":      batch_idx,
        "probabilities":    {c: round(float(p), 6) for c, p in zip(classes, mean_proba)},
        "row_predictions":  [classes[i] for i in row_pred_idx],
        "n_rows":           int(len(X)),
    }


# ---------------------------------------------------------------------------
# Single-row prediction
# ---------------------------------------------------------------------------


def predict_row(
    row: Union[pd.Series, Dict[str, float]],
    model: Any,
    feature_cols: List[str],
    classes: List[str] = CLASSES,
    *,
    do_mean: float = 0.006,
    do_std:  float = 0.001,
) -> Dict[str, Any]:
    """Predict the regime for a single observation (row of sensor readings).

    .. note::
        Rolling features cannot be computed from a single row.  All rolling
        statistics will default to the raw signal value (mean) or 0 (std/slope).
        For meaningful predictions use ``predict_batch`` with the full time series.

    Parameters
    ----------
    row : dict or pd.Series
        Mapping of column name → sensor value.  Must contain at least the
        columns in ``src.features.RAW_SIGNALS``.

    Returns
    -------
    dict with keys:
        ``predicted_class``  — string class label
        ``class_index``      — integer index
        ``probabilities``    — dict of {class_label: probability}
    """
    if isinstance(row, dict):
        row_s = pd.Series(row)
    else:
        row_s = row.copy()

    # Build a minimal two-row DataFrame so derivatives and rolling work
    df_single = pd.DataFrame([row_s, row_s]).reset_index(drop=True)
    # Force timestamp to [0, dt] so gradient doesn't divide by zero
    if "timestamp" not in df_single.columns:
        df_single["timestamp"] = [0.0, 1.0 / 60.0]
    if "regime" not in df_single.columns:
        df_single["regime"] = "Unknown"
    if "t_onset" not in df_single.columns:
        df_single["t_onset"] = np.nan

    feat_df = build_features(df_single, do_mean=do_mean, do_std=do_std, drop_nan_rows=False)
    # Take only the first row for the actual input
    feat_row = feat_df.iloc[[0]]
    X = _align_features(feat_row, feature_cols)

    proba     = model.predict_proba(X)[0]
    class_idx = int(np.argmax(proba))

    return {
        "predicted_class": classes[class_idx],
        "class_index":     class_idx,
        "probabilities":   {c: round(float(p), 6) for c, p in zip(classes, proba)},
    }
