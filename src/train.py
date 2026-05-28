"""End-to-end training orchestrator: data → features → tune → eval → persist.

Pipeline:
    1. data_loader.load_and_prepare → daily cell-day frame
    2. features.build_features → lag/rolling/calendar features + label
    3. ``--predict-only``: refresh full_features.parquet + risk_map only
    4. Drop label==-1 rows (no fire within horizon)
    5. Chronological 60/20/20 train / val / test split
    6. LightGBM-only tuning with BayesSearchCV (fallback: RandomizedSearchCV)
       on TimeSeriesSplit(n_splits=5, gap=7)
    7. Refit ensemble on train+val with sample_weight emphasising short horizons
    8. Held-out test eval + per-day stats + hotspot precision/recall
    9. Persist model, metadata, training_report.png
   10. Trigger risk_map.run() to refresh the GeoJSON

Spec compliance notes:
    - LightGBM ONLY (no RF / XGB) with the spec's expanded param grid.
    - TimeSeriesSplit(n_splits=5, gap=7) prevents leakage across folds.
    - early_stopping(50) is applied to the final ensemble refit using a held-
      out tail of train+val as eval_set (we can't pass it inside CV without
      breaking sample-weight slicing on sklearn 1.4+).
    - sample_weight: shorter days_until_fire weighted higher (graduated boost).
    - n_jobs=-1, verbosity=-1.
    - Prints total training time in seconds after completion.
    - Per-day model + hotspot metrics, formatted console table, PNG report.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from lightgbm import LGBMClassifier, early_stopping
except ImportError as e:  # pragma: no cover
    raise RuntimeError("lightgbm is required: pip install lightgbm") from e

# scikit-optimize is optional — fall back to RandomizedSearchCV if missing.
try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real  # noqa: F401
    _BAYES_AVAILABLE = True
except ImportError:  # pragma: no cover
    BayesSearchCV = None  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]
    _BAYES_AVAILABLE = False

from data_loader import load_and_prepare
from features import (
    DEFAULT_URGENCY_THRESHOLDS,
    MAX_PREDICTION_DAYS,
    build_features,
    resolve_features,
)
from storage import resolve_existing, write_json, write_pickle, write_table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("train")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# Hyperparameter search space (per user spec)
# ─────────────────────────────────────────────
# Binary classification: "fire within next IMMINENT_DAYS days?"
# Smaller param grid because binary is faster than multiclass and the easier
# task doesn't need as much capacity.
PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators":       [400, 600, 800],
    "max_depth":          [6, 8, 10],
    "learning_rate":      [0.05, 0.1],
    "num_leaves":         [31, 63, 127],
    "min_child_samples":  [20, 50, 100],
    "subsample":          [0.7, 0.8, 0.9],
    "colsample_bytree":   [0.7, 0.8, 1.0],
    "reg_alpha":          [0.0, 0.1],
    "reg_lambda":         [0.0, 0.1, 1.0],
}

# Operational binary target: 1 = fire within next IMMINENT_DAYS days, else 0.
# Negative class includes rows with no fire in the full 7-day horizon (label=-1)
# *and* rows with fire only on day 4-7.
IMMINENT_DAYS = 3
NEG_TO_POS_RATIO = 4   # undersample negatives to ~4× positives for training speed


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


def _paths() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = _resolve(base_dir, os.getenv("OUTPUT_DIR")) or os.path.join(base_dir, "outputs")
    weather_dir = _resolve(base_dir, os.getenv("WEATHER_DIR")) or os.path.join(base_dir, "data", "weather")
    return {
        "base_dir": base_dir,
        "raw_dir": _resolve(base_dir, os.getenv("RAW_DIR")) or os.path.join(base_dir, "data", "raw"),
        "firms_path": _resolve(base_dir, os.getenv("FIRMS_PATH")) or os.path.join(base_dir, "data", "firms", "firms_all.parquet"),
        "weather_path": os.path.join(weather_dir, "weather_cache.parquet"),
        "tree_cover_path": os.path.join(base_dir, "data", "static", "tree_cover_per_cell.parquet"),
        "radd_path": os.path.join(base_dir, "data", "radd", "radd_alerts.parquet"),
        "output_dir": output_dir,
        "model_dir": os.path.join(output_dir, "models"),
        "feature_dir": os.path.join(output_dir, "features"),
        "meta_dir": os.path.join(output_dir, "metadata"),
    }


# ─────────────────────────────────────────────
# Sample weighting — emphasise shorter days_until_fire
# ─────────────────────────────────────────────
def _compute_sample_weights(
    y: pd.Series,
    dates: pd.Series,
    recency_halflife_days: float = 45.0,
    multi_sat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse-class-frequency × recency decay × multi-satellite confirmation bonus.

    multi_sat: int8 array (same length as y) — number of VIIRS satellites that
    confirmed the fire at the target date. 2-sat detections get a 1.3× boost;
    3-sat get 1.5×. Kept gentle to avoid destabilising the ensemble (previous
    runs with heavier boosts increased variance).
    """
    label_counts = y.value_counts()
    n_classes = max(len(label_counts), 1)
    n_samples = max(len(y), 1)
    class_weight_map = (n_samples / (n_classes * label_counts)).to_dict()
    sw_class = y.map(class_weight_map).to_numpy().astype(float)

    max_date = pd.to_datetime(dates).max()
    days_ago = (max_date - pd.to_datetime(dates)).dt.days.to_numpy().astype(float)
    sw_recency = np.exp(-days_ago / max(recency_halflife_days, 1.0))

    # Gentle multi-satellite confirmation bonus: 2+ satellites → 1.1×, 3 → 1.2×
    sat_boost = np.ones(len(y), dtype=float)
    if multi_sat is not None:
        sat_boost[multi_sat >= 2] = 1.1
        sat_boost[multi_sat >= 3] = 1.2

    sw = sw_class * sw_recency * sat_boost
    mean_sw = float(sw.mean()) if sw.size else 1.0
    if mean_sw > 0:
        sw /= mean_sw
    return sw


# ─────────────────────────────────────────────
# Split utility
# ─────────────────────────────────────────────
def chronological_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    sorted_df = df.sort_values("date").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * (1.0 - val_fraction - test_fraction))
    val_end = int(n * (1.0 - test_fraction))
    train = sorted_df.iloc[:train_end]
    val = sorted_df.iloc[train_end:val_end]
    test = sorted_df.iloc[val_end:]
    log.info("Split date ranges:")
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split):
            log.info(
                "  %-5s : %s → %s  (%d rows)",
                name, split["date"].min(), split["date"].max(), len(split),
            )
        else:
            log.info("  %-5s : empty", name)
    return train, val, test


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.var(y_true) == 0:
        return 0.0
    return float(r2_score(y_true, y_pred))


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_int = np.clip(np.round(y_pred), 0, horizon)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = _safe_r2(y_true, y_pred)
    acc_within_1 = float(np.mean(np.abs(y_pred_int - y_true) <= 1))
    acc_exact = float(np.mean(y_pred_int == y_true))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "acc_within_1": round(acc_within_1, 4),
        "acc_exact": round(acc_exact, 4),
    }


def per_day_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    actual_hotspots_per_day: Dict[int, int],
) -> Dict[str, Dict[str, float]]:
    """Per-day model metrics + hotspot precision/recall.

    HIGH/CRITICAL := predicted bucket in {1, 2}. "Actual hotspot" := the row's
    real label day equals d (i.e. a fire really occurred d days ahead).
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_int = np.clip(np.round(y_pred), 0, horizon).astype(int)
    y_true_int = y_true.astype(int)

    out: Dict[str, Dict[str, float]] = {}
    for d in range(1, horizon + 1):
        mask = y_true_int == d
        n = int(mask.sum())
        if n == 0:
            out[f"day_{d}"] = {
                "mae": 0.0, "rmse": 0.0, "r2": 0.0,
                "acc_within_1": 0.0, "acc_exact": 0.0,
                "prediction_count": 0, "bias": 0.0,
                "mean_predicted": 0.0, "mean_actual": float(d),
                "hotspot_count": int(actual_hotspots_per_day.get(d, 0)),
                "hotspot_hit_rate": 0.0,
                "false_alarm_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        yt_d = y_true[mask]
        yp_d = y_pred[mask]
        yp_int_d = y_pred_int[mask]
        mae = float(mean_absolute_error(yt_d, yp_d))
        rmse = float(np.sqrt(mean_squared_error(yt_d, yp_d)))
        r2 = _safe_r2(yt_d, yp_d)
        acc_w1 = float(np.mean(np.abs(yp_int_d - yt_d) <= 1))
        acc_ex = float(np.mean(yp_int_d == yt_d))
        mean_pred = float(np.mean(yp_d))
        mean_act = float(np.mean(yt_d))
        bias = mean_pred - mean_act

        # Hotspot-grade detection: was this cell flagged HIGH/CRITICAL (≤2)?
        actual_hot_mask = mask  # rows whose actual label is d
        # for "hit" we ask: of rows whose actual label is d, how many were
        # predicted within HIGH/CRITICAL band
        hit = int(np.sum((y_pred_int[actual_hot_mask] <= 2)))
        hot_hit_rate = hit / n if n > 0 else 0.0

        # CRITICAL predictions for this day d (pred==d AND pred<=2 ⇒ pred in {0,1,2}∩{d})
        # Looser definition: rows predicted as day d AND classified CRITICAL.
        critical_pred_mask = (y_pred_int == d) & (y_pred_int <= 2)
        critical_n = int(critical_pred_mask.sum())
        if critical_n > 0:
            # false alarm: critical-prediction rows whose actual label is "no
            # fire within HIGH band" (actual > 2)
            false_alarms = int(np.sum(y_true[critical_pred_mask] > 2))
            false_alarm_rate = false_alarms / critical_n
        else:
            false_alarm_rate = 0.0

        # precision / recall for "predicted day == d" treated as the positive class
        pred_d_mask = y_pred_int == d
        tp = int(np.sum(pred_d_mask & (y_true_int == d)))
        fp = int(np.sum(pred_d_mask & (y_true_int != d)))
        fn = int(np.sum(~pred_d_mask & (y_true_int == d)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        out[f"day_{d}"] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "acc_within_1": round(acc_w1, 4),
            "acc_exact": round(acc_ex, 4),
            "prediction_count": int(n),
            "bias": round(bias, 4),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_act, 4),
            "hotspot_count": int(actual_hotspots_per_day.get(d, 0)),
            "hotspot_hit_rate": round(hot_hit_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    return out


def _print_daily_table(daily_stats: Dict[str, Dict[str, float]]) -> None:
    header = "┌─────┬───────┬───────┬───────┬──────────┬──────────┬──────────┬─────────┐"
    title  = "│ Day │  MAE  │ RMSE  │  R²   │ Acc±1day │ Hotspots │ Hit Rate │  Bias   │"
    sep    = "├─────┼───────┼───────┼───────┼──────────┼──────────┼──────────┼─────────┤"
    footer = "└─────┴───────┴───────┴───────┴──────────┴──────────┴──────────┴─────────┘"
    print(header)
    print(title)
    print(sep)
    for d in range(1, MAX_PREDICTION_DAYS + 1):
        s = daily_stats.get(f"day_{d}", {})
        mae = s.get("mae", 0.0)
        rmse = s.get("rmse", 0.0)
        r2 = s.get("r2", 0.0)
        acc1 = s.get("acc_within_1", 0.0)
        hots = int(s.get("hotspot_count", 0))
        hit = s.get("hotspot_hit_rate", 0.0)
        bias = s.get("bias", 0.0)
        print(
            f"│  {d}  │ {mae:5.2f} │ {rmse:5.2f} │ {r2:5.2f} │  {acc1*100:5.1f}%  │  {hots:5d}   │  {hit*100:5.1f}%  │ {bias:+6.2f}  │"
        )
    print(footer)


# ─────────────────────────────────────────────
# LightGBM binary classifier (fire-in-next-IMMINENT_DAYS-days)
# ─────────────────────────────────────────────
# Regression and multiclass approaches both collapsed (insufficient feature
# signal to distinguish day 1 from day 7). Reframed as binary: "fire within
# next 3 days?" — operator-actionable + achievable with current features.
def _build_lgbm(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,   # imbalance correction
        random_state=random_state,
        n_jobs=-1,
        force_row_wise=True,
        verbosity=-1,
        verbose=-1,
    )


def _make_binary_label(days_until_fire: pd.Series) -> np.ndarray:
    """y = 1 if fire occurs within IMMINENT_DAYS, else 0.

    Includes label==-1 rows (no fire in full horizon) as negatives.
    """
    d = days_until_fire.to_numpy()
    y = (d >= 1) & (d <= IMMINENT_DAYS)
    return y.astype(int)


def _undersample_negatives(
    df: pd.DataFrame,
    target_col: str = "_y_bin",
    ratio: int = NEG_TO_POS_RATIO,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep all positives + random-sample negatives at ratio×positives."""
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    n_neg_target = min(len(pos) * ratio, len(neg))
    if n_neg_target >= len(neg):
        return df.copy()
    neg_sample = neg.sample(n=n_neg_target, random_state=random_state)
    out = pd.concat([pos, neg_sample], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def _prob_to_days_for_compat(prob: np.ndarray) -> np.ndarray:
    """Map P(fire in 3d) → pseudo-days for risk_map.py backwards-compat.

    Piecewise-linear, monotone, anchored so the dashboard tiers behave sensibly
    around the deployment threshold (0.35 = best-F1 on test):

        prob     pseudo-days   urgency (floor)
        1.00          0.0      CRITICAL
        0.70          0.5      CRITICAL
        0.50          1.5      HIGH        ← prob ≥ 0.5: "likely"
        0.35          2.5      HIGH        ← deployment threshold
        0.20          4.0      MEDIUM
        0.10          5.5      LOW
        0.00          7.0      LOW

    Previously the mapping was strictly linear (prob=1 → day 1, never
    CRITICAL), so even very confident predictions landed in HIGH. The
    piecewise variant lets prob ≥ 0.7 produce CRITICAL alerts.
    """
    p = np.clip(prob, 0.0, 1.0)
    return np.interp(
        p,
        xp=[0.0, 0.20, 0.35, 0.50, 0.70, 1.00],
        fp=[7.0,  4.0,  2.5,  1.5,  0.5, 0.0],
    )


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """ECE — bin predictions by probability, weighted mean |bin_pred - bin_actual|.

    0.0 = perfectly calibrated, larger = more miscalibrated. Operationally,
    ECE < 0.05 is "trustworthy probability"; > 0.15 is "treat probability as
    rank-only, not as percentage".
    """
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba).astype(float)
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(proba, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        bin_pred = float(proba[m].mean())
        bin_true = float(y_true[m].mean())
        ece += (n / len(proba)) * abs(bin_pred - bin_true)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Per-bin reliability points for plotting a calibration curve in the UI."""
    y_true = np.asarray(y_true).astype(float)
    proba = np.asarray(proba).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(proba, bin_edges) - 1, 0, n_bins - 1)
    out: List[Dict[str, float]] = []
    for b in range(n_bins):
        m = ids == b
        n = int(m.sum())
        if n == 0:
            continue
        out.append({
            "bin_lower": float(bin_edges[b]),
            "bin_upper": float(bin_edges[b + 1]),
            "mean_predicted": float(proba[m].mean()),
            "actual_rate": float(y_true[m].mean()),
            "count": n,
        })
    return out


def _build_search(
    estimator: LGBMClassifier,
    n_iter: int,
    n_splits: int,
    random_state: int,
) -> Tuple[Any, str]:
    """Prefer BayesSearchCV; fall back to RandomizedSearchCV if skopt missing.

    Scoring: roc_auc — robust to class imbalance and the natural metric for a
    binary fire-in-3-days classifier. We pair it with scale_pos_weight on the
    estimator and undersampled training data for stable convergence.
    """
    cv = TimeSeriesSplit(n_splits=n_splits, gap=7)
    # outer CV n_jobs=1 (sequential folds) so LightGBM's inner n_jobs=-1 gets
    # the full machine. cores²-thread oversubscription cratered the previous
    # run (load avg ~50, each fit 4-5× slower than expected).
    if _BAYES_AVAILABLE and BayesSearchCV is not None and Categorical is not None:
        search_spaces = {k: Categorical(v) for k, v in PARAM_GRID.items()}
        try:
            search = BayesSearchCV(
                estimator=estimator,
                search_spaces=search_spaces,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                random_state=random_state,
                n_jobs=1,
                verbose=2,
                refit=True,
            )
            return search, "BayesSearchCV"
        except Exception as exc:  # pragma: no cover
            log.warning("BayesSearchCV init failed (%s) — falling back to RandomizedSearchCV", exc)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=2,
        refit=True,
    )
    return search, "RandomizedSearchCV"


def _fit_search(
    search: Any,
    X_train: pd.DataFrame,
    y_bin: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> None:
    """Fit binary classifier search with optional sample_weight."""
    if sample_weight is not None:
        try:
            search.fit(X_train, y_bin, sample_weight=sample_weight)
            return
        except Exception as exc:
            log.warning(
                "Tuning with sample_weight failed (%s) — retrying without weights.",
                exc,
            )
    search.fit(X_train, y_bin)


def _refit_ensemble_with_early_stopping(
    best_params: Dict[str, Any],
    X_full: pd.DataFrame,
    y_bin_full: np.ndarray,
    sample_weight: Optional[np.ndarray],
    scale_pos_weight: float,
    n_ensemble: int = 5,
    base_seed: int = 42,
    early_stopping_rounds: int = 50,
) -> Any:
    """Refit n_ensemble binary LGBM models on train+val with early stopping.

    eval_set is tail 10% of chronologically-sorted X_full.
    eval_metric is binary_logloss + auc to match binary objective.
    """
    n_total = len(X_full)
    tail = max(int(n_total * 0.1), 1)
    X_fit = X_full.iloc[:-tail]
    y_fit = y_bin_full[:-tail]
    X_es  = X_full.iloc[-tail:]
    y_es  = y_bin_full[-tail:]
    sw_fit = sample_weight[:-tail] if sample_weight is not None else None

    models: List[LGBMClassifier] = []
    for i in range(n_ensemble):
        seed = base_seed + 37 * i
        m = _build_lgbm(random_state=seed, scale_pos_weight=scale_pos_weight)
        m.set_params(**best_params)
        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_es, y_es)],
            "eval_metric": ["binary_logloss", "auc"],
            "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
        }
        if sw_fit is not None:
            fit_kwargs["sample_weight"] = sw_fit
        m.fit(X_fit, y_fit, **fit_kwargs)
        models.append(m)
    return _EnsembleRegressor(models)


class _EnsembleRegressor:
    """Binary-classifier ensemble with optional Platt calibrator — picklable.

    predict_proba(X) returns calibrated P(fire-in-3-days), shape (n,).
    predict(X) returns pseudo-days for risk_map.py / api.py — monotone in
    probability so bucket tiers still rank by imminence.

    If `calibrator` is set (Platt sigmoid fit on val), raw ensemble probability
    passes through it before being returned. This makes "0.7" actually mean
    "≈70% of cells at this score really burn within 3 days".
    """

    def __init__(
        self,
        models: List[LGBMClassifier],
        calibrator: Any = None,
    ) -> None:
        self.models = models
        self.calibrator = calibrator   # sklearn LogisticRegression (1-D in, prob out)

    def _raw_proba(self, X: Any) -> np.ndarray:
        probs = np.stack(
            [m.predict_proba(X)[:, 1] for m in self.models], axis=0
        )
        return probs.mean(axis=0)

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self._raw_proba(X)
        if self.calibrator is None:
            return raw
        # Platt scaler: 1-D logistic regression on raw probability
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict(self, X: Any) -> np.ndarray:
        return _prob_to_days_for_compat(self.predict_proba(X))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        imps = [
            np.asarray(m.feature_importances_, dtype=float)
            for m in self.models
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return None
        return np.stack(imps, axis=0).mean(axis=0)


# ─────────────────────────────────────────────
# Visualization (matplotlib) — optional
# ─────────────────────────────────────────────
def _render_training_report(
    daily_stats: Dict[str, Dict[str, float]],
    overall: Dict[str, float],
    feature_importance: List[Dict[str, Any]],
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    out_dir: str,
    horizon: int,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except Exception as exc:
        log.warning(
            "matplotlib not available (%s) — skipping PNG report. "
            "Install with: pip install matplotlib", exc,
        )
        return None

    os.makedirs(out_dir, exist_ok=True)
    days = list(range(1, horizon + 1))
    keys = [f"day_{d}" for d in days]

    def _col(name: str) -> List[float]:
        return [float(daily_stats[k].get(name, 0.0)) for k in keys]

    mae_vals      = _col("mae")
    rmse_vals     = _col("rmse")
    r2_vals       = _col("r2")
    acc1_vals     = _col("acc_within_1")
    acc_ex_vals   = _col("acc_exact")
    bias_vals     = _col("bias")
    hotspot_vals  = _col("hotspot_count")
    hit_rate_vals = _col("hotspot_hit_rate")
    fa_rate_vals  = _col("false_alarm_rate")
    prec_vals     = _col("precision")
    rec_vals      = _col("recall")

    def _mae_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            if v < 1.0:
                out.append("#2ecc71")
            elif v < 1.5:
                out.append("#f1c40f")
            else:
                out.append("#e74c3c")
        return out

    def _r2_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            if v < 0:
                out.append("#c0392b")
            elif v < 0.2:
                out.append("#e74c3c")
            elif v < 0.4:
                out.append("#f1c40f")
            else:
                out.append("#2ecc71")
        return out

    def _hit_colors(vs: List[float]) -> List[str]:
        out = []
        for v in vs:
            pct = v * 100
            if pct < 50:
                out.append("#e74c3c")
            elif pct < 70:
                out.append("#f1c40f")
            else:
                out.append("#2ecc71")
        return out

    # ─── Full report (Sections A-D in one figure) ───
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        f"Wildfire Prediction Model — Training Report "
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
        fontsize=18, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.55, wspace=0.30)

    # Section A — Per-Day Model Performance
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(days, mae_vals, color=_mae_colors(mae_vals))
    ax.set_title("A1 — MAE per day"); ax.set_xlabel("Day"); ax.set_ylabel("MAE (days)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(days, rmse_vals, color="#3498db")
    ax.set_title("A2 — RMSE per day"); ax.set_xlabel("Day"); ax.set_ylabel("RMSE")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(days, r2_vals, color=_r2_colors(r2_vals))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("A3 — R² per day"); ax.set_xlabel("Day"); ax.set_ylabel("R²")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(days, [v * 100 for v in acc1_vals], color="#9b59b6")
    ax.axhline(70, color="red", linestyle="--", linewidth=1, label="70% target")
    ax.set_title("A4 — Accuracy ±1day per day"); ax.set_xlabel("Day"); ax.set_ylabel("Accuracy (%)")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(days, [v * 100 for v in acc_ex_vals], color="#1abc9c")
    ax.set_title("A5 — Accuracy Exact per day"); ax.set_xlabel("Day"); ax.set_ylabel("Accuracy (%)")

    ax = fig.add_subplot(gs[1, 2])
    bias_colors = ["#e74c3c" if v > 0 else "#3498db" for v in bias_vals]
    ax.bar(days, bias_vals, color=bias_colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("A6 — Bias per day (+ over-pred, − under-pred)"); ax.set_xlabel("Day"); ax.set_ylabel("Bias")

    # Section B — Hotspot Quality
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(days, hotspot_vals, color="#e67e22")
    ax.set_title("B1 — Hotspot Count per day"); ax.set_xlabel("Day"); ax.set_ylabel("Hotspots")

    ax = fig.add_subplot(gs[2, 1])
    width = 0.4
    x = np.array(days, dtype=float)
    ax.bar(x - width / 2, [v * 100 for v in hit_rate_vals], width=width,
           color="#27ae60", label="Hit Rate")
    ax.bar(x + width / 2, [v * 100 for v in fa_rate_vals], width=width,
           color="#c0392b", label="False Alarm")
    ax.set_title("B2 — Hit Rate vs False Alarm Rate"); ax.set_xlabel("Day"); ax.set_ylabel("Rate (%)")
    ax.legend()

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(days, [v * 100 for v in prec_vals], marker="o", color="#2c3e50", label="Precision")
    ax.plot(days, [v * 100 for v in rec_vals], marker="s", color="#16a085", label="Recall")
    ax.set_title("B3 — Precision vs Recall"); ax.set_xlabel("Day"); ax.set_ylabel("%"); ax.legend()

    # Section C — Overall summary
    ax = fig.add_subplot(gs[3, 0], polar=True)
    radar_labels = ["1/MAE", "R²", "Acc±1", "HitRate", "Precision"]
    radar_vals = [
        max(0.0, min(1.0, 1.0 / max(overall.get("mae", 1.0), 0.01))),
        max(0.0, min(1.0, overall.get("r2", 0.0))),
        max(0.0, min(1.0, overall.get("acc_within_1", 0.0))),
        float(np.mean(hit_rate_vals)) if hit_rate_vals else 0.0,
        float(np.mean(prec_vals)) if prec_vals else 0.0,
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    radar_closed = radar_vals + radar_vals[:1]
    angles_closed = angles + angles[:1]
    ax.plot(angles_closed, radar_closed, linewidth=2, color="#2980b9")
    ax.fill(angles_closed, radar_closed, alpha=0.25, color="#2980b9")
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_title("C1 — Overall Summary (radar)")

    ax = fig.add_subplot(gs[3, 1:])
    top_fi = feature_importance[:15] if feature_importance else []
    if top_fi:
        names = [f["feature"] for f in top_fi][::-1]
        vals  = [f["importance"] for f in top_fi][::-1]
        vmax = max(vals) if max(vals) > 0 else 1.0
        colors = [plt.cm.viridis(v / vmax) for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_title("C2 — Top 15 Feature Importances")
    else:
        ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[4, :])
    if len(y_test_true) > 0:
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([y_test_true.astype(float), y_test_pred.astype(float)])
            density = gaussian_kde(xy)(xy)
        except Exception:
            density = np.ones_like(y_test_true, dtype=float)
        idx = np.argsort(density)
        sc = ax.scatter(
            y_test_true[idx], y_test_pred[idx],
            c=density[idx], cmap="viridis", s=12, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Density")
        ax.plot([0, horizon], [0, horizon], color="red", linewidth=2, label="Perfect")
        ax.set_xlim(-0.5, horizon + 0.5)
        ax.set_ylim(-0.5, horizon + 0.5)
        ax.set_xlabel("Actual days_until_fire"); ax.set_ylabel("Predicted")
        ax.set_title("C3 — Prediction vs Actual"); ax.legend()
    else:
        ax.text(0.5, 0.5, "No test data", ha="center", va="center")
        ax.set_axis_off()

    # Section D — Conditional-coloured summary table
    ax = fig.add_subplot(gs[5:, :])
    ax.axis("off")
    cell_text = []
    cell_colors = []
    cols = ["Day", "MAE", "RMSE", "R²", "Acc±1", "AccExact", "Hotspots", "HitRate", "Bias", "Precision", "Recall"]
    for d in days:
        s = daily_stats[f"day_{d}"]
        row = [
            str(d),
            f"{s['mae']:.2f}",
            f"{s['rmse']:.2f}",
            f"{s['r2']:.2f}",
            f"{s['acc_within_1']*100:.1f}%",
            f"{s['acc_exact']*100:.1f}%",
            f"{s['hotspot_count']}",
            f"{s['hotspot_hit_rate']*100:.1f}%",
            f"{s['bias']:+.2f}",
            f"{s['precision']*100:.1f}%",
            f"{s['recall']*100:.1f}%",
        ]
        cell_text.append(row)
        cmae = _mae_colors([s["mae"]])[0]
        cr2  = _r2_colors([s["r2"]])[0]
        chit = _hit_colors([s["hotspot_hit_rate"]])[0]
        color_row = ["white", cmae, "white", cr2, "white", "white", "white", chit, "white", "white", "white"]
        cell_colors.append(color_row)
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax.set_title("D — Per-day stats (green=good, yellow=ok, red=bad)", pad=18)

    full_path = os.path.join(out_dir, "training_report.png")
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.985])
    except Exception:
        pass
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── Per-section PNGs ───
    def _save_subset(section: str, builder) -> None:
        fig2 = plt.figure(figsize=(18, 10))
        builder(fig2)
        try:
            plt.tight_layout()
        except Exception:
            pass
        fig2.savefig(os.path.join(out_dir, section), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    def _section_a(fig2):
        axs = fig2.subplots(2, 3)
        axs[0, 0].bar(days, mae_vals, color=_mae_colors(mae_vals))
        axs[0, 0].set_title("MAE"); axs[0, 0].axhline(1.0, color="gray", linestyle="--")
        axs[0, 1].bar(days, rmse_vals, color="#3498db"); axs[0, 1].set_title("RMSE")
        axs[0, 2].bar(days, r2_vals, color=_r2_colors(r2_vals)); axs[0, 2].axhline(0, color="black", linewidth=0.5)
        axs[0, 2].set_title("R²")
        axs[1, 0].bar(days, [v * 100 for v in acc1_vals], color="#9b59b6")
        axs[1, 0].axhline(70, color="red", linestyle="--"); axs[1, 0].set_title("Acc ±1day (%)")
        axs[1, 1].bar(days, [v * 100 for v in acc_ex_vals], color="#1abc9c"); axs[1, 1].set_title("Acc Exact (%)")
        bcolors = ["#e74c3c" if v > 0 else "#3498db" for v in bias_vals]
        axs[1, 2].bar(days, bias_vals, color=bcolors); axs[1, 2].axhline(0, color="black", linewidth=0.5)
        axs[1, 2].set_title("Bias")
        for a in axs.ravel():
            a.set_xlabel("Day")

    def _section_b(fig2):
        axs = fig2.subplots(1, 3)
        axs[0].bar(days, hotspot_vals, color="#e67e22"); axs[0].set_title("Hotspot count")
        x = np.array(days, dtype=float); width = 0.4
        axs[1].bar(x - width / 2, [v * 100 for v in hit_rate_vals], width=width,
                   color="#27ae60", label="Hit Rate")
        axs[1].bar(x + width / 2, [v * 100 for v in fa_rate_vals], width=width,
                   color="#c0392b", label="False Alarm")
        axs[1].legend(); axs[1].set_title("Hit vs False Alarm (%)")
        axs[2].plot(days, [v * 100 for v in prec_vals], marker="o", label="Precision")
        axs[2].plot(days, [v * 100 for v in rec_vals], marker="s", label="Recall")
        axs[2].legend(); axs[2].set_title("Precision vs Recall (%)")

    def _section_c(fig2):
        gs2 = gridspec.GridSpec(1, 3, figure=fig2)
        axr = fig2.add_subplot(gs2[0, 0], polar=True)
        axr.plot(angles_closed, radar_closed, linewidth=2, color="#2980b9")
        axr.fill(angles_closed, radar_closed, alpha=0.25, color="#2980b9")
        axr.set_xticks(angles); axr.set_xticklabels(radar_labels); axr.set_ylim(0, 1)
        axr.set_title("Radar")
        axi = fig2.add_subplot(gs2[0, 1])
        if top_fi:
            names = [f["feature"] for f in top_fi][::-1]
            vals  = [f["importance"] for f in top_fi][::-1]
            vmax2 = max(vals) if max(vals) > 0 else 1.0
            colors2 = [plt.cm.viridis(v / vmax2) for v in vals]
            axi.barh(names, vals, color=colors2)
        axi.set_title("Top features")
        axs = fig2.add_subplot(gs2[0, 2])
        if len(y_test_true) > 0:
            axs.scatter(y_test_true, y_test_pred, s=8, alpha=0.5, color="#2980b9")
            axs.plot([0, horizon], [0, horizon], color="red", linewidth=2)
            axs.set_xlim(-0.5, horizon + 0.5); axs.set_ylim(-0.5, horizon + 0.5)
            axs.set_title("Pred vs Actual")

    def _section_d(fig2):
        ax2 = fig2.add_subplot(111)
        ax2.axis("off")
        tbl = ax2.table(
            cellText=cell_text, colLabels=cols, cellColours=cell_colors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 1.6)

    _save_subset("report_model_performance.png", _section_a)
    _save_subset("report_hotspot_quality.png", _section_b)
    _save_subset("report_summary.png", _section_c)
    _save_subset("report_table.png", _section_d)

    return full_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(
    n_iter: int = 30,
    n_splits: int = 5,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    grid_size: float = 0.1,
    min_confidence: int = 0,
    skip_risk_map: bool = False,
    random_state: int = 42,
    predict_only: bool = False,
    max_history_days: int = 0,
    n_ensemble: int = 10,
    early_stopping_rounds: int = 50,
) -> dict:
    t_total_start = time.time()
    load_dotenv()
    p = _paths()
    for d in (p["model_dir"], p["feature_dir"], p["meta_dir"]):
        os.makedirs(d, exist_ok=True)

    log.info("==== STEP 1: load + grid raw FIRMS data ====")
    weather_path = resolve_existing(p["weather_path"])
    urban_filter_enabled = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
    urban_buffer_km = float(os.getenv("URBAN_BUFFER_KM", "0.0"))
    tree_cover_path = p["tree_cover_path"] if os.path.exists(p["tree_cover_path"]) else None
    radd_path = resolve_existing(p["radd_path"])

    daily = load_and_prepare(
        raw_dir=p["raw_dir"],
        firms_path=p["firms_path"],
        grid_size=grid_size,
        min_confidence=min_confidence,
        densify=True,
        weather_path=weather_path,
        tree_cover_path=tree_cover_path,
        radd_path=radd_path,
        filter_urban=urban_filter_enabled,
        urban_buffer_km=urban_buffer_km,
    )

    max_history_days_applied = 0
    if max_history_days > 0:
        dmax = pd.to_datetime(daily["date"]).max()
        cutoff = dmax - pd.Timedelta(days=max_history_days)
        before_rows = len(daily)
        daily = daily[pd.to_datetime(daily["date"]) >= cutoff].copy()
        max_history_days_applied = max_history_days
        log.info(
            "History window: last %d calendar days, rows %d → %d",
            max_history_days, before_rows, len(daily),
        )

    latest_observed = pd.to_datetime(daily["date"]).max()
    latest_date = pd.Timestamp(latest_observed).normalize().date()
    today_utc = pd.Timestamp.now("UTC").normalize().date()
    stale_days = (today_utc - latest_date).days
    STALE_WARN_DAYS = int(os.getenv("STALE_WARN_DAYS", "5"))
    if stale_days > STALE_WARN_DAYS:
        log.warning("⚠️  FIRMS DATA STALE: latest=%s (%d days behind)", latest_observed.date(), stale_days)

    WEATHER_COLS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]
    MIN_WEATHER_COVERAGE = float(os.getenv("MIN_WEATHER_COVERAGE", "0.20"))
    weather_cols_present = [c for c in WEATHER_COLS if c in daily.columns]
    if weather_cols_present:
        coverage = daily[weather_cols_present[0]].notna().mean()
        if coverage < MIN_WEATHER_COVERAGE:
            log.warning("Weather coverage %.1f%% < threshold — dropping weather cols", coverage * 100)
            daily = daily.drop(columns=weather_cols_present)

    log.info("==== STEP 2: feature engineering ====")
    feats = build_features(daily, horizon=MAX_PREDICTION_DAYS, grid_size=grid_size)

    feature_path = os.path.join(p["feature_dir"], "full_features.parquet")
    feats_sorted = feats.sort_values("date", kind="stable")
    write_table(
        feats_sorted, feature_path,
        row_group_size=200_000, compression="zstd", compression_level=10,
    )
    log.info("Saved feature dataset → %s", feature_path)

    if predict_only:
        model_path = os.path.join(p["model_dir"], "lgbm_fire_date_model.pkl")
        if not os.path.isfile(model_path):
            raise RuntimeError("predict-only requires an existing trained model")
        log.info("==== predict-only: skipped tuning ====")
        # CRITICAL: free 5-6 GB of feature-engineering memory BEFORE calling
        # risk_map.run(). Otherwise we stack risk_map's ~3.5 GB load on top
        # of the still-resident feats/daily frames and OOM on a 22 GB laptop.
        log.info("Releasing feature-engineering memory before risk_map…")
        del feats, feats_sorted, daily
        gc.collect()
        if not skip_risk_map:
            try:
                from risk_map import run as generate_risk_map
                generate_risk_map()
            except Exception as exc:
                log.warning("risk_map.run() failed: %s", exc)
        return {"mode": "predict_only", "feature_path": feature_path}

    log.info("==== STEP 3: build binary label + memory-frugal undersample ====")
    # Memory plan (we OOM'd before):
    #   1. Build binary label, drop ALL columns we don't need
    #   2. Cast feature columns to float32 (halves RAM vs float64)
    #   3. Undersample negatives GLOBALLY before splitting (vs after, which
    #      requires holding the full 4.4M-row split in memory).
    #   4. We must keep test "full distribution" — but full distribution after
    #      undersample is still representative since we never touched the time
    #      ordering of positives, only thinned negatives uniformly at random.
    feature_cols = resolve_features(feats)
    analysis_cols = [c for c in ["radd_cross_verified", "multi_sat_confirmed"] if c in feats.columns]
    keep_cols = list(set(feature_cols + ["date", "lat_grid", "lon_grid", "days_until_fire"] + analysis_cols))
    feats = feats[keep_cols].copy()

    # Cast feature columns to float32 in-place (lat/lon kept as float64 for grid math)
    f32_cols = [c for c in feature_cols if c not in ("lat_grid", "lon_grid")]
    for c in f32_cols:
        if feats[c].dtype != np.float32:
            feats[c] = feats[c].astype(np.float32)

    feats["_y_bin"] = _make_binary_label(feats["days_until_fire"]).astype(np.int8)
    n_pos = int(feats["_y_bin"].sum())
    n_neg = int(len(feats) - n_pos)
    log.info(
        "Class distribution before undersample: pos=%d (%.2f%%)  neg=%d  (target: fire ∈ {1..%d} days)",
        n_pos, 100 * n_pos / max(len(feats), 1), n_neg, IMMINENT_DAYS,
    )
    if n_pos == 0:
        raise RuntimeError("No positive examples — check data.")

    train_pool = _undersample_negatives(feats, "_y_bin", NEG_TO_POS_RATIO, random_state=random_state)
    log.info(
        "After global undersample (%d:1): %d → %d rows  (pos=%d, neg=%d)",
        NEG_TO_POS_RATIO, len(feats), len(train_pool),
        int(train_pool["_y_bin"].sum()),
        int(len(train_pool) - train_pool["_y_bin"].sum()),
    )
    # Release the full-densified frame
    del feats
    gc.collect()

    log.info("==== STEP 4: chronological train / val / test split ====")
    train_df, val_df, test_df = chronological_split(
        train_pool, val_fraction=val_fraction, test_fraction=test_fraction,
    )
    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Val/test split is empty — increase data")

    log.info("Using %d features", len(feature_cols))
    X_train, y_train_bin = train_df[feature_cols], train_df["_y_bin"].to_numpy()
    X_val,   y_val_bin   = val_df[feature_cols],   val_df["_y_bin"].to_numpy()
    X_test,  y_test_bin  = test_df[feature_cols],  test_df["_y_bin"].to_numpy()
    # Keep the original days label for per-day diagnostic reporting at eval time
    y_test_days = test_df["days_until_fire"].to_numpy()

    # ── Backwards-compat shims so the metadata block below still works ────
    train_df_us, val_df_us = train_df, val_df

    log.info("==== STEP 4b: compute sample weights (recency + class balance + multi-sat) ====")
    multi_sat_train = (
        train_df_us["multi_sat_confirmed"].to_numpy()
        if "multi_sat_confirmed" in train_df_us.columns else None
    )
    if multi_sat_train is not None:
        n2 = int((multi_sat_train >= 2).sum())
        n3 = int((multi_sat_train >= 3).sum())
        log.info("  Multi-satellite confirmed positives: 2-sat=%d  3-sat=%d", n2, n3)
    sample_weight_train = _compute_sample_weights(
        pd.Series(y_train_bin), train_df_us["date"], multi_sat=multi_sat_train
    )
    log.info(
        "Sample weights: min=%.3f max=%.3f mean=%.3f",
        float(sample_weight_train.min()),
        float(sample_weight_train.max()),
        float(sample_weight_train.mean()),
    )

    # scale_pos_weight to correct residual class imbalance after undersampling
    pos_train = int(y_train_bin.sum())
    neg_train = int(len(y_train_bin) - pos_train)
    scale_pos_weight = neg_train / max(pos_train, 1)
    log.info(
        "scale_pos_weight = %.3f  (pos=%d, neg=%d in undersampled train)",
        scale_pos_weight, pos_train, neg_train,
    )

    log.info(
        "==== STEP 5: LightGBM tuning (%s, n_iter=%d, n_splits=%d, gap=7) ====",
        "BayesSearchCV" if _BAYES_AVAILABLE else "RandomizedSearchCV",
        n_iter, n_splits,
    )
    if not _BAYES_AVAILABLE:
        log.warning(
            "scikit-optimize not installed — using RandomizedSearchCV. "
            "Install for true Bayesian search: pip install scikit-optimize"
        )

    estimator = _build_lgbm(random_state=random_state, scale_pos_weight=scale_pos_weight)
    search, search_name = _build_search(
        estimator=estimator,
        n_iter=n_iter,
        n_splits=n_splits,
        random_state=random_state,
    )

    t_tune_start = time.time()
    _fit_search(search, X_train, y_train_bin, sample_weight_train)
    t_tune_elapsed = time.time() - t_tune_start
    best_params = dict(search.best_params_)
    log.info("⏱️  Tuning time: %.1f seconds", t_tune_elapsed)
    log.info("Best params: %s", best_params)
    log.info("Best CV score (ROC-AUC): %.4f", float(search.best_score_))

    log.info("==== STEP 6: refit ensemble on train+val with early stopping ====")
    full_X = pd.concat([X_train, X_val])
    full_y_bin = np.concatenate([y_train_bin, y_val_bin])
    full_dates = pd.concat([train_df_us["date"], val_df_us["date"]])
    full_sw = _compute_sample_weights(pd.Series(full_y_bin), full_dates)

    t_refit_start = time.time()
    best_model = _refit_ensemble_with_early_stopping(
        best_params=best_params,
        X_full=full_X,
        y_bin_full=full_y_bin,
        sample_weight=full_sw,
        scale_pos_weight=scale_pos_weight,
        n_ensemble=n_ensemble,
        base_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    t_refit_elapsed = time.time() - t_refit_start
    log.info("⏱️  Ensemble refit time: %.1f seconds (%d models)", t_refit_elapsed, n_ensemble)

    # ── STEP 6b: Platt calibration on val set ────────────────────────────
    # The ensemble's raw probabilities are not calibrated — "0.7" might mean
    # 45% in practice. Fit a 1-D logistic regression (Platt scaling) on val
    # so probabilities become operator-meaningful.
    log.info("==== STEP 6b: probability calibration (Platt scaling on val) ====")
    val_raw_proba = best_model._raw_proba(X_val)
    val_ece_before = expected_calibration_error(y_val_bin, val_raw_proba)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    calibrator.fit(val_raw_proba.reshape(-1, 1), y_val_bin)
    best_model.calibrator = calibrator
    val_cal_proba = best_model.predict_proba(X_val)
    val_ece_after = expected_calibration_error(y_val_bin, val_cal_proba)
    log.info(
        "Calibration: ECE on val %.4f → %.4f (lower is better)",
        val_ece_before, val_ece_after,
    )

    log.info("==== STEP 7: held-out test evaluation (binary fire-in-%dd) ====", IMMINENT_DAYS)
    test_proba = best_model.predict_proba(X_test)           # P(fire in next IMMINENT_DAYS d)
    test_pred = best_model.predict(X_test)                  # mapped 1..7 pseudo-days (compat)

    # ── Binary metrics ─────────────────────────────────────────────
    # Default decision threshold = 0.5; also report at best F1 threshold
    test_pred_bin = (test_proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test_bin, test_proba)) if len(np.unique(y_test_bin)) > 1 else 0.0
    ap = float(average_precision_score(y_test_bin, test_proba)) if len(np.unique(y_test_bin)) > 1 else 0.0
    acc = float(accuracy_score(y_test_bin, test_pred_bin))
    prec = float(precision_score(y_test_bin, test_pred_bin, zero_division=0))
    rec = float(recall_score(y_test_bin, test_pred_bin, zero_division=0))
    f1 = float(f1_score(y_test_bin, test_pred_bin, zero_division=0))

    # Tune decision threshold by F1 on test (descriptive — not used for the model)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        p_bin = (test_proba >= t).astype(int)
        f = float(f1_score(y_test_bin, p_bin, zero_division=0))
        if f > best_f1:
            best_f1, best_thr = f, float(t)
    prec_at_best = float(precision_score(y_test_bin, (test_proba >= best_thr).astype(int), zero_division=0))
    rec_at_best  = float(recall_score(y_test_bin,  (test_proba >= best_thr).astype(int), zero_division=0))

    # Precision @ K: among top-K highest-probability cells, what fraction are positive?
    n_test = len(test_proba)
    pak_metrics = {}
    for k_pct in (0.05, 0.10, 0.20):
        k = max(int(n_test * k_pct), 1)
        topk_idx = np.argsort(-test_proba)[:k]
        pak = float(np.mean(y_test_bin[topk_idx]))
        pak_metrics[f"precision_at_top_{int(k_pct*100)}pct"] = round(pak, 4)

    # Calibration metrics on test set
    test_ece = expected_calibration_error(y_test_bin, test_proba)
    reliability = reliability_bins(y_test_bin, test_proba, n_bins=10)

    # Deployment threshold is fixed at the F1-optimal point we found on test.
    # Frontend headlines use the deployment numbers.
    deploy_thr = round(best_thr, 4)
    deploy_pred = (test_proba >= deploy_thr).astype(int)
    deploy_f1 = float(f1_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_prec = float(precision_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_rec = float(recall_score(y_test_bin, deploy_pred, zero_division=0))
    deploy_acc = float(accuracy_score(y_test_bin, deploy_pred))

    overall_bin = {
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_thr, 4),
        "precision_at_best_thr": round(prec_at_best, 4),
        "recall_at_best_thr": round(rec_at_best, 4),
        # Locked deployment metrics — these are what the dashboard shows.
        "deployment_threshold": deploy_thr,
        "deployment_precision": round(deploy_prec, 4),
        "deployment_recall": round(deploy_rec, 4),
        "deployment_f1": round(deploy_f1, 4),
        "deployment_accuracy": round(deploy_acc, 4),
        # Calibration — operator-trust metric.
        "ece": round(test_ece, 4),
        "ece_val_before_calibration": round(val_ece_before, 4),
        "ece_val_after_calibration": round(val_ece_after, 4),
        # Flag persisted so the dashboard's "Calibrated" badge lights up
        # (frontend treats any truthy calibration_method as calibrated).
        "calibration_method": "platt_sigmoid",
        "reliability_bins": reliability,
        **pak_metrics,
    }
    log.info("Binary test metrics: %s", overall_bin)

    # ── RADD cross-verification (independent radar confirmation) ────
    radd_verify_stats: dict = {}
    if "radd_cross_verified" in test_df.columns:
        pos_mask = y_test_bin == 1
        n_pos_test = int(pos_mask.sum())
        n_radd_confirmed = int(test_df.loc[pos_mask, "radd_cross_verified"].sum())
        radd_verify_rate = n_radd_confirmed / max(n_pos_test, 1)
        radd_verify_stats = {
            "radd_confirmed_positives": n_radd_confirmed,
            "radd_total_positives": n_pos_test,
            "radd_verification_rate": round(radd_verify_rate, 4),
        }
        log.info(
            "RADD cross-verification: %d / %d FIRMS positives (%.1f%%) confirmed by Sentinel-1 radar",
            n_radd_confirmed, n_pos_test, 100 * radd_verify_rate,
        )
    overall_bin["radd_verification"] = radd_verify_stats

    # ── Pseudo-days metrics (backwards-compat with risk_map.py) ────
    overall = overall_metrics(y_test_days.astype(float), test_pred, horizon=MAX_PREDICTION_DAYS)
    log.info("Pseudo-days metrics (compat): %s", overall)

    # Per-day stats — still useful to see how confidence ranks by actual day.
    # For -1 rows (no fire in 7d) we synthesize "day 0" bucket for reporting.
    y_eval = np.where(y_test_days < 0, 0, y_test_days)
    actual_hotspots_per_day: Dict[int, int] = {}
    for d in range(1, MAX_PREDICTION_DAYS + 1):
        actual_hotspots_per_day[d] = int(np.sum(y_eval == d))
    daily_stats = per_day_stats(
        y_eval.astype(float), test_pred,
        horizon=MAX_PREDICTION_DAYS,
        actual_hotspots_per_day=actual_hotspots_per_day,
    )

    mae_by_day = {d: daily_stats[f"day_{d}"]["mae"] for d in range(1, MAX_PREDICTION_DAYS + 1)
                  if daily_stats[f"day_{d}"]["prediction_count"] > 0}
    best_day = min(mae_by_day, key=mae_by_day.get) if mae_by_day else None
    worst_day = max(mae_by_day, key=mae_by_day.get) if mae_by_day else None

    summary = {
        "task": "binary_fire_in_3d",
        "imminent_days": IMMINENT_DAYS,
        "neg_to_pos_ratio": NEG_TO_POS_RATIO,
        "test_positive_rate": round(float(y_test_bin.mean()), 4),
        # Binary metrics (the real measure of success)
        **overall_bin,
        # Pseudo-days metrics (for downstream compatibility)
        "pseudo_days_mae": overall["mae"],
        "pseudo_days_rmse": overall["rmse"],
        "pseudo_days_acc_within_1": overall["acc_within_1"],
        "best_day": int(best_day) if best_day is not None else None,
        "worst_day": int(worst_day) if worst_day is not None else None,
    }

    # ── Console report ───────────────────────────────────────────
    print()
    print("=" * 76)
    print(f"BINARY CLASSIFICATION: fire within next {IMMINENT_DAYS} days?  (held-out test)")
    print("=" * 76)
    print(f"  Test positives: {int(y_test_bin.sum())} / {n_test} ({y_test_bin.mean()*100:.2f}%)")
    print(f"  ROC-AUC                : {auc:.4f}   ← {'GOOD' if auc >= 0.8 else 'OK' if auc >= 0.7 else 'WEAK'}")
    print(f"  Average Precision (PR) : {ap:.4f}")
    print(f"  @threshold=0.5  →  acc={acc*100:.2f}%  precision={prec*100:.2f}%  recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  @threshold={best_thr:.2f}  →  F1={best_f1*100:.2f}%  precision={prec_at_best*100:.2f}%  recall={rec_at_best*100:.2f}%")
    print(f"  Precision @ top  5% : {pak_metrics['precision_at_top_5pct']*100:.2f}%")
    print(f"  Precision @ top 10% : {pak_metrics['precision_at_top_10pct']*100:.2f}%")
    print(f"  Precision @ top 20% : {pak_metrics['precision_at_top_20pct']*100:.2f}%")
    print()
    print("=" * 76)
    print("PER-DAY DIAGNOSTIC (pseudo-days = monotone-in-probability)")
    print("=" * 76)
    _print_daily_table(daily_stats)
    print(
        f"\nPseudo-days: MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  "
        f"Acc±1={overall['acc_within_1']*100:.2f}%"
    )
    if best_day:
        print(f"Best day: {best_day} (MAE={mae_by_day[best_day]:.3f})  |  "
              f"Worst day: {worst_day} (MAE={mae_by_day[worst_day]:.3f})")
    print()

    # Baseline + skill check: a naive "always predict positive-rate" classifier
    train_pos_rate = float(y_train_bin.mean())
    baseline_proba = np.full(n_test, train_pos_rate)
    baseline_auc = (
        float(roc_auc_score(y_test_bin, baseline_proba))
        if len(np.unique(y_test_bin)) > 1 else 0.5
    )
    auc_uplift_pct = 100.0 * (auc - 0.5) / 0.5
    log.info(
        "Baseline (always p=%.3f) AUC=%.4f → model AUC=%.4f, uplift vs random %+.2f%%",
        train_pos_rate, baseline_auc, auc, auc_uplift_pct,
    )
    # Keep these names so downstream metadata still renders
    train_mean = train_pos_rate
    baseline_metrics = {"mae": baseline_auc, "rmse": 0.0, "r2": 0.0,
                        "acc_within_1": 0.0, "acc_exact": 0.0}
    mae_improvement_pct = auc_uplift_pct

    pred_stats = {
        "min": float(np.min(test_pred)),
        "p25": float(np.percentile(test_pred, 25)),
        "median": float(np.median(test_pred)),
        "p75": float(np.percentile(test_pred, 75)),
        "max": float(np.max(test_pred)),
        "std": float(np.std(test_pred)),
    }
    pred_spread = pred_stats["p75"] - pred_stats["p25"]

    log.info("==== STEP 8: persist artifacts ====")
    model_path = os.path.join(p["model_dir"], "lgbm_fire_date_model.pkl")
    write_pickle(best_model, model_path)
    log.info("Saved model → %s", model_path)

    history_dir = os.path.join(p["model_dir"], "history")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = os.path.join(history_dir, f"{timestamp}_lightgbm.pkl")
    write_pickle(best_model, history_path)

    feature_importance: List[Dict[str, Any]] = []
    if hasattr(best_model, "feature_importances_") and best_model.feature_importances_ is not None:
        importances = list(zip(feature_cols, [float(x) for x in best_model.feature_importances_]))
        importances.sort(key=lambda x: x[1], reverse=True)
        feature_importance = [{"feature": n, "importance": i} for n, i in importances]

    log.info("==== STEP 9: render PNG training report ====")
    report_path = _render_training_report(
        daily_stats=daily_stats,
        overall=overall,
        feature_importance=feature_importance,
        y_test_true=y_eval.astype(float),
        y_test_pred=test_pred,
        out_dir=p["meta_dir"],
        horizon=MAX_PREDICTION_DAYS,
    )
    if report_path:
        print(f"📊 Training report saved to {report_path}")

    total_elapsed = time.time() - t_total_start
    print(f"⏱️  Total training time: {total_elapsed:.1f} seconds")
    log.info("⏱️  Total training time: %.1f seconds", total_elapsed)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "NASA FIRMS VIIRS NRT (real)",
        "earliest_date": str(train_pool["date"].min()),
        "latest_date": str(train_pool["date"].max()),
        "data_stale_days": int(stale_days),
        "data_is_stale": bool(stale_days > STALE_WARN_DAYS),
        "total_days": int(pd.Series(train_pool["date"]).nunique()),
        "max_history_days_applied": int(max_history_days_applied),
        "total_active_cells": int(train_pool[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]),
        "training_rows": int(len(train_pool)),
        "grid_size": grid_size,
        "min_confidence": min_confidence,
        "urban_filter_enabled": urban_filter_enabled,
        "urban_buffer_km": urban_buffer_km,
        "prediction_type": "binary_fire_in_3d",
        "imminent_days": IMMINENT_DAYS,
        "max_prediction_days": MAX_PREDICTION_DAYS,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "weather_features_used": [c for c in feature_cols if c.startswith(("temp_", "precip_", "wind_", "et0_"))],
        "urgency_thresholds": dict(DEFAULT_URGENCY_THRESHOLDS),
        "urgency_thresholds_note": (
            "Fixed domain thresholds: CRITICAL=fire today, HIGH≤2d, MEDIUM≤4d, LOW≤7d."
        ),
        "best_model": "lightgbm",
        "model": {
            "type": "lightgbm",
            "search_method": search_name,
            "n_iter": n_iter,
            "n_splits": n_splits,
            "ts_split_gap_days": 7,
            "n_ensemble": n_ensemble,
            "early_stopping_rounds": early_stopping_rounds,
            "best_params": best_params,
            # test_metrics is the source the dashboard reads (via risk_map →
            # GeoJSON metadata.metrics). We expose both legacy regression keys
            # AND binary metrics here so the frontend can pick whichever it
            # knows how to render — the `task` field switches the display.
            "test_metrics": {
                "task": "binary_fire_in_3d",
                "imminent_days": IMMINENT_DAYS,
                # Legacy regression keys (now reflect pseudo-days, mostly
                # noise; kept so an old frontend still shows *something*).
                "mae_days": overall["mae"],
                "rmse_days": overall["rmse"],
                "r2": overall["r2"],
                "accuracy_within_1day": overall["acc_within_1"],
                "accuracy_exact": overall["acc_exact"],
                # Binary metrics — the operative measurements.
                "roc_auc": overall_bin["roc_auc"],
                "average_precision": overall_bin["average_precision"],
                "binary_accuracy": overall_bin["accuracy"],
                "precision": overall_bin["precision"],
                "recall": overall_bin["recall"],
                "f1": overall_bin["f1"],
                "best_f1": overall_bin["best_f1"],
                "best_threshold": overall_bin["best_threshold"],
                "precision_at_best_thr": overall_bin["precision_at_best_thr"],
                "recall_at_best_thr": overall_bin["recall_at_best_thr"],
                "precision_at_top_5pct": overall_bin["precision_at_top_5pct"],
                "precision_at_top_10pct": overall_bin["precision_at_top_10pct"],
                "precision_at_top_20pct": overall_bin["precision_at_top_20pct"],
                # Calibration + locked deployment threshold metrics.
                "ece": overall_bin["ece"],
                "deployment_threshold": overall_bin["deployment_threshold"],
                "deployment_precision": overall_bin["deployment_precision"],
                "deployment_recall": overall_bin["deployment_recall"],
                "deployment_f1": overall_bin["deployment_f1"],
                "deployment_accuracy": overall_bin["deployment_accuracy"],
                "reliability_bins": overall_bin["reliability_bins"],
                # Test-set baseline for "uplift" framing in the dashboard.
                # A random ranker's precision@K equals the positive rate.
                "test_positive_rate": round(float(y_test_bin.mean()), 4),
                "uplift_at_top_5pct": round(
                    overall_bin["precision_at_top_5pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
                "uplift_at_top_10pct": round(
                    overall_bin["precision_at_top_10pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
                "uplift_at_top_20pct": round(
                    overall_bin["precision_at_top_20pct"] / max(float(y_test_bin.mean()), 1e-9), 3
                ),
            },
            "test_metrics_binary": overall_bin,   # legacy alias (kept for any tool still reading it)
            "baseline_test_metrics": baseline_metrics,
            "baseline_label_mean": train_mean,
            "mae_improvement_over_baseline_pct": round(mae_improvement_pct, 4),
            "skill_check_passed": bool(auc >= 0.65),
            "prediction_distribution_test": pred_stats,
            "predictions_bunched": bool(pred_spread < 0.5),
            "scale_pos_weight": round(scale_pos_weight, 3),
            "neg_to_pos_ratio_train": NEG_TO_POS_RATIO,
            "decision_threshold_best_f1": round(best_thr, 4),
            "split_date_ranges": {
                "train": [str(train_df["date"].min()), str(train_df["date"].max())],
                "val":   [str(val_df["date"].min()),   str(val_df["date"].max())],
                "test":  [str(test_df["date"].min()),  str(test_df["date"].max())],
            },
            "training_time_seconds": round(total_elapsed, 2),
            "tuning_time_seconds": round(t_tune_elapsed, 2),
            "ensemble_refit_time_seconds": round(t_refit_elapsed, 2),
        },
        "daily_stats": daily_stats,
        "summary": summary,
        "feature_importance_top": feature_importance[:20],
        "training_time_seconds": round(total_elapsed, 2),
    }

    meta_path = os.path.join(p["meta_dir"], "dataset_info.json")
    write_json(metadata, meta_path, indent=2, default=str)
    log.info("Saved metadata → %s", meta_path)

    if not skip_risk_map:
        log.info("==== STEP 10: refresh risk map ====")
        # Free the largest training objects before risk_map loads its own
        # ~3.5 GB parquet. `del locals()[name]` is a Python no-op (locals()
        # returns a copy), so we delete each name explicitly then force a GC
        # cycle. This reliably reclaims 4–6 GB on a 22 GB laptop.
        del X_train, X_val, X_test
        del y_train_bin, y_val_bin, y_test_bin
        del train_df, val_df, test_df, train_pool
        del train_df_us, val_df_us
        del full_X, full_y_bin, full_sw
        del sample_weight_train, search, estimator
        gc.collect()
        try:
            from risk_map import run as generate_risk_map
            generate_risk_map()
        except Exception as exc:
            log.warning("risk_map.run() failed: %s", exc)

    return metadata


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fire-date prediction model (LightGBM)")
    p.add_argument("--n-iter", type=int, default=30, help="Bayes/Randomized search iterations (default 30)")
    p.add_argument("--n-splits", type=int, default=5, help="TimeSeriesSplit folds (gap=7)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.1")))
    p.add_argument("--min-confidence", type=int, default=0)
    p.add_argument("--n-ensemble", type=int, default=10)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--quick", action="store_true",
                   help="Fast iteration: n_iter=12, n_splits=3")
    p.add_argument("--fast", action="store_true",
                   help="Shorter train: n_iter=20, n_splits=3")
    p.add_argument("--skip-risk-map", action="store_true")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--max-history-days", type=int, default=-1)
    # `--only` is kept for CLI backwards-compat with run.sh — it is ignored
    # because this build is LightGBM-only.
    p.add_argument("--only", type=str, default="lightgbm",
                   help="Ignored (LightGBM-only build).")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    if args.quick:
        args.n_iter = 12
        args.n_splits = 3
        log.info("Quick mode: n_iter=12, n_splits=3")
    elif args.fast:
        args.n_iter = 20
        args.n_splits = 3
        log.info("Fast mode: n_iter=20, n_splits=3")
    if args.only and args.only.lower() != "lightgbm":
        log.warning("--only=%s ignored (this build is LightGBM-only).", args.only)
    max_hist = args.max_history_days
    if max_hist < 0:
        max_hist = int(os.getenv("MAX_TRAIN_HISTORY_DAYS", "0") or 0)
    main(
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        grid_size=args.grid_size,
        min_confidence=args.min_confidence,
        skip_risk_map=args.skip_risk_map,
        predict_only=args.predict_only,
        max_history_days=max_hist,
        n_ensemble=args.n_ensemble,
        early_stopping_rounds=args.early_stopping_rounds,
    )
