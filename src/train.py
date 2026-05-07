"""End-to-end training orchestrator: data → features → tune → eval → persist.

Pipeline:
    1. data_loader.load_and_prepare → daily cell-day frame
    2. features.build_features → lag/rolling/calendar features + label
    3. Drop training-invalid rows (label = -1, no fire within horizon)
    4. Chronological train / val / test split
       - train : first 60 %
       - val   : next  20 %  (used for model selection only)
       - test  : last  20 %  (held-out; never seen during tuning or selection)
    5. model.select_best → tune RF / LightGBM / XGBoost on TimeSeriesSplit,
       pick best val MAE
    6. Refit best model on train+val, evaluate on held-out test set
    7. Persist best model, feature CSV, and metadata
    8. Trigger risk_map.run() to refresh the GeoJSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import load_and_prepare
from features import (
    DEFAULT_URGENCY_THRESHOLDS,  # FIX: BUG 3 — fixed domain thresholds replace quantile calibration
    MAX_PREDICTION_DAYS,
    build_features,
    resolve_features,
)
from io_utils import resolve_existing, write_table
from model import evaluate, select_best

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("train")


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    """Resolve an env-supplied path against the project root if it's relative."""
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
        "output_dir": output_dir,
        "model_dir": os.path.join(output_dir, "models"),
        "feature_dir": os.path.join(output_dir, "features"),
        "meta_dir": os.path.join(output_dir, "metadata"),
    }


def chronological_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way chronological split so no future data leaks into training.

    Returns:
        train  – first ``(1 - val_fraction - test_fraction)`` of rows by date
        val    – next ``val_fraction`` rows  → used for model selection
        test   – last ``test_fraction`` rows → held-out final evaluation
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    sorted_df = df.sort_values("date").reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * (1.0 - val_fraction - test_fraction))
    val_end = int(n * (1.0 - test_fraction))

    train = sorted_df.iloc[:train_end]
    val = sorted_df.iloc[train_end:val_end]
    test = sorted_df.iloc[val_end:]

    # FIX: BUG 4 — explicit split-date ranges + label distribution per slice so
    # the train/val/test seasonal mismatch (train pre-Nov, test in peak burn
    # season) is visible in the log instead of being hidden behind a one-line
    # summary. Makes the apparently-higher test R² interpretable.
    log.info("Split date ranges:")
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split):
            log.info(
                "  %-5s : %s → %s  (%d rows)",
                name,
                split["date"].min(),
                split["date"].max(),
                len(split),
            )
        else:
            log.info("  %-5s : empty", name)

    if "days_until_fire" in sorted_df.columns:
        for name, split in (("train", train), ("val", val), ("test", test)):
            if not len(split):
                continue
            dist = split["days_until_fire"].value_counts().sort_index()
            log.info("  %s label dist:\n%s", name, dist.to_string())

    return train, val, test


def main(
    n_iter: int = 20,
    n_splits: int = 5,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    grid_size: float = 0.1,
    # Empirically, min_confidence=30 hurt MAE (-1.8 %) and acc±1 (-2.2 pp)
    # because VIIRS "low" doesn't mean "false detection" — it just means the
    # surrounding pixels were ambiguous. Dropping those rows costs ~1.3 % of
    # training data and the model loses more signal than noise. Stay at 0.
    min_confidence: int = 0,
    # Default skips random_forest: it has consistently lost to xgboost on val
    # MAE while accounting for ~95% of total tuning time (200+ deep trees fit
    # sequentially). Pass `--only random_forest,lightgbm,xgboost` to reinstate.
    only: Optional[Tuple[str, ...]] = ("lightgbm", "xgboost"),
    skip_risk_map: bool = False,
    random_state: int = 42,
) -> dict:
    load_dotenv()
    p = _paths()
    for d in (p["model_dir"], p["feature_dir"], p["meta_dir"]):
        os.makedirs(d, exist_ok=True)

    log.info("==== STEP 1: load + grid raw FIRMS data ====")
    weather_path = resolve_existing(p["weather_path"])
    if weather_path:
        log.info("Real ERA5 weather cache detected → %s", weather_path)
    else:
        log.info("No weather cache at %s — training without weather features.", p["weather_path"])

    urban_filter_enabled = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
    urban_buffer_km = float(os.getenv("URBAN_BUFFER_KM", "0.0"))
    log.info(
        "Urban filter: enabled=%s, buffer_km=%.1f (consistent with risk_map.py)",
        urban_filter_enabled, urban_buffer_km,
    )

    daily = load_and_prepare(
        raw_dir=p["raw_dir"],
        firms_path=p["firms_path"],
        grid_size=grid_size,
        min_confidence=min_confidence,
        densify=True,
        weather_path=weather_path,
        filter_urban=urban_filter_enabled,
        urban_buffer_km=urban_buffer_km,
    )

    # Stale-data check: warn if FIRMS hasn't been refreshed recently. The
    # model can still train on old data, but inference on stale state means
    # the dashboard's predictions are for a window that has already passed.
    latest_observed = pd.to_datetime(daily["date"]).max()
    stale_days = (pd.Timestamp.utcnow().normalize().tz_localize(None) - latest_observed).days
    STALE_WARN_DAYS = int(os.getenv("STALE_WARN_DAYS", "5"))
    if stale_days > STALE_WARN_DAYS:
        log.warning(
            "⚠️  FIRMS DATA STALE: latest observation is %s (%d days behind today). "
            "Predictions will be for a window starting %d days ago. "
            "Run `python fetch_firms.py --days 5` (or `./run.sh --fresh`) to refresh.",
            latest_observed.date(), stale_days, stale_days,
        )

    # Auto-drop weather features when coverage is too sparse to be useful.
    # With <20% coverage the lag/roll weather features are mostly NaN→0 fill,
    # which adds 30 noise columns that drag the model's MAE without contributing
    # signal. Better to train without weather than to train with broken weather.
    WEATHER_COLS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]
    MIN_WEATHER_COVERAGE = float(os.getenv("MIN_WEATHER_COVERAGE", "0.20"))
    weather_cols_present = [c for c in WEATHER_COLS if c in daily.columns]
    if weather_cols_present:
        coverage = daily[weather_cols_present[0]].notna().mean()
        if coverage < MIN_WEATHER_COVERAGE:
            log.warning(
                "Weather coverage %.1f%% < %.0f%% threshold — dropping weather "
                "columns to avoid feeding noise to the model. Run fetch_weather.py "
                "to completion (set MIN_WEATHER_COVERAGE in .env to override).",
                coverage * 100, MIN_WEATHER_COVERAGE * 100,
            )
            daily = daily.drop(columns=weather_cols_present)
        else:
            log.info(
                "Weather coverage %.1f%% (≥%.0f%% threshold) — keeping weather features.",
                coverage * 100, MIN_WEATHER_COVERAGE * 100,
            )

    log.info("==== STEP 2: feature engineering ====")
    feats = build_features(daily, horizon=MAX_PREDICTION_DAYS, grid_size=grid_size)

    feature_path = os.path.join(p["feature_dir"], "full_features.parquet")
    write_table(feats, feature_path)
    log.info("Saved feature dataset → %s", feature_path)

    log.info("==== STEP 3: filter to labelled rows ====")
    train_pool = feats[feats["days_until_fire"] >= 0].copy()
    if train_pool.empty:
        raise RuntimeError(
            "No labelled rows. Either no fires were observed within the horizon, "
            "or the dataset is too short. Fetch more days."
        )
    log.info(
        "Label distribution:\n%s",
        train_pool["days_until_fire"].value_counts().sort_index().to_string(),
    )

    log.info("==== STEP 4: chronological train / val / test split ====")
    train_df, val_df, test_df = chronological_split(
        train_pool,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty — increase data or adjust fractions")
    if len(test_df) == 0:
        raise RuntimeError("Test split is empty — increase data or adjust fractions")

    feature_cols = resolve_features(train_pool)
    log.info("Using %d features (weather present: %s)",
             len(feature_cols),
             any(c.startswith(("temp_", "precip_", "wind_", "et0_")) for c in feature_cols))
    X_train, y_train = train_df[feature_cols], train_df["days_until_fire"]
    X_val,   y_val   = val_df[feature_cols],   val_df["days_until_fire"]
    X_test,  y_test  = test_df[feature_cols],  test_df["days_until_fire"]

    log.info("==== STEP 5: tune candidate models on TimeSeriesSplit ====")
    selection = select_best(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        horizon=MAX_PREDICTION_DAYS,
        n_iter=n_iter,
        n_splits=n_splits,
        random_state=random_state,
        only=only,
    )
    best_name = selection["best_name"]
    best_model = selection["best_model"]

    # ── STEP 6 ──────────────────────────────────────────────────────────────
    # Evaluate on the held-out test set BEFORE refitting so that test metrics
    # are genuinely unseen.  Then refit on train+val to maximise the data the
    # deployed model learns from.
    # ────────────────────────────────────────────────────────────────────────
    log.info("==== STEP 6: held-out test evaluation then refit on train+val ====")

    # 6a. Test evaluation (model was selected on val, never saw test)
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test.to_numpy(), test_pred, horizon=MAX_PREDICTION_DAYS)
    log.info("Test metrics (held-out): %s", test_metrics)

    # Honest comparison against a "predict the train mean" baseline. If model
    # MAE ≈ baseline MAE, the model is essentially predicting the prior — a
    # 1.52-day MAE looks impressive in isolation but means nothing when a
    # constant predictor lands at 1.5 days too. This makes that visible.
    train_mean = float(y_train.mean())
    baseline_pred = np.full(len(y_test), train_mean)
    baseline_metrics = evaluate(
        y_test.to_numpy(), baseline_pred, horizon=MAX_PREDICTION_DAYS,
    )
    log.info(
        "Baseline (predict train mean=%.2f) test metrics: %s", train_mean, baseline_metrics,
    )
    mae_improvement = baseline_metrics["mae_days"] - test_metrics["mae_days"]
    mae_improvement_pct = (
        100.0 * mae_improvement / baseline_metrics["mae_days"]
        if baseline_metrics["mae_days"] > 0 else 0.0
    )
    log.info(
        "Model improves MAE by %+.4f days (%+.2f%%) over predict-mean baseline.",
        mae_improvement, mae_improvement_pct,
    )
    if mae_improvement_pct < 5.0:
        log.warning(
            "⚠️  MODEL SKILL CHECK FAILED: model only beats baseline by %.2f%% "
            "(< 5%% threshold). Likely causes: (1) features lack signal, "
            "(2) target framing too hard, (3) data quality issue. Review "
            "feature_importance_top + label distribution before deploying.",
            mae_improvement_pct,
        )

    # Sanity-check the prediction distribution. A useful regressor produces
    # outputs with non-trivial spread; if predictions collapse to a narrow
    # band, the dashboard's tier-coloring will be effectively single-color
    # and the model is regressing to the mean.
    pred_stats = {
        "min": float(np.min(test_pred)),
        "p25": float(np.percentile(test_pred, 25)),
        "median": float(np.median(test_pred)),
        "p75": float(np.percentile(test_pred, 75)),
        "max": float(np.max(test_pred)),
        "std": float(np.std(test_pred)),
    }
    pred_spread = pred_stats["p75"] - pred_stats["p25"]
    log.info("Prediction distribution on test: %s", pred_stats)
    if pred_spread < 0.5:
        log.warning(
            "⚠️  PREDICTIONS BUNCHED: IQR = %.2f days (< 0.5 threshold). "
            "Most cells will fall in the same urgency tier on the dashboard. "
            "Model is essentially predicting the prior — try richer features.",
            pred_spread,
        )

    # 6b. Use fixed-domain urgency thresholds.  # FIX: BUG 3 — quantile calibration
    # forced every active cell into a tier (NONE=0); fixed cutoffs make NONE
    # meaningful again and align the pipeline with operator-facing semantics.
    urgency_thresholds = dict(DEFAULT_URGENCY_THRESHOLDS)
    log.info(
        "Urgency thresholds (fixed domain): CRITICAL=0d, HIGH≤2d, MEDIUM≤4d, LOW≤7d"
    )

    # 6c. Refit on train + val to produce the final deployed artefact
    full_X = pd.concat([X_train, X_val])
    full_y = pd.concat([y_train, y_val])
    best_model.fit(full_X, full_y)
    log.info(
        "Refit complete on %d rows (train + val). "
        "Final artefact will NOT be evaluated again on val to avoid leakage.",
        len(full_X),
    )

    log.info("==== STEP 7: persist artifacts ====")
    model_path = os.path.join(p["model_dir"], "lgbm_fire_date_model.pkl")
    joblib.dump(best_model, model_path)
    log.info("Saved model → %s", model_path)

    # Versioned snapshot for rollback / A/B testing. Filename includes the
    # UTC timestamp + best model name so it's obvious which artefact is
    # which without opening dataset_info.json.
    history_dir = os.path.join(p["model_dir"], "history")
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = os.path.join(history_dir, f"{timestamp}_{best_name}.pkl")
    joblib.dump(best_model, history_path)
    log.info("Versioned snapshot → %s", history_path)

    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        importances = list(zip(feature_cols, [float(x) for x in best_model.feature_importances_]))
        importances.sort(key=lambda x: x[1], reverse=True)
        feature_importance = [{"feature": n, "importance": i} for n, i in importances]

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "NASA FIRMS VIIRS NRT (real)",
        "earliest_date": str(feats["date"].min()),
        "latest_date": str(feats["date"].max()),
        "data_stale_days": int(stale_days),
        "data_is_stale": bool(stale_days > STALE_WARN_DAYS),
        "total_days": int(pd.Series(feats["date"]).nunique()),
        "total_active_cells": int(feats[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]),
        "training_rows": int(len(train_pool)),
        "grid_size": grid_size,
        "min_confidence": min_confidence,
        "urban_filter_enabled": urban_filter_enabled,
        "urban_buffer_km": urban_buffer_km,
        "prediction_type": "fire_date",
        "max_prediction_days": MAX_PREDICTION_DAYS,
        "features": feature_cols,
        "weather_features_used": [
            c for c in feature_cols
            if c.startswith(("temp_", "precip_", "wind_", "et0_"))
        ],
        "urgency_thresholds": urgency_thresholds,
        "urgency_thresholds_note": (
            # FIX: BUG 3 — switched from quantile calibration to fixed domain cutoffs
            "Fixed domain thresholds: CRITICAL=fire today, HIGH≤2d, MEDIUM≤4d, "
            "LOW≤7d, NONE>7d. Quantile calibration was removed because it forced "
            "every active cell into a tier and made NONE structurally impossible."
        ),
        "best_model": best_name,
        # val metrics = used for model selection (before refit)
        # test metrics = genuinely held-out, never used in any fitting decision
        "model": {
            "type": best_name,
            "val_metrics": selection["all_results"][best_name]["metrics"],
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
            "baseline_label_mean": train_mean,
            "mae_improvement_over_baseline_pct": mae_improvement_pct,
            "skill_check_passed": mae_improvement_pct >= 5.0,
            "prediction_distribution_test": pred_stats,
            "predictions_bunched": pred_spread < 0.5,
            "best_params": selection["all_results"][best_name]["best_params"],
            # FIX: BUG 4 — persist split date ranges so downstream consumers can
            # interpret val/test metric gaps in light of seasonal coverage.
            "split_date_ranges": {
                "train": [str(train_df["date"].min()), str(train_df["date"].max())],
                "val":   [str(val_df["date"].min()),   str(val_df["date"].max())],
                "test":  [str(test_df["date"].min()),  str(test_df["date"].max())],
            },
            "note": (
                "val_metrics: model selection on 20% val split. "
                "test_metrics: held-out 20% test split, evaluated before refit. "
                "Deployed artefact is refit on train+val."
            ),
        },
        "all_candidates": selection["all_results"],
        "feature_importance_top": feature_importance[:20],
    }

    meta_path = os.path.join(p["meta_dir"], "dataset_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved metadata → %s", meta_path)

    if not skip_risk_map:
        log.info("==== STEP 8: refresh risk map ====")
        try:
            from risk_map import run as generate_risk_map

            generate_risk_map()
        except Exception as exc:
            log.warning("risk_map.run() failed: %s", exc)

    return metadata


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fire-date prediction model")
    p.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations")
    p.add_argument("--n-splits", type=int, default=5, help="TimeSeriesSplit folds")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.1")))
    p.add_argument("--min-confidence", type=int, default=0,
                   help="VIIRS confidence floor. 0 (default) keeps everything; "
                        "30 drops 'low' but empirically hurts MAE — only useful "
                        "if you suspect specific false-positive sources.")
    p.add_argument(
        "--only",
        type=str,
        default="lightgbm,xgboost",
        help="Comma-separated subset (default: lightgbm,xgboost — random_forest "
             "is excluded because it loses on val MAE while taking ~95%% of tuning time)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Fast iteration: --n-iter 10 --n-splits 3 --only lightgbm (~3-5 min)",
    )
    p.add_argument("--skip-risk-map", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    if args.quick:
        args.n_iter = 10
        args.n_splits = 3
        args.only = "lightgbm"
        log.info("Quick mode: n_iter=10, n_splits=3, only=lightgbm")
    only = tuple(s.strip() for s in args.only.split(",")) if args.only else None
    main(
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        grid_size=args.grid_size,
        min_confidence=args.min_confidence,
        only=only,
        skip_risk_map=args.skip_risk_map,
    )