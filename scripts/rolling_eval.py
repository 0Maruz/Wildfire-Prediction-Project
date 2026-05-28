#!/usr/bin/env python
"""Rolling monthly evaluation of the deployed model.

Slides a 1-month evaluation window over the entire feature timeline and
records AUC, F1, precision, recall, ECE, and class-positive rate for each
month. Result is written to `outputs/metadata/rolling_eval.json` so the
operator can see *stability* of model quality over time — does the model
hold up off-burn-season? Does AUC drift?

Memory-frugal like post_calibrate.py: predicts probabilities chunk-by-chunk
so the full 4.4M-row feature matrix is never resident in memory.

Run from project root:
    .venv/bin/python scripts/rolling_eval.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from sklearn.metrics import (  # noqa: E402
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

import train  # noqa: E402
from features import resolve_features  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "outputs", "models", "lgbm_fire_date_model.pkl")
FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "full_features.parquet")
OUT_PATH = os.path.join(ROOT, "outputs", "metadata", "rolling_eval.json")
CHUNK_SIZE = 200_000


def _eval_window(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {
            "n": int(len(y)),
            "positive_rate": float(y.mean()) if len(y) else 0.0,
            "auc": 0.0,
            "average_precision": 0.0,
            "ece": 0.0,
            "precision_at_thr": 0.0,
            "recall_at_thr": 0.0,
            "f1_at_thr": 0.0,
            "warning": "single_class",
        }
    pred = (p >= thr).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "average_precision": round(float(average_precision_score(y, p)), 4),
        "ece": round(train.expected_calibration_error(y, p), 4),
        "precision_at_thr": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_at_thr": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1_at_thr": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def main() -> None:
    t0 = time.time()
    print("[1/3] Loading model + scanning parquet...")
    # train.py defines _EnsembleRegressor; when train.py runs as __main__
    # joblib pickles it with __module__ = "__main__". Other entry points
    # (this script, api.py, standalone risk_map.py) need the class
    # aliased into their own __main__ before unpickling.
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None:
        for _name in ("_EnsembleRegressor", "_prob_to_days_for_compat"):
            if hasattr(train, _name) and not hasattr(_main_mod, _name):
                setattr(_main_mod, _name, getattr(train, _name))
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, train._EnsembleRegressor):
        raise SystemExit(f"Not _EnsembleRegressor: {type(model)}")

    pf = pq.ParquetFile(FEATURES_PATH)
    schema_cols = pf.schema_arrow.names
    head = pf.read_row_group(0).to_pandas().head(0)
    feature_cols = resolve_features(head)
    needed = list(set(feature_cols + ["date", "days_until_fire"]))
    needed = [c for c in needed if c in schema_cols]
    f32_cols = [c for c in feature_cols if c not in ("lat_grid", "lon_grid")]

    print("[2/3] Predicting probabilities chunk-by-chunk...")
    rows: List[pd.DataFrame] = []
    n_done = 0
    t_pred = time.time()
    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=needed):
        chunk = batch.to_pandas()
        for c in f32_cols:
            if c in chunk.columns and chunk[c].dtype != np.float32:
                chunk[c] = chunk[c].astype(np.float32)
        X = chunk[feature_cols]
        proba = model.predict_proba(X)
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(chunk["date"].to_numpy()),
            "y_bin": train._make_binary_label(chunk["days_until_fire"]).astype(np.int8),
            "proba": proba.astype(np.float32),
        }))
        n_done += len(chunk)
        del chunk, X, proba, batch
        gc.collect()
        print(f"  {n_done:>9,} rows  ({time.time() - t_pred:.1f}s)", end="\r")
    print()

    df = pd.concat(rows, ignore_index=True)
    del rows
    gc.collect()
    print(f"  {len(df):,} (date, label, proba) rows")

    # Deployment threshold pulled from metadata (best-F1 on full-distribution test)
    meta_path = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
    with open(meta_path) as f:
        meta = json.load(f)
    thr = float(meta["model"]["test_metrics"].get("deployment_threshold", 0.5))
    print(f"  Using deployment threshold = {thr:.2f}")

    print("[3/3] Per-month rolling evaluation...")
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())
    results: List[Dict[str, Any]] = []
    for m in months:
        sub = df[df["month"] == m]
        y = sub["y_bin"].to_numpy()
        p = sub["proba"].to_numpy().astype(np.float64)
        row = {"month": str(m), **_eval_window(y, p, thr)}
        results.append(row)
        warn = f" [{row.get('warning')}]" if row.get("warning") else ""
        print(f"  {m}: n={row['n']:>7,}  pos%={row['positive_rate']*100:5.2f}  "
              f"AUC={row['auc']:.3f}  ECE={row['ece']:.3f}  "
              f"P@thr={row['precision_at_thr']*100:5.1f}  R@thr={row['recall_at_thr']*100:5.1f}{warn}")

    aucs = [r["auc"] for r in results if not r.get("warning")]
    summary = {
        "deployment_threshold": thr,
        "months_evaluated": len(months),
        "valid_months": len(aucs),
        "auc_mean": round(float(np.mean(aucs)), 4) if aucs else 0.0,
        "auc_std":  round(float(np.std(aucs)), 4)  if aucs else 0.0,
        "auc_min":  round(float(np.min(aucs)), 4)  if aucs else 0.0,
        "auc_max":  round(float(np.max(aucs)), 4)  if aucs else 0.0,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }
    print()
    print(f"AUC across {len(aucs)} valid months: "
          f"mean={summary['auc_mean']:.3f}  std={summary['auc_std']:.3f}  "
          f"min={summary['auc_min']:.3f}  max={summary['auc_max']:.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "months": results}, f, indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")

    # Patch dataset_info.json + GeoJSON so the dashboard's Reports page can
    # render the rolling AUC chart without a separate fetch — mirrors what
    # scripts/scientific_stats.py does for its bootstrap CIs.
    inline_months = [
        {"month": r["month"], "auc": r["auc"], "positive_rate": r["positive_rate"], "n": r["n"]}
        for r in results
    ]
    try:
        meta_path = os.path.join(ROOT, "outputs", "metadata", "dataset_info.json")
        with open(meta_path) as f:
            meta = json.load(f)
        tm = meta.setdefault("model", {}).setdefault("test_metrics", {})
        tm["rolling_by_month"] = inline_months
        tm["stability_months"] = summary["months_evaluated"]
        tm["stability_valid_months"] = summary["valid_months"]
        tm["stability_auc_mean"] = summary["auc_mean"]
        tm["stability_auc_std"] = summary["auc_std"]
        tm["stability_auc_min"] = summary["auc_min"]
        tm["stability_auc_max"] = summary["auc_max"]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"✓ Patched {meta_path}")

        gj_path = os.path.join(ROOT, "outputs", "riskmap", "fire_dates_all.geojson")
        if os.path.exists(gj_path):
            with open(gj_path) as f:
                gj = json.load(f)
            gj_metrics = gj.setdefault("metadata", {}).setdefault("metrics", {})
            gj_metrics["rolling_by_month"] = inline_months
            gj_metrics["stability_months"] = summary["months_evaluated"]
            gj_metrics["stability_valid_months"] = summary["valid_months"]
            gj_metrics["stability_auc_mean"] = summary["auc_mean"]
            gj_metrics["stability_auc_std"] = summary["auc_std"]
            gj_metrics["stability_auc_min"] = summary["auc_min"]
            gj_metrics["stability_auc_max"] = summary["auc_max"]
            with open(gj_path, "w") as f:
                json.dump(gj, f)
            print(f"✓ Patched {gj_path}")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠ Could not patch dataset_info/geojson: {exc}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
