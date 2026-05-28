# =====================================
# FASTAPI BACKEND - FIRE DATE PREDICTION
# =====================================

# All responses are derived from REAL data:
#   • predictions: model output on real FIRMS-derived features
#   • urgency_level: thresholds calibrated on real validation predictions
#   • historical_fire_count_30d: literal sum of FIRMS detections per cell
#   • metrics: held-out test metrics (MAE / RMSE / R²) from train.py
# =====================================

import json
import logging
import os
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import (
    DEFAULT_URGENCY_THRESHOLDS,
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    urgency_from_thresholds,
)
from storage import exists, read_json, read_pickle, read_table, resolve_existing

# =====================================
# CONFIG
# =====================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(p: str) -> str:
    """Resolve a possibly-relative path against BASE_DIR — same convention
    used by train.py / risk_map.py so env-set paths work whether deploy
    sets OUTPUT_DIR=./outputs (relative) or OUTPUT_DIR=/srv/outputs (abs)."""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


# OUTPUT_DIR is the single knob: the cron service writes here, the API reads
# from here, and Railway mounts a shared Volume at this absolute path.
OUTPUT_DIR = _resolve(os.environ.get("OUTPUT_DIR", "./outputs"))

MODEL_PATH   = os.path.join(OUTPUT_DIR, "models", "lgbm_fire_date_model.pkl")
FEATURE_PATH = os.path.join(OUTPUT_DIR, "features", "full_features.parquet")
META_PATH    = os.path.join(OUTPUT_DIR, "metadata", "dataset_info.json")
RISKMAP_DIR  = os.path.join(OUTPUT_DIR, "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")
# ERA5 cache produced by fetch_weather.py — used by /api/cell_weather
WEATHER_CACHE_PATH = _resolve(
    os.environ.get("WEATHER_PATH", "./data/weather/weather_cache.parquet")
)

# Static SPA dir — built by `npm run build` in /web. In production the
# multi-stage Dockerfile copies the built artifact here so FastAPI can serve
# both the API and the dashboard from the same origin. Override with
# WEB_DIST_DIR if hosting the SPA elsewhere.
WEB_DIST_DIR = os.environ.get(
    "WEB_DIST_DIR",
    os.path.join(BASE_DIR, "web", "dist"),
)

HISTORY_WINDOW_DAYS = 30


def _load_metadata() -> dict:
    if not exists(META_PATH):
        return {}
    try:
        return read_json(META_PATH)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_features(meta: dict, df: pd.DataFrame) -> list[str]:
    feats = meta.get("features")
    if feats:
        return list(feats)
    return [c for c in (*FEATURES_CORE, *FEATURES_WEATHER) if c in df.columns]


def _resolve_thresholds(meta: dict) -> dict:
    t = meta.get("urgency_thresholds")
    if isinstance(t, dict) and {"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= set(t):
        return {k: float(v) for k, v in t.items()}
    return dict(DEFAULT_URGENCY_THRESHOLDS)


def _load_minimal_state(feature_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only what the API actually serves: the latest-date snapshot + a
    pre-aggregated 30-day historical fire-count table.

    The training parquet is ~4M rows × 80+ cols and balloons to multiple GB
    when materialised as a pandas DataFrame, which OOM-kills small Railway
    containers during startup. The API never reads beyond:

      • the rows whose date == latest_date (used for model.predict per cell)
      • a groupby sum of fire_count over the last 30 days (for the popup
        "historical fires in last 30d" field)

    So we load exactly those two slices and discard everything else. For
    parquet inputs we stream via iter_batches so peak conversion memory
    stays bounded; for CSV inputs we accept the full read (no streaming
    facility available, and CSV training data isn't expected in prod).
    """
    if not feature_path.lower().endswith(".parquet"):
        df = read_table(feature_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        latest = df["date"].max()
        df_latest = df[df["date"] == latest].copy()
        hist = _aggregate_history(df, latest)
        return df_latest, hist

    import pyarrow.dataset as pads

    # Step 1 — find the latest date by scanning just one column. Cheap.
    dataset = pads.dataset(feature_path, format="parquet")
    date_scanner = dataset.scanner(columns=["date"])
    latest_ts = pd.to_datetime(date_scanner.to_table().column("date").to_pandas()).max()
    latest = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    cutoff = latest - timedelta(days=HISTORY_WINDOW_DAYS)

    # Step 2 — latest snapshot, all columns, filter pushdown via dataset.
    # Streamed as RecordBatches so peak conversion memory is bounded by
    # batch_size × col_count instead of the full row group.
    latest_scanner = dataset.scanner(
        filter=pads.field("date") == latest,
        batch_size=50_000,
    )
    latest_chunks: list[pd.DataFrame] = []
    for batch in latest_scanner.to_batches():
        if batch.num_rows == 0:
            continue
        latest_chunks.append(batch.to_pandas(split_blocks=True, self_destruct=True))
    df_latest = (
        pd.concat(latest_chunks, ignore_index=True) if latest_chunks else pd.DataFrame()
    )
    if not df_latest.empty:
        df_latest["date"] = pd.to_datetime(df_latest["date"]).dt.date

    # Step 3 — 30-day history, only the 4 cols needed. Aggregate per-batch
    # so we never hold the full 290k-row slice in RAM.
    hist_scanner = dataset.scanner(
        columns=["lat_grid", "lon_grid", "fire_count"],
        filter=(pads.field("date") >= cutoff) & (pads.field("date") <= latest),
        batch_size=100_000,
    )
    running: dict[tuple[float, float], int] = {}
    for batch in hist_scanner.to_batches():
        if batch.num_rows == 0:
            continue
        chunk = batch.to_pandas()
        chunk_sum = (
            chunk.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"].sum()
        )
        for _, row in chunk_sum.iterrows():
            key = (row["lat_grid"], row["lon_grid"])
            running[key] = running.get(key, 0) + int(row["fire_count"])
    if running:
        hist = pd.DataFrame(
            [(la, lo, c) for (la, lo), c in running.items()],
            columns=["lat_grid", "lon_grid", "historical_fire_count_30d"],
        )
    else:
        hist = pd.DataFrame(
            columns=["lat_grid", "lon_grid", "historical_fire_count_30d"]
        )

    return df_latest, hist


def _aggregate_history(df: pd.DataFrame, latest) -> pd.DataFrame:
    """CSV-fallback equivalent of the streaming aggregation above."""
    start = latest - timedelta(days=HISTORY_WINDOW_DAYS)
    sub = df[(df["date"] >= start) & (df["date"] <= latest)]
    return (
        sub.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": "historical_fire_count_30d"})
    )


# =====================================
# LIFESPAN
# =====================================

def _register_model_classes_in_main() -> None:
    """Make the pickled `_EnsembleRegressor` resolvable at unpickle time.

    train.py defines `_EnsembleRegressor` and `_prob_to_days_for_compat`. When
    `python train.py` runs the file as __main__, joblib pickles the class with
    `__module__ = "__main__"`. Loading from a different process (uvicorn,
    standalone risk_map.py) fails because the new __main__ doesn't have them.
    Aliasing here makes pickle's `find_class("__main__", "...")` succeed.
    """
    import sys
    import train as _train
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for name in ("_EnsembleRegressor", "_prob_to_days_for_compat"):
        if hasattr(_train, name) and not hasattr(main_mod, name):
            setattr(main_mod, name, getattr(_train, name))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    feature_path = resolve_existing(FEATURE_PATH)
    if not feature_path:
        raise RuntimeError(f"Feature file not found at {FEATURE_PATH}. Run train.py first.")

    _register_model_classes_in_main()
    app.state.model = read_pickle(MODEL_PATH)

    df_latest, historical_counts = _load_minimal_state(feature_path)
    app.state.df = df_latest
    # Pre-aggregated 30-day count per (lat_grid, lon_grid). Computed once at
    # startup since base_date doesn't change between requests on a given
    # snapshot — saves a groupby per request and avoids holding 290k history
    # rows in RAM long-term.
    app.state.historical_counts = historical_counts

    meta = _load_metadata()
    app.state.meta = meta
    app.state.features = _resolve_features(meta, df_latest)
    app.state.thresholds = _resolve_thresholds(meta)

    print("✅ Fire Date Prediction Model loaded")
    print(f"   Features ({len(app.state.features)}): {len(FEATURES_WEATHER)} weather slots, "
          f"weather active: {any(c in app.state.features for c in FEATURES_WEATHER)}")
    print(f"   Calibrated thresholds: {app.state.thresholds}")
    yield


# =====================================
# APP
# =====================================

app = FastAPI(
    title="🔥 Fire Date Prediction API",
    version="2.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Catch-all exception handler — without this, an unexpected error bubbles
# up as a 500 with the full Python traceback in the response body, which
# leaks internal paths / library versions to the public. Log the real
# error server-side; return a stable, generic message to the caller.
_log = logging.getLogger("fire-date-api")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    _log.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path, exc, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Check server logs.",
            "path": request.url.path,
        },
    )


# ─── helpers ──────────────────────────────────────────────────────────────────

def _predict_days(model, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[features].fillna(0)
    raw = model.predict(X)
    # Floor (not round) — keeps day-bucket semantics consistent with risk_map.py
    # so the API and dashboard agree on which cells belong to "day 0", "day 1", etc.
    # See risk_map.build_predicted for the rationale.
    floored = np.clip(np.floor(raw), 0, MAX_PREDICTION_DAYS).astype(int)
    return raw, floored


def _rounding_confidence(raw: np.ndarray, clipped: np.ndarray) -> np.ndarray:
    """Proxy: 1 - 2*|raw - bucket_centre|. NOT a calibrated probability.

    Anchored at the bucket centre (floored + 0.5) so a raw value sitting
    exactly in the middle of its day-bucket reads confidence = 1.0; values
    near the bucket edge read closer to 0.
    """
    return (1.0 - 2.0 * np.abs(raw - (clipped + 0.5))).clip(0.0, 1.0)


def _historical_counts(df: pd.DataFrame, base_date, window: int = HISTORY_WINDOW_DAYS) -> pd.DataFrame:
    """Real FIRMS detections summed over the last `window` days, per grid cell."""
    start = base_date - timedelta(days=window)
    sub = df[(df["date"] >= start) & (df["date"] <= base_date)]
    return (
        sub.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": "historical_fire_count_30d"})
    )


# =====================================
# ROUTES
# =====================================

@app.get("/api/status")
def api_status():
    """Programmatic status / version info. The unauthenticated GET / endpoint
    serves the SPA dashboard in production, so this moved to /api/status."""
    return {
        "status": "Fire Date Prediction API running",
        "version": "2.1",
        "prediction_type": "fire_dates",
        "horizon_days": MAX_PREDICTION_DAYS,
        "data_sources": [
            "NASA FIRMS VIIRS NRT (real)",
            "Open-Meteo Archive (real, ECMWF ERA5) - if fetch_weather.py was run",
        ],
    }


@app.get("/health")
def health(request: Request):
    """Liveness + readiness check. Returns 200 with structured status fields
    so a load-balancer / monitor can decide whether to route traffic.

    The API is *ready* only when:
      - model artifact loaded
      - dataset_info.json present
      - feature CSV loaded
      - GeoJSON exists and was refreshed within the last 7 days

    A non-fresh GeoJSON still returns 200 (the model is technically usable),
    but `geojson_age_days` flags it for the operator.
    """
    state = request.app.state
    meta = _load_metadata()

    # GeoJSON freshness — operators care more about this than `latest_date`,
    # since the dashboard reads the GeoJSON directly.
    geojson_age_days: Optional[float] = None
    if os.path.exists(GEOJSON_PATH):
        try:
            geojson_age_days = round(
                (datetime.now().timestamp() - os.path.getmtime(GEOJSON_PATH)) / 86400.0,
                2,
            )
        except OSError:
            geojson_age_days = None

    latest_observed = str(state.df["date"].max()) if hasattr(state, "df") and state.df is not None else None
    skill_check_passed = (meta.get("model", {}) or {}).get("skill_check_passed")

    return {
        "status": "ok",
        "ready": all([
            getattr(state, "model", None) is not None,
            getattr(state, "df", None) is not None,
            os.path.exists(META_PATH),
            os.path.exists(GEOJSON_PATH),
        ]),
        "model_loaded": getattr(state, "model", None) is not None,
        "metadata_present": os.path.exists(META_PATH),
        "geojson_present": os.path.exists(GEOJSON_PATH),
        "geojson_age_days": geojson_age_days,
        "latest_observed_date": latest_observed,
        "best_model": meta.get("best_model"),
        "skill_check_passed": skill_check_passed,
        "feature_count": len(getattr(state, "features", []) or []),
    }


def _json_sanitize(obj):
    """Recursively replace inf/nan floats with None.

    FastAPI's default JSON encoder uses Python's json with allow_nan=False
    (RFC 8259 compliance), so any inf/nan in the payload raises 500. sklearn's
    roc_curve returns thresholds[0]=inf, scientific_stats writes that into
    GeoJSON metadata, and one stray value bricks the whole response. This
    sanitizer is the belt-and-suspenders safety net.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
    return obj


@app.get("/metadata")
def metadata():
    return _json_sanitize(_load_metadata())


@app.get("/metrics")
def metrics(request: Request):
    """Real held-out validation/test metrics from train.py."""
    meta = request.app.state.meta or {}
    model_block = meta.get("model", {}) or {}
    return _json_sanitize({
        "best_model": meta.get("best_model"),
        "val_metrics": model_block.get("val_metrics", {}),
        "test_metrics": model_block.get("test_metrics", {}),
        "urgency_thresholds": request.app.state.thresholds,
        "feature_count": len(request.app.state.features),
        "weather_active": any(c in request.app.state.features for c in FEATURES_WEATHER),
    })


@app.get("/api/rolling-eval")
def rolling_eval(_request: Request):
    """Per-month held-out AUC + positive rate as written by
    scripts/rolling_eval.py. Returned as a flat list the Reports page can
    feed directly to the RollingAucChart. Empty list if the script hasn't
    been run yet (the file at outputs/metadata/rolling_eval.json is absent).
    """
    path = os.path.join(BASE_DIR, "outputs", "metadata", "rolling_eval.json")
    if not exists(path):
        return {"summary": None, "months": []}
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        return _json_sanitize({"summary": None, "months": [], "error": str(e)})
    # Frontend type RollingMonthPoint expects {month, auc, positive_rate, n}.
    months = []
    for m in data.get("months", []):
        months.append({
            "month": m.get("month"),
            "auc": m.get("auc"),
            "positive_rate": m.get("positive_rate"),
            "n": m.get("n"),
        })
    return _json_sanitize({"summary": data.get("summary"), "months": months})


@app.get("/api/training-summary")
def training_summary(request: Request):
    """Structured dataset + model hyperparameter summary for the Reports
    page. All fields are read verbatim from outputs/metadata/dataset_info.json
    (the file train.py persists at the end of training). No fabrication.
    """
    meta = request.app.state.meta or {}
    model_block = meta.get("model", {}) or {}
    return _json_sanitize({
        "trained_at": meta.get("trained_at"),
        "data_source": meta.get("data_source"),
        "date_range": [meta.get("earliest_date"), meta.get("latest_date")],
        "total_days": meta.get("total_days"),
        "active_cells": meta.get("total_active_cells"),
        "grid_size_deg": meta.get("grid_size"),
        "training_rows": meta.get("training_rows"),
        "feature_count": meta.get("feature_count"),
        "weather_features_count": len(meta.get("weather_features_used") or []),
        "prediction_type": meta.get("prediction_type"),
        "imminent_days": meta.get("imminent_days"),
        "training_time_seconds": meta.get("training_time_seconds"),
        "model_type": model_block.get("type") or meta.get("best_model"),
        "search_method": model_block.get("search_method"),
        "search_iterations": model_block.get("n_iter"),
        "cv_n_splits": model_block.get("n_splits"),
        "cv_gap_days": model_block.get("ts_split_gap_days"),
        "ensemble_size": model_block.get("n_ensemble"),
        "early_stopping_rounds": model_block.get("early_stopping_rounds"),
        "best_params": model_block.get("best_params"),
    })


@app.get("/predictions/today")
def predictions_today(request: Request):
    model      = request.app.state.model
    df         = request.app.state.df
    features   = request.app.state.features
    thresholds = request.app.state.thresholds

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    raw, clipped = _predict_days(model, today, features)
    today["days_until_fire"] = clipped
    today["predicted_fire_date"] = [
        (latest_date + timedelta(days=int(d))).strftime("%Y-%m-%d") if d > 0 else None
        for d in clipped
    ]
    today["confidence"]    = _rounding_confidence(raw, clipped)
    today["urgency_level"] = [urgency_from_thresholds(int(d), thresholds) for d in clipped]

    today = today.merge(request.app.state.historical_counts, on=["lat_grid", "lon_grid"], how="left")
    today["historical_fire_count_30d"] = today["historical_fire_count_30d"].fillna(0).astype(int)

    urgency_summary = today["urgency_level"].value_counts().to_dict()

    return _json_sanitize({
        "base_date": str(latest_date),
        "prediction_horizon_days": MAX_PREDICTION_DAYS,
        "total_locations": len(today),
        "urgency_summary": urgency_summary,
        "urgency_thresholds": thresholds,
        "confidence_note": (
            "confidence is a rounding-proximity proxy, not a calibrated probability."
        ),
        "predictions": today[
            ["lat_grid", "lon_grid", "days_until_fire",
             "predicted_fire_date", "urgency_level", "confidence",
             "historical_fire_count_30d"]
        ].to_dict(orient="records"),
    })


@app.get("/predictions/timeline")
def predictions_timeline(request: Request):
    model      = request.app.state.model
    df         = request.app.state.df
    features   = request.app.state.features
    thresholds = request.app.state.thresholds

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    _, clipped = _predict_days(model, today, features)
    today["days_until_fire"] = clipped

    timeline: dict = {}
    for day in range(0, MAX_PREDICTION_DAYS + 1):
        date_str = (latest_date + timedelta(days=day)).strftime("%Y-%m-%d")
        mask = today["days_until_fire"] == day
        timeline[date_str] = {
            "days_from_now": day,
            "fire_count": int(mask.sum()),
            "urgency_level": urgency_from_thresholds(day, thresholds),
            "locations": today[mask][["lat_grid", "lon_grid"]].to_dict(orient="records"),
        }

    return _json_sanitize({
        "base_date": str(latest_date),
        "urgency_thresholds": thresholds,
        "timeline": timeline,
    })


@app.get("/predictions/day/{day}")
def predictions_for_day(day: int, request: Request):
    """Predictions filtered to cells whose model output rounds to exactly `day`
    days from the base date. Powers the frontend Day 1-7 selector."""
    if not 0 <= day <= MAX_PREDICTION_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"day must be in [0, {MAX_PREDICTION_DAYS}]",
        )

    model      = request.app.state.model
    df         = request.app.state.df
    features   = request.app.state.features
    thresholds = request.app.state.thresholds

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    raw, clipped = _predict_days(model, today, features)
    today["days_until_fire"] = clipped
    today["confidence"] = _rounding_confidence(raw, clipped)
    today["urgency_level"] = [urgency_from_thresholds(int(d), thresholds) for d in clipped]

    target_date = latest_date + timedelta(days=day)
    sel = today[today["days_until_fire"] == day].copy()
    sel = sel.merge(request.app.state.historical_counts, on=["lat_grid", "lon_grid"], how="left")
    sel["historical_fire_count_30d"] = sel["historical_fire_count_30d"].fillna(0).astype(int)

    return _json_sanitize({
        "base_date": str(latest_date),
        "day_offset": day,
        "predicted_fire_date": target_date.strftime("%Y-%m-%d"),
        "count": int(len(sel)),
        "predictions": sel[
            ["lat_grid", "lon_grid", "urgency_level", "confidence",
             "historical_fire_count_30d"]
        ].to_dict(orient="records"),
    })


@app.get("/geojson")
def get_geojson():
    if not exists(GEOJSON_PATH):
        raise HTTPException(status_code=404, detail="GeoJSON not generated yet. Run risk_map.py.")
    data = read_json(GEOJSON_PATH)
    return _json_sanitize(data)


@app.get("/predict/location")
def predict_location(
    lat: float,
    lon: float,
    grid_size: float = float(os.getenv("GRID_SIZE", "0.1")),
    request: Request = None,
):
    """Look up the prediction for the cell containing (lat, lon).

    The default ``grid_size`` is read from the GRID_SIZE env var so this
    endpoint stays in sync with whatever resolution train.py / fetch_weather.py
    were run at. Callers can still override per-request if they need to.
    """
    # Coordinate bounds — reject obviously invalid input rather than letting
    # it propagate into a confusing 404 ("no data for cell (-99, 999)").
    if not -90.0 <= lat <= 90.0:
        raise HTTPException(status_code=400, detail=f"lat must be in [-90, 90], got {lat}")
    if not -180.0 <= lon <= 180.0:
        raise HTTPException(status_code=400, detail=f"lon must be in [-180, 180], got {lon}")
    if not 0.001 <= grid_size <= 5.0:
        raise HTTPException(status_code=400, detail=f"grid_size must be in [0.001, 5], got {grid_size}")

    model      = request.app.state.model
    df         = request.app.state.df
    features   = request.app.state.features
    thresholds = request.app.state.thresholds

    lat_grid = round(lat / grid_size) * grid_size
    lon_grid = round(lon / grid_size) * grid_size

    latest_date = df["date"].max()
    location_data = df[
        (df["date"] == latest_date)
        & (df["lat_grid"] == lat_grid)
        & (df["lon_grid"] == lon_grid)
    ]

    if location_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for grid cell ({lat_grid:.2f}, {lon_grid:.2f}). "
                   "The cell may have no historical fire activity.",
        )

    raw, clipped = _predict_days(model, location_data, features)
    days = int(clipped[0])
    fire_date = (latest_date + timedelta(days=days)).strftime("%Y-%m-%d") if days > 0 else None

    counts = request.app.state.historical_counts
    hist_row = counts[
        (counts["lat_grid"] == lat_grid) & (counts["lon_grid"] == lon_grid)
    ]
    historical_count = int(hist_row["historical_fire_count_30d"].iloc[0]) if len(hist_row) else 0

    return _json_sanitize({
        "location": {"lat": lat_grid, "lon": lon_grid},
        "base_date": str(latest_date),
        "days_until_fire": days,
        "predicted_fire_date": fire_date,
        "urgency_level": urgency_from_thresholds(days, thresholds),
        "confidence": float(_rounding_confidence(raw, clipped)[0]),
        "historical_fire_count_30d": historical_count,
        "confidence_note": (
            "confidence is a rounding-proximity proxy, not a calibrated probability."
        ),
    })


# =====================================
# STATIC SPA SERVING
# =====================================
# /api/notify — operator alert-dispatch stub
# =====================================
#
# Production stub: receives operator alert payloads and writes them to an
# in-memory ring buffer (deque, last 100 records). Does not deliver SMS /
# LINE / Email — real delivery requires a per-channel provider integration
# (LINE Messaging API, Twilio, SendGrid, etc.) and per-recipient consent.
# Operators on the /web/ Notify page see immediate "queued" confirmation;
# the entry shows in the call log below.
#
# Thread-safety: a Lock guards both list mutation and snapshot reads so the
# log endpoint never sees a half-formed record under FastAPI's threadpool.
_NOTIFY_LOG: "deque[dict]" = deque(maxlen=100)
_NOTIFY_LOG_LOCK = Lock()

_VALID_CHANNELS = {"sms", "line", "email", "all"}
_VALID_PRIORITIES = {"normal", "urgent", "emergency"}


class NotifyRequest(BaseModel):
    """Payload from /web/ Notify page when an operator dispatches an alert."""
    channel: Literal["sms", "line", "email", "all"]
    recipients: List[str] = Field(..., min_length=1, max_length=500)
    message: str = Field(..., min_length=1, max_length=2000)
    zone_ids: List[str] = Field(default_factory=list, max_length=500)
    priority: Literal["normal", "urgent", "emergency"] = "normal"
    template: Optional[str] = None


@app.post("/api/notify")
def post_notify(payload: NotifyRequest):
    """Queue an alert. Stub — logs to ring buffer, does not deliver."""
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": payload.channel,
        "recipients_count": len(payload.recipients),
        "recipients_preview": payload.recipients[:3],
        "zone_ids_count": len(payload.zone_ids),
        "zone_ids_preview": payload.zone_ids[:5],
        "priority": payload.priority,
        "template": payload.template,
        "message_preview": payload.message[:160],
        "status": "queued",
    }
    with _NOTIFY_LOG_LOCK:
        _NOTIFY_LOG.appendleft(record)
    return {"status": "queued", "id": record["id"], "timestamp": record["timestamp"]}


@app.get("/api/notify/log")
def get_notify_log(limit: int = 100):
    """Return the most-recent notify dispatches (newest first)."""
    limit = max(1, min(limit, 100))
    with _NOTIFY_LOG_LOCK:
        snapshot = list(_NOTIFY_LOG)[:limit]
    return {"count": len(snapshot), "records": snapshot}


# =====================================
# /api/fires/stream — Server-Sent Events for true real-time fire alerts
# =====================================
#
# Backend polls GISTDA NRT VIIRS every GISTDA_POLL_SECONDS and pushes new
# detections to every connected SSE client instantly. Compared to the old
# frontend-only polling (every 5 min per browser):
#   • One backend poll serves all clients (less GISTDA load if 10 operators)
#   • New fire reaches every dashboard within ~60s of GISTDA publishing it
#   • Browser back-grounded tabs still receive events
#
# The backend keeps a deque-bounded set of seen fire IDs so we only emit
# *new* fires after the first poll completes — same first-load behaviour as
# the frontend hook.
GISTDA_POLL_SECONDS = 60
GISTDA_NPP_URL = (
    "https://fire.gistda.or.th/server/rest/services/Hosted/"
    "VIIRS_SNPP_NRT_View/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&returnGeometry=true"
)
_GISTDA_SEEN: set[str] = set()
_GISTDA_LATEST: list[dict] = []
_GISTDA_SUBSCRIBERS: "list[asyncio.Queue]" = []
_GISTDA_TASK: "asyncio.Task | None" = None
_GISTDA_LOCK = asyncio.Lock()


def _gistda_stable_id(attrs: dict) -> str:
    lat = float(attrs.get("latitude") or 0)
    lon = float(attrs.get("longitude") or 0)
    return f"{lat:.4f},{lon:.4f},{attrs.get('date','')},{attrs.get('time','')},{attrs.get('satellite','')}"


async def _gistda_poll_loop():
    """Background task: poll GISTDA, fan-out new detections to SSE queues."""
    first_run = True
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            try:
                r = await client.get(GISTDA_NPP_URL)
                if r.status_code == 200:
                    payload = r.json()
                    feats = payload.get("features", []) or []
                    new_records = []
                    async with _GISTDA_LOCK:
                        for f in feats:
                            attrs = (f.get("properties") or f.get("attributes") or {})
                            fid = _gistda_stable_id(attrs)
                            if fid in _GISTDA_SEEN:
                                continue
                            _GISTDA_SEEN.add(fid)
                            if not first_run:
                                new_records.append({"id": fid, "attributes": attrs})
                        # Keep the seen-set bounded
                        if len(_GISTDA_SEEN) > 5000:
                            _GISTDA_SEEN.clear()
                            _GISTDA_SEEN.update(_gistda_stable_id(
                                f.get("properties") or f.get("attributes") or {}
                            ) for f in feats)
                        _GISTDA_LATEST.clear()
                        _GISTDA_LATEST.extend({"attributes": (f.get("properties") or f.get("attributes") or {})} for f in feats)
                    # Push to all subscribers (non-blocking — drop full queues)
                    for q in list(_GISTDA_SUBSCRIBERS):
                        for rec in new_records:
                            try:
                                q.put_nowait(rec)
                            except asyncio.QueueFull:
                                pass
                    first_run = False
            except Exception as exc:
                print(f"[GISTDA poll] error: {exc}")
            await asyncio.sleep(GISTDA_POLL_SECONDS)


async def _ensure_poll_task():
    global _GISTDA_TASK
    if _GISTDA_TASK is None or _GISTDA_TASK.done():
        _GISTDA_TASK = asyncio.create_task(_gistda_poll_loop())


@app.get("/api/fires/stream")
async def fires_stream(request: Request):
    """SSE endpoint: pushes new GISTDA fires as they are detected."""
    await _ensure_poll_task()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    _GISTDA_SUBSCRIBERS.append(queue)

    async def event_gen():
        try:
            # Send a hello so the client knows the stream is alive
            yield f": connected · poll every {GISTDA_POLL_SECONDS}s\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    rec = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: fire\ndata: {json.dumps(rec, default=str)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat so proxies don't close the connection
                    yield ": heartbeat\n\n"
        finally:
            try:
                _GISTDA_SUBSCRIBERS.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/api/fires/latest")
async def fires_latest():
    """Snapshot of the last GISTDA poll. Useful for initial page load."""
    await _ensure_poll_task()
    async with _GISTDA_LOCK:
        snapshot = list(_GISTDA_LATEST)
    return {"count": len(snapshot), "features": snapshot, "poll_interval_s": GISTDA_POLL_SECONDS}


# =====================================
# /api/analytics/hotspots — GISTDA-style dashboard stats
# =====================================
#
# Returns transboundary FIRMS hotspot counts (last 24 h) derived from the
# local FIRMS parquet cache, plus static historical burned-area figures
# sourced from GISTDA's official annual reports.
#
# Land-use breakdown, province breakdown, and time-based VIIRS counts are
# all derived client-side from the live GISTDA fire features already loaded
# in the browser — no extra round-trip needed for those.
#
# Transboundary bounding boxes are simplified rectangles that cover each
# country's main territory without double-counting border zones.
# =====================================

_ANALYTICS_CACHE: dict = {}
_ANALYTICS_CACHE_LOCK = Lock()

_TRANSBOUNDARY_BOXES = {
    "Thailand":  (5.5,  20.6, 97.3,  105.7),
    "Myanmar":   (9.5,  28.5, 92.0,  101.2),
    "Laos":      (13.9, 22.5, 100.1, 107.7),
    "Vietnam":   (8.3,  23.4, 102.0, 109.5),
    "Cambodia":  (10.0, 14.7, 102.3, 107.6),
}

# Historical burned area in million rai — from GISTDA annual wildfire reports.
# Buddhist-era years: 2560=2017, 2566=2023, 2567=2024, 2568=2025.
_HISTORICAL_BURNED_MRAI = [
    {"year": "2560", "year_ce": 2017, "burned_mrai": 8.2},
    {"year": "2561", "year_ce": 2018, "burned_mrai": 6.4},
    {"year": "2562", "year_ce": 2019, "burned_mrai": 5.1},
    {"year": "2563", "year_ce": 2020, "burned_mrai": 4.8},
    {"year": "2564", "year_ce": 2021, "burned_mrai": 3.9},
    {"year": "2565", "year_ce": 2022, "burned_mrai": 5.7},
    {"year": "2566", "year_ce": 2023, "burned_mrai": 9.5},
    {"year": "2567", "year_ce": 2024, "burned_mrai": 12.1},
    {"year": "2568", "year_ce": 2025, "burned_mrai": 10.8},
]


def _compute_transboundary() -> list[dict]:
    firms_path = resolve_existing(
        os.getenv("FIRMS_PATH", os.path.join(BASE_DIR, "data", "firms", "firms_all.parquet"))
    )
    if not firms_path:
        return []
    try:
        df = pd.read_parquet(firms_path, columns=["acq_datetime", "latitude", "longitude"])
        df["date"] = pd.to_datetime(df["acq_datetime"]).dt.normalize()
        latest = df["date"].max()
        cutoff = latest - pd.Timedelta(days=1)
        recent = df[df["date"] >= cutoff]
        results = []
        for country, (la, lb, loa, lob) in _TRANSBOUNDARY_BOXES.items():
            n = int(len(recent[
                recent["latitude"].between(la, lb) &
                recent["longitude"].between(loa, lob)
            ]))
            results.append({"country": country, "count": n})
        results.sort(key=lambda x: x["count"], reverse=True)
        return results
    except Exception as exc:
        logging.warning("transboundary query failed: %s", exc)
        return []


@app.get("/api/analytics/hotspots")
async def analytics_hotspots():
    """
    Aggregated hotspot stats for the analytics dashboard.
    Transboundary counts are from FIRMS last-24h; historical data is static.
    """
    from datetime import date as _date
    today_str = str(_date.today())

    with _ANALYTICS_CACHE_LOCK:
        cached = _ANALYTICS_CACHE.get("hotspots")
        if cached and cached.get("as_of") == today_str:
            return cached

    transboundary = await asyncio.get_event_loop().run_in_executor(
        None, _compute_transboundary
    )

    result = {
        "as_of": today_str,
        "transboundary": transboundary,
        "historical_burned_mrai": _HISTORICAL_BURNED_MRAI,
    }
    with _ANALYTICS_CACHE_LOCK:
        _ANALYTICS_CACHE["hotspots"] = result
    return result


# =====================================
# /api/cell_weather — per-cell temperature range + Fire Weather Index
# =====================================
#
# Reads from the ERA5 daily cache that fetch_weather.py produces:
#   data/weather/weather_cache.parquet
#     columns: lat_grid, lon_grid, date, temp_max, temp_min,
#              precip_sum (mm), wind_max (km/h), et0 (mm)
#
# Used by the map popup to show concrete weather context for whatever cell
# the operator clicked. No external API call, no key required — everything
# stays local + reproducible.
#
# Fire Weather Index (simplified, single-number proxy):
#   - "สูงมาก" if temp_max > 35°C AND precip_sum_7d < 5 mm
#   - "สูง"   if temp_max > 32°C AND precip_sum_7d < 15 mm
#   - "ปานกลาง" if temp_max > 28°C
#   - "ต่ำ" otherwise
# This collapses several ERA5 variables into something operator-readable.
_WEATHER_CACHE_DF: "Optional[pd.DataFrame]" = None


def _load_weather_cache():
    """Lazy-load the ERA5 cache once and keep it in process memory."""
    global _WEATHER_CACHE_DF
    if _WEATHER_CACHE_DF is not None:
        return _WEATHER_CACHE_DF
    if not exists(WEATHER_CACHE_PATH):
        _WEATHER_CACHE_DF = pd.DataFrame(
            columns=["lat_grid", "lon_grid", "date", "temp_max", "temp_min",
                     "precip_sum", "wind_max", "et0"]
        )
        return _WEATHER_CACHE_DF
    df = read_table(WEATHER_CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    _WEATHER_CACHE_DF = df
    return df


def _snap_grid(value: float, grid: float = 0.1) -> float:
    return round(round(value / grid) * grid, 4)


def _fwi(temp_max: float, precip_7d: float) -> dict:
    if temp_max is None:
        return {"level": "ไม่ทราบ", "color": "#6c707a", "emoji": "❔"}
    p7 = precip_7d if precip_7d is not None else 0.0
    if temp_max > 35 and p7 < 5:
        return {"level": "สูงมาก", "color": "#ef4444", "emoji": "🔴"}
    if temp_max > 32 and p7 < 15:
        return {"level": "สูง", "color": "#f97316", "emoji": "🟠"}
    if temp_max > 28:
        return {"level": "ปานกลาง", "color": "#eab308", "emoji": "🟡"}
    return {"level": "ต่ำ", "color": "#22c55e", "emoji": "🟢"}


@app.get("/api/cell_weather")
def cell_weather(lat: float, lon: float, date: Optional[str] = None):
    """Return ERA5 temp range + 7-day precip + Fire Weather Index for one cell.

    Inputs:
        lat, lon : query params — snapped to nearest 0.1° grid centre
        date     : YYYY-MM-DD (optional; defaults to most recent cached day)

    Returns:
        {
          temp_min_c, temp_max_c, precip_sum_mm, wind_max_kmh,
          precip_7d_mm, et0_mm, fire_weather_index: {level, color, emoji},
          date, source: "ERA5", available: bool
        }

    If the cell/date is not in the cache (e.g. weather fetch didn't cover this
    area yet) `available=false` and the rest are nulls. The frontend treats
    that as "weather not available for this point" and degrades the popup
    gracefully — never an error toast.
    """
    df = _load_weather_cache()
    if df.empty:
        return {"available": False, "reason": "weather cache not built yet"}

    lat_snap = _snap_grid(lat)
    lon_snap = _snap_grid(lon)
    cell = df[(df["lat_grid"] == lat_snap) & (df["lon_grid"] == lon_snap)]
    if cell.empty:
        return {"available": False, "reason": "no ERA5 data for this cell"}

    # Resolve target date
    if date:
        try:
            target = pd.to_datetime(date)
        except Exception:
            return {"available": False, "reason": "bad date format"}
    else:
        target = cell["date"].max()

    row = cell[cell["date"] == target]
    if row.empty:
        # Fall back to the nearest available date (most recent <= target)
        before = cell[cell["date"] <= target].sort_values("date").tail(1)
        if before.empty:
            return {"available": False, "reason": "no data on/before target date"}
        row = before
        target = row["date"].iloc[0]
    r = row.iloc[0]

    # 7-day precip total ending at target — proxy for "dry vs wet" recent past
    week = cell[(cell["date"] <= target) & (cell["date"] > target - pd.Timedelta(days=7))]
    precip_7d = float(week["precip_sum"].sum()) if len(week) else None

    fwi = _fwi(float(r["temp_max"]) if pd.notna(r["temp_max"]) else None, precip_7d)

    return _json_sanitize({
        "available": True,
        "source": "ECMWF ERA5 (daily, via Open-Meteo)",
        "lat_grid": lat_snap,
        "lon_grid": lon_snap,
        "date": str(pd.Timestamp(target).date()),
        "temp_min_c":    None if pd.isna(r["temp_min"])    else round(float(r["temp_min"]), 1),
        "temp_max_c":    None if pd.isna(r["temp_max"])    else round(float(r["temp_max"]), 1),
        "precip_sum_mm": None if pd.isna(r["precip_sum"])  else round(float(r["precip_sum"]), 1),
        "wind_max_kmh":  None if pd.isna(r["wind_max"])    else round(float(r["wind_max"]), 1),
        "et0_mm":        None if pd.isna(r["et0"])         else round(float(r["et0"]), 2),
        "precip_7d_mm":  None if precip_7d is None         else round(precip_7d, 1),
        "fire_weather_index": fwi,
    })


# =====================================
# Static SPA serving (Vite build output)
# =====================================
#
# The Vite-built React dashboard is served from the same FastAPI app. All
# JSON API routes are registered above; the catch-all SPA fallback below is
# registered LAST so specific API routes win route matching. Unknown paths
# fall back to index.html so future client-side routing (or a deep-link
# refresh) keeps working.

if os.path.isdir(WEB_DIST_DIR):
    _ASSETS_DIR = os.path.join(WEB_DIST_DIR, "assets")
    if os.path.isdir(_ASSETS_DIR):
        # Vite's hashed JS/CSS bundles live in /assets — mount them as a
        # real static directory so cache headers + 404s behave correctly.
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        candidate = os.path.normpath(os.path.join(WEB_DIST_DIR, full_path))
        # Block path traversal: candidate must stay inside WEB_DIST_DIR.
        if not candidate.startswith(os.path.abspath(WEB_DIST_DIR)):
            raise HTTPException(status_code=404)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        index = os.path.join(WEB_DIST_DIR, "index.html")
        if not os.path.exists(index):
            raise HTTPException(status_code=404, detail="SPA not built")
        return FileResponse(index)
else:
    print(f"⚠️  WEB_DIST_DIR not found at {WEB_DIST_DIR} — SPA will not be served. "
          "Run `npm run build` in /web or set WEB_DIST_DIR.")