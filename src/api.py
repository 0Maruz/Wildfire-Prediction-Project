# =====================================
# FASTAPI BACKEND - FIRE DATE PREDICTION
# =====================================
#
# All responses are derived from REAL data:
#   • predictions: model output on real FIRMS-derived features
#   • urgency_level: thresholds calibrated on real validation predictions
#   • historical_fire_count_30d: literal sum of FIRMS detections per cell
#   • metrics: held-out test metrics (MAE / RMSE / R²) from train.py
# =====================================

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from features import (
    DEFAULT_URGENCY_THRESHOLDS,
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    urgency_from_thresholds,
)
from io_utils import read_table, resolve_existing

# =====================================
# CONFIG
# =====================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH   = os.path.join(BASE_DIR, "outputs", "models", "lgbm_fire_date_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "outputs", "features", "full_features.parquet")
META_PATH    = os.path.join(BASE_DIR, "outputs", "metadata", "dataset_info.json")
RISKMAP_DIR  = os.path.join(BASE_DIR, "outputs", "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")

HISTORY_WINDOW_DAYS = 30


def _load_metadata() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
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


# =====================================
# LIFESPAN
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    feature_path = resolve_existing(FEATURE_PATH)
    if not feature_path:
        raise RuntimeError(f"Feature file not found at {FEATURE_PATH}. Run train.py first.")

    app.state.model = joblib.load(MODEL_PATH)

    df = read_table(feature_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    app.state.df = df

    meta = _load_metadata()
    app.state.meta = meta
    app.state.features = _resolve_features(meta, df)
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
import logging
import traceback
from fastapi.responses import JSONResponse

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

@app.get("/")
def root():
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


@app.get("/metadata")
def metadata():
    return _load_metadata()


@app.get("/metrics")
def metrics(request: Request):
    """Real held-out validation/test metrics from train.py."""
    meta = request.app.state.meta or {}
    model_block = meta.get("model", {}) or {}
    return {
        "best_model": meta.get("best_model"),
        "val_metrics": model_block.get("val_metrics", {}),
        "test_metrics": model_block.get("test_metrics", {}),
        "urgency_thresholds": request.app.state.thresholds,
        "feature_count": len(request.app.state.features),
        "weather_active": any(c in request.app.state.features for c in FEATURES_WEATHER),
    }


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

    counts = _historical_counts(df, latest_date)
    today = today.merge(counts, on=["lat_grid", "lon_grid"], how="left")
    today["historical_fire_count_30d"] = today["historical_fire_count_30d"].fillna(0).astype(int)

    urgency_summary = today["urgency_level"].value_counts().to_dict()

    return {
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
    }


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

    return {
        "base_date": str(latest_date),
        "urgency_thresholds": thresholds,
        "timeline": timeline,
    }


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
    counts = _historical_counts(df, latest_date)
    sel = sel.merge(counts, on=["lat_grid", "lon_grid"], how="left")
    sel["historical_fire_count_30d"] = sel["historical_fire_count_30d"].fillna(0).astype(int)

    return {
        "base_date": str(latest_date),
        "day_offset": day,
        "predicted_fire_date": target_date.strftime("%Y-%m-%d"),
        "count": int(len(sel)),
        "predictions": sel[
            ["lat_grid", "lon_grid", "urgency_level", "confidence",
             "historical_fire_count_30d"]
        ].to_dict(orient="records"),
    }


@app.get("/geojson")
def get_geojson():
    if not os.path.exists(GEOJSON_PATH):
        raise HTTPException(status_code=404, detail="GeoJSON not generated yet. Run risk_map.py.")
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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

    counts = _historical_counts(df, latest_date)
    hist_row = counts[
        (counts["lat_grid"] == lat_grid) & (counts["lon_grid"] == lon_grid)
    ]
    historical_count = int(hist_row["historical_fire_count_30d"].iloc[0]) if len(hist_row) else 0

    return {
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
    }