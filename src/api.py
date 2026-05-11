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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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


# How many days of history to keep resident in RAM. The API's deepest reach
# back is `_historical_counts` over the last 30 days; a 60-day buffer leaves
# headroom without paying the multi-GB cost of holding the full 4M-row
# training frame in memory.
API_FEATURE_WINDOW_DAYS = int(os.environ.get("API_FEATURE_WINDOW_DAYS", "60"))


def _load_features_recent(feature_path: str) -> pd.DataFrame:
    """Load just the recent slice of the features parquet.

    The training parquet (~4M rows × 80+ cols) blows up to multiple GB when
    fully materialised as a pandas DataFrame, which OOMs container memory
    on small Railway plans. We use pyarrow's row-group predicate pushdown
    to read only rows in the last ``API_FEATURE_WINDOW_DAYS`` days — enough
    for the latest snapshot + the 30-day historical-count window.

    Falls back to a full read for CSV inputs (no pushdown available there).
    """
    if not feature_path.lower().endswith(".parquet"):
        df = read_table(feature_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    import pyarrow.parquet as pq
    from datetime import date

    # Find the latest date by reading just the date column (cheap — parquet
    # is column-oriented). Casting to pyarrow.compute would be even cheaper
    # but pandas → date keeps the code small and works for date32 / string.
    date_only = pq.read_table(feature_path, columns=["date"]).column("date")
    latest_ts = pd.to_datetime(date_only.to_pandas()).max()
    latest: date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    cutoff = latest - timedelta(days=API_FEATURE_WINDOW_DAYS)

    table = pq.read_table(feature_path, filters=[("date", ">=", cutoff)])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


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

    df = _load_features_recent(feature_path)
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


# =====================================
# STATIC SPA SERVING
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