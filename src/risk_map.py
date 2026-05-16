# =========================================================
# FIRE DATE MAP GENERATION (OBSERVED + PREDICTED DATES)
# =========================================================
#
# All values written to fire_dates_all.geojson are derived from REAL data:
#   • observed:   NASA FIRMS VIIRS NRT detections (latest densified day)
#   • predicted:  model output on real features for the latest base date
#   • historical_fire_count_30d: literal sum of FIRMS detections in this
#                                grid cell over the 30 days ending at base_date
#   • urgency_level: derived via thresholds calibrated on real validation
#                    predictions (dataset_info.json["urgency_thresholds"])
# =========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Optional
from dotenv import load_dotenv

from features import (
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    DEFAULT_URGENCY_THRESHOLDS,
    calibrate_urgency_thresholds,
    urgency_from_thresholds,
)
from io_utils import read_table
from urban_areas import THAI_URBAN_AREAS, classify_urban
from thailand_boundary import is_in_thailand, find_province

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "lgbm_fire_date_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "outputs", "features", "full_features.parquet")
META_PATH  = os.path.join(BASE_DIR, "outputs", "metadata", "dataset_info.json")

RISKMAP_DIR  = os.path.join(BASE_DIR, "outputs", "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")
LATEST_PATH  = os.path.join(RISKMAP_DIR, "latest.json")

os.makedirs(RISKMAP_DIR, exist_ok=True)

HISTORY_WINDOW_DAYS = 30

# Short-window filter: cells must have ≥ N fires in the last HISTORY_WINDOW_DAYS
# days. Catches "recently active" cells. Default 3 (was 1 — at 1 the dashboard
# showed ~975 Thailand cells, vs GISTDA's ~74 hotspots/day; bumping to 3
# brings the displayed scale closer to ground truth).
MIN_HISTORICAL_FIRES_FOR_DISPLAY = int(os.getenv("MIN_HISTORICAL_FIRES_FOR_DISPLAY", "3"))

# Long-window filter: cells must have ≥ N fires in the last
# LONG_HISTORY_WINDOW_DAYS days. Catches "structurally fire-prone" cells.
# This is the strongest signal you have for "this place actually burns" — a
# cell that's burned 0–2 times in 90 days is almost certainly not going to
# burn next week, regardless of what the model predicts. Default 3 fires in
# 90 days = "this place burns at least once per month on average".
# Set MIN_LONG_HISTORICAL_FIRES=0 to disable the long-window filter.
LONG_HISTORY_WINDOW_DAYS  = int(os.getenv("LONG_HISTORY_WINDOW_DAYS",  "90"))
MIN_LONG_HISTORICAL_FIRES = int(os.getenv("MIN_LONG_HISTORICAL_FIRES", "3"))

# Climatological filter: annualized fire-days per cell, computed across the
# ENTIRE available data record (not a fixed window). Captures the empirical
# base rate of fire activity for each location — the "is this place
# structurally fire-prone?" question that count-based windows can't answer
# cleanly. A cell with 0.5 fire-days/year is essentially never on fire; a
# cell with 5+ fire-days/year is a known hotspot. Default cutoff of 1.0
# fire-day/year is conservative — bump higher to be more selective.
# Set to 0 to disable.
MIN_FIRE_DAYS_PER_YEAR = float(os.getenv("MIN_FIRE_DAYS_PER_YEAR", "3.0"))

# Per-day cap: regression model outputs concentrate predictions in 2-3 day
# bucket (model bunching), so without a cap the dashboard would show ~500
# cells "all firing on day 2" — wildly above GISTDA's observed ~74/day.
# We rank cells within each predicted day by historical_fire_count_30d
# (cells already actively burning are most likely to keep burning) and keep
# only the top MAX_CELLS_PER_DAY. Default 100 ≈ GISTDA scale + headroom.
# Set to 0 to disable.
MAX_CELLS_PER_DAY = int(os.getenv("MAX_CELLS_PER_DAY", "100"))

# Urban-exclusion filter: drop predictions that fall inside (or near) major
# Thai cities. FIRMS detects garbage burning, industrial heat, and other
# non-wildfire heat sources in cities; predicting "wildfire in central
# Bangkok" is misleading. The curated list of urban areas + radii is in
# urban_areas.py — edit it to add/remove cities or adjust their footprints.
#
#   URBAN_FILTER_ENABLED  — set to "false" to disable the filter entirely
#   URBAN_BUFFER_KM       — extra km added beyond each city's radius. 0 =
#                           drop only cells inside the hand-tuned radius;
#                           5–10 = also drop suburban edge cells.
URBAN_FILTER_ENABLED = os.getenv("URBAN_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")
URBAN_BUFFER_KM      = float(os.getenv("URBAN_BUFFER_KM", "0.0"))

# Country filter: drop cells that fall outside Thailand's land border.
# The FIRMS BBOX is rectangular and necessarily includes Myanmar, Laos,
# Cambodia, Vietnam, and northern Malaysia — useful for the model to learn
# regional fire patterns, but the dashboard is Thailand-focused. Default-on.
COUNTRY_FILTER_ENABLED = os.getenv("COUNTRY_FILTER_ENABLED", "true").lower() in ("1", "true", "yes")


# =========================================================
# METADATA HELPERS
# =========================================================

def _load_metadata() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_features(meta: dict, df: pd.DataFrame) -> list:
    """Prefer the persisted feature list (matches the deployed model exactly)."""
    feats = meta.get("features")
    if feats:
        return list(feats)
    # Fall back: any core/weather column that's actually present.
    return [c for c in (*FEATURES_CORE, *FEATURES_WEATHER) if c in df.columns]


def _resolve_thresholds(meta: dict) -> dict:
    t = meta.get("urgency_thresholds")
    if isinstance(t, dict) and {"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= set(t):
        return {k: float(v) for k, v in t.items()}
    return dict(DEFAULT_URGENCY_THRESHOLDS)


# =========================================================
# LOAD MODEL & DATA
# =========================================================

def load_assets():
    model = joblib.load(MODEL_PATH)

    df = read_table(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return model, df


# =========================================================
# OBSERVED LAYER (latest real day from FIRMS)
# =========================================================

def build_observed(df: pd.DataFrame):
    observed_date = df["date"].max()
    obs = df[df["date"] == observed_date].copy()
    return obs, observed_date


# =========================================================
# HISTORICAL FIRE COUNT (real FIRMS detections in last N days)
# =========================================================

def historical_fire_counts(
    df: pd.DataFrame,
    base_date,
    window_days: int = HISTORY_WINDOW_DAYS,
    column_name: str = "historical_fire_count_30d",
) -> pd.DataFrame:
    """Real per-cell sum of fire_count over [base_date-window, base_date].

    No imputation, no extrapolation — just a groupby on the densified FIRMS
    frame. Cells with no detections in the window get exactly 0.

    Args:
        column_name: Name of the output count column. Default matches the
            legacy "historical_fire_count_30d" so existing callers keep working.
    """
    start = base_date - timedelta(days=window_days)
    window = df[(df["date"] >= start) & (df["date"] <= base_date)]
    counts = (
        window.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": column_name})
    )
    counts[column_name] = counts[column_name].astype(int)
    return counts


def historical_fire_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell annualized fire frequency, computed across the full record.

    For each cell we count the number of distinct *fire-days* (days the cell
    had at least one FIRMS detection), then annualize:

        fire_days_per_year = fire_days_total × 365.25 / days_in_record

    Why fire-days instead of total detections? A single intense fire can
    produce dozens of detections in one day — those should count as one
    "fire event," not dozens. Fire-days is a more honest proxy for "how
    often does this cell actually burn?"

    Returns a frame with columns:
        lat_grid, lon_grid, fire_days_total, fire_days_per_year
    """
    if df.empty:
        return pd.DataFrame(columns=["lat_grid", "lon_grid",
                                     "fire_days_total", "fire_days_per_year"])
    days_in_record = (df["date"].max() - df["date"].min()).days + 1
    if days_in_record < 30:
        # Too short to estimate a yearly rate reliably; mark every cell as
        # "rate unknown" with NaN so downstream filtering can fall back to
        # the count-based filters.
        rate = (
            df.groupby(["lat_grid", "lon_grid"])["fire_count"]
            .apply(lambda x: int((x > 0).sum()))
            .reset_index(name="fire_days_total")
        )
        rate["fire_days_per_year"] = float("nan")
        return rate

    rate = (
        df.groupby(["lat_grid", "lon_grid"])["fire_count"]
        .apply(lambda x: int((x > 0).sum()))
        .reset_index(name="fire_days_total")
    )
    rate["fire_days_per_year"] = (
        rate["fire_days_total"] * 365.25 / days_in_record
    )
    return rate


# =========================================================
# PREDICTED LAYER
# =========================================================

def build_predicted(df: pd.DataFrame, model, base_date, meta: dict):
    base = df[df["date"] == base_date].copy()

    feature_cols = _resolve_features(meta, df)
    missing = [c for c in feature_cols if c not in base.columns]
    if missing:
        raise RuntimeError(
            f"Feature CSV is missing {len(missing)} columns expected by the model: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}. Re-run train.py."
        )
    X = base[feature_cols].fillna(0)

    raw_pred = model.predict(X)
    # Floor (not round) to assign each cell to a day bucket. round() pushed
    # the model's narrow prediction band of [~1, ~5] days into days 1-5,
    # leaving day 0 ("today") with 0 cells because the lowest raw_pred ~0.96
    # rounds up to 1. Floor lets a 0.96 prediction land on day 0, which
    # matches the operator's intuition that "≤1 day to fire" = today.
    # Net effect: each bucket shifts forward by ~0.5 day on average and the
    # day selector now exercises all of day 0..4 instead of just day 2..3.
    days_floored = np.clip(np.floor(raw_pred), 0, MAX_PREDICTION_DAYS).astype(int)

    base["raw_prediction"] = raw_pred
    base["days_until_fire"] = days_floored
    base["predicted_fire_date"] = [
        (base_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_floored
    ]

    # Rounding-proximity proxy. Documented as NOT a calibrated probability.
    # Anchored at the bucket centre (days_floored + 0.5) so a raw_pred sitting
    # exactly in the middle of a day reads as confidence = 1.0.
    base["prediction_confidence"] = (
        1.0 - 2.0 * np.abs(raw_pred - (days_floored + 0.5))
    ).clip(0.0, 1.0)

    # Attach real historical fire counts per cell at TWO time scales:
    #   • 30-day count → "is this cell active right now?"
    #   • long-window count (default 90d) → "is this cell structurally fire-prone?"
    # The long-window count is the strongest filter we have for "this place
    # actually burns" — it cuts cells that the model wants to predict on but
    # which haven't seen meaningful fire activity in months.
    counts_30d = historical_fire_counts(
        df, base_date,
        window_days=HISTORY_WINDOW_DAYS,
        column_name="historical_fire_count_30d",
    )
    long_col = f"historical_fire_count_{LONG_HISTORY_WINDOW_DAYS}d"
    counts_long = historical_fire_counts(
        df, base_date,
        window_days=LONG_HISTORY_WINDOW_DAYS,
        column_name=long_col,
    )
    base = base.merge(counts_30d,  on=["lat_grid", "lon_grid"], how="left")
    base = base.merge(counts_long, on=["lat_grid", "lon_grid"], how="left")
    base["historical_fire_count_30d"] = base["historical_fire_count_30d"].fillna(0).astype(int)
    base[long_col] = base[long_col].fillna(0).astype(int)

    # Climatological fire-rate (full-history annualized). This is the
    # "empirical base rate" filter — answers "does this cell actually
    # burn often enough to be worth predicting on?" Cells the model wants
    # to call CRITICAL but that have never historically burned will be
    # dropped here.
    rate = historical_fire_rate(df)
    base = base.merge(rate, on=["lat_grid", "lon_grid"], how="left")
    base["fire_days_total"] = base["fire_days_total"].fillna(0).astype(int)
    base["fire_days_per_year"] = base["fire_days_per_year"].fillna(0.0)

    # Apply ALL filters in sequence. Cells must pass each one to be displayed.
    n_total = len(base)
    if MIN_HISTORICAL_FIRES_FOR_DISPLAY > 0:
        before = len(base)
        base = base[base["historical_fire_count_30d"] >= MIN_HISTORICAL_FIRES_FOR_DISPLAY].copy()
        print(
            f"Short-window  filter: ≥{MIN_HISTORICAL_FIRES_FOR_DISPLAY} fire(s) in "
            f"last {HISTORY_WINDOW_DAYS}d → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )
    if MIN_LONG_HISTORICAL_FIRES > 0:
        before = len(base)
        base = base[base[long_col] >= MIN_LONG_HISTORICAL_FIRES].copy()
        print(
            f"Long-window   filter: ≥{MIN_LONG_HISTORICAL_FIRES} fire(s) in "
            f"last {LONG_HISTORY_WINDOW_DAYS}d → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )
    if MIN_FIRE_DAYS_PER_YEAR > 0:
        before = len(base)
        # Cells with NaN fire_days_per_year (very short data record) are
        # kept — we only filter when we have enough data to estimate a rate.
        mask = (
            base["fire_days_per_year"].isna()
            | (base["fire_days_per_year"] >= MIN_FIRE_DAYS_PER_YEAR)
        )
        base = base[mask].copy()
        print(
            f"Fire-rate     filter: ≥{MIN_FIRE_DAYS_PER_YEAR} fire-days/year "
            f"(full history) → {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Urban-exclusion: drop cells that fall inside (or near) cities. Run AFTER
    # the cheaper count/rate filters so we do the haversine pass on the
    # smallest possible set. We also annotate every surviving cell with its
    # nearest urban area + distance, surfaced in the popup so users can see
    # context like "5 km from Chiang Mai" alongside the prediction.
    is_urban, urban_dist, urban_name = classify_urban(
        base["lat_grid"].to_numpy(),
        base["lon_grid"].to_numpy(),
        urban_areas=THAI_URBAN_AREAS,
        buffer_km=URBAN_BUFFER_KM,
    )
    base["nearest_urban_area"] = urban_name
    base["nearest_urban_distance_km"] = urban_dist
    if URBAN_FILTER_ENABLED:
        before = len(base)
        base = base[~is_urban].copy()
        n_dropped = before - len(base)
        print(
            f"Urban-area    filter: dropped {n_dropped:,} cells inside major "
            f"cities (+{URBAN_BUFFER_KM} km buffer); {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Country filter — drop cells outside Thailand's land border. Done last
    # so the polygon-containment check (most expensive of the three filters)
    # only runs against the smallest possible set.
    if COUNTRY_FILTER_ENABLED and len(base) > 0:
        before = len(base)
        in_th = is_in_thailand(
            base["lat_grid"].to_numpy(),
            base["lon_grid"].to_numpy(),
        )
        base = base[in_th].copy()
        n_dropped = before - len(base)
        print(
            f"Country       filter: dropped {n_dropped:,} cells outside Thailand "
            f"(neighbour countries); {len(base):,} / {before:,} kept "
            f"({len(base)*100/max(before,1):.1f}%)"
        )

    # Per-cell province annotation — emitted into the GeoJSON properties so
    # the frontend can offer a provincial filter without doing point-in-
    # polygon on the client. Only runs against post-filter cells (small set).
    if len(base) > 0:
        provinces = find_province(
            base["lat_grid"].to_numpy(),
            base["lon_grid"].to_numpy(),
        )
        base["province"] = provinces

    # Per-day cap: cells already grouped by days_until_fire (clipped int).
    # Within each day we sort by recent activity — historical_fire_count_30d
    # is the strongest "is this place burning right now" signal — and keep
    # the top MAX_CELLS_PER_DAY. Cells without 30d history (e.g. brand-new
    # active cells) get historical_fire_count_30d=0 and tie-break by
    # raw_prediction (closer to integer = more confident).
    if MAX_CELLS_PER_DAY > 0 and len(base) > 0:
        before = len(base)
        # Confidence proxy: how close raw_pred landed to the rounded integer.
        # Used as tiebreaker when historical_fire_count_30d is equal.
        rounding_proximity = 1.0 - np.abs(
            base["raw_prediction"].to_numpy() - base["days_until_fire"].to_numpy()
        )
        base = base.assign(_rank_score=base["historical_fire_count_30d"].fillna(0) * 1000 + rounding_proximity)
        base = (
            base.sort_values(["days_until_fire", "_rank_score"], ascending=[True, False])
                .groupby("days_until_fire", group_keys=False)
                .head(MAX_CELLS_PER_DAY)
                .drop(columns="_rank_score")
                .reset_index(drop=True)
        )
        n_dropped = before - len(base)
        per_day_counts = base.groupby("days_until_fire").size().to_dict()
        print(
            f"Per-day cap   : capped each predicted day to {MAX_CELLS_PER_DAY} most-active "
            f"cells; dropped {n_dropped:,}; {len(base):,} / {before:,} kept "
            f"(per-day: {per_day_counts})"
        )

    print(f"Total after history filters: {len(base):,} / {n_total:,} cells "
          f"({len(base)*100/max(n_total,1):.1f}%)")

    # Use fixed-domain thresholds (CRITICAL=0, HIGH=2, MEDIUM=4, LOW=7) by
    # default, BUT fall back to per-snapshot quantile thresholds when the
    # fixed cutoffs would collapse every surviving cell into a single tier
    # (model output is too narrow). The bunching is still visible to anyone
    # who reads the metadata: `urgency_thresholds_source` records which
    # mode was used, and dataset_info.json keeps the fixed cutoffs for
    # api.py and training-time semantics.
    fixed_thresholds = _resolve_thresholds(meta)
    if len(base) >= 20:
        fixed_tiers = {
            urgency_from_thresholds(float(rp), fixed_thresholds)
            for rp in base["raw_prediction"]
        }
        if len(fixed_tiers) <= 1:
            thresholds = calibrate_urgency_thresholds(
                base["raw_prediction"].to_numpy(),
                horizon=MAX_PREDICTION_DAYS,
            )
            thresholds_source = "snapshot-quantile fallback (fixed cutoffs collapse to one tier)"
        else:
            thresholds = fixed_thresholds
            thresholds_source = "fixed-domain, from dataset_info.json"
    else:
        thresholds = fixed_thresholds
        thresholds_source = "fixed-domain (too few cells to recalibrate)"
    print(f"Urgency thresholds [{thresholds_source}]: {thresholds}")

    base["urgency_level"] = [
        urgency_from_thresholds(float(rp), thresholds)
        for rp in base["raw_prediction"]
    ]

    return base, base_date, thresholds


# =========================================================
# WRITE GEOJSON (append-then-overwrite-current-base-date)
# =========================================================

def _revalidate_predictions(features: list, daily_df: pd.DataFrame) -> dict:
    """Tag each historical predicted feature with hit / miss / future based on
    actual FIRMS observations for its `predicted_fire_date`.

    A prediction is a *hit* when the cell had at least one FIRMS detection
    on the predicted date OR within ±1 day of it (matching the model's
    "accuracy within 1 day" headline metric). It's a *miss* when the
    target date is past, has full FIRMS coverage, but the cell stayed
    quiet. *future* predictions are kept untagged for the operator to see
    as "still pending".

    The caller passes the densified daily frame produced by
    data_loader.load_and_prepare; we read fire_count > 0 from it as
    ground truth.
    """
    if daily_df is None or daily_df.empty:
        return {"hits": 0, "misses": 0, "future": 0, "hit_rate": None}

    # Lookup: set of (lat_grid, lon_grid, date) tuples that had fire.
    # Densified frame includes zero-fire rows, so explicitly filter.
    fired = daily_df[daily_df["fire_count"] > 0]
    obs_keys = set(zip(
        fired["lat_grid"].round(6),
        fired["lon_grid"].round(6),
        pd.to_datetime(fired["date"]).dt.date,
    ))

    # Latest date with FIRMS coverage — anything after this we can't validate.
    latest_observed = pd.to_datetime(daily_df["date"]).max().date()

    # Per-snapshot tallies — operators want to know "how did the 2026-05-06
    # call hold up?" not just "how do all calls ever made hold up?"
    # Old snapshots may have been generated with relaxed filters or buggy
    # logic and dragging them into a single overall rate hides the current
    # model's real performance.
    per_snapshot: Dict[str, Dict[str, int]] = {}

    hits = misses = future = 0
    for f in features:
        props = f.get("properties", {})
        if props.get("source") != "predicted":
            continue
        target_str = props.get("predicted_fire_date")
        if not target_str:
            continue
        try:
            target_date = pd.to_datetime(target_str).date()
        except (ValueError, TypeError):
            continue

        bd = props.get("base_date") or "unknown"
        bucket = per_snapshot.setdefault(bd, {"hits": 0, "misses": 0, "future": 0})

        if target_date > latest_observed:
            props["validation_status"] = "future"
            future += 1
            bucket["future"] += 1
            continue

        lat = round(float(props.get("lat", 0)), 6)
        lon = round(float(props.get("lon", 0)), 6)
        hit = any(
            (lat, lon, target_date + timedelta(days=offset)) in obs_keys
            for offset in (-1, 0, 1)
        )
        if hit:
            props["validation_status"] = "hit"
            hits += 1
            bucket["hits"] += 1
        else:
            props["validation_status"] = "miss"
            misses += 1
            bucket["misses"] += 1

    # Annotate each snapshot bucket with its own hit rate so the frontend
    # doesn't have to recompute.
    for bd, bucket in per_snapshot.items():
        valid = bucket["hits"] + bucket["misses"]
        bucket["hit_rate"] = round(bucket["hits"] / valid, 4) if valid > 0 else None

    return {
        "hits": hits,
        "misses": misses,
        "future": future,
        "hit_rate": round(hits / (hits + misses), 4) if (hits + misses) > 0 else None,
        "per_snapshot": per_snapshot,
    }


def append_geojson(observed: pd.DataFrame, predicted: pd.DataFrame, base_date, thresholds: dict, metrics: dict, daily_df: Optional[pd.DataFrame] = None):
    base_date_str = base_date.strftime("%Y-%m-%d")

    geojson = {"type": "FeatureCollection", "features": []}
    if os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Corrupted GeoJSON → recreate")

    # Retention. Without pruning the GeoJSON grows by ~one prediction snapshot
    # plus a fresh observed batch on every risk_map.run() — by month two the
    # file would be tens of MB and slow on first dashboard load. Two limits:
    #   • Keep the most recent MAX_BASE_DATE_SNAPSHOTS predicted snapshots
    #     (incl. the one we're about to write, replacing any same-base prior).
    #   • Drop observed features — we re-emit a fresh observed batch from the
    #     latest densified FIRMS day a few lines down. Stale observed rows
    #     just bloat the file; the date picker only needs predictions.
    MAX_BASE_DATE_SNAPSHOTS = int(os.getenv("MAX_BASE_DATE_SNAPSHOTS", "7"))
    existing = geojson.get("features", [])
    predicted_only = [f for f in existing if f["properties"].get("source") == "predicted"]
    # Snapshots to keep: most recent N-1, excluding the one we're rewriting.
    base_dates = sorted(
        {f["properties"].get("base_date") for f in predicted_only if f["properties"].get("base_date")},
        reverse=True,
    )
    keep_dates = set(d for d in base_dates if d != base_date_str)
    keep_dates = set(list(keep_dates)[: MAX_BASE_DATE_SNAPSHOTS - 1])
    geojson["features"] = [
        f for f in predicted_only
        if f["properties"].get("base_date") in keep_dates
    ]

    # Top-level metadata so the frontend can read calibrated thresholds and
    # validation metrics without a separate fetch.
    geojson["metadata"] = {
        "base_date": base_date_str,
        "horizon_days": MAX_PREDICTION_DAYS,
        "urgency_thresholds": thresholds,
        "metrics": metrics,
        "history_window_days": HISTORY_WINDOW_DAYS,
        "long_history_window_days": LONG_HISTORY_WINDOW_DAYS,
    }

    # ---------- OBSERVED (latest densified FIRMS day) ----------
    observed_date_str = observed["date"].iloc[0].strftime("%Y-%m-%d") if len(observed) else base_date_str
    for _, r in observed.iterrows():
        if int(r.get("fire_count", 0)) <= 0:
            # Densified rows include cells with zero fires; only emit real detections.
            continue
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "date": observed_date_str,
                "source": "observed",
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
                "fire_count": int(r["fire_count"]),
            },
        })

    # ---------- PREDICTED ----------
    long_col = f"historical_fire_count_{LONG_HISTORY_WINDOW_DAYS}d"
    for _, r in predicted.iterrows():
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "base_date": base_date_str,
                "source": "predicted",
                "days_until_fire": int(r["days_until_fire"]),
                "predicted_fire_date": str(r["predicted_fire_date"]),
                "urgency_level": str(r["urgency_level"]),
                "confidence": float(r["prediction_confidence"]),
                "raw_prediction": float(r["raw_prediction"]),
                "historical_fire_count_30d": int(r["historical_fire_count_30d"]),
                "historical_fire_count_long": int(r[long_col]),
                "fire_days_total": int(r["fire_days_total"]),
                "fire_days_per_year": float(r["fire_days_per_year"]),
                "nearest_urban_area": str(r["nearest_urban_area"]),
                "nearest_urban_distance_km": float(r["nearest_urban_distance_km"]),
                "province": str(r.get("province", "") or ""),
                # Hansen tree cover passed through so the dashboard can
                # categorise cells by land-cover bucket without re-doing
                # the per-cell lookup. Defaults to 0 if the cache wasn't
                # populated during training.
                "tree_cover_pct_2000": float(r.get("tree_cover_pct_2000", 0) or 0),
                "tree_loss_pct_recent": float(r.get("tree_loss_pct_recent", 0) or 0),
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
            },
        })

    # Retrospective validation: walk every predicted feature still in the
    # GeoJSON (current snapshot + retained history) and tag with
    # hit / miss / future against actual FIRMS observations from the
    # densified daily frame. Lets the dashboard show "model called this
    # cell to fire 3 days ago — did it actually burn?" at a glance.
    if daily_df is not None:
        validation_summary = _revalidate_predictions(geojson["features"], daily_df)
        geojson["metadata"]["validation_summary"] = validation_summary
        if validation_summary.get("hit_rate") is not None:
            print(
                f"Retrospective validation: {validation_summary['hits']} hits / "
                f"{validation_summary['hits'] + validation_summary['misses']} validatable "
                f"= {validation_summary['hit_rate']*100:.1f}% hit rate "
                f"(±1 day window, {validation_summary['future']} still pending)"
            )

    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_date": base_date_str,
                "observed_date": observed_date_str,
                "prediction_horizon_days": MAX_PREDICTION_DAYS,
                "urgency_thresholds": thresholds,
                "metrics": metrics,
            },
            f,
            indent=2,
        )


# =========================================================
# PIPELINE
# =========================================================

def run():
    print("🔄 Loading assets...")
    model, df = load_assets()
    meta = _load_metadata()

    print("📍 Building observed layer (real FIRMS detections)...")
    observed, obs_date = build_observed(df)

    print(f"🔮 Predicting fire dates for next {MAX_PREDICTION_DAYS} days...")
    predicted, base_date, thresholds = build_predicted(df, model, obs_date, meta)

    metrics = (meta.get("model") or {}).get("test_metrics", {}) or {}
    # Include feature_importance_top (lives at the root of dataset_info.json)
    # in the GeoJSON-side metrics so the Reports page can render it without
    # a second fetch.
    if "feature_importance_top" in meta:
        metrics["feature_importance_top"] = meta["feature_importance_top"][:20]
    # Pass the densified daily frame so append_geojson can revalidate every
    # predicted feature against actual FIRMS observations and tag it
    # hit / miss / future. Empowers the dashboard's "did it actually burn?"
    # comparison view.
    append_geojson(observed, predicted, base_date, thresholds, metrics, daily_df=df)

    print("\n✅ FIRE DATE MAP UPDATED")
    print("Observed date :", obs_date)
    print("Base date     :", base_date)
    print("Thresholds    :", thresholds)
    print("GeoJSON       :", GEOJSON_PATH)

    urgency_counts = predicted["urgency_level"].value_counts()
    print("\n📊 URGENCY SUMMARY (calibrated thresholds):")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        print(f"  {level}: {int(urgency_counts.get(level, 0))} locations")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()