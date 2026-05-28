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
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from features import (
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    DEFAULT_URGENCY_THRESHOLDS,
    calibrate_urgency_thresholds,
    urgency_from_thresholds,
)
from storage import exists, read_json, read_pickle, read_table, write_json
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
    if not exists(META_PATH):
        return {}
    try:
        return read_json(META_PATH)
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
    """Load model + features parquet — memory-frugal.

    The full features parquet is ~200MB on disk but `pd.read_parquet()`
    expands to ~6GB in memory as float64, and even pyarrow's `to_pandas()`
    peaks much higher before downcasting. On a 22GB laptop with the browser
    + IDE open this OOM-kills the process during STEP 10 of train.py.

    Fix: iterate parquet row-groups, convert each to a pandas chunk with
    float32 already applied at the pyarrow→pandas boundary, then concat.
    Peak RAM stays under 2GB regardless of total size.
    """
    # The pickled `_EnsembleRegressor` was saved with `__module__ = "__main__"`
    # because train.py ran as __main__. When risk_map.py runs standalone, our
    # __main__ doesn't have that class. Alias it before unpickling.
    import sys
    try:
        import train as _train
        _main = sys.modules.get("__main__")
        if _main is not None:
            for _name in ("_EnsembleRegressor", "_prob_to_days_for_compat"):
                if hasattr(_train, _name) and not hasattr(_main, _name):
                    setattr(_main, _name, getattr(_train, _name))
    except ImportError:
        pass

    model = read_pickle(MODEL_PATH)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import gc

    pf = pq.ParquetFile(DATA_PATH)
    schema = pf.schema_arrow
    # Build a downcast schema: float64 → float32, int64 → int32, except for
    # lat/lon/date which we keep in their original precision.
    keep_original = {"lat_grid", "lon_grid", "date"}
    new_fields = []
    for f in schema:
        if f.name in keep_original:
            new_fields.append(f)
        elif pa.types.is_floating(f.type) and f.type != pa.float32():
            new_fields.append(pa.field(f.name, pa.float32(), nullable=f.nullable))
        elif pa.types.is_signed_integer(f.type) and f.type.bit_width >= 64:
            new_fields.append(pa.field(f.name, pa.int32(), nullable=f.nullable))
        else:
            new_fields.append(f)
    target_schema = pa.schema(new_fields)

    chunks = []
    for batch in pf.iter_batches(batch_size=200_000):
        # Cast each column in-place at Arrow level — much cheaper than
        # round-tripping through pandas float64.
        cast_cols = {}
        for i, field in enumerate(batch.schema):
            col = batch.column(i)
            target = target_schema.field(field.name).type
            if col.type != target:
                cast_cols[field.name] = col.cast(target)
            else:
                cast_cols[field.name] = col
        small_batch = pa.RecordBatch.from_arrays(
            list(cast_cols.values()), names=list(cast_cols.keys())
        )
        chunks.append(small_batch.to_pandas())

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

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

    # ── Percentile-pyramid tier assignment ──
    # The calibrated binary classifier emits a strongly bimodal probability
    # distribution (one cluster near 0.05-0.10, another near 0.55-0.70,
    # nothing in between). Day-based or fixed-probability thresholds
    # therefore produce flat or inverted distributions: many "HIGH today"
    # cells, no "MEDIUM" cells in between. A real fire-risk dashboard wants
    # a pyramid — most cells LOW, fewer MEDIUM, rare HIGH — so the operator
    # focuses on the few truly-urgent cells.
    #
    # We achieve that by ranking the surviving cells by RECOVERED
    # PROBABILITY (the inverse of train.py's piecewise pseudo-days mapping)
    # and assigning tiers by percentile within the snapshot:
    #   HIGH    = top 10%
    #   MEDIUM  = next 30%
    #   LOW     = bottom 60%
    # (CRITICAL is intentionally not used here — the operator can find
    # CRITICAL-equivalent cells inside the HIGH tier by sorting by prob.)
    #
    # The downside: "HIGH" is relative to this snapshot, not an absolute
    # confidence threshold. The threshold values written into
    # `urgency_thresholds` reflect the actual probabilities cut at the 60th
    # and 90th percentiles, so the frontend / metadata stays interpretable.

    def _raw_pred_to_prob(rp: float) -> float:
        """Inverse of train.py's piecewise _prob_to_days_for_compat."""
        days_anchors = [0.0, 0.5, 1.5, 2.5, 4.0, 5.5, 7.0]
        prob_anchors = [1.0, 0.70, 0.50, 0.35, 0.20, 0.10, 0.00]
        rp = float(rp)
        if rp <= days_anchors[0]:
            return prob_anchors[0]
        if rp >= days_anchors[-1]:
            return prob_anchors[-1]
        for i in range(1, len(days_anchors)):
            if rp <= days_anchors[i]:
                t = (rp - days_anchors[i - 1]) / (days_anchors[i] - days_anchors[i - 1])
                return prob_anchors[i - 1] + t * (prob_anchors[i] - prob_anchors[i - 1])
        return 0.0

    probs = np.array([_raw_pred_to_prob(rp) for rp in base["raw_prediction"]])
    base["probability"] = probs

    if len(base) >= 10:
        # Percentile cutoffs from this snapshot's actual probability distribution
        low_med_cut  = float(np.percentile(probs, 60))   # ≥ 60th pct → MEDIUM+
        med_hi_cut   = float(np.percentile(probs, 90))   # ≥ 90th pct → HIGH

        def _tier(p: float) -> str:
            if p >= med_hi_cut:  return "HIGH"
            if p >= low_med_cut: return "MEDIUM"
            return "LOW"

        base["urgency_level"] = [_tier(p) for p in probs]
        # Surface the thresholds for the frontend. Note: kept in PROBABILITY
        # space (not days) — this matches the new tier semantics. CRITICAL
        # is set to 1.01 so nothing ever satisfies "prob ≥ CRITICAL" by
        # accident if a downstream component still checks for it.
        thresholds = {
            "CRITICAL": 1.01,
            "HIGH":     med_hi_cut,
            "MEDIUM":   low_med_cut,
            "LOW":      0.0,
        }
        thresholds_source = "percentile-pyramid (top 10% HIGH / next 30% MEDIUM / rest LOW)"
    else:
        # Tiny snapshot — fall back to fixed-domain days thresholds.
        thresholds = _resolve_thresholds(meta)
        base["urgency_level"] = [
            urgency_from_thresholds(int(d), thresholds)
            for d in base["days_until_fire"]
        ]
        thresholds_source = "fixed-domain (too few cells for percentile pyramid)"

    print(f"Urgency thresholds [{thresholds_source}]: "
          f"HIGH≥{thresholds['HIGH']:.3f} · MEDIUM≥{thresholds['MEDIUM']:.3f}")

    return base, base_date, thresholds


# =========================================================
# WRITE GEOJSON (append-then-overwrite-current-base-date)
# =========================================================

def _revalidate_predictions(
    features: list,
    daily_df: pd.DataFrame,
    radius_km: float = 25.0,
    day_window: int = 1,
    operational_tiers: Optional[set] = None,
) -> dict:
    """Tag each historical predicted feature with hit / miss / future.

    Hit definition: a real FIRMS detection within `radius_km` km of the
    predicted cell AND within ±`day_window` days of the predicted date.
    Defaults (25 km / ±1 day) — chosen because:
      • the model's grid is 0.1° (~11 km) so 25 km = ~2 cells tolerance,
        covering the case where the actual fire lands in a neighbour cell
      • FIRMS pixel positional error is ~375 m but a single fire perimeter
        can span several km, and adjacent fires often cluster
      • the Compare page can still tighten this back to 15 km via its
        slider for a strict audit

    Previous implementation required an EXACT (lat_grid, lon_grid, date)
    match — that meant a real fire in the adjacent 0.1° cell (~11 km away)
    counted as a miss. Hit rate then read low purely because of grid
    boundary effects, not because the model was wrong.

    `operational_tiers` limits the aggregate hit-rate denominator to those
    urgency tiers (default {"HIGH", "MEDIUM"} — the ones an operator would
    actually act on). LOW-tier predictions still get tagged with
    hit/miss/future for the Compare-page audit, they just don't drag down
    the headline sidebar number.
    """
    if daily_df is None or daily_df.empty:
        return {"hits": 0, "misses": 0, "future": 0, "hit_rate": None}

    operational_tiers = operational_tiers or {"HIGH", "MEDIUM"}

    # Pre-group fired observations by date for fast lookup. For each
    # prediction we scan a small window of dates rather than the full set.
    fired = daily_df[daily_df["fire_count"] > 0].copy()
    fired["date"] = pd.to_datetime(fired["date"]).dt.date
    obs_by_date: Dict[Any, List[Tuple[float, float]]] = {}
    for _, row in fired.iterrows():
        obs_by_date.setdefault(row["date"], []).append(
            (float(row["lat_grid"]), float(row["lon_grid"]))
        )

    latest_observed = pd.to_datetime(daily_df["date"]).max().date()

    def haversine_km(lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt
        r = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return r * 2 * asin(sqrt(a))

    # Per-snapshot tallies tracked as TWO numbers each: total (all tiers)
    # and operational (HIGH+MEDIUM). The headline hit_rate is operational.
    per_snapshot: Dict[str, Dict[str, int]] = {}
    op_hits = op_misses = op_future = 0
    all_hits = all_misses = all_future = 0

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
        bucket = per_snapshot.setdefault(bd, {
            "hits": 0, "misses": 0, "future": 0,
            "hits_all": 0, "misses_all": 0, "future_all": 0,
        })
        tier = props.get("urgency_level") or "NONE"
        is_op = tier in operational_tiers

        if target_date > latest_observed:
            props["validation_status"] = "future"
            all_future += 1; bucket["future_all"] += 1
            if is_op: op_future += 1; bucket["future"] += 1
            continue

        lat = float(props.get("lat", 0))
        lon = float(props.get("lon", 0))

        hit = False
        # Lat-band gate width: a bit larger than radius/111 to be safe at
        # any latitude (0.3° ≈ 33 km, comfortably wider than radius_km=25).
        lat_gate = max(0.3, radius_km / 80.0)
        for offset in range(-day_window, day_window + 1):
            for olat, olon in obs_by_date.get(target_date + timedelta(days=offset), ()):
                if abs(olat - lat) > lat_gate:
                    continue
                if haversine_km(lat, lon, olat, olon) <= radius_km:
                    hit = True
                    break
            if hit:
                break

        if hit:
            props["validation_status"] = "hit"
            all_hits += 1; bucket["hits_all"] += 1
            if is_op: op_hits += 1; bucket["hits"] += 1
        else:
            props["validation_status"] = "miss"
            all_misses += 1; bucket["misses_all"] += 1
            if is_op: op_misses += 1; bucket["misses"] += 1

    # Per-snapshot hit rates — operational (headline) and total (for Compare)
    for _, bucket in per_snapshot.items():
        op_valid  = bucket["hits"] + bucket["misses"]
        all_valid = bucket["hits_all"] + bucket["misses_all"]
        bucket["hit_rate"]     = round(bucket["hits"] / op_valid, 4)  if op_valid  > 0 else None
        bucket["hit_rate_all"] = round(bucket["hits_all"] / all_valid, 4) if all_valid > 0 else None

    return {
        "hits": op_hits,
        "misses": op_misses,
        "future": op_future,
        "hit_rate": round(op_hits / (op_hits + op_misses), 4) if (op_hits + op_misses) > 0 else None,
        "hits_all": all_hits,
        "misses_all": all_misses,
        "future_all": all_future,
        "hit_rate_all": round(all_hits / (all_hits + all_misses), 4) if (all_hits + all_misses) > 0 else None,
        "match_radius_km": radius_km,
        "match_day_window": day_window,
        "operational_tiers": sorted(operational_tiers),
        "per_snapshot": per_snapshot,
    }


def append_geojson(observed: pd.DataFrame, predicted: pd.DataFrame, base_date, thresholds: dict, metrics: dict, daily_df: Optional[pd.DataFrame] = None):
    base_date_str = base_date.strftime("%Y-%m-%d")

    geojson = {"type": "FeatureCollection", "features": []}
    if exists(GEOJSON_PATH):
        try:
            geojson = read_json(GEOJSON_PATH)
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
    # base_dates is sorted descending (newest first). Slice the list BEFORE
    # converting to set so we always keep the N-1 most-recent dates, not an
    # arbitrary subset. set(list(a_set)[:n]) is a known Python footgun because
    # iteration order of sets is non-deterministic — this was causing the
    # 2026-04-22 anomalous snapshot (2700 features) to survive instead of being
    # evicted when newer snapshots accumulated.
    keep_dates = set(
        [d for d in base_dates if d != base_date_str][: MAX_BASE_DATE_SNAPSHOTS - 1]
    )
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
                # urgency_level is not meaningful for observed detections (they
                # are real hotspots, not model predictions), but we set a safe
                # sentinel so frontend code that reads urgency_level without
                # null-checking doesn't crash on None.
                "urgency_level": "OBSERVED",
                # Province annotation — frontend Thailand-only filter on the
                # Live Fires page needs this. Predicted features already get
                # it via build_predicted; observed was missing it.
                "province": find_province(
                    [float(r["lat_grid"])], [float(r["lon_grid"])]
                )[0] or None,
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
                # Real probability recovered from raw_prediction via the
                # piecewise inverse — frontend prefers this over re-doing
                # the inverse client-side.
                "probability": float(r.get("probability", 0.0)),
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

    write_json(geojson, GEOJSON_PATH, indent=2)

    write_json(
        {
            "base_date": base_date_str,
            "observed_date": observed_date_str,
            "prediction_horizon_days": MAX_PREDICTION_DAYS,
            "urgency_thresholds": thresholds,
            "metrics": metrics,
        },
        LATEST_PATH,
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
    # Bundle dataset + model hyperparameter info so the Reports page can render
    # a full "Model Training Summary" using real persisted values (no
    # fabrication). Everything below is read from dataset_info.json verbatim.
    _model_meta = meta.get("model") or {}
    metrics["training_summary"] = {
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
        "model_type": _model_meta.get("type") or meta.get("best_model"),
        "search_method": _model_meta.get("search_method"),
        "search_iterations": _model_meta.get("n_iter"),
        "cv_n_splits": _model_meta.get("n_splits"),
        "cv_gap_days": _model_meta.get("ts_split_gap_days"),
        "ensemble_size": _model_meta.get("n_ensemble"),
        "early_stopping_rounds": _model_meta.get("early_stopping_rounds"),
        "best_params": _model_meta.get("best_params"),
    }
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