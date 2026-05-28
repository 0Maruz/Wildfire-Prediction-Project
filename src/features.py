"""Time-series + spatial + calendar feature engineering and label construction.

LEAKAGE AUDIT (per user request — see comments tagged ``# CAUSAL``):
- Every rolling window now uses ``.shift(1)`` BEFORE ``.rolling(w)`` so that
  ``fire_sum_7d`` answers "last 7 days NOT including today". The model still
  sees today's data via separate ``*_today`` features.
- Every lag uses ``.shift(lag)`` with ``lag >= 1`` (past only).
- Streak counters and dry-streak use shifted fire flags so today's fire never
  contributes to today's counter.
- Trend/diff features (``frp_trend_*``) use ``shift(1).diff()`` semantics so
  the comparison is between two past days, not today vs the past.
- Same-week-last-year, season cumulatives, expanding fire rate — all use
  ``.shift(1)`` on the cumulative path.

All features are derived from REAL data sources only:
    - NASA FIRMS VIIRS NRT hotspots → fire_count, frp_*, bright_*, confidence_*
    - Spatial neighbourhood = 8 + 16 surrounding grid cells from same FIRMS frame
    - Calendar = real datetime values
    - Optional weather (when present): real ECMWF ERA5 via Open-Meteo Archive

No synthetic / random / interpolated features. If a real source isn't available
for a given column, the column is simply not added to FEATURES.

`FEATURES_CORE` is the always-on input contract. Use `resolve_features(df)` (or
read `dataset_info.json["features"]`) to get the exact list the deployed model
was trained on, since weather columns are added dynamically.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from urban_areas import classify_urban

log = logging.getLogger("features")

MAX_PREDICTION_DAYS = 7

LAG_DAYS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 30)
ROLL_WINDOWS: Tuple[int, ...] = (3, 5, 7, 14, 21, 30, 60)  # 60 added per spec
NEIGHBOR_LAGS: Tuple[int, ...] = (1, 2, 3, 5, 7)
NEIGHBOR_ROLLS: Tuple[int, ...] = (3, 5, 7, 14)

GROUP_KEYS = ["lat_grid", "lon_grid"]

WEATHER_COLUMNS: Tuple[str, ...] = (
    "temp_max",
    "temp_min",
    "precip_sum",
    "wind_max",
    "et0",
)
WEATHER_LAGS: Tuple[int, ...] = (1, 3, 7)
WEATHER_ROLLS: Tuple[int, ...] = (3, 7)

# RADD roll windows — wider than fire rolls because RADD has a 6–12 day revisit
RADD_COLUMNS: Tuple[str, ...] = ("radd_alert_count", "radd_confidence_max")
RADD_ROLLS: Tuple[int, ...] = (14, 30, 90)
RADD_LAGS: Tuple[int, ...] = (7, 14, 30)

UrgencyLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]

DEFAULT_URGENCY_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 0.0,
    "HIGH": 2.0,
    "MEDIUM": 4.0,
    "LOW": float(MAX_PREDICTION_DAYS),
}


# ─────────────────────────────────────────────
# Urgency calibration
# ─────────────────────────────────────────────
def calibrate_urgency_thresholds(
    val_predictions: np.ndarray,
    horizon: int = MAX_PREDICTION_DAYS,
    quantiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> Dict[str, float]:
    arr = np.asarray(val_predictions, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(DEFAULT_URGENCY_THRESHOLDS)
    q1, q2, q3 = np.quantile(arr, quantiles)
    q1 = float(min(q1, horizon))
    q2 = float(min(max(q2, q1), horizon))
    q3 = float(min(max(q3, q2), horizon))
    return {"CRITICAL": q1, "HIGH": q2, "MEDIUM": q3, "LOW": float(horizon)}


def urgency_from_thresholds(
    days: float, thresholds: Optional[Dict[str, float]] = None
) -> UrgencyLevel:
    t = thresholds or DEFAULT_URGENCY_THRESHOLDS
    d = float(days)
    if d <= t["CRITICAL"]:
        return "CRITICAL"
    if d <= t["HIGH"]:
        return "HIGH"
    if d <= t["MEDIUM"]:
        return "MEDIUM"
    if d <= t["LOW"]:
        return "LOW"
    return "NONE"


def get_urgency(days: int) -> UrgencyLevel:
    return urgency_from_thresholds(days, DEFAULT_URGENCY_THRESHOLDS)


# ─────────────────────────────────────────────
# Helpers — past-only rolling
# ─────────────────────────────────────────────
def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)


def _past_roll(
    df: pd.DataFrame, col: str, window: int, agg: str = "sum"
) -> np.ndarray:
    """CAUSAL: shift(1) before rolling — answers 'last <window> days, excluding today'.

    Returns a 1-D numpy array aligned with df.index (df must already be sorted
    by GROUP_KEYS + date).
    """
    grp = df.groupby(GROUP_KEYS, sort=False)[col]
    shifted = grp.shift(1)
    regrouped = shifted.groupby(
        [df["lat_grid"], df["lon_grid"]], sort=False
    ).rolling(window, min_periods=1)
    if agg == "sum":
        rolled = regrouped.sum()
    elif agg == "max":
        rolled = regrouped.max()
    elif agg == "mean":
        rolled = regrouped.mean()
    elif agg == "std":
        rolled = regrouped.std()
    elif agg == "active":
        rolled = regrouped.apply(lambda x: float((x > 0).sum()), raw=True)
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    return (
        rolled.reset_index(level=[0, 1], drop=True)
        .reindex(df.index)
        .fillna(0)
        .to_numpy()
    )


# ─────────────────────────────────────────────
# Spatial neighbours (current-day aggregates — used downstream with shift)
# ─────────────────────────────────────────────
def add_neighbor_features(daily: pd.DataFrame, grid_size: float = 0.1) -> pd.DataFrame:
    """Per-cell spatial aggregates over the surrounding grid (3×3 and 5×5 rings).

    NOTE: produces *_today columns. The downstream add_temporal_features turns
    them into past-only lag/roll features via shift(1).
    """
    df = daily.copy()
    df["lat_grid"] = df["lat_grid"].round(6)
    df["lon_grid"] = df["lon_grid"].round(6)

    base = df[["lat_grid", "lon_grid", "date", "fire_count", "frp_sum"]].copy()

    df["neighbor_fire_today"] = 0.0
    df["neighbor_frp_today"] = 0.0
    df["wide_neighbor_fire_today"] = 0.0
    df["wide_neighbor_frp_today"] = 0.0

    inner_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    outer_offsets = [
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2),                             (-1, 2),
        ( 0, -2),                             ( 0, 2),
        ( 1, -2),                             ( 1, 2),
        ( 2, -2), ( 2, -1), ( 2, 0), ( 2, 1), ( 2, 2),
    ]

    def _accumulate(offsets, fire_col, frp_col):
        for dlat, dlon in offsets:
            shifted = base.copy()
            shifted["lat_grid"] = (shifted["lat_grid"] - dlat * grid_size).round(6)
            shifted["lon_grid"] = (shifted["lon_grid"] - dlon * grid_size).round(6)
            shifted = shifted.rename(columns={"fire_count": "_nfire", "frp_sum": "_nfrp"})
            merged = df.merge(
                shifted[["lat_grid", "lon_grid", "date", "_nfire", "_nfrp"]],
                on=["lat_grid", "lon_grid", "date"],
                how="left",
            )
            df[fire_col] = df[fire_col].to_numpy() + merged["_nfire"].fillna(0).to_numpy()
            df[frp_col]  = df[frp_col].to_numpy()  + merged["_nfrp"].fillna(0).to_numpy()

    _accumulate(inner_offsets, "neighbor_fire_today", "neighbor_frp_today")
    _accumulate(outer_offsets, "wide_neighbor_fire_today", "wide_neighbor_frp_today")

    return df


# ─────────────────────────────────────────────
# Temporal features — ALL rolling windows are past-only
# ─────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling-window, and trend features per cell.

    CAUSAL audit: every rolling/trend feature here uses .shift(1) before
    aggregation, so no row contains its own day in the aggregate.
    """
    df = _ensure_sorted(df).copy()
    grp = df.groupby(GROUP_KEYS, sort=False)

    # ── per-cell self lags (already past via shift(lag), lag>=1) ──
    for lag in LAG_DAYS:
        df[f"fire_lag_{lag}"] = grp["fire_count"].shift(lag).fillna(0)  # CAUSAL: past only
        df[f"frp_lag_{lag}"]  = grp["frp_sum"].shift(lag).fillna(0)     # CAUSAL: past only

    # ── per-cell rolling windows (shift(1) then rolling — past only) ──
    for w in ROLL_WINDOWS:
        df[f"fire_sum_{w}d"]    = _past_roll(df, "fire_count", w, "sum")     # CAUSAL
        df[f"frp_sum_{w}d"]     = _past_roll(df, "frp_sum",    w, "sum")     # CAUSAL
        df[f"frp_max_{w}d"]     = _past_roll(df, "frp_max",    w, "max")     # CAUSAL
        df[f"active_days_{w}d"] = _past_roll(df, "fire_count", w, "active")  # CAUSAL

    # ── trend = lag1 - lag(k+1), so today is never used ──
    df["frp_trend_1d"] = (df["frp_lag_1"].to_numpy() - df["frp_lag_2"].to_numpy())   # CAUSAL
    df["frp_trend_3d"] = (df["frp_lag_1"].to_numpy() - df["frp_lag_4"].to_numpy())   # CAUSAL
    df["frp_trend_7d"] = (df["frp_lag_1"].to_numpy() - df["frp_lag_7"].to_numpy() if "frp_lag_7" in df.columns
                          else (df["frp_lag_1"].to_numpy() - grp["frp_sum"].shift(7).fillna(0).to_numpy()))  # CAUSAL

    df["fire_std_7d"] = _past_roll(df, "fire_count", 7, "std")  # CAUSAL

    # ── today-only signals (known at inference time, not leakage) ──
    df["fire_count_today"]       = df["fire_count"].fillna(0)
    df["frp_sum_today"]          = df["frp_sum"].fillna(0)
    df["bright_mean_today"]      = df["bright_mean"].fillna(0)
    df["confidence_mean_today"]  = df["confidence_mean"].fillna(0)

    for opt_col in (
        "night_fire_count",
        "afternoon_fire_count",
        "n_satellites_today",
        "tree_cover_pct_2000",
        "tree_loss_pct_recent",
    ):
        if opt_col in df.columns:
            df[opt_col] = df[opt_col].fillna(0)

    # ── derived features (past-only inputs ⇒ derived also past-only) ──
    df["fire_acceleration"] = (
        df["fire_sum_7d"].to_numpy() / 7.0 - df["fire_sum_14d"].to_numpy() / 14.0
    )                                                                    # CAUSAL (past windows)
    df["frp_intensity_7d"] = (
        df["frp_sum_7d"].to_numpy() / np.clip(df["fire_sum_7d"].to_numpy(), 1, None)
    ).astype(float)                                                      # CAUSAL

    if "night_fire_count" in df.columns:
        df["night_fire_ratio"] = (
            df["night_fire_count"].to_numpy()
            / np.clip(df["fire_count_today"].to_numpy(), 1, None)
        ).astype(float)
        df["night_fire_ratio"] = np.nan_to_num(df["night_fire_ratio"].to_numpy(), nan=0.0)

    # ── fire_streak_today: consecutive past fire days ENDING yesterday ──
    # CAUSAL: shift fire flag by 1 so today's fire never lights its own streak.
    fire_flag_past = (grp["fire_count"].shift(1).fillna(0) > 0).astype(int).to_numpy()
    lats = df["lat_grid"].to_numpy()
    lons = df["lon_grid"].to_numpy()
    streak_vals = np.zeros(len(df), dtype=np.float32)
    run = 0
    prev_key = None
    for i in range(len(df)):
        key = (lats[i], lons[i])
        if key != prev_key:
            run = 0
            prev_key = key
        run = run + 1 if fire_flag_past[i] else 0
        streak_vals[i] = run
    df["fire_streak_today"] = streak_vals  # CAUSAL: counts ≤ yesterday only

    # ── fire_momentum_7_30: past 7d rate vs past 30d rate ──
    df["fire_momentum_7_30"] = (
        (df["fire_sum_7d"].to_numpy() / 7.0)
        / np.clip(df["fire_sum_30d"].to_numpy() / 30.0, 0.01, None)
    ).astype(float)                                                      # CAUSAL
    df["fire_momentum_7_30"] = np.clip(df["fire_momentum_7_30"].to_numpy(), 0.0, 10.0)

    # ── frp_per_fire_today: ratio of two known-today values, not leakage ──
    df["frp_per_fire_today"] = (
        df["frp_sum_today"].to_numpy()
        / np.clip(df["fire_count_today"].to_numpy(), 1, None)
    ).astype(float)

    # ─────────────────────────────────────────
    # NEW FEATURES (per user spec)
    # ─────────────────────────────────────────
    # fire_count_60d: rolling sum of fire_count over past 60 days (exclusive of today)
    if "fire_sum_60d" not in df.columns:
        df["fire_sum_60d"] = _past_roll(df, "fire_count", 60, "sum")     # CAUSAL
    df["fire_count_60d"] = df["fire_sum_60d"].astype(float)              # alias per spec

    # fire_frequency_rate: fire-active days / total days in past 60d window
    active_days_60d = _past_roll(df, "fire_count", 60, "active")         # CAUSAL
    df["fire_frequency_rate"] = (active_days_60d / 60.0).astype(float)   # in [0, 1]
    # ─────────────────────────────────────────

    # ── inner-ring neighbour lags / rolls ──
    if "neighbor_fire_today" in df.columns:
        for lag in NEIGHBOR_LAGS:
            df[f"neighbor_fire_lag_{lag}"] = grp["neighbor_fire_today"].shift(lag).fillna(0)  # CAUSAL
            df[f"neighbor_frp_lag_{lag}"]  = grp["neighbor_frp_today"].shift(lag).fillna(0)   # CAUSAL
        for w in NEIGHBOR_ROLLS:
            df[f"neighbor_fire_sum_{w}d"] = _past_roll(df, "neighbor_fire_today", w, "sum")   # CAUSAL
            df[f"neighbor_frp_sum_{w}d"]  = _past_roll(df, "neighbor_frp_today",  w, "sum")   # CAUSAL
        df["neighbor_fire_velocity_3d"] = (
            df["neighbor_fire_lag_1"].to_numpy() - df["neighbor_fire_lag_3"].to_numpy()
        )  # CAUSAL: yesterday vs 3 days ago, no today
        df["neighbor_frp_velocity_3d"] = (
            df["neighbor_frp_lag_1"].to_numpy() - df["neighbor_frp_lag_3"].to_numpy()
        )  # CAUSAL

    # ── outer-ring (5×5) neighbour lags / rolls ──
    if "wide_neighbor_fire_today" in df.columns:
        for lag in NEIGHBOR_LAGS:
            df[f"wide_neighbor_fire_lag_{lag}"] = grp["wide_neighbor_fire_today"].shift(lag).fillna(0)  # CAUSAL
        for w in NEIGHBOR_ROLLS:
            df[f"wide_neighbor_fire_sum_{w}d"] = _past_roll(df, "wide_neighbor_fire_today", w, "sum")   # CAUSAL
        df["wide_neighbor_fire_velocity_3d"] = (
            df["wide_neighbor_fire_lag_1"].to_numpy() - df["wide_neighbor_fire_lag_3"].to_numpy()
        )  # CAUSAL

    if "wide_neighbor_frp_today" in df.columns:
        for lag in NEIGHBOR_LAGS:
            df[f"wide_neighbor_frp_lag_{lag}"] = grp["wide_neighbor_frp_today"].shift(lag).fillna(0)  # CAUSAL
        for w in NEIGHBOR_ROLLS:
            df[f"wide_neighbor_frp_sum_{w}d"] = _past_roll(df, "wide_neighbor_frp_today", w, "sum")    # CAUSAL

    # ── weather lags / rolls (past-only) ──
    weather_cols_present = [c for c in WEATHER_COLUMNS if c in df.columns]
    if weather_cols_present:
        for col in weather_cols_present:
            df[f"{col}_today"] = df[col].fillna(0)
            for lag in WEATHER_LAGS:
                df[f"{col}_lag_{lag}"] = grp[col].shift(lag).fillna(0)         # CAUSAL
            for w in WEATHER_ROLLS:
                df[f"{col}_mean_{w}d"] = _past_roll(df, col, w, "mean")        # CAUSAL

    return df


# ─────────────────────────────────────────────
# Calendar features
# ─────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings and Thai burn/dry-season flags. CAUSAL: depends only on date."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)   # CAUSAL: pure calendar
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)   # CAUSAL
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.0)  # CAUSAL
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.0)  # CAUSAL

    # is_burn_season: Thai burn months Jan–Apr (kept for backwards compat)
    df["is_burn_season"] = dt.dt.month.between(1, 4).astype(int)            # CAUSAL

    # NEW: is_dry_season per user spec — months Feb–May
    df["is_dry_season"] = dt.dt.month.isin([2, 3, 4, 5]).astype(int)        # CAUSAL

    BURN_PEAK_DOY = 75
    df["days_from_burn_peak"] = (df["day_of_year"] - BURN_PEAK_DOY).abs().clip(upper=100)  # CAUSAL

    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)          # CAUSAL
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)          # CAUSAL
    df["day_of_week"] = dt.dt.dayofweek.astype(int)                          # CAUSAL
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)                   # CAUSAL

    return df


# ─────────────────────────────────────────────
# days_since_last_fire — CAUSAL
# ─────────────────────────────────────────────
def add_dry_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell days-since-last-fire counter.

    CAUSAL: uses the fire flag shifted by 1, so today's fire never resets
    today's counter. On the day of a fire, days_since_last_fire reflects how
    long since the *previous* fire — not zero.
    """
    df = _ensure_sorted(df).copy()
    grp_fire = df.groupby(GROUP_KEYS, sort=False)["fire_count"]
    fire_flag_past = (grp_fire.shift(1).fillna(0) > 0).astype(int)  # CAUSAL: yesterday's flag

    # block id increments each time a past fire occurred — within block the
    # cumcount is the number of days since that fire.
    block = fire_flag_past.groupby([df["lat_grid"], df["lon_grid"]]).cumsum()
    df["_dry_block"] = block.to_numpy()
    df["days_since_last_fire"] = (
        df.groupby(GROUP_KEYS + ["_dry_block"]).cumcount() + 1
    ).astype(int)
    # Before any past fire has occurred (block==0), there is no "last fire" —
    # set to a large sentinel so the model can distinguish "never burned" from
    # "burned recently".
    df.loc[df["_dry_block"] == 0, "days_since_last_fire"] = 999
    df.drop(columns=["_dry_block"], inplace=True)
    return df


def add_cumulative_fire_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Annualized fire-day rate per cell. CAUSAL: shift(1) on cumsum."""
    df = _ensure_sorted(df).copy()
    fire_flag = (df["fire_count"].fillna(0) > 0).astype(int)
    cum_fires = (
        fire_flag.groupby([df["lat_grid"], df["lon_grid"]]).cumsum()
        .groupby([df["lat_grid"], df["lon_grid"]]).shift(1).fillna(0)
    )  # CAUSAL: cumulative up to yesterday
    days_before = df.groupby(GROUP_KEYS).cumcount()
    rate = np.where(
        days_before >= 30,
        cum_fires.to_numpy() * 365.25 / np.clip(days_before.to_numpy(), 1, None),
        0.0,
    )
    df["fire_days_per_year_so_far"] = rate.astype(float)
    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-year fire count + days into burn season. CAUSAL on cumulative."""
    df = _ensure_sorted(df).copy()
    dt = pd.to_datetime(df["date"])
    df["_season_year"] = dt.dt.year

    in_season = dt.dt.month.between(1, 4)
    df["days_into_burn_season"] = np.where(in_season, dt.dt.dayofyear, 0).astype(float)  # CAUSAL

    fire_flag = (df["fire_count"].fillna(0) > 0).astype(float)
    cum = (
        fire_flag.groupby([df["lat_grid"], df["lon_grid"], df["_season_year"]]).cumsum()
        .groupby([df["lat_grid"], df["lon_grid"], df["_season_year"]]).shift(1).fillna(0)
    )  # CAUSAL: up to yesterday
    df["season_fire_count_so_far"] = cum.to_numpy().astype(float)
    df.drop(columns=["_season_year"], inplace=True)
    return df


def add_urban_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell great-circle distance to nearest Thai city. CAUSAL: static geography."""
    df = df.copy()
    cells = df[["lat_grid", "lon_grid"]].drop_duplicates().reset_index(drop=True)
    _, urban_dist, _ = classify_urban(
        cells["lat_grid"].to_numpy(),
        cells["lon_grid"].to_numpy(),
    )
    cells["distance_to_nearest_city_km"] = urban_dist
    df = df.merge(cells, on=["lat_grid", "lon_grid"], how="left")
    return df


def add_fire_recurrence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell fire periodicity + same-week-last-year signal.

    CAUSAL: median_fire_gap is an EXPANDING median over past gaps; the
    same-week-last-year merge uses _prev_year so today's row only learns from
    prior calendar years; the per-month historical mean uses .shift(1) on the
    expanding mean so the current year doesn't leak.
    """
    df = _ensure_sorted(df).copy()
    dt = pd.to_datetime(df["date"])

    df["_fire_flag"] = (df["fire_count"].fillna(0) > 0).astype(int)
    df["_doy"] = dt.dt.dayofyear
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    df["_row_date"] = dt

    prev_fire_date = (
        df.where(df["_fire_flag"] == 1)
        .groupby(GROUP_KEYS)["_row_date"]
        .shift(1)
    )
    df["_gap_days"] = (df["_row_date"] - prev_fire_date).dt.days

    def _expanding_gap_med(grp):
        ser = grp["_gap_days"].where(grp["_fire_flag"] == 1)
        return ser.expanding(min_periods=2).median().ffill()

    gap_med = df.groupby(GROUP_KEYS, group_keys=False).apply(_expanding_gap_med)
    df["median_fire_gap"] = gap_med.to_numpy()
    df["median_fire_gap"] = df["median_fire_gap"].fillna(30.0)

    df["days_since_last_fire_norm"] = (
        df.get("days_since_last_fire", pd.Series(0, index=df.index))
        / df["median_fire_gap"].clip(lower=1.0)
    ).clip(upper=10.0)

    # ── same ±7-day window, previous calendar year ──
    fire_rows = df[df["_fire_flag"] == 1][["lat_grid", "lon_grid", "_year", "_doy"]].copy()
    expanded = []
    for offset in range(-7, 8):
        tmp = fire_rows.copy()
        tmp["_doy"] = (tmp["_doy"] + offset - 1) % 365 + 1
        expanded.append(tmp)
    fire_window = pd.concat(expanded, ignore_index=True).drop_duplicates()
    fire_window["_had_fire_window"] = 1

    df["_prev_year"] = df["_year"] - 1
    df = df.merge(
        fire_window.rename(columns={"_year": "_prev_year"})[
            ["lat_grid", "lon_grid", "_prev_year", "_doy", "_had_fire_window"]
        ],
        on=["lat_grid", "lon_grid", "_prev_year", "_doy"],
        how="left",
    )
    df["same_week_fire_last_year"] = df["_had_fire_window"].fillna(0).astype(float)

    # ── historical avg fire count per month (prior years only) ──
    monthly = (
        df[df["_fire_flag"] == 1]
        .groupby(["lat_grid", "lon_grid", "_year", "_month"])
        .size()
        .reset_index(name="_mf")
    )
    monthly = monthly.sort_values(["lat_grid", "lon_grid", "_month", "_year"])
    monthly["_hist_mean"] = (
        monthly.groupby(["lat_grid", "lon_grid", "_month"])["_mf"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1, 2], drop=True)
    )

    df = df.merge(
        monthly[["lat_grid", "lon_grid", "_year", "_month", "_hist_mean"]],
        on=["lat_grid", "lon_grid", "_year", "_month"],
        how="left",
    )
    df["fire_count_same_month_hist"] = df["_hist_mean"].fillna(0).astype(float)

    drop_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    return df


def add_radd_features(df: pd.DataFrame) -> pd.DataFrame:
    """CAUSAL past-only RADD alert features.

    RADD has a ~6–12 day revisit cycle so individual cell-days are mostly zero.
    Rolling windows of 14–90 days capture whether this cell has been confirmed as
    an active disturbance area by Sentinel-1 radar, independent of optical satellites.
    """
    if not any(c in df.columns for c in RADD_COLUMNS):
        return df

    df = _ensure_sorted(df)

    # Fill NaN with 0 — absence of RADD data means no confirmed alert
    for col in RADD_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("float32")

    if "radd_alert_count" in df.columns:
        for w in RADD_ROLLS:
            # CAUSAL: shift(1) before rolling so today's value is excluded  # CAUSAL
            df[f"radd_sum_{w}d"] = _past_roll(df, "radd_alert_count", w, "sum")
            df[f"radd_active_{w}d"] = _past_roll(df, "radd_alert_count", w, "active")

        for lag in RADD_LAGS:
            df[f"radd_lag_{lag}"] = (  # CAUSAL
                df.groupby(GROUP_KEYS, sort=False)["radd_alert_count"]
                .shift(lag)
                .fillna(0)
                .astype("float32")
                .to_numpy()
            )

    if "radd_confidence_max" in df.columns:
        for w in RADD_ROLLS:
            df[f"radd_conf_max_{w}d"] = _past_roll(df, "radd_confidence_max", w, "max")  # CAUSAL

    return df


def add_radd_cross_verified(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add radd_cross_verified analysis column (NOT a model feature — forward-looking).

    1 = at least one RADD alert appears in the same cell within `window` days of
    the FIRMS-labelled fire (i.e. the label target date). Used in train.py to
    report the fraction of FIRMS positives that are radar-confirmed.

    NEVER add radd_cross_verified to FEATURES_* — it reads future RADD data.
    """
    df = _ensure_sorted(df).copy()

    if "radd_alert_count" not in df.columns or "days_until_fire" not in df.columns:
        df["radd_cross_verified"] = np.int8(0)
        return df

    has_radd = (df["radd_alert_count"].fillna(0) > 0).astype("float32")
    df["_has_radd"] = has_radd

    # Forward sum: does this cell have any RADD alert in the next `window` days?
    fwd_sum = pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    for k in range(1, window + 1):
        fwd_sum += df.groupby(GROUP_KEYS, sort=False)["_has_radd"].shift(-k).fillna(0)

    is_positive = df["days_until_fire"].between(1, MAX_PREDICTION_DAYS)
    df["radd_cross_verified"] = ((fwd_sum > 0) & is_positive).astype(np.int8)
    df.drop(columns=["_has_radd"], inplace=True)
    return df


def add_multi_sat_confirmed(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi_sat_confirmed analysis column (NOT a model feature — forward-looking).

    For each positive label row (days_until_fire ∈ {1..IMMINENT_DAYS}), records
    how many independent VIIRS satellites confirmed the fire at the target date.
    Used in train.py to upweight high-confidence positive labels.

    NEVER add multi_sat_confirmed to FEATURES_* — it reads the future fire date.
    """
    if "n_satellites_today" not in df.columns or "days_until_fire" not in df.columns:
        df["multi_sat_confirmed"] = np.int8(0)
        return df

    df = _ensure_sorted(df).copy()
    sat_confirmed = pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)

    for k in range(1, MAX_PREDICTION_DAYS + 1):
        sat_at_k = (
            df.groupby(GROUP_KEYS, sort=False)["n_satellites_today"]
            .shift(-k)
            .fillna(0)
        )
        mask = df["days_until_fire"] == k
        sat_confirmed = sat_confirmed.where(~mask, sat_at_k)

    df["multi_sat_confirmed"] = sat_confirmed.astype(np.int8)
    return df


def make_label_days_until_fire(
    df: pd.DataFrame, horizon: int = MAX_PREDICTION_DAYS
) -> pd.DataFrame:
    """Vectorized label: days until next strict-future fire in this cell, ≤ horizon."""
    df = _ensure_sorted(df).copy()
    labels = np.full(len(df), -1, dtype=np.int8)
    fire_today = (df["fire_count"].fillna(0) > 0).astype(np.int8)

    for k in range(1, horizon + 1):
        shifted = (
            df.groupby(GROUP_KEYS, sort=False)["fire_count"]
            .shift(-k)
            .fillna(0)
            .to_numpy()
        )
        future_fire = shifted > 0
        unset = labels == -1
        labels = np.where(future_fire & unset, k, labels)

    df["days_until_fire"] = labels.astype(int)
    df["fire_today"] = fire_today.astype(int)
    return df


def build_features(
    daily: pd.DataFrame,
    horizon: int = MAX_PREDICTION_DAYS,
    grid_size: float = 0.1,
) -> pd.DataFrame:
    # ── Downcast input to float32 BEFORE feature engineering so every
    # downstream `.shift().rolling()` produces float32 Series too. This
    # halves PEAK memory during build (10-12 GB → 5-6 GB on a 4.4M-row
    # frame) which is what was OOM-killing predict-only on a 22 GB laptop.
    # lat_grid / lon_grid stay float64 for exact grid arithmetic.
    import gc
    daily = daily.copy()
    keep_64 = {"lat_grid", "lon_grid", "date"}
    for c in daily.columns:
        if c in keep_64:
            continue
        if daily[c].dtype == "float64":
            daily[c] = daily[c].astype("float32")
    gc.collect()

    df = add_neighbor_features(daily, grid_size=grid_size)
    df = add_temporal_features(df)
    df = add_calendar_features(df)
    df = add_dry_streak(df)
    df = add_cumulative_fire_rate(df)
    df = add_season_features(df)
    df = add_urban_distance(df)
    df = add_fire_recurrence_features(df)
    df = add_radd_features(df)
    df = make_label_days_until_fire(df, horizon=horizon)
    df = add_radd_cross_verified(df)
    df = add_multi_sat_confirmed(df)

    # ── Downcast float64 → float32 across all feature columns ──
    # On a 4.4M × ~180-col frame this halves memory (≈3.5 GB → ≈1.8 GB)
    # without affecting LightGBM accuracy (the histogram binner doesn't care
    # about the extra precision). lat_grid / lon_grid stay float64 for
    # exact grid arithmetic; the label is int.
    import gc
    keep_64 = {"lat_grid", "lon_grid"}
    for c in df.columns:
        if c in keep_64:
            continue
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64" and c != "days_until_fire":
            df[c] = df[c].astype("int32")
    gc.collect()

    log.info(
        "Built features for %d rows, %d positive labels (fire within %d days) — float32",
        len(df),
        int((df["days_until_fire"] >= 0).sum()),
        horizon,
    )
    return df


# ─────────────────────────────────────────────
# Feature contract — single source of truth
# ─────────────────────────────────────────────
def _build_core_feature_list() -> List[str]:
    cols: List[str] = []
    cols += [f"fire_lag_{l}" for l in LAG_DAYS]
    cols += [f"frp_lag_{l}" for l in LAG_DAYS]
    for w in ROLL_WINDOWS:
        cols += [
            f"fire_sum_{w}d",
            f"frp_sum_{w}d",
            f"frp_max_{w}d",
            f"active_days_{w}d",
        ]
    cols += ["frp_trend_1d", "frp_trend_3d", "frp_trend_7d", "fire_std_7d"]
    cols += [
        "fire_count_today",
        "frp_sum_today",
        "bright_mean_today",
        "confidence_mean_today",
        "night_fire_count",
        "afternoon_fire_count",
        "n_satellites_today",
    ]
    cols += [
        "fire_acceleration",
        "frp_intensity_7d",
        "night_fire_ratio",
        "fire_streak_today",
        "fire_momentum_7_30",
        "frp_per_fire_today",
        # NEW per spec
        "fire_count_60d",
        "fire_frequency_rate",
    ]
    # spatial inner ring
    cols += ["neighbor_fire_today", "neighbor_frp_today"]
    cols += [f"neighbor_fire_lag_{l}" for l in NEIGHBOR_LAGS]
    cols += [f"neighbor_frp_lag_{l}" for l in NEIGHBOR_LAGS]
    for w in NEIGHBOR_ROLLS:
        cols += [f"neighbor_fire_sum_{w}d", f"neighbor_frp_sum_{w}d"]
    cols += ["neighbor_fire_velocity_3d", "neighbor_frp_velocity_3d"]
    # outer ring
    cols += ["wide_neighbor_fire_today", "wide_neighbor_frp_today"]
    cols += [f"wide_neighbor_fire_lag_{l}" for l in NEIGHBOR_LAGS]
    cols += [f"wide_neighbor_frp_lag_{l}" for l in NEIGHBOR_LAGS]
    for w in NEIGHBOR_ROLLS:
        cols += [f"wide_neighbor_fire_sum_{w}d", f"wide_neighbor_frp_sum_{w}d"]
    cols += ["wide_neighbor_fire_velocity_3d"]
    # calendar
    cols += [
        "month_sin", "month_cos", "doy_sin", "doy_cos",
        "is_burn_season", "is_dry_season",   # is_dry_season is NEW per spec
        "days_from_burn_peak",
        "week_sin", "week_cos",
        "day_of_week", "is_weekend",
    ]
    # per-cell static + causal derivatives
    cols += [
        "distance_to_nearest_city_km",
        "fire_days_per_year_so_far",
        "days_since_last_fire",
        "season_fire_count_so_far",
        "days_into_burn_season",
        "median_fire_gap",
        "days_since_last_fire_norm",
        "same_week_fire_last_year",
        "fire_count_same_month_hist",
    ]
    # vegetation
    cols += [
        "tree_cover_pct_2000",
        "tree_loss_pct_recent",
    ]
    cols += ["lat_grid", "lon_grid"]
    return cols


def _build_radd_feature_list() -> List[str]:
    cols: List[str] = []
    for w in RADD_ROLLS:
        cols += [f"radd_sum_{w}d", f"radd_active_{w}d", f"radd_conf_max_{w}d"]
    for lag in RADD_LAGS:
        cols.append(f"radd_lag_{lag}")
    return cols


def _build_weather_feature_list() -> List[str]:
    cols: List[str] = []
    for c in WEATHER_COLUMNS:
        cols.append(f"{c}_today")
        for lag in WEATHER_LAGS:
            cols.append(f"{c}_lag_{lag}")
        for w in WEATHER_ROLLS:
            cols.append(f"{c}_mean_{w}d")
    return cols


FEATURES_CORE: Tuple[str, ...] = tuple(_build_core_feature_list())
FEATURES_WEATHER: Tuple[str, ...] = tuple(_build_weather_feature_list())
FEATURES_RADD: Tuple[str, ...] = tuple(_build_radd_feature_list())
FEATURES: Tuple[str, ...] = FEATURES_CORE


def resolve_features(df: pd.DataFrame) -> List[str]:
    """Return the feature list actually present in `df`, including optional columns."""
    feats: List[str] = [c for c in FEATURES_CORE if c in df.columns]
    feats += [c for c in FEATURES_WEATHER if c in df.columns]
    feats += [c for c in FEATURES_RADD if c in df.columns]
    return feats
