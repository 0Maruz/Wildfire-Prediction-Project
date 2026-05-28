"""Load, clean, grid, and aggregate NASA FIRMS / raw hotspot CSVs.

End product: a daily cell-day dataframe with columns
    [lat_grid, lon_grid, date, fire_count, frp_sum, frp_mean, frp_max,
     bright_mean, bright_max, confidence_mean]

Densification: by default we expand sparse fire-only rows into a dense
(active-cell × every-day) grid where days with no detection are filled with
zeros. This gives correct "yesterday/last-week" semantics for rolling features.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from storage import list_tables, read_table, resolve_existing
from urban_areas import classify_urban

log = logging.getLogger("data_loader")

DEFAULT_GRID = 0.1

AGG_COLUMNS = [
    "fire_count",
    "frp_sum",
    "frp_mean",
    "frp_max",
    "bright_mean",
    "bright_max",
    "confidence_mean",
    # Time-of-day stratification (Thai local hours, derived from real UTC).
    # `night_fire_count` = detections at hour 22-23 or 0-5 (Thai local) — these
    # tend to be larger fires that survived dusk; `afternoon_fire_count` =
    # 12-17 = peak agricultural-burn window.
    "night_fire_count",
    "afternoon_fire_count",
    # How many of the 3 VIIRS satellites (SNPP, NOAA20, NOAA21) detected this
    # cell on this date — multi-satellite consensus is a confidence signal.
    "n_satellites_today",
]

# Names produced by fetch_weather.py — must match features.WEATHER_COLUMNS.
WEATHER_COLUMNS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]

# Names produced by fetch_radd_alerts.py — must match features.RADD_COLUMNS.
RADD_COLUMNS = ["radd_alert_count", "radd_confidence_max"]


def _coerce_acq_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Build a datetime from FIRMS' split acq_date / acq_time columns."""
    df = df.copy()
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str)
        + " "
        + df["acq_time"].astype(str).str.zfill(4),
        errors="coerce",
    )
    return df


def _parse_confidence(series: pd.Series) -> pd.Series:
    """VIIRS reports confidence as l/n/h letters; MODIS reports 0-100 integers."""
    letter_map = {"l": 0, "n": 50, "h": 100}
    mapped = series.astype(str).str.lower().map(letter_map)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


# Minimum schema every FIRMS file must satisfy. Bulk archives use
# `latitude`/`longitude` while NRT exports also include the same. Bright
# columns vary (`bright_ti4` for VIIRS, `bright` for MODIS) so we require
# at least one — `clean_hotspots` then derives `bright_main` from whichever
# is present.
FIRMS_REQUIRED = {"latitude", "longitude", "acq_date", "acq_time", "frp"}
# VIIRS uses `bright_ti4`; MODIS bulk archives use `brightness`. Older NRT
# exports sometimes use `bright`. Any one is enough for clean_hotspots to
# derive `bright_main`.
FIRMS_BRIGHT_ALTERNATIVES = ("bright_ti4", "brightness", "bright")


def _validate_firms_schema(df: pd.DataFrame, source: str) -> None:
    """Fail-fast schema check. Catches truncated downloads and corrupt files
    before we burn 30 minutes on feature engineering only to crash mid-train.
    """
    cols = set(df.columns)
    missing = FIRMS_REQUIRED - cols
    if missing:
        raise ValueError(
            f"FIRMS file '{source}' is missing required columns {sorted(missing)}. "
            f"Got columns: {sorted(cols)}. Re-run fetch_firms.py or replace the file."
        )
    if not any(c in cols for c in FIRMS_BRIGHT_ALTERNATIVES):
        raise ValueError(
            f"FIRMS file '{source}' has no brightness column "
            f"(expected one of {FIRMS_BRIGHT_ALTERNATIVES}). Got: {sorted(cols)}."
        )
    if df.empty:
        raise ValueError(f"FIRMS file '{source}' is empty.")


def _infer_dataset_from_filename(path: str) -> str:
    """Map FIRMS bulk-archive filenames to the same dataset labels fetch_firms.py
    uses, so multi-satellite consensus features work consistently across the
    historical archive and live NRT data.

      fire_*_J1V-C2_*.parquet → VIIRS_NOAA20_NRT
      fire_*_J2V-C2_*.parquet → VIIRS_NOAA21_NRT
      fire_*_SV-C2_*.parquet  → VIIRS_SNPP_NRT
    """
    name = os.path.basename(path).upper()
    if "J1V" in name:
        return "VIIRS_NOAA20_NRT"
    if "J2V" in name:
        return "VIIRS_NOAA21_NRT"
    if "SV-C2" in name or "_SV_" in name:
        return "VIIRS_SNPP_NRT"
    return "UNKNOWN"


def load_firms_csv(paths_or_globs: Iterable[str]) -> pd.DataFrame:
    """Load FIRMS hotspots from any mix of CSV / Parquet files, dirs, or globs."""
    files = list_tables(paths_or_globs)
    if not files:
        raise FileNotFoundError(
            f"No FIRMS CSV/Parquet files found at: {list(paths_or_globs)}. Run fetch_firms.py first."
        )

    log.info("Loading %d FIRMS file(s)", len(files))
    frames = []
    for f in files:
        df = read_table(f)
        _validate_firms_schema(df, f)
        # Bulk archive files don't carry a `dataset` column (one satellite per
        # file). Derive it from the filename so downstream multi-satellite
        # aggregations work without special-casing.
        if "dataset" not in df.columns:
            df = df.copy()
            df["dataset"] = _infer_dataset_from_filename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_hotspots(
    df: pd.DataFrame,
    min_confidence: int = 0,
    drop_frp_outliers: bool = True,
    frp_quantile: float = 0.999,
) -> pd.DataFrame:
    """Standardize columns, coerce numerics, apply quality + outlier filters."""
    df = _coerce_acq_datetime(df)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    df["bright_main"] = np.nan
    if "bright_ti4" in df.columns:
        df["bright_main"] = df["bright_ti4"]
    if "brightness" in df.columns:
        df["bright_main"] = df["bright_main"].fillna(df["brightness"])
    if "bright" in df.columns:
        df["bright_main"] = df["bright_main"].fillna(df["bright"])

    if "confidence" in df.columns:
        df["confidence"] = _parse_confidence(df["confidence"])
    else:
        df["confidence"] = np.nan

    for col in ["lat", "lon", "frp", "bright_main", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = df["acq_datetime"].dt.date
    df = df.dropna(subset=["lat", "lon", "date", "frp"])

    df = df[(df["frp"] >= 0) & (df["frp"].notna())]
    if drop_frp_outliers and len(df) > 1000:
        cap = df["frp"].quantile(frp_quantile)
        n_before = len(df)
        df = df[df["frp"] <= cap]
        log.info(
            "Capped FRP at q%.3f=%.1f, dropped %d rows",
            frp_quantile,
            cap,
            n_before - len(df),
        )

    if min_confidence > 0:
        n_before = len(df)
        df = df[df["confidence"].fillna(0) >= min_confidence]
        log.info(
            "Confidence filter ≥%d dropped %d rows",
            min_confidence,
            n_before - len(df),
        )

    # Thai-local hour, derived from the real UTC acq_datetime. FIRMS reports
    # acq_time in UTC; Thailand is UTC+7. Used downstream for night-vs-day
    # fire stratification.
    df["thai_hour"] = (df["acq_datetime"].dt.hour + 7) % 24
    if "dataset" not in df.columns:
        df["dataset"] = "UNKNOWN"

    keep = ["lat", "lon", "date", "frp", "bright_main", "confidence",
            "acq_datetime", "thai_hour", "dataset"]
    return df[keep].reset_index(drop=True)


def filter_urban_hotspots(
    df: pd.DataFrame,
    buffer_km: float = 0.0,
) -> pd.DataFrame:
    """Drop hotspots that fall inside any curated Thai urban-area exclusion zone.

    Cities produce non-wildfire FIRMS detections (garbage burning, industrial
    heat, structure fires) that bias the model toward predicting "wildfire"
    in Bangkok / Chiang Mai / etc. The curated list + radii live in
    ``urban_areas.py``.
    """
    if df.empty:
        return df
    is_urban, _, _ = classify_urban(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        buffer_km=buffer_km,
    )
    n_before = len(df)
    out = df.loc[~is_urban].reset_index(drop=True)
    log.info(
        "Urban filter (buffer=%.1f km): dropped %d / %d hotspots",
        buffer_km, n_before - len(out), n_before,
    )
    return out


def grid_and_aggregate(
    df: pd.DataFrame, grid_size: float = DEFAULT_GRID
) -> pd.DataFrame:
    """Snap each detection to a lat/lon grid cell and aggregate to one row per cell-day."""
    df = df.copy()
    df["lat_grid"] = (df["lat"] / grid_size).round() * grid_size
    df["lon_grid"] = (df["lon"] / grid_size).round() * grid_size

    # Time-of-day flags computed *before* the groupby so `.sum` over them
    # gives per-cell-day counts. `night` = Thai 22:00-05:59;
    # `afternoon` = Thai 12:00-17:59 (peak agri-burn window).
    night_hours = df["thai_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    afternoon_hours = df["thai_hour"].between(12, 17)
    df["_is_night"] = night_hours.astype(int)
    df["_is_afternoon"] = afternoon_hours.astype(int)

    daily = df.groupby(["lat_grid", "lon_grid", "date"], as_index=False).agg(
        fire_count=("frp", "count"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        bright_mean=("bright_main", "mean"),
        bright_max=("bright_main", "max"),
        confidence_mean=("confidence", "mean"),
        night_fire_count=("_is_night", "sum"),
        afternoon_fire_count=("_is_afternoon", "sum"),
        n_satellites_today=("dataset", "nunique"),
    )

    return daily.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)


def densify_active_cells(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand to (active-cell × every-day-in-range) so rolling features see real zero-days.

    Active cell = cell with at least one detection in the dataset. Inactive cells
    are excluded entirely (they have no signal to learn from).
    """
    if daily.empty:
        return daily

    cells = daily[["lat_grid", "lon_grid"]].drop_duplicates().reset_index(drop=True)
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    dense_idx = cells.assign(_k=1).merge(
        pd.DataFrame({"date": date_range, "_k": 1}), on="_k"
    ).drop(columns="_k")

    dense_idx["date"] = pd.to_datetime(dense_idx["date"]).dt.date
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    out = dense_idx.merge(daily, on=["lat_grid", "lon_grid", "date"], how="left")
    out[AGG_COLUMNS] = out[AGG_COLUMNS].fillna(0.0)

    log.info(
        "Densified %d active cells × %d days = %d rows",
        len(cells),
        len(date_range),
        len(out),
    )
    return out.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)


def merge_tree_cover(daily: pd.DataFrame, tree_cover_path: Optional[str]) -> pd.DataFrame:
    """Left-join Hansen GFC tree cover features (per-cell, static).

    Cache columns: lat_grid, lon_grid, tree_cover_pct_2000, tree_loss_pct_recent.
    No-op if cache missing — features.py will simply not see the columns
    and resolve_features will skip them.
    """
    if not tree_cover_path or not os.path.exists(tree_cover_path):
        return daily

    try:
        tc = read_table(tree_cover_path)
    except Exception as exc:
        log.warning("Could not read tree_cover cache (%s): %s — skipping merge.",
                    tree_cover_path, exc)
        return daily

    expected = {"lat_grid", "lon_grid", "tree_cover_pct_2000", "tree_loss_pct_recent"}
    missing = expected - set(tc.columns)
    if missing:
        log.warning("tree_cover cache missing columns %s — skipping merge.", sorted(missing))
        return daily

    tc = tc.copy()
    tc["lat_grid"] = tc["lat_grid"].round(6)
    tc["lon_grid"] = tc["lon_grid"].round(6)

    daily = daily.copy()
    daily["lat_grid"] = daily["lat_grid"].round(6)
    daily["lon_grid"] = daily["lon_grid"].round(6)

    merged = daily.merge(
        tc[["lat_grid", "lon_grid", "tree_cover_pct_2000", "tree_loss_pct_recent"]],
        on=["lat_grid", "lon_grid"],
        how="left",
    )
    n_with_cover = merged["tree_cover_pct_2000"].notna().sum()
    log.info(
        "Merged Hansen tree cover: %d / %d rows have tree_cover values",
        int(n_with_cover), len(merged),
    )
    return merged


def merge_weather(daily: pd.DataFrame, weather_path: Optional[str]) -> pd.DataFrame:
    """Left-join real ERA5 weather (from fetch_weather.py cache) onto the daily frame.

    No-op if weather_path is missing or the file doesn't exist. The cache must
    have columns [lat_grid, lon_grid, date, temp_max, temp_min, precip_sum,
    wind_max, et0]. Cells / dates not in the cache get NaN, which features.py
    converts to 0 only at model-input time.
    """
    actual = resolve_existing(weather_path) if weather_path else None
    if not actual:
        return daily

    try:
        wx = read_table(actual)
    except Exception as exc:
        log.warning("Could not read weather cache (%s): %s — skipping merge.", actual, exc)
        return daily

    if wx.empty:
        return daily

    expected = {"lat_grid", "lon_grid", "date", *WEATHER_COLUMNS}
    missing = expected - set(wx.columns)
    if missing:
        log.warning(
            "Weather cache is missing columns %s — skipping merge.", sorted(missing)
        )
        return daily

    wx["date"] = pd.to_datetime(wx["date"]).dt.date
    wx["lat_grid"] = wx["lat_grid"].round(6)
    wx["lon_grid"] = wx["lon_grid"].round(6)

    daily = daily.copy()
    daily["lat_grid"] = daily["lat_grid"].round(6)
    daily["lon_grid"] = daily["lon_grid"].round(6)

    merged = daily.merge(
        wx[["lat_grid", "lon_grid", "date", *WEATHER_COLUMNS]],
        on=["lat_grid", "lon_grid", "date"],
        how="left",
    )
    log.info(
        "Merged weather: %d / %d rows have real ERA5 values",
        int(merged[WEATHER_COLUMNS[0]].notna().sum()),
        len(merged),
    )
    return merged


def merge_radd(daily: pd.DataFrame, radd_path: Optional[str]) -> pd.DataFrame:
    """Left-join RADD alert aggregates (from fetch_radd_alerts.py) onto the daily frame.

    No-op if radd_path is missing or the file doesn't exist. Cells / dates not
    in the cache get NaN (treated as 0 in features.py). RADD only covers forest
    cells and has a ~6–12 day revisit, so most rows will be NaN — that's expected.
    """
    actual = resolve_existing(radd_path) if radd_path else None
    if not actual:
        return daily

    try:
        radd = read_table(actual)
    except Exception as exc:
        log.warning("Could not read RADD cache (%s): %s — skipping merge.", actual, exc)
        return daily

    if radd.empty:
        return daily

    expected = {"lat_grid", "lon_grid", "date", *RADD_COLUMNS}
    missing = expected - set(radd.columns)
    if missing:
        log.warning("RADD cache missing columns %s — skipping merge.", sorted(missing))
        return daily

    radd["date"] = pd.to_datetime(radd["date"]).dt.date
    radd["lat_grid"] = radd["lat_grid"].round(6)
    radd["lon_grid"] = radd["lon_grid"].round(6)

    daily = daily.copy()
    daily["lat_grid"] = daily["lat_grid"].round(6)
    daily["lon_grid"] = daily["lon_grid"].round(6)

    merged = daily.merge(
        radd[["lat_grid", "lon_grid", "date", *RADD_COLUMNS]],
        on=["lat_grid", "lon_grid", "date"],
        how="left",
    )
    hit = int(merged[RADD_COLUMNS[0]].notna().sum())
    log.info(
        "Merged RADD: %d / %d rows have alert data (%.1f%%)",
        hit, len(merged), 100.0 * hit / max(len(merged), 1),
    )
    return merged


def load_and_prepare(
    raw_dir: Optional[str],
    firms_path: Optional[str],
    grid_size: float = DEFAULT_GRID,
    min_confidence: int = 0,
    densify: bool = True,
    weather_path: Optional[str] = None,
    tree_cover_path: Optional[str] = None,
    radd_path: Optional[str] = None,
    filter_urban: bool = True,
    urban_buffer_km: float = 0.0,
) -> pd.DataFrame:
    """One-shot: glob raw_dir + firms_path → cleaned, gridded, daily aggregate.

    If ``weather_path`` is provided and exists, real ERA5 daily aggregates are
    left-joined onto the densified frame. Run ``fetch_weather.py`` to populate.

    When ``filter_urban=True`` (default), hotspots inside any curated Thai
    urban exclusion zone (``urban_areas.THAI_URBAN_AREAS``) are dropped before
    gridding so the model trains on wildfire-only signal.
    """
    sources: List[str] = []
    if raw_dir:
        sources.append(raw_dir)
    if firms_path:
        sources.append(firms_path)

    raw = load_firms_csv(sources)
    log.info("Loaded %d raw hotspot rows", len(raw))

    cleaned = clean_hotspots(raw, min_confidence=min_confidence)
    log.info("After cleaning: %d rows", len(cleaned))

    if filter_urban:
        cleaned = filter_urban_hotspots(cleaned, buffer_km=urban_buffer_km)

    daily = grid_and_aggregate(cleaned, grid_size=grid_size)
    log.info(
        "Aggregated to %d cell-day rows across %d cells",
        len(daily),
        daily[["lat_grid", "lon_grid"]].drop_duplicates().shape[0],
    )

    if densify:
        daily = densify_active_cells(daily)

    daily = merge_tree_cover(daily, tree_cover_path)
    daily = merge_weather(daily, weather_path)
    daily = merge_radd(daily, radd_path)
    return daily
