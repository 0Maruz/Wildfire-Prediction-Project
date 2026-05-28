"""GFW RADD (Radar Alert for Detecting Deforestation) fetcher for Thailand.

Sentinel-1 radar backscatter — independent of optical satellites (VIIRS/MODIS).
Used as a cross-verification source for FIRMS hotspot labels: a cell where both
FIRMS and RADD fire independently, radar confirms the disturbance actually happened.

Service: GFW Integrated Deforestation Alerts (no auth required)
  https://services8.arcgis.com/oTalEaSXAuyNT7xf/ArcGIS/rest/services/
          Deforestation_Alerts/FeatureServer/0

Output schema (data/radd/radd_alerts.parquet):
  lat_grid, lon_grid  — snapped to same 0.1° grid as FIRMS
  date                — alert date (Python date)
  radd_alert_count    — number of RADD alert pixels in this cell on this date
  radd_confidence_max — highest confidence in the cell (2=nominal, 3=high, 4=highest)

Note: RADD detects ANY forest disturbance (logging, fire, clearing).  Cross-checking
against FIRMS narrows it to fire-origin disturbances in data_loader / features.

Coverage: Southeast Asia from 2020-01-01 onward.
Spatial resolution: 10 m (Sentinel-1 native), aggregated here to 0.1° cells.
Temporal cadence: ~6–12 day revisit per area; new alerts published within days.

Usage::
    cd src && python fetch_radd_alerts.py
    cd src && python fetch_radd_alerts.py --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from storage import read_table, resolve_existing, write_table

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetch_radd_alerts")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(BASE_DIR, "data", "radd", "radd_alerts.parquet")

FEATURE_SERVICE = (
    "https://services8.arcgis.com/oTalEaSXAuyNT7xf/ArcGIS/rest/services"
    "/Deforestation_Alerts/FeatureServer/0/query"
)

# Thailand bounding box in ArcGIS envelope format
TH_BBOX = {"xmin": 96, "ymin": 4, "xmax": 107, "ymax": 22, "spatialReference": {"wkid": 4326}}

GRID_SIZE = float(os.getenv("GRID_SIZE", "0.1"))
PAGE_SIZE = 1000  # ArcGIS default max per request

# Confidence string → numeric (matches RADD specification)
CONF_MAP = {"nominal": 2, "high": 3, "highest": 4}

# Fields as named in the GFW Integrated Alerts feature service
DATE_FIELD = "wur_radd_alerts__date"
CONF_FIELD = "wur_radd_alerts__confidence"
OUT_FIELDS = f"latitude,longitude,{DATE_FIELD},{CONF_FIELD}"


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _snap(v: float) -> float:
    return round(round(v / GRID_SIZE) * GRID_SIZE, 6)


def _parse_features(features: list[dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        a = feat.get("attributes", {})
        lat_raw = a.get("latitude")
        lon_raw = a.get("longitude")
        date_raw = a.get(DATE_FIELD)
        conf_raw = str(a.get(CONF_FIELD) or "").strip().lower()

        if lat_raw is None or lon_raw is None or date_raw is None:
            continue

        # date_raw may be epoch-ms (int) or ISO string
        if isinstance(date_raw, (int, float)):
            alert_date = datetime.utcfromtimestamp(date_raw / 1000).date()
        else:
            try:
                alert_date = datetime.strptime(str(date_raw)[:10], "%Y-%m-%d").date()
            except ValueError:
                continue

        conf = CONF_MAP.get(conf_raw)
        if conf is None:
            # Skip "not_detected" and unrecognised values
            continue

        rows.append({
            "lat_grid":  _snap(float(lat_raw)),
            "lon_grid":  _snap(float(lon_raw)),
            "date":       alert_date,
            "radd_confidence": conf,
        })

    return pd.DataFrame(rows)


def _build_where(start: Optional[date], end: Optional[date]) -> str:
    # Base: only RADD-sourced alerts (not GLAD or other integrated sources)
    clauses = [f"{CONF_FIELD} IS NOT NULL", f"{CONF_FIELD} <> 'not_detected'"]
    if start:
        clauses.append(f"{DATE_FIELD} >= '{start.isoformat()}'")
    if end:
        clauses.append(f"{DATE_FIELD} <= '{end.isoformat()}'")
    return " AND ".join(clauses)


def fetch_radd(
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch RADD alerts for Thailand, return aggregated per-cell-day frame."""
    session = _make_session()
    where = _build_where(start, end)
    frames: list[pd.DataFrame] = []
    offset = 0

    log.info("Fetching RADD alerts (where: %s) …", where)
    while True:
        params = {
            "where": where,
            "geometry": str(TH_BBOX).replace("'", '"'),
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": OUT_FIELDS,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        try:
            resp = session.get(FEATURE_SERVICE, params=params, timeout=90)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            log.error("Request failed at offset %d: %s", offset, exc)
            break
        except Exception as exc:
            log.error("JSON parse error at offset %d: %s", offset, exc)
            break

        if "error" in payload:
            log.error("ArcGIS error: %s", payload["error"])
            break

        features = payload.get("features", [])
        if not features:
            break

        df = _parse_features(features)
        if not df.empty:
            frames.append(df)
        log.info("  offset %d: %d raw → %d parsed", offset, len(features), len(df))

        exceeded = payload.get("exceededTransferLimit", False)
        if exceeded or len(features) == PAGE_SIZE:
            offset += PAGE_SIZE
        else:
            break

    if not frames:
        log.warning("No RADD alerts fetched for this range.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date

    # Aggregate pixel-level records to 0.1° cell-day
    agg = (
        raw.groupby(["lat_grid", "lon_grid", "date"], as_index=False)
        .agg(
            radd_alert_count=("radd_confidence", "count"),
            radd_confidence_max=("radd_confidence", "max"),
        )
    )
    log.info("Fetched %d raw alert pixels → %d cell-day rows", len(raw), len(agg))
    return agg


def update_radd(
    start: Optional[date] = None,
    end: Optional[date] = None,
    out_file: Optional[str] = None,
) -> int:
    """Fetch new RADD data, merge with on-disk cache, dedup, persist."""
    out_file = out_file or OUT_FILE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Incremental: if cache exists and no start given, resume from last cached date
    existing_path = resolve_existing(out_file)
    old_df = pd.DataFrame()
    if existing_path and os.path.getsize(existing_path) > 0:
        try:
            old_df = read_table(existing_path)
            old_df["date"] = pd.to_datetime(old_df["date"]).dt.date
            if start is None:
                last = old_df["date"].max()
                start = last - timedelta(days=7)  # 7-day overlap to catch late alerts
                log.info("Incremental mode: resuming from %s", start)
        except Exception as exc:
            log.warning("Could not read existing cache (%s) — starting fresh", exc)
            old_df = pd.DataFrame()

    new_df = fetch_radd(start=start, end=end)
    if new_df.empty and old_df.empty:
        log.warning("Nothing to save.")
        return 0

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    combined = combined.dropna(subset=["lat_grid", "lon_grid", "date"])

    before = len(combined)
    combined = (
        combined
        .sort_values(["lat_grid", "lon_grid", "date"])
        .drop_duplicates(subset=["lat_grid", "lon_grid", "date"], keep="last")
        .reset_index(drop=True)
    )

    write_table(combined, out_file)
    log.info(
        "Saved %d rows (deduped %d) → %s", len(combined), before - len(combined), out_file
    )
    log.info(
        "  date range: %s → %s",
        combined["date"].min(),
        combined["date"].max(),
    )
    conf_dist = combined["radd_confidence_max"].value_counts().to_dict()
    log.info("  confidence distribution: %s", conf_dist)
    return len(combined)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch GFW RADD alerts for Thailand")
    p.add_argument("--start", metavar="YYYY-MM-DD", help="Start date (inclusive)")
    p.add_argument("--end",   metavar="YYYY-MM-DD", help="End date (inclusive)")
    p.add_argument("--out",   default=OUT_FILE, help="Output parquet path")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    start = date.fromisoformat(args.start) if args.start else None
    end   = date.fromisoformat(args.end)   if args.end   else None
    update_radd(start=start, end=end, out_file=args.out)
