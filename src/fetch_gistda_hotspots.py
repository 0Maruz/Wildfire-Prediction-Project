"""GISTDA ArcGIS REST hotspot fetcher (VIIRS NPP + MODIS).

GISTDA (Thailand's Geo-Informatics and Space Technology Development Agency)
re-processes VIIRS Suomi NPP and MODIS fire detections and serves them via
ArcGIS MapServer REST APIs updated ~12-hourly. These are the same satellite
sources as NASA FIRMS but independently processed, with Thai land-use
classification (``lu_name``) and administrative names already attached to
each detection.

Services (no auth required):
  FR_Fire/hotspot_npp_daily/MapServer/0  — VIIRS-N (SNPP), max 12 000 rec
  FR_Fire/hotspot_daily/MapServer/0       — MODIS (Terra/Aqua), max 1 000 rec

Note: GISTDA does not expose FRP or brightness temperature. Those columns
are filled with NaN so the output file remains join-compatible with
data/firms/firms_all.parquet. In data_loader.py the two sources are kept
separate; frp/brightness features come exclusively from FIRMS.

Unique value-add over FIRMS:
  • ``lu_name`` — Thai land-use category per detection (national forest,
    conservation forest, agriculture, etc.). Lets data_loader compute
    "fraction of historic fires that were agricultural burns" per cell.
  • Thai administrative names already attached (pv_tn/ap_tn/tb_tn).

Output schema:
  latitude, longitude, acq_datetime, frp (NaN), bright_ti4 (NaN),
  confidence, dataset, satellite, lu_name, pv_tn, ap_tn, tb_tn

Cache: data/gistda/gistda_hotspots.parquet (accumulative; same merge
  pattern as fetch_firms.py — new rows appended, deduped by
  lat/lon/acq_datetime).

Usage::
    cd src && python fetch_gistda_hotspots.py
    cd src && python fetch_gistda_hotspots.py --start 2025-01-01 --end 2025-03-31
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from storage import read_table, resolve_existing, write_table

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetch_gistda_hotspots")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GISTDA_BASE = "https://gistdaportal.gistda.or.th/data/rest/services/FR_Fire"
TH_GEOMETRY = "96,4,107,22"  # minX,minY,maxX,maxY (ArcGIS envelope bbox)

OUT_FILE = os.path.join(BASE_DIR, "data", "gistda", "gistda_hotspots.parquet")

# ArcGIS MapServer layer endpoints
VIIRS_NPP_URL = f"{GISTDA_BASE}/hotspot_npp_daily/MapServer/0"
MODIS_URL = f"{GISTDA_BASE}/hotspot_daily/MapServer/0"

# GISTDA VIIRS confident field is categorical; map to 0–100 numeric
CONF_STR_MAP = {"high": 80, "nominal": 50, "low": 20}

# Chunk size for date-range fetching — keeps each query under the 12 000
# record ceiling even during peak fire season (March–April).
CHUNK_DAYS = 3


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _epoch_ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _date_to_epoch_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _query_layer(
    session: requests.Session,
    url: str,
    where: str = "1=1",
) -> list[dict]:
    """Issue one ArcGIS /query request; return list of feature dicts."""
    params = {
        "where": where,
        "geometry": TH_GEOMETRY,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "json",
    }
    try:
        res = session.get(f"{url}/query", params=params, timeout=60)
        res.raise_for_status()
    except requests.RequestException as exc:
        log.error("Request failed for %s: %s", url, exc)
        return []

    try:
        payload = res.json()
    except Exception as exc:
        log.error("JSON parse failed for %s: %s", url, exc)
        return []

    if "error" in payload:
        log.error("ArcGIS error for %s: %s", url, payload["error"])
        return []

    features = payload.get("features", [])
    exceeded = payload.get("exceededTransferLimit", False)
    if exceeded:
        log.warning("%s: transfer limit exceeded — result is truncated", url)
    return features


def _parse_viirs_npp(features: list[dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        a = feat.get("attributes", {})
        raw_date = a.get("date")
        if raw_date is None:
            continue
        dt = _epoch_ms_to_datetime(raw_date)

        time_str = str(a.get("time") or "").strip()
        if time_str:
            try:
                h, m, s = time_str.split(":")
                dt = dt.replace(hour=int(h), minute=int(m), second=int(s))
            except (ValueError, TypeError):
                pass

        conf_raw = str(a.get("confident") or "").lower()
        conf = CONF_STR_MAP.get(conf_raw)

        rows.append({
            "latitude":    a.get("latitude"),
            "longitude":   a.get("longitude"),
            "acq_datetime": dt,
            "frp":         float("nan"),
            "bright_ti4":  float("nan"),
            "confidence":  float(conf) if conf is not None else float("nan"),
            "dataset":     "GISTDA_VIIRS_NPP",
            "satellite":   str(a.get("satellite") or "N"),
            "lu_name":     str(a.get("lu_name") or ""),
            "pv_tn":       str(a.get("pv_tn") or ""),
            "ap_tn":       str(a.get("ap_tn") or ""),
            "tb_tn":       str(a.get("tb_tn") or ""),
        })
    return pd.DataFrame(rows)


def _parse_modis(features: list[dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        a = feat.get("attributes", {})
        raw_date = a.get("datetime")
        if raw_date is None:
            continue
        dt = _epoch_ms_to_datetime(raw_date)

        conf_raw = a.get("confident")
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = float("nan")

        rows.append({
            "latitude":    a.get("latitude"),
            "longitude":   a.get("longitude"),
            "acq_datetime": dt,
            "frp":         float("nan"),
            "bright_ti4":  float("nan"),
            "confidence":  conf,
            "dataset":     "GISTDA_MODIS",
            "satellite":   str(a.get("satellite") or ""),
            "lu_name":     str(a.get("lu_name") or ""),
            "pv_tn":       str(a.get("pv_tn") or ""),
            "ap_tn":       str(a.get("ap_tn") or ""),
            "tb_tn":       str(a.get("tb_tb") or a.get("tb_tn") or ""),
        })
    return pd.DataFrame(rows)


def _fetch_layer_chunked(
    session: requests.Session,
    url: str,
    date_field: str,
    parse_fn,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> pd.DataFrame:
    """Fetch a layer, splitting into CHUNK_DAYS windows if a range is given."""
    if start_ms is None or end_ms is None:
        features = _query_layer(session, url)
        df = parse_fn(features)
        log.info("  %s: %d records (no date filter)", url.rsplit("/", 2)[-2], len(df))
        return df

    frames: list[pd.DataFrame] = []
    cursor_ms = start_ms
    chunk_ms = CHUNK_DAYS * 86_400_000

    while cursor_ms < end_ms:
        chunk_end_ms = min(cursor_ms + chunk_ms, end_ms)
        where = f"{date_field} >= {cursor_ms} AND {date_field} < {chunk_end_ms}"
        features = _query_layer(session, url, where=where)
        df = parse_fn(features)
        if not df.empty:
            frames.append(df)
        chunk_start_dt = _epoch_ms_to_datetime(cursor_ms).strftime("%Y-%m-%d")
        chunk_end_dt = _epoch_ms_to_datetime(chunk_end_ms - 1).strftime("%Y-%m-%d")
        log.info(
            "  %s %s → %s: %d records",
            url.rsplit("/", 2)[-2], chunk_start_dt, chunk_end_dt, len(df),
        )
        cursor_ms = chunk_end_ms

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_gistda(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch GISTDA hotspots for the given date range (or all available if None)."""
    session = _make_session()

    start_ms = _date_to_epoch_ms(start) if start else None
    end_ms = _date_to_epoch_ms(end) + 86_400_000 if end else None  # inclusive end day

    log.info("Fetching GISTDA VIIRS NPP hotspots…")
    viirs = _fetch_layer_chunked(
        session, VIIRS_NPP_URL, "date", _parse_viirs_npp, start_ms, end_ms
    )

    log.info("Fetching GISTDA MODIS hotspots…")
    modis = _fetch_layer_chunked(
        session, MODIS_URL, "datetime", _parse_modis, start_ms, end_ms
    )

    frames = [f for f in [viirs, modis] if not f.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Coerce geometry columns
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
    combined = combined.dropna(subset=["latitude", "longitude", "acq_datetime"])
    return combined


def update_gistda(
    start: Optional[str] = None,
    end: Optional[str] = None,
    out_file: Optional[str] = None,
) -> int:
    """Fetch new GISTDA data, merge with on-disk cache, dedup, and persist."""
    out_file = out_file or OUT_FILE
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    new_df = fetch_gistda(start=start, end=end)
    if new_df.empty:
        log.warning("No new data fetched; cache unchanged")
        return 0

    existing = resolve_existing(out_file)
    if existing and os.path.getsize(existing) > 0:
        try:
            old_df = read_table(existing)
            old_df["acq_datetime"] = pd.to_datetime(old_df["acq_datetime"], errors="coerce", utc=True)
        except Exception as exc:
            log.warning("Failed to read existing cache (%s); starting fresh", exc)
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["acq_datetime"] = pd.to_datetime(combined["acq_datetime"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["acq_datetime"])

    before = len(combined)
    combined = combined.drop_duplicates(subset=["latitude", "longitude", "acq_datetime"])
    combined = combined.sort_values("acq_datetime").reset_index(drop=True)

    write_table(combined, out_file)

    lu_dist = combined["lu_name"].value_counts().head(6).to_dict()
    log.info(
        "Saved %d rows (deduped %d) → %s", len(combined), before - len(combined), out_file
    )
    log.info("  lu_name distribution (top 6): %s", lu_dist)
    log.info(
        "  date range: %s → %s",
        combined["acq_datetime"].min().date(),
        combined["acq_datetime"].max().date(),
    )
    return len(combined)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch GISTDA NRT hotspots (VIIRS NPP + MODIS)")
    p.add_argument("--start", metavar="YYYY-MM-DD", help="Start date (inclusive)")
    p.add_argument("--end",   metavar="YYYY-MM-DD", help="End date (inclusive)")
    p.add_argument("--out",   default=OUT_FILE, help="Output parquet path")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    update_gistda(start=args.start, end=args.end, out_file=args.out)
