"""NASA FIRMS NRT hotspot fetcher (VIIRS) with retries and accumulative caching."""

import argparse
import logging
import os
from io import StringIO
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from io_utils import read_table, resolve_existing, write_table

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("fetch_firms")

FIRMS_API_KEY = os.getenv("FIRMS_API_KEY")
TH_BBOX = os.getenv("FIRMS_BBOX", "96,4,107,22")
DEFAULT_DAYS = int(os.getenv("FIRMS_DAYS", "2"))

# Resolve relative env-supplied paths against the project root, not the
# script's cwd. Without this, running `python fetch_firms.py` from src/ writes
# to `src/data/firms/` while training reads from the project's `data/firms/`,
# creating a silent dual-cache split. Same _resolve pattern as train.py.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


DATA_DIR = _resolve(BASE_DIR, os.getenv("DATA_DIR")) or os.path.join(BASE_DIR, "data")
OUT_FILE = _resolve(BASE_DIR, os.getenv("FIRMS_PATH")) or os.path.join(BASE_DIR, "data", "firms", "firms_all.parquet")

DATASETS = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

BASE_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp",
    "confidence",
]

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def _make_session() -> requests.Session:
    """Session with exponential-backoff retries on transient HTTP errors."""
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


def fetch_firms(days: int = DEFAULT_DAYS, bbox: str = TH_BBOX) -> pd.DataFrame:
    """Fetch the last `days` of VIIRS hotspots across all configured datasets."""
    if not FIRMS_API_KEY:
        raise RuntimeError("FIRMS_API_KEY not set in environment / .env")
    if not 1 <= days <= 10:
        raise ValueError("FIRMS API supports days in [1, 10]")

    session = _make_session()
    frames: List[pd.DataFrame] = []

    for dataset in DATASETS:
        url = f"{FIRMS_URL}/{FIRMS_API_KEY}/{dataset}/{bbox}/{days}"
        log.info("Fetching %s (last %d day(s))", dataset, days)
        try:
            res = session.get(url, timeout=60)
            res.raise_for_status()
        except requests.RequestException as exc:
            log.error("Request failed for %s: %s", dataset, exc)
            continue

        body = res.text.strip()
        if not body or body.lower().startswith(("invalid", "error")):
            log.warning("Empty / error response from %s: %.120s", dataset, body)
            continue

        try:
            df = pd.read_csv(StringIO(body))
        except pd.errors.ParserError as exc:
            log.error("CSV parse failed for %s: %s", dataset, exc)
            continue

        if df.empty:
            log.warning("No rows from %s", dataset)
            continue

        missing = set(BASE_COLUMNS) - set(df.columns)
        if missing:
            log.warning("Schema mismatch from %s, missing %s", dataset, missing)
            continue

        df = df[BASE_COLUMNS].copy()
        df["dataset"] = dataset
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, normalize confidence, drop unparseable rows."""
    df = df.copy()
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str)
        + " "
        + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )

    conf_letter_map = {"l": 0, "n": 50, "h": 100}
    mapped = df["confidence"].astype(str).str.lower().map(conf_letter_map)
    numeric = pd.to_numeric(df["confidence"], errors="coerce")
    df["confidence"] = mapped.fillna(numeric)

    numeric_cols = [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "frp",
        "confidence",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df.dropna(subset=["acq_datetime", "latitude", "longitude", "frp"])


def update_firms(days: int = DEFAULT_DAYS, out_file: Optional[str] = None) -> int:
    """Fetch new NRT data, merge with on-disk cache, dedupe, and persist."""
    out_file = out_file or OUT_FILE
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    new_df = fetch_firms(days=days)
    if new_df.empty:
        log.warning("No new data fetched; cache unchanged")
        return 0

    new_df = clean_firms(new_df)

    existing = resolve_existing(out_file)
    if existing and os.path.getsize(existing) > 0:
        try:
            old_df = read_table(existing)
        except Exception as exc:
            log.warning("Failed to read existing cache (%s); starting fresh", exc)
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["acq_datetime"] = pd.to_datetime(
        combined["acq_datetime"], errors="coerce"
    )
    combined = combined.dropna(subset=["acq_datetime"])

    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["latitude", "longitude", "acq_datetime"]
    )
    combined = combined.sort_values("acq_datetime").reset_index(drop=True)

    write_table(combined, out_file)
    log.info(
        "Saved %d rows (deduped %d) → %s", len(combined), before - len(combined), out_file
    )
    return len(combined)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch NASA FIRMS VIIRS NRT hotspots")
    p.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Days of history to fetch (1-10)",
    )
    p.add_argument(
        "--out", default=OUT_FILE, help="Output CSV path (accumulative cache)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    update_firms(days=args.days, out_file=args.out)
