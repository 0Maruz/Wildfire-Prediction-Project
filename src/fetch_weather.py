"""Open-Meteo weather fetcher — REAL ECMWF ERA5 reanalysis only.

Fetches daily temperature / precipitation / wind / evapotranspiration values
for every active FIRMS grid cell over the dataset's full date range and caches
the result to ``data/weather/weather_cache.parquet``.

Key properties (per the project rule: real data only):
    • Source = Open-Meteo's free Archive API, which serves ECMWF ERA5 / ERA5-Land
      reanalysis (https://open-meteo.com/en/docs/historical-weather-api).
      No API key, no commercial restriction for non-commercial use.
    • Each cached row corresponds to a real (lat_grid, lon_grid, date) tuple
      that appears in the densified FIRMS dataset.
    • No interpolation, no fabrication. If the API returns a NULL for a given
      (cell, date), the column is left empty and downstream features.py will
      treat it as a 0-fill *only* at model-input time — the cache itself
      preserves the genuine missing-data signal.

Performance
-----------
Three stacking optimizations compared to the naive sequential approach:

1. **Per-cell missing-date detection** — before any HTTP call we compute,
   for each cell, the smallest date window that actually needs fetching
   based on the existing cache. Fully-cached cells are skipped entirely.
   For a daily refresh this typically reduces work by 99%+.

2. **Concurrent fetching** — cells (or batches of cells) are fetched in
   parallel via ``ThreadPoolExecutor``. The default of **3** workers (override
   with ``OPEN_METEO_MAX_WORKERS`` or ``--workers``) stays under Open-Meteo's
   free-tier rate limits; raise workers only after the cache fills in.

3. **Optional location batching** — Open-Meteo accepts up to 1000
   comma-separated coordinates per request. Enable with ``--batch-size N``
   (default 1 = single-location, the safe baseline). With ``--batch-size 100``
   and ``--workers 5`` the wall-clock cost drops by another order of
   magnitude.

Usage::

    cd src && python fetch_weather.py [--limit-cells N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                      [--workers N] [--batch-size N]
                                      [--quiet-hours START-END]

``--quiet-hours 2-8`` waits until the clock in ``TIMEZONE`` (default Asia/Bangkok)
is between 02:00 and 08:59 inclusive before issuing HTTP requests — a single long
run can sleep first, then fetch everything in one session.

Checkpoint flushes (``OPEN_METEO_FLUSH_EVERY``) persist partial progress so a
multi-hour run does not lose all rows if interrupted; failed batches are retried
at the end (``OPEN_METEO_FINAL_RETRY_ROUNDS``).

After running, train.py / data_loader.py automatically detect the cache and
merge it onto the daily frame, so weather features become part of the model
input contract on the next training run.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_loader import grid_and_aggregate, clean_hotspots, load_firms_csv
from storage import read_table, resolve_existing, write_table

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("fetch_weather")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


RAW_DIR     = _resolve(BASE_DIR, os.getenv("RAW_DIR"))     or os.path.join(BASE_DIR, "data", "raw")
FIRMS_PATH  = _resolve(BASE_DIR, os.getenv("FIRMS_PATH"))  or os.path.join(BASE_DIR, "data", "firms", "firms_all.parquet")
WEATHER_DIR = _resolve(BASE_DIR, os.getenv("WEATHER_DIR")) or os.path.join(BASE_DIR, "data", "weather")
WEATHER_CACHE_PATH = os.path.join(WEATHER_DIR, "weather_cache.parquet")

# ECMWF ERA5 archive — no key required, ~5-day lag for the most recent dates.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Real, measurable daily aggregates from ERA5 reanalysis.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]
RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precip_sum",
    "wind_speed_10m_max": "wind_max",
    "et0_fao_evapotranspiration": "et0",
}

TIMEZONE = os.getenv("TIMEZONE", "Asia/Bangkok")
GRID_SIZE = float(os.getenv("GRID_SIZE", "0.1"))
ARCHIVE_LAG_DAYS = 5  # ERA5T preliminary data lag

# Open-Meteo accepts up to 1000 coordinates per call.
MAX_BATCH_SIZE = 1000

# Defaults tuned for the free Archive API (no API key). 429 = rate limit —
# lower concurrency + explicit backoff beats "stuck" retries on a shared pool.
_DEFAULT_WORKERS = max(1, int(os.getenv("OPEN_METEO_MAX_WORKERS", "3")))
_DEFAULT_BATCH_SIZE = max(1, min(MAX_BATCH_SIZE, int(os.getenv("OPEN_METEO_BATCH_SIZE", "1"))))
# Application-level GET retries after urllib3 gives up (mostly 429 storms).
_ARCHIVE_APP_RETRIES = max(1, int(os.getenv("OPEN_METEO_APP_RETRIES", "8")))
# Write merged parquet every N successful HTTP units (survive long runs / SIGINT).
_FLUSH_EVERY = max(1, int(os.getenv("OPEN_METEO_FLUSH_EVERY", "20")))
# After the main pool, re-submit failed units with backoff (same process).
_FINAL_RETRY_ROUNDS = max(0, int(os.getenv("OPEN_METEO_FINAL_RETRY_ROUNDS", "15")))
_RETRY_ROUND_SLEEP_BASE = float(os.getenv("OPEN_METEO_RETRY_ROUND_SLEEP_BASE", "45"))

_thread_local = threading.local()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────────────────────

def _make_session(pool_size: int = 16) -> requests.Session:
    """Create a session with retries and a connection pool large enough for
    the worker count. Without this, ``requests`` uses the default pool size
    of 10 and silently serializes excess concurrent requests."""
    s = requests.Session()
    s.headers.setdefault(
        "User-Agent",
        "Science-Project-version-3/fetch_weather (Open-Meteo archive; contact: local)",
    )
    retry = Retry(
        total=6,
        backoff_factor=2.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _thread_session() -> requests.Session:
    """One Session per worker thread — ``requests`` pools are not fully
    thread-safe under bursty parallel GETs; sharing one session caused
    intermittent stalls and odd 429 behaviour in practice."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = _make_session(pool_size=8)
        _thread_local.session = s
    return s


def _get_archive(
    session: requests.Session,
    params: dict,
    *,
    timeout: float,
) -> requests.Response:
    """GET the archive endpoint with **manual** 429 handling.

    urllib3's automatic 429 retries fight Open-Meteo's ``Retry-After`` when many
    workers hammer the API — we sleep once per response and re-issue a fresh GET.
    """
    last_err: Optional[Exception] = None
    for attempt in range(_ARCHIVE_APP_RETRIES):
        try:
            res = session.get(ARCHIVE_URL, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            wait = min(120.0, 2.0 ** attempt + 0.25 * attempt)
            log.warning("HTTP error (%s); retry in %.1fs (%d/%d)", exc, wait, attempt + 1, _ARCHIVE_APP_RETRIES)
            time.sleep(wait)
            continue

        if res.status_code == 429:
            ra = res.headers.get("Retry-After")
            try:
                wait = float(ra) if ra is not None else min(180.0, 15.0 * (attempt + 1))
            except ValueError:
                wait = min(180.0, 15.0 * (attempt + 1))
            wait = max(5.0, wait)
            log.warning(
                "429 Too Many Requests from Open-Meteo — sleeping %.0fs (attempt %d/%d)",
                wait,
                attempt + 1,
                _ARCHIVE_APP_RETRIES,
            )
            time.sleep(wait)
            continue

        if res.status_code >= 400:
            res.raise_for_status()

        return res

    if last_err is not None:
        raise last_err
    raise RuntimeError("Open-Meteo archive GET failed without response")


# ─────────────────────────────────────────────────────────────────────────────
# FIRMS → active cells / date range
# ─────────────────────────────────────────────────────────────────────────────

def discover_active_cells() -> tuple[pd.DataFrame, date, date]:
    """Return (cells_df, min_date, max_date) from the densified FIRMS frame."""
    sources = []
    if RAW_DIR:
        sources.append(RAW_DIR)
    if FIRMS_PATH:
        sources.append(FIRMS_PATH)

    raw = load_firms_csv(sources)
    cleaned = clean_hotspots(raw, min_confidence=0)
    daily = grid_and_aggregate(cleaned, grid_size=GRID_SIZE)

    cells = (
        daily[["lat_grid", "lon_grid"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return cells, daily["date"].min(), daily["date"].max()


# ─────────────────────────────────────────────────────────────────────────────
# Cache analysis — what do we actually need to fetch?
# ─────────────────────────────────────────────────────────────────────────────

CellTodo = Tuple[float, float, date, date]   # (lat, lon, fetch_start, fetch_end)


def _build_cache_index(
    cache: pd.DataFrame,
) -> Tuple[Dict[Tuple[float, float], Set[date]], Set[Tuple[float, float, date]]]:
    """Build two lookup structures from the cache:
        • ``cached_by_cell``: (lat, lon) → set of cached dates
        • ``existing``: set of (lat, lon, date) — used for post-fetch dedupe

    Single pass over the cache; both structures share the same source data.
    """
    cached_by_cell: Dict[Tuple[float, float], Set[date]] = defaultdict(set)
    existing: Set[Tuple[float, float, date]] = set()
    if cache.empty:
        return cached_by_cell, existing

    lats = cache["lat_grid"].round(6).tolist()
    lons = cache["lon_grid"].round(6).tolist()
    dates = cache["date"].tolist()
    for la, lo, d in zip(lats, lons, dates):
        cached_by_cell[(la, lo)].add(d)
        existing.add((la, lo, d))
    return cached_by_cell, existing


def _compute_todo(
    cells: pd.DataFrame,
    start: date,
    end: date,
    cached_by_cell: Dict[Tuple[float, float], Set[date]],
) -> List[CellTodo]:
    """For each cell, narrow the (start, end) window to the smallest range
    that covers every missing date. Skip cells already fully cached.

    The returned range is the *bounding window* of missing dates — there may
    still be cached dates inside the window, which the post-fetch dedupe will
    drop. This is correct behavior: re-fetching a few interior dates is far
    cheaper than splitting one cell into multiple sub-range requests.
    """
    todo: List[CellTodo] = []
    full_set = set(pd.date_range(start, end, freq="D").date)

    for _, row in cells.iterrows():
        lat = round(float(row["lat_grid"]), 6)
        lon = round(float(row["lon_grid"]), 6)
        cached = cached_by_cell.get((lat, lon), set())
        missing = full_set - cached
        if not missing:
            continue
        todo.append((lat, lon, min(missing), max(missing)))
    return todo


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo fetch (real ERA5 reanalysis)
# ─────────────────────────────────────────────────────────────────────────────

def _payload_to_df(payload: dict, lat: float, lon: float) -> pd.DataFrame:
    """Convert a single Open-Meteo location payload to a flat DataFrame.
    Returns empty DataFrame if no daily data is present."""
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns=RENAME)
    df["lat_grid"] = round(lat, 6)
    df["lon_grid"] = round(lon, 6)
    keep = ["lat_grid", "lon_grid", "date", *RENAME.values()]
    # Tolerate variables the API didn't return (e.g. transient outages).
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def fetch_one_cell(
    session: requests.Session,
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch a single grid cell's ERA5 daily aggregates over [start, end]."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    res = _get_archive(session, params, timeout=60)
    return _payload_to_df(res.json(), lat, lon)


def fetch_batch(
    session: requests.Session,
    batch: Sequence[Tuple[float, float]],
    start: date,
    end: date,
) -> List[pd.DataFrame]:
    """Fetch a batch of cells sharing the same date window in one HTTP call.

    Open-Meteo accepts comma-separated coordinates and returns either:
      • a single JSON object (when 1 location was requested), or
      • a JSON array of per-location objects (when >1 was requested).

    We handle both shapes. If the API ever falls back to single-object even
    for multi-coord requests, we degrade gracefully — the worker stays alive
    and the next batch proceeds.
    """
    if not batch:
        return []
    if len(batch) == 1:
        lat, lon = batch[0]
        return [fetch_one_cell(session, lat, lon, start, end)]

    lats = ",".join(f"{lat:.6f}" for lat, _ in batch)
    lons = ",".join(f"{lon:.6f}" for _, lon in batch)
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    res = _get_archive(session, params, timeout=120)
    payload = res.json()

    # Multi-location → list of dicts; single → dict. Normalise to list.
    if isinstance(payload, dict):
        # API didn't honor the multi-loc request — treat as one location.
        log.debug("Batched request returned single-object response; falling back.")
        lat, lon = batch[0]
        return [_payload_to_df(payload, lat, lon)]

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload type: {type(payload).__name__}")

    if len(payload) != len(batch):
        log.warning(
            "Batched response length (%d) != batch size (%d); pairing by index.",
            len(payload), len(batch),
        )

    out: List[pd.DataFrame] = []
    for (lat, lon), p in zip(batch, payload):
        # Each item may itself be an error object — tolerate per-location errors.
        if isinstance(p, dict) and p.get("error"):
            log.error("Per-location error for (%.3f, %.3f): %s", lat, lon, p.get("reason"))
            continue
        out.append(_payload_to_df(p, lat, lon))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Idempotent cache update
# ─────────────────────────────────────────────────────────────────────────────

WorkUnit = Tuple[Tuple[date, date], List[Tuple[float, float]]]


def parse_quiet_hours(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse ``"2-8"`` → inclusive local hours 02–08 (``TIMEZONE`` wall clock)."""
    if spec is None or not str(spec).strip():
        return None
    s = str(spec).strip()
    if "-" not in s:
        raise ValueError(
            "OPEN_METEO_QUIET_HOURS / --quiet-hours must look like START-END "
            f"(e.g. 2-8), got {spec!r}"
        )
    a_s, b_s = s.split("-", 1)
    a, b = int(a_s.strip()), int(b_s.strip())
    if not (0 <= a <= 23 and 0 <= b <= 23):
        raise ValueError(f"Quiet hours must be 0–23, got {a}-{b}")
    if a > b:
        raise ValueError(
            f"Quiet start ({a}) must be <= end ({b}). "
            "Overnight windows are not supported — split into two cron invocations."
        )
    return (a, b)


def wait_for_quiet_hours(quiet: Tuple[int, int], zone_name: str) -> None:
    """Block until local wall-clock hour is inside ``quiet`` (inclusive)."""
    from zoneinfo import ZoneInfo

    z = ZoneInfo(zone_name)
    a, b = quiet[0], quiet[1]
    while True:
        now = datetime.now(z)
        h = now.hour
        if a <= h <= b:
            log.info(
                "Preferred fetch window: %s hour=%d is inside [%d,%d] — starting HTTP.",
                zone_name,
                h,
                a,
                b,
            )
            return
        if h < a:
            target = now.replace(hour=a, minute=0, second=0, microsecond=0)
        else:
            target = (now + timedelta(days=1)).replace(hour=a, minute=0, second=0, microsecond=0)
        sec = max(1.0, (target - now).total_seconds())
        log.info(
            "Outside preferred hours [%d–%d] %s (hour=%d) — sleeping %.0f min until %s",
            a,
            b,
            zone_name,
            h,
            sec / 60.0,
            target.strftime("%Y-%m-%d %H:%M %Z"),
        )
        time.sleep(sec)


def _write_cache_merge(
    cache: pd.DataFrame,
    fresh_chunks: List[pd.DataFrame],
    existing: Set[Tuple[float, float, date]],
    cache_path: str,
) -> Tuple[pd.DataFrame, int]:
    """Merge ``fresh_chunks`` into ``cache``, write parquet, extend ``existing``."""
    if not fresh_chunks:
        return cache, 0
    fresh = pd.concat(fresh_chunks, ignore_index=True)
    fresh = fresh.drop_duplicates(subset=["lat_grid", "lon_grid", "date"])
    if fresh.empty:
        return cache, 0
    for la, lo, d in zip(
        fresh["lat_grid"].round(6),
        fresh["lon_grid"].round(6),
        fresh["date"],
    ):
        existing.add((float(la), float(lo), d))
    if not cache.empty:
        combined = pd.concat([cache, fresh], ignore_index=True)
    else:
        combined = fresh
    combined = combined.drop_duplicates(subset=["lat_grid", "lon_grid", "date"])
    combined = combined.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)
    write_table(combined, cache_path)
    return combined, len(fresh)


def _group_cells_by_window(
    todo: List[CellTodo],
) -> Dict[Tuple[date, date], List[Tuple[float, float]]]:
    """Group cells that share the same fetch (start, end) window.
    Cells in the same group can be batched into a single API call."""
    groups: Dict[Tuple[date, date], List[Tuple[float, float]]] = defaultdict(list)
    for lat, lon, s, e in todo:
        groups[(s, e)].append((lat, lon))
    return groups


def update_cache(
    cells: pd.DataFrame,
    start: date,
    end: date,
    cache_path: str = WEATHER_CACHE_PATH,
    sleep_between_calls: float = 0.0,
    limit_cells: Optional[int] = None,
    max_workers: int = _DEFAULT_WORKERS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    quiet_hours: Optional[Tuple[int, int]] = None,
    flush_every: Optional[int] = None,
) -> int:
    """Update the weather cache with any (cell, date) tuples missing from it.

    One invocation keeps working until all units succeed or retries are exhausted.
    Periodic checkpoints (``OPEN_METEO_FLUSH_EVERY``) write partial progress.

    Args:
        cells: DataFrame with ``lat_grid`` / ``lon_grid`` columns.
        start, end: Inclusive date range to cover.
        cache_path: Where to read/write the parquet cache.
        sleep_between_calls: Optional seconds to sleep between API submissions.
            With concurrency this is rarely needed; default 0.
        limit_cells: Debug — only process the first N cells.
        max_workers: Number of concurrent HTTP workers (env: ``OPEN_METEO_MAX_WORKERS``).
        batch_size: How many cells to pack into one HTTP call (1–1000; env:
            ``OPEN_METEO_BATCH_SIZE``). ``batch_size=1`` is the safest baseline.
            ``batch_size=50`` with ``workers=2`` is a good speed / 429 trade-off
            once the cache is warm.
        quiet_hours: Optional ``(start_hour, end_hour)`` inclusive in ``TIMEZONE``
            local time — the process sleeps until the clock enters this window
            before any HTTP (good for off-peak Open-Meteo load).
        flush_every: Override rows checkpoint interval (``0`` = only flush at end).

    Returns:
        Number of new rows added to the cache during this run (including checkpoints).
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    _flush_cfg = _FLUSH_EVERY if flush_every is None else max(0, int(flush_every))

    # Cap end at archive availability (ERA5T has a ~5-day lag).
    today = datetime.now(timezone.utc).date()
    end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
    if start > end:
        log.warning("Start (%s) is after archive-available end (%s); nothing to fetch.", start, end)
        return 0

    # ── Load existing cache and build lookup indices ─────────────────────────
    cache_existing = resolve_existing(cache_path)
    if cache_existing and os.path.getsize(cache_existing) > 0:
        cache = read_table(cache_existing)
        cache["date"] = pd.to_datetime(cache["date"]).dt.date
    else:
        cache = pd.DataFrame()

    cached_by_cell, existing = _build_cache_index(cache)

    cells = cells.copy()
    cells["lat_grid"] = cells["lat_grid"].round(6)
    cells["lon_grid"] = cells["lon_grid"].round(6)
    if limit_cells:
        cells = cells.head(limit_cells)

    # ── Compute per-cell todo (skipping fully-cached cells) ──────────────────
    todo = _compute_todo(cells, start, end, cached_by_cell)
    log.info(
        "Need to fetch %d / %d cells over %s → %s (timezone=%s) — %d already fully cached",
        len(todo), len(cells), start, end, TIMEZONE, len(cells) - len(todo),
    )
    if not todo:
        log.info("Cache already up to date — nothing to fetch.")
        return 0

    # ── Group into batches by shared date window ─────────────────────────────
    groups = _group_cells_by_window(todo)
    work_units: List[WorkUnit] = []
    for (s, e), pts in groups.items():
        for i in range(0, len(pts), batch_size):
            work_units.append(((s, e), pts[i : i + batch_size]))

    total_cells_in_units = sum(len(b) for _, b in work_units)
    log.info(
        "Submitting %d HTTP request(s) covering %d cell-windows (workers=%d, batch_size=%d)",
        len(work_units),
        total_cells_in_units,
        max_workers,
        batch_size,
    )

    if quiet_hours is not None:
        wait_for_quiet_hours(quiet_hours, TIMEZONE)

    # ── Concurrent fetch + checkpoints + final retry waves ───────────────────
    total_new = 0
    new_frames: List[pd.DataFrame] = []
    units_since_flush = 0
    flush_lock = threading.Lock()
    submission_lock = threading.Lock()
    unit_retries = max(1, int(os.getenv("OPEN_METEO_UNIT_RETRIES", "5")))

    def _do_unit(window: Tuple[date, date], batch: List[Tuple[float, float]]) -> List[pd.DataFrame]:
        if sleep_between_calls > 0:
            with submission_lock:
                time.sleep(sleep_between_calls)
        s, e = window
        ts = _thread_session()
        last_exc: Optional[Exception] = None
        for attempt in range(unit_retries):
            try:
                return fetch_batch(ts, batch, s, e)
            except requests.RequestException as exc:
                last_exc = exc
                wait = min(120.0, 2.5 * (2**attempt))
                log.warning(
                    "Batch fetch failed (%s); retry in %.1fs (%d/%d)",
                    exc,
                    wait,
                    attempt + 1,
                    unit_retries,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def checkpoint(reason: str) -> None:
        nonlocal cache, new_frames, total_new, units_since_flush
        with flush_lock:
            if not new_frames:
                return
            chunks = new_frames[:]
            new_frames.clear()
            units_since_flush = 0
        cache, n_add = _write_cache_merge(cache, chunks, existing, cache_path)
        total_new += n_add
        log.info(
            "Checkpoint (%s): +%d rows → %d total → %s",
            reason,
            n_add,
            len(cache),
            cache_path,
        )

    def process_success(dfs: List[pd.DataFrame]) -> None:
        nonlocal new_frames, units_since_flush
        for df in dfs:
            if df.empty:
                continue
            keys = list(zip(df["lat_grid"].round(6), df["lon_grid"].round(6), df["date"]))
            mask = [k not in existing for k in keys]
            df = df[mask]
            if not df.empty:
                new_frames.append(df)
        units_since_flush += 1
        if _flush_cfg > 0 and units_since_flush >= _flush_cfg:
            checkpoint("interval")

    def run_pool(work_list: List[WorkUnit], workers: int) -> List[WorkUnit]:
        if not work_list:
            return []
        failed_local: List[WorkUnit] = []
        done_n = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fmap = {pool.submit(_do_unit, w, b): (w, b) for w, b in work_list}
            for fut in as_completed(fmap):
                win, bat = fmap[fut]
                try:
                    dfs = fut.result()
                except Exception as exc:
                    log.warning(
                        "Work unit failed window=%s batch=%d: %s",
                        win,
                        len(bat),
                        exc,
                    )
                    failed_local.append((win, bat))
                else:
                    process_success(dfs)
                finally:
                    done_n += 1
                    if done_n % 25 == 0 or done_n == len(work_list):
                        log.info("  progress: %d / %d units", done_n, len(work_list))
        return failed_local

    failed = run_pool(work_units, max_workers)
    for rnd in range(_FINAL_RETRY_ROUNDS):
        if not failed:
            break
        pause = _RETRY_ROUND_SLEEP_BASE * (1.4**rnd)
        log.info(
            "Retry wave %d/%d: %d failed unit(s) — sleeping %.0fs then retrying (workers=%d)",
            rnd + 1,
            _FINAL_RETRY_ROUNDS,
            len(failed),
            pause,
            min(2, max_workers),
        )
        time.sleep(pause)
        failed = run_pool(failed, min(2, max(1, max_workers)))

    if failed:
        checkpoint("pre-failure-save")
        raise RuntimeError(
            f"{len(failed)} weather fetch unit(s) still failed after all retry waves. "
            "Partial progress was checkpointed if any rows succeeded — wait for API quota, "
            "then re-run the same command (idempotent)."
        )

    checkpoint("final")
    if total_new == 0:
        log.info("No new weather rows (tuples already present in cache).")
    else:
        log.info(
            "Weather fetch run complete: +%d new row(s) this run → %s",
            total_new,
            cache_path,
        )
    return total_new


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch real ERA5 daily weather (Open-Meteo Archive) for "
                    "every active FIRMS grid cell."
    )
    p.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    p.add_argument("--end",   help="Override end date (YYYY-MM-DD)")
    p.add_argument("--limit-cells", type=int, default=None,
                   help="Fetch only the first N active cells (debugging)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Optional seconds to sleep between API calls (default 0; "
                        "429s are handled with Retry-After sleeps inside each GET)")
    p.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Concurrent HTTP workers (default {_DEFAULT_WORKERS} from OPEN_METEO_MAX_WORKERS)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Cells per HTTP request (1–{MAX_BATCH_SIZE}, default {_DEFAULT_BATCH_SIZE} from OPEN_METEO_BATCH_SIZE). "
        "Try 50–100 with workers=2 once stable.",
    )
    p.add_argument(
        "--quiet-hours",
        type=str,
        default=None,
        metavar="START-END",
        help="Local clock in TIMEZONE (default Asia/Bangkok): wait until hour is "
        "within START–END inclusive before HTTP (e.g. 2-8 = 02:00–08:59). "
        "Env: OPEN_METEO_QUIET_HOURS=2-8",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=None,
        metavar="N",
        help="Checkpoint parquet every N successful HTTP units (default from "
        "OPEN_METEO_FLUSH_EVERY; 0 = only at end of run).",
    )
    return p.parse_args()


def main() -> None:
    args = _cli()
    cells, dmin, dmax = discover_active_cells()

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else dmin
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end   else dmax

    qh = parse_quiet_hours(args.quiet_hours)
    if qh is None:
        qh = parse_quiet_hours(os.getenv("OPEN_METEO_QUIET_HOURS"))

    t0 = time.perf_counter()
    update_cache(
        cells=cells,
        start=start,
        end=end,
        sleep_between_calls=args.sleep,
        limit_cells=args.limit_cells,
        max_workers=args.workers,
        batch_size=args.batch_size,
        quiet_hours=qh,
        flush_every=args.flush_every,
    )
    log.info("Total elapsed: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
