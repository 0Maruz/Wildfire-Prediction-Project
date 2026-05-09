"""Hansen Global Forest Change tree cover → per-cell aggregates.

Streams treecover2000 + lossyear from Hansen GFC v1.11 via windowed HTTP
range reads (no full-tile download), aggregates each 30 m source pixel
into the project's 0.1° grid, and persists per-cell statistics to
``data/static/tree_cover_per_cell.parquet``.

Why Hansen:
    The current model has no vegetation context — two cells with identical
    fire history but different land cover (dense forest vs grassland) burn
    very differently in reality, but our regressor sees them as identical.
    Adding `tree_cover_pct_2000` (canopy density baseline) and
    `tree_loss_pct_recent` (fraction of pixels in the cell that lost
    forest 2018-2023, post-deforestation fire-risk indicator) gives the
    model the "what's there to burn" signal.

Why streaming:
    Each Hansen tile is ~520 MB compressed and decompresses to ~6 GB in
    memory. Six tiles cover Thailand. We use rasterio's `/vsicurl/` driver
    + windowed reads so only the bytes covering the Thailand BBOX leave
    Google Cloud Storage — typical run transfers ~80-150 MB total.

Output schema:
    lat_grid, lon_grid, tree_cover_pct_2000, tree_loss_pct_recent

Usage::

    cd src && python fetch_treecover.py
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetch_treecover")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "static", "tree_cover_per_cell.parquet")

# Hansen GFC v1.11 (2023 release). Filename encodes the tile's TOP-LEFT
# corner (north, west). Thailand spans 5-22°N, 96-107°E so we need the
# six 10°×10° tiles covering 0-30°N, 90-110°E.
HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen"
TILES = [
    ("30N", "090E"),
    ("30N", "100E"),
    ("20N", "090E"),
    ("20N", "100E"),
    ("10N", "090E"),
    ("10N", "100E"),
]

# Output grid. The pipeline uses `round(coord / 0.1) * 0.1` to snap FIRMS
# detections to grid cells, so cells are labelled by the integer multiple
# of 0.1° and a cell named (22.0, 100.0) covers lat ∈ [21.95, 22.05) and
# lon ∈ [99.95, 100.05). To make the rasterio output cells land on the
# same labels, we shift the BBOX by 0.05° in each direction so that
# `from_bounds(...)` puts pixel CENTRES on integer multiples of 0.1°.
GRID_SIZE = 0.1
TH_BBOX = (95.95, 3.95, 107.05, 22.05)  # half-cell expansion of (96, 4, 107, 22)

# Hansen lossyear codes: 0 = no loss, 1 = 2001 ... 23 = 2023. The "recent"
# window captures fires that follow recent deforestation (slash-and-burn,
# post-clearance regrowth that's exceptionally fire-prone).
RECENT_LOSS_START_YEAR = 18
RECENT_LOSS_END_YEAR = 23


def _tile_url(layer: str, lat_tag: str, lon_tag: str) -> str:
    return (
        f"{HANSEN_BASE}/{HANSEN_VERSION}/"
        f"Hansen_{HANSEN_VERSION}_{layer}_{lat_tag}_{lon_tag}.tif"
    )


def _vsicurl(url: str) -> str:
    return f"/vsicurl/{url}"


def _build_grid() -> Tuple[np.ndarray, "rasterio.Affine", np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = TH_BBOX
    width = int(round((max_lon - min_lon) / GRID_SIZE))
    height = int(round((max_lat - min_lat) / GRID_SIZE))
    transform = rasterio.transform.from_bounds(
        min_lon, min_lat, max_lon, max_lat, width, height
    )
    arr = np.zeros((height, width), dtype=np.float32)
    lon_centres = min_lon + (np.arange(width) + 0.5) * GRID_SIZE
    lat_centres = max_lat - (np.arange(height) + 0.5) * GRID_SIZE  # row 0 = north
    return arr, transform, lat_centres, lon_centres


def _bbox_window(src) -> Window:
    """Source-pixel window covering Thailand BBOX, clipped to the tile.

    Returns a Window with zero width/height when the tile doesn't overlap
    the BBOX (caller skips it). Otherwise the window is the smallest
    rectangle of source pixels that fully contains the BBOX.
    """
    full = Window(0, 0, src.width, src.height)
    bbox_window = src.window(*TH_BBOX)
    inter = bbox_window.intersection(full)
    return inter


def _stream_tile_into(
    layer: str,
    lat_tag: str,
    lon_tag: str,
    dst: np.ndarray,
    dst_transform,
    encode_loss: bool,
) -> None:
    url = _tile_url(layer, lat_tag, lon_tag)
    log.info("  reading %s", url.rsplit("/", 1)[-1])

    with rasterio.open(_vsicurl(url)) as src:
        try:
            window = _bbox_window(src)
        except rasterio.errors.WindowError:
            log.info("    tile does not overlap BBOX — skipping")
            return
        if window.width <= 0 or window.height <= 0:
            log.info("    tile does not overlap BBOX — skipping")
            return

        src_window_data = src.read(1, window=window)
        src_window_transform = src.window_transform(window)

        if encode_loss:
            # Encode each source pixel as 100 if it lost forest in the
            # recent window, else 0 — so the average resampler returns
            # "% pixels that lost forest in this cell" directly.
            encoded = (
                (src_window_data >= RECENT_LOSS_START_YEAR)
                & (src_window_data <= RECENT_LOSS_END_YEAR)
            ).astype(np.float32) * 100.0
            source_payload = encoded
        else:
            source_payload = src_window_data.astype(np.float32)

        intermediate = np.zeros_like(dst, dtype=np.float32)
        reproject(
            source=source_payload,
            destination=intermediate,
            src_transform=src_window_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
        # Only overwrite cells that this tile actually contributed to.
        # Tiles cover non-overlapping 10°×10° quadrants, so any cell that
        # received a non-zero value from this tile is genuinely from here.
        mask = intermediate > 0
        dst[mask] = intermediate[mask]


def _process_layer(layer: str, encode_loss: bool = False) -> np.ndarray:
    arr, transform, _, _ = _build_grid()
    for lat_tag, lon_tag in TILES:
        _stream_tile_into(layer, lat_tag, lon_tag, arr, transform, encode_loss)
    return arr


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    log.info("Streaming Hansen treecover2000 → 0.1° Thailand grid (windowed reads)…")
    treecover = _process_layer("treecover2000", encode_loss=False)

    log.info(
        "Streaming Hansen lossyear → recent-window (20%d-20%d) fraction per cell…",
        RECENT_LOSS_START_YEAR, RECENT_LOSS_END_YEAR,
    )
    loss = _process_layer("lossyear", encode_loss=True)

    _, _, lat_centres, lon_centres = _build_grid()
    lat_grid, lon_grid = np.meshgrid(lat_centres, lon_centres, indexing="ij")
    df = pd.DataFrame({
        "lat_grid": np.round(lat_grid.flatten(), 6),
        "lon_grid": np.round(lon_grid.flatten(), 6),
        "tree_cover_pct_2000": treecover.flatten().astype(float),
        "tree_loss_pct_recent": loss.flatten().astype(float),
    })

    df.to_parquet(OUT_PATH, index=False)
    log.info(
        "Saved %d cells → %s",
        len(df), OUT_PATH,
    )
    log.info(
        "  tree_cover_pct_2000:  mean=%.1f, p25=%.1f, p75=%.1f, max=%.1f",
        df["tree_cover_pct_2000"].mean(),
        df["tree_cover_pct_2000"].quantile(0.25),
        df["tree_cover_pct_2000"].quantile(0.75),
        df["tree_cover_pct_2000"].max(),
    )
    log.info(
        "  tree_loss_pct_recent: mean=%.2f, p75=%.2f, max=%.2f",
        df["tree_loss_pct_recent"].mean(),
        df["tree_loss_pct_recent"].quantile(0.75),
        df["tree_loss_pct_recent"].max(),
    )


if __name__ == "__main__":
    main()
