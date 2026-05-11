"""GISTDA Thai forest classification + crop type → per-cell aggregates.

Fetches two GISTDA ArcGIS MapServer polygon datasets and rasterises them
onto the project's 0.1° grid, producing binary presence flags and area
estimates per cell.

Datasets (no auth required):
  Mhesi/3P_Forest/MapServer — 5 Thai forest-management polygon layers
      Layer 0: ALRO forest reserves
      Layer 1: Economic/commercial plantations
      Layer 2: Mangrove forests
      Layer 3: National protected forests (ป่าสงวนแห่งชาติ)
      Layer 4: Conservation forests (ป่าอนุรักษ์)

  Mhesi/agriculture/MapServer — Crop-type polygons (UTM Zone 47N, EPSG:32647)
      Layer 0: Maize (ข้าวโพด)
      Layer 1: Rice (ข้าว)
      Layer 2: Aggregate agriculture (irrigation projects)

Why this matters:
  Hansen GFC tree_cover_pct_2000 distinguishes "forest vs open land" but
  cannot tell apart a national protected forest from a commercial oil-palm
  plantation, or an agricultural burn from a wildfire. These flags give the
  model that context:
    • conservation / national forest → fire = ecological event, high severity
    • plantation / ALRO → fire = managed burn or encroachment indicator
    • maize / rice → fire = post-harvest residue burn, highly seasonal

Rasterisation approach:
  For each polygon, we enumerate all 0.1° grid cells whose centre falls
  inside the polygon using shapely. This "rasterise by centre-point" method
  is simple and safe; partial coverage is ignored but cells near polygon
  edges are captured if the centre crosses. The approach is appropriate
  because the polygons are large administrative / management units that
  typically cover many grid cells rather than individual parcels.

Output schema (data/static/gistda_lulc_per_cell.parquet):
  lat_grid, lon_grid,
  in_alro_forest, in_plantation, in_mangrove,
  in_national_forest, in_conservation_forest,
  in_maize, in_rice,
  forest_area_rai, maize_area_rai, rice_area_rai

Usage::
    cd src && python fetch_gistda_lulc.py          # full Thailand
    cd src && python fetch_gistda_lulc.py --dry-run # count polygons, no write
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.prepared import prep
from shapely.strtree import STRtree
from urllib3.util.retry import Retry

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetch_gistda_lulc")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "static", "gistda_lulc_per_cell.parquet")

GISTDA_BASE = "https://gistdaportal.gistda.or.th/data/rest/services"
TH_GEOMETRY = "96,4,107,22"  # minX,minY,maxX,maxY
TH_BBOX = (96.0, 4.0, 107.0, 22.0)

GRID_SIZE = 0.1
MAX_RECORDS_PER_PAGE = 1000  # ArcGIS service hard limit

# 3P_Forest layers: (layer_id, output_flag_column, output_area_column)
FOREST_LAYERS = [
    (0, "in_alro_forest",         "alro_forest_area_rai"),
    (1, "in_plantation",          "plantation_area_rai"),
    (2, "in_mangrove",            "mangrove_area_rai"),
    (3, "in_national_forest",     "national_forest_area_rai"),
    (4, "in_conservation_forest", "conservation_forest_area_rai"),
]

# Agriculture layers: (layer_id, output_flag_column, output_area_column)
AGRI_LAYERS = [
    (0, "in_maize", "maize_area_rai"),
    (1, "in_rice",  "rice_area_rai"),
]


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


def _query_all_features(
    session: requests.Session,
    url: str,
    extra_params: Optional[dict] = None,
) -> list[dict]:
    """Paginate through ArcGIS /query using resultOffset until exhausted."""
    all_features: list[dict] = []
    offset = 0

    while True:
        params: dict = {
            "where": "1=1",
            "geometry": TH_GEOMETRY,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "resultRecordCount": MAX_RECORDS_PER_PAGE,
            "resultOffset": offset,
            "f": "json",
        }
        if extra_params:
            params.update(extra_params)

        try:
            res = session.get(f"{url}/query", params=params, timeout=120)
            res.raise_for_status()
            payload = res.json()
        except Exception as exc:
            log.error("Request failed for %s (offset=%d): %s", url, offset, exc)
            break

        if "error" in payload:
            code = payload["error"].get("code", "?")
            msg  = payload["error"].get("message", "")
            # ArcGIS returns code 400 when resultOffset is not supported;
            # fall back to a single non-paginated query.
            if code in (400, "400") and offset == 0:
                log.warning("%s: pagination not supported, fetching single page", url)
                params.pop("resultOffset", None)
                params.pop("resultRecordCount", None)
                try:
                    res = session.get(f"{url}/query", params=params, timeout=120)
                    payload = res.json()
                    all_features.extend(payload.get("features", []))
                except Exception as exc2:
                    log.error("Fallback query failed: %s", exc2)
            else:
                log.error("ArcGIS error (code=%s): %s", code, msg)
            break

        batch = payload.get("features", [])
        all_features.extend(batch)
        exceeded = payload.get("exceededTransferLimit", False)

        if len(batch) < MAX_RECORDS_PER_PAGE and not exceeded:
            break  # last page
        offset += len(batch)
        log.debug("  paginated: %d so far (offset=%d)", len(all_features), offset)

    return all_features


def _rings_to_shapely(rings: list[list[list[float]]]) -> Optional[Polygon]:
    """Convert ArcGIS ring list to a shapely Polygon (or MultiPolygon)."""
    if not rings:
        return None
    try:
        exterior = rings[0]
        holes    = rings[1:]
        return Polygon(exterior, holes)
    except Exception:
        return None


def _esri_geom_to_shapely(geom: dict, to_wgs84=None) -> Optional[Polygon | MultiPolygon]:
    """Convert an esriGeometryPolygon dict to a shapely geometry.

    If ``to_wgs84`` is a pyproj Transformer, coordinates are reprojected
    from the source CRS (e.g. UTM47N) to WGS84 before building the shape.
    """
    rings = geom.get("rings", [])
    if not rings:
        return None

    if to_wgs84 is not None:
        reprojected = []
        for ring in rings:
            xs, ys = zip(*ring)
            lons, lats = to_wgs84.transform(xs, ys)
            reprojected.append(list(zip(lons, lats)))
        rings = reprojected

    return _rings_to_shapely(rings)


def _build_grid_centres() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_centres, lon_centres) 1-D arrays for the Thailand grid."""
    lons = np.arange(TH_BBOX[0], TH_BBOX[2], GRID_SIZE) + GRID_SIZE / 2
    lats = np.arange(TH_BBOX[1], TH_BBOX[3], GRID_SIZE) + GRID_SIZE / 2
    return lats, lons


def _rasterise_polygons(
    polygons: list[tuple[object, float]],  # (shapely_geom, area_rai)
    flag_col: str,
    area_col: str,
) -> pd.DataFrame:
    """Mark grid cells whose centre falls inside any of the polygons.

    Returns a DataFrame with columns:
      lat_grid, lon_grid, {flag_col} (bool), {area_col} (float, rai)
    """
    lats, lons = _build_grid_centres()

    valid = [(g, a) for g, a in polygons if g is not None and g.is_valid]
    if not valid:
        log.warning("No valid polygons to rasterise for %s", flag_col)
        return pd.DataFrame()

    geoms, areas = zip(*valid)

    # STRtree for fast spatial indexing
    tree = STRtree(geoms)

    flag_cells: dict[tuple[float, float], float] = {}  # (lat, lon) → area_rai

    for g_idx, geom in enumerate(geoms):
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        # candidate grid cells whose centre could be inside this polygon
        col_lo = int((bounds[0] - TH_BBOX[0]) / GRID_SIZE)
        col_hi = int((bounds[2] - TH_BBOX[0]) / GRID_SIZE) + 1
        row_lo = int((bounds[1] - TH_BBOX[1]) / GRID_SIZE)
        row_hi = int((bounds[3] - TH_BBOX[1]) / GRID_SIZE) + 1

        col_lo = max(col_lo, 0)
        row_lo = max(row_lo, 0)
        col_hi = min(col_hi, len(lons))
        row_hi = min(row_hi, len(lats))

        pgeom = prep(geom)
        area_rai = areas[g_idx]

        for ri in range(row_lo, row_hi):
            lat_c = round(lats[ri], 6)
            for ci in range(col_lo, col_hi):
                lon_c = round(lons[ci], 6)
                from shapely.geometry import Point
                if pgeom.contains(Point(lon_c, lat_c)):
                    key = (lat_c, lon_c)
                    # accumulate area (a cell can overlap multiple polygons
                    # of the same type, e.g. fragmented national forest)
                    flag_cells[key] = flag_cells.get(key, 0.0) + area_rai

    if not flag_cells:
        log.info("  %s: 0 grid cells matched", flag_col)
        return pd.DataFrame()

    rows = [
        {"lat_grid": k[0], "lon_grid": k[1], flag_col: True, area_col: v}
        for k, v in flag_cells.items()
    ]
    df = pd.DataFrame(rows)
    log.info("  %s: %d grid cells matched", flag_col, len(df))
    return df


def _fetch_forest_layer(
    session: requests.Session,
    layer_id: int,
    flag_col: str,
    area_col: str,
) -> pd.DataFrame:
    url = f"{GISTDA_BASE}/Mhesi/3P_Forest/MapServer/{layer_id}"
    log.info("Fetching 3P_Forest layer %d (%s)…", layer_id, flag_col)
    features = _query_all_features(session, url)
    log.info("  %d polygon features returned", len(features))

    polygons: list[tuple] = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        attrs = feat.get("attributes", {})
        area_rai = float(attrs.get("AREA_RAI") or attrs.get("Shape_Area") or 0)
        shapely_geom = _esri_geom_to_shapely(geom)
        if shapely_geom and not shapely_geom.is_empty:
            polygons.append((shapely_geom, area_rai))

    return _rasterise_polygons(polygons, flag_col, area_col)


def _fetch_agri_layer(
    session: requests.Session,
    layer_id: int,
    flag_col: str,
    area_col: str,
) -> pd.DataFrame:
    url = f"{GISTDA_BASE}/Mhesi/agriculture/MapServer/{layer_id}"
    log.info("Fetching agriculture layer %d (%s, UTM47N → WGS84)…", layer_id, flag_col)

    if not _HAS_PYPROJ:
        log.warning("pyproj not available — skipping UTM agriculture layer %d", layer_id)
        return pd.DataFrame()

    to_wgs84 = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    features = _query_all_features(session, url)
    log.info("  %d polygon features returned", len(features))

    polygons: list[tuple] = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        attrs = feat.get("attributes", {})
        area_rai = float(attrs.get("rai") or attrs.get("area") or attrs.get("Shape_Area") or 0)
        shapely_geom = _esri_geom_to_shapely(geom, to_wgs84=to_wgs84)
        if shapely_geom and not shapely_geom.is_empty:
            polygons.append((shapely_geom, area_rai))

    return _rasterise_polygons(polygons, flag_col, area_col)


def run(dry_run: bool = False) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    session = _make_session()

    # Collect per-layer DataFrames, then merge them onto the full grid.
    # Start from a grid of all possible cells (all flagged False / area 0);
    # the per-layer frames set the matching cells to True.
    lats, lons = _build_grid_centres()
    lat_arr, lon_arr = np.meshgrid(lats, lons, indexing="ij")
    base = pd.DataFrame({
        "lat_grid": np.round(lat_arr.flatten(), 6),
        "lon_grid": np.round(lon_arr.flatten(), 6),
    })

    all_flag_cols = (
        [fc for _, fc, _ in FOREST_LAYERS] +
        [fc for _, fc, _ in AGRI_LAYERS]
    )
    all_area_cols = (
        [ac for _, _, ac in FOREST_LAYERS] +
        [ac for _, _, ac in AGRI_LAYERS]
    )
    for c in all_flag_cols:
        base[c] = False
    for c in all_area_cols:
        base[c] = 0.0

    if dry_run:
        log.info("--dry-run: skipping actual polygon fetch; grid has %d cells", len(base))
        return

    # Forest layers (WGS84)
    for layer_id, flag_col, area_col in FOREST_LAYERS:
        layer_df = _fetch_forest_layer(session, layer_id, flag_col, area_col)
        if layer_df.empty:
            continue
        base = base.merge(
            layer_df.rename(columns={flag_col: f"_{flag_col}", area_col: f"_{area_col}"}),
            on=["lat_grid", "lon_grid"],
            how="left",
        )
        mask = base[f"_{flag_col}"].notna()
        base.loc[mask, flag_col] = True
        base.loc[mask, area_col] = base.loc[mask, f"_{area_col}"].fillna(0.0)
        base.drop(columns=[f"_{flag_col}", f"_{area_col}"], inplace=True)

    # Agriculture layers (UTM47N → WGS84)
    for layer_id, flag_col, area_col in AGRI_LAYERS:
        layer_df = _fetch_agri_layer(session, layer_id, flag_col, area_col)
        if layer_df.empty:
            continue
        base = base.merge(
            layer_df.rename(columns={flag_col: f"_{flag_col}", area_col: f"_{area_col}"}),
            on=["lat_grid", "lon_grid"],
            how="left",
        )
        mask = base[f"_{flag_col}"].notna()
        base.loc[mask, flag_col] = True
        base.loc[mask, area_col] = base.loc[mask, f"_{area_col}"].fillna(0.0)
        base.drop(columns=[f"_{flag_col}", f"_{area_col}"], inplace=True)

    # Only write cells that have at least one flag set (sparse output)
    result = base[base[all_flag_cols].any(axis=1)].reset_index(drop=True)

    result.to_parquet(OUT_PATH, index=False)
    log.info("Saved %d cells with LULC flags → %s", len(result), OUT_PATH)
    for fc, ac in zip(all_flag_cols, all_area_cols):
        count = result[fc].sum()
        if count > 0:
            log.info("  %-30s %d cells, total area ≈ %.0f rai", fc + ":", count, result[ac].sum())


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch GISTDA Thai LULC → 0.1° grid")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Count polygons without writing output",
    )
    p.add_argument("--out", default=OUT_PATH, help="Output parquet path")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    if args.out != OUT_PATH:
        OUT_PATH = args.out
    run(dry_run=args.dry_run)
