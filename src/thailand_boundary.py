"""Thailand-only spatial filter.

The FIRMS BBOX (`96,4,107,22`) is a rectangle that necessarily includes
parts of Myanmar, Laos, Cambodia, Vietnam, and northern Malaysia. For a
Thailand-focused dashboard those neighbour-country cells are noise — the
model trained on them, but the operator only cares about the Thai outcome.

This module loads the 77-province Thailand boundary GeoJSON, merges it into
a single MultiPolygon at import time, and exposes a vectorised
``is_in_thailand(lats, lons)`` predicate. Used to drop predictions outside
the Thai land border at inference time.

Boundary source: https://github.com/apisit/thailand.json (CC BY 4.0)
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from storage import exists, read_json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY_PATH = os.path.join(BASE_DIR, "data", "boundaries", "thailand.geojson")


def _load_provinces():
    """Return list of (name, geometry) per Thai province from the bundled
    GeoJSON, plus the union of all provinces as the country boundary."""
    if not exists(BOUNDARY_PATH):
        raise FileNotFoundError(
            f"Thailand boundary GeoJSON not found at {BOUNDARY_PATH}. "
            "Restore from https://github.com/apisit/thailand.json/blob/master/thailand.json"
        )
    gj = read_json(BOUNDARY_PATH)
    provinces = []
    geoms = []
    for feat in gj["features"]:
        name = feat["properties"].get("name", "Unknown")
        geom = shape(feat["geometry"])
        provinces.append((name, geom))
        geoms.append(geom)
    return provinces, unary_union(geoms)


# Eagerly build per-province + country-level shapes. Prepared geometries
# turn repeated `.contains()` checks into O(log N) instead of O(N).
_PROVINCES, _THAILAND = _load_provinces()
_PROVINCE_PREPARED = [(name, prep(geom), geom.bounds) for name, geom in _PROVINCES]
_THAILAND_PREPARED = prep(_THAILAND)
THAILAND_BOUNDS = _THAILAND.bounds  # (min_lon, min_lat, max_lon, max_lat)
PROVINCE_NAMES = sorted({name for name, _ in _PROVINCES})


def is_in_thailand(lats: Iterable[float], lons: Iterable[float]) -> np.ndarray:
    """Return a boolean array — True where (lat, lon) is inside Thailand's
    land border (any of the 77 provinces)."""
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    if lats_arr.shape != lons_arr.shape:
        raise ValueError("lats and lons must have the same length")

    # Cheap bbox short-circuit before the polygon query — eliminates points
    # that are obviously outside (most cells in the wider FIRMS BBOX).
    min_lon, min_lat, max_lon, max_lat = THAILAND_BOUNDS
    inside = (
        (lats_arr >= min_lat) & (lats_arr <= max_lat)
        & (lons_arr >= min_lon) & (lons_arr <= max_lon)
    )

    # Polygon test only for points that survive the bbox cut.
    candidates = np.where(inside)[0]
    for idx in candidates:
        if not _THAILAND_PREPARED.contains(Point(lons_arr[idx], lats_arr[idx])):
            inside[idx] = False
    return inside


def find_province(lats: Iterable[float], lons: Iterable[float]) -> np.ndarray:
    """Per-cell province lookup. Returns an object array of strings —
    province name for points inside Thailand, empty string for points
    outside any province polygon.

    Two-stage filter for speed: a cheap per-province bbox check
    eliminates the obvious misses, then prepared-geometry .contains()
    confirms the survivors. The expensive call runs at most once per
    cell because we break out of the province loop on the first match.
    """
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    if lats_arr.shape != lons_arr.shape:
        raise ValueError("lats and lons must have the same length")

    out = np.array([""] * lats_arr.size, dtype=object)
    for i in range(lats_arr.size):
        lat, lon = lats_arr[i], lons_arr[i]
        for name, prepared, bounds in _PROVINCE_PREPARED:
            min_lon, min_lat, max_lon, max_lat = bounds
            if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
                continue
            if prepared.contains(Point(lon, lat)):
                out[i] = name
                break
    return out
