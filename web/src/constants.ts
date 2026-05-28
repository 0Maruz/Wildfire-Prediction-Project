import type { UrgencyLevel } from "./types";

export const URGENCY_COLORS: Record<UrgencyLevel, string> = {
  CRITICAL: "#dc2626",
  HIGH: "#ea580c",
  MEDIUM: "#f59e0b",
  LOW: "#10b981",
  NONE: "#6b7280",
};

// Per-tier dot fractions of cell-width. Sized so HIGH dots touch their
// neighbors at the centre (no lattice gap), MEDIUM and LOW stay clearly
// dot-shaped (visible separation). Earlier values (0.95/0.78) made cells
// overlap into solid blobs; smaller values (0.4/0.32) left visible empty
// "grid stripes" between dots. This middle ground reads as a heatmap
// near hotspots and individual points elsewhere.
export const URGENCY_DOT_FRAC: Record<UrgencyLevel, number> = {
  CRITICAL: 0.55,
  HIGH:     0.46,
  MEDIUM:   0.38,
  LOW:      0.30,
  NONE:     0.24,
};

export const OBSERVED_DOT_FRAC = 0.36;
export const APPROX_METERS_PER_DEGREE = 111320;

// Pixel-size clamps applied across zoom levels — without these a 2 km
// CRITICAL dot becomes ~900 px wide at zoom 14 and ~1 px at zoom 6.
export const DOT_CLAMP_MIN_PX_REF = 3;
export const DOT_CLAMP_MAX_PX_REF = 28;
export const DOT_CLAMP_FRAC_REF = 0.4;

// Thailand bounding box. The model trains across the full SEA region but the
// dashboard is Thailand-only — any prediction or observation outside this box
// gets dropped before rendering.
export const THAILAND_BBOX = {
  latMin: 5.5,
  latMax: 20.5,
  lonMin: 97.5,
  lonMax: 105.7,
} as const;

export function isInThailandBbox(lat: number, lon: number): boolean {
  return (
    lat >= THAILAND_BBOX.latMin &&
    lat <= THAILAND_BBOX.latMax &&
    lon >= THAILAND_BBOX.lonMin &&
    lon <= THAILAND_BBOX.lonMax
  );
}
