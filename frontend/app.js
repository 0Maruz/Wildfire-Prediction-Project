// ===================================
// FIRE DATE PREDICTION DASHBOARD
//
// All values shown to the user come from REAL sources:
//   • observed:  NASA FIRMS detections at the latest base date
//   • predicted: model output (real features) for the next 7 days
//   • thresholds: calibrated from real validation predictions
//   • metrics:   real held-out test metrics from train.py
//   • historical fire count: literal sum of FIRMS detections per cell
// No synthetic / faked / interpolated data anywhere in the UI.
// ===================================

// Thailand centroid (~15.0°N, 101.0°E) at zoom 6 fits the full country
// from Mae Sai down to Narathiwat in a typical 16:9 viewport.
const map = L.map("map", { center: [15.0, 101.0], zoom: 6 });

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  { attribution: "&copy; OpenStreetMap contributors" }
).addTo(map);

// ===================================
// State
// ===================================
const state = {
  geojson: null,
  selectedDay: "all", // "all" | 0..7
  thresholds: null,   // { CRITICAL, HIGH, MEDIUM, LOW }
  metrics: null,      // held-out test metrics
  layers: { observed: null, predicted: null, heatmap: null },
};

const URGENCY_COLORS = {
  CRITICAL: "#dc2626",
  HIGH: "#ea580c",
  MEDIUM: "#f59e0b",
  LOW: "#10b981",
  NONE: "#6b7280",
};

// ===================================
// Init
// ===================================
async function init() {
  try {
    const response = await fetch("../outputs/riskmap/fire_dates_all.geojson");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.geojson = await response.json();
  } catch (err) {
    console.error("Failed to load GeoJSON:", err);
    alert("Failed to load fire prediction data. Run train.py + risk_map.py first.");
    return;
  }

  // GeoJSON top-level metadata is written by risk_map.append_geojson.
  const meta = state.geojson.metadata || {};
  state.thresholds = meta.urgency_thresholds || null;
  state.metrics = meta.metrics || null;

  renderThresholds();
  renderMetrics();
  bindEvents();
  displayData();
}

// ===================================
// Render calibrated threshold ranges in urgency cards
// ===================================
function renderThresholds() {
  const t = state.thresholds;
  if (!t) {
    document.getElementById("thresholdNote").textContent =
      "No calibrated thresholds in metadata — falling back to legacy 0/2/4/7 cutoffs.";
    document.getElementById("criticalRange").textContent = "≤ 0 d";
    document.getElementById("highRange").textContent = "≤ 2 d";
    document.getElementById("mediumRange").textContent = "≤ 4 d";
    document.getElementById("lowRange").textContent = "≤ 7 d";
    return;
  }
  const fmt = (v) => `≤ ${Number(v).toFixed(1)} d`;
  document.getElementById("criticalRange").textContent = fmt(t.CRITICAL);
  document.getElementById("highRange").textContent = fmt(t.HIGH);
  document.getElementById("mediumRange").textContent = fmt(t.MEDIUM);
  document.getElementById("lowRange").textContent = fmt(t.LOW);
}

// ===================================
// Render real held-out test metrics
// ===================================
function renderMetrics() {
  const m = state.metrics || {};
  const fmt = (v, digits = 3) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "—";
  document.getElementById("metricMAE").textContent = fmt(m.mae_days);
  document.getElementById("metricRMSE").textContent = fmt(m.rmse_days);
  document.getElementById("metricR2").textContent = fmt(m.r2);
  const acc = m.accuracy_within_1day;
  document.getElementById("metricAcc").textContent =
    typeof acc === "number" ? `${(acc * 100).toFixed(1)}%` : "—";
  if (!Object.keys(m).length) {
    document.getElementById("metricsNote").textContent =
      "No metrics in GeoJSON metadata — re-run train.py to populate.";
  }
}

// ===================================
// Display
// ===================================
function displayData() {
  if (!state.geojson || !state.geojson.features) return;
  clearLayers();

  const observed  = state.geojson.features.filter(f => f.properties.source === "observed");
  let   predicted = state.geojson.features.filter(f => f.properties.source === "predicted");

  // Latest base_date wins (file may contain history of prior base dates).
  const latestBaseDate = predicted
    .map(f => f.properties.base_date)
    .filter(Boolean)
    .sort()
    .pop() || "N/A";
  document.getElementById("baseDate").textContent = latestBaseDate;
  renderFreshness(latestBaseDate);
  predicted = predicted.filter(f => f.properties.base_date === latestBaseDate);

  // Apply day filter (real model output, no smoothing)
  const daySelected = state.selectedDay;
  const dayFiltered = daySelected === "all"
    ? predicted
    : predicted.filter(f => f.properties.days_until_fire === Number(daySelected));

  document.getElementById("daySelectorInfo").textContent =
    daySelected === "all"
      ? `Showing all ${predicted.length} predicted cells.`
      : `Showing ${dayFiltered.length} cells predicted to fire on ` +
        `${dateAdd(latestBaseDate, Number(daySelected))} (Day +${daySelected}).`;

  if (document.getElementById("showObserved").checked) {
    state.layers.observed = createObservedLayer(observed);
  }
  // The heatmap is the primary prediction visualization. The clickable cell
  // pins are an optional overlay for inspecting individual cells via popups.
  if (document.getElementById("showPredicted").checked) {
    state.layers.heatmap = createHeatmapLayer(dayFiltered);
    if (document.getElementById("showCellPins").checked) {
      state.layers.predicted = createPredictedLayer(dayFiltered);
    }
  }

  updateStatistics(predicted); // urgency summary always reflects all 7 days
  updateTimeline(predicted, latestBaseDate);
  addLayersToMap();
}

function dateAdd(isoDate, days) {
  const d = new Date(isoDate);
  if (isNaN(d.getTime())) return "—";
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

// ===================================
// Data freshness — how stale is the FIRMS data we trained / predicted on?
// FIRMS NRT typically arrives within hours but can lag by a day or more on
// quota-throttled days. ERA5 weather has a built-in 5-day lag that we accept.
// We surface this so the user knows whether "Today" on the timeline really
// means the calendar today or is shifted because the upstream feed is behind.
// ===================================
function renderFreshness(baseDateIso) {
  const badge = document.getElementById("freshnessBadge");
  const note  = document.getElementById("freshnessNote");
  if (!baseDateIso || baseDateIso === "N/A") {
    badge.textContent = "no data";
    badge.className = "freshness-badge expired";
    note.textContent = "";
    return;
  }
  const base = new Date(baseDateIso);
  const today = new Date();
  // Strip time-of-day so the diff is whole-day units.
  base.setHours(0, 0, 0, 0);
  today.setHours(0, 0, 0, 0);
  const lagDays = Math.round((today - base) / 86400000);

  let cls, label, msg;
  if (lagDays <= 0) {
    cls = "fresh";   label = "live";
    msg = "Data is current — Day +1 = real tomorrow.";
  } else if (lagDays === 1) {
    cls = "fresh";   label = "1 day old";
    msg = "Yesterday's data — Day +1 = real today.";
  } else if (lagDays <= 3) {
    cls = "stale";   label = `${lagDays} d behind`;
    msg = `Last FIRMS pull was ${lagDays} days ago. Run fetch_firms.py + risk_map.py to refresh.`;
  } else {
    cls = "expired"; label = `${lagDays} d behind`;
    msg = `Data is ${lagDays} days stale — predictions may not reflect current conditions. Refresh with fetch_firms.py.`;
  }
  badge.textContent = label;
  badge.className = `freshness-badge ${cls}`;
  note.textContent = msg;
}

// ===================================
// GRID-AWARE DOT SIZING
//
// Detect the grid resolution from the data (smallest non-zero difference
// between consecutive unique latitudes) and size each dot as a fraction of
// the cell width. Because L.circle takes radius in METERS (not pixels):
//   • Dots scale naturally with zoom — Leaflet does the conversion.
//   • A dot of radius < cell_half_width can never overlap a neighbor.
//   • Sizing self-adjusts when GRID_SIZE changes — drop in a finer grid
//     and dots get proportionally smaller, automatically.
//
// Per-tier fractions are chosen so even adjacent CRITICAL+CRITICAL dots
// (the largest pair) leave ~20% of the cell width as edge-to-edge gap.
// ===================================
const URGENCY_DOT_FRAC = {
  CRITICAL: 0.40,
  HIGH:     0.32,
  MEDIUM:   0.25,
  LOW:      0.18,
  NONE:     0.14,
};
const OBSERVED_DOT_FRAC = 0.30;       // observed-fire dots use a fixed fraction
const APPROX_METERS_PER_DEGREE = 111320;

// Pixel-size clamps applied across zoom levels. Without these, a CRITICAL dot
// (radius ~2 km in meters) becomes ~900 px wide at zoom 14 — covering most of
// the screen — and ~1 px at zoom 6 — invisible. The clamp keeps dots in a
// reasonable pixel range while still letting the meters-based scaling show
// through in mid-zoom levels.
//   • Per-tier scaling: CRITICAL gets the biggest min/max, NONE the smallest,
//     proportional to URGENCY_DOT_FRAC. The 0.40 reference matches CRITICAL.
const DOT_CLAMP_MIN_PX_REF = 3;       // CRITICAL min px at extreme zoom-out
const DOT_CLAMP_MAX_PX_REF = 28;      // CRITICAL max px at extreme zoom-in
const DOT_CLAMP_FRAC_REF   = 0.40;    // reference fraction (CRITICAL)

function _detectGridSizeDegrees(features) {
  if (!features || features.length < 2) return 0.1;
  const lats = [...new Set(features.map(f => f.geometry.coordinates[1]))]
    .sort((a, b) => a - b);
  let minDiff = Infinity;
  for (let i = 1; i < lats.length; i++) {
    const d = lats[i] - lats[i - 1];
    if (d > 1e-6 && d < minDiff) minDiff = d;
  }
  // Sanity-clamp to a plausible range (between 0.005° and 0.5°). If the
  // detection fails we fall back to the project default of 0.1°.
  if (!isFinite(minDiff) || minDiff < 0.005 || minDiff > 0.5) return 0.1;
  return minDiff;
}

function _gridSizeMeters() {
  // Detect against ALL prediction features in the GeoJSON (stable across
  // day-filter changes), then convert degrees → metres along latitude.
  const all = ((state.geojson && state.geojson.features) || [])
    .filter(f => f.properties.source === "predicted");
  return _detectGridSizeDegrees(all) * APPROX_METERS_PER_DEGREE;
}

function _dotRadiusMeters(fraction, gridMeters) {
  return (gridMeters / 2) * fraction;
}

// Web Mercator meters-per-pixel at a given latitude and the map's current zoom.
function _metersPerPixel(lat) {
  return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, map.getZoom());
}

function _clampPxForFrac(frac) {
  const scale = frac / DOT_CLAMP_FRAC_REF;
  return {
    min: DOT_CLAMP_MIN_PX_REF * scale,
    max: DOT_CLAMP_MAX_PX_REF * scale,
  };
}

// Convert a meters-radius into a meters-radius that, at the current zoom and
// latitude, falls inside [minPx, maxPx]. The natural meters-based behaviour
// is preserved in mid-zoom levels; only the extremes are clamped.
function _clampedRadiusMeters(lat, baseM, minPx, maxPx) {
  const mPerPx = _metersPerPixel(lat);
  let px = baseM / mPerPx;
  if (px > maxPx) px = maxPx;
  if (px < minPx) px = minPx;
  return px * mPerPx;
}

// ===================================
// Layers
// ===================================
function createObservedLayer(features) {
  const cluster = document.getElementById("clusterMarkers").checked;
  const layer = cluster ? L.markerClusterGroup() : L.layerGroup();

  // Shared canvas renderer = much faster than the default SVG renderer when
  // there are many markers (8k+ cells at finer grid sizes).
  const renderer = L.canvas({ padding: 0.5 });
  const gridM = _gridSizeMeters();
  const baseM = _dotRadiusMeters(OBSERVED_DOT_FRAC, gridM);
  const px = _clampPxForFrac(OBSERVED_DOT_FRAC);

  features.forEach(f => {
    const [lon, lat] = f.geometry.coordinates;
    const props = f.properties;
    const radiusM = _clampedRadiusMeters(lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer: renderer,
      fillColor: "#ff5722", color: "#fff", weight: 1, fillOpacity: 0.8,
    });
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;
    marker.bindPopup(`
      <div class="popup">
        <b style="color:#ff5722;">🔥 Observed Fire (FIRMS)</b><br>
        <small>Date: ${props.date}</small><br>
        <small>FIRMS detections: ${props.fire_count ?? "—"}</small><br>
        <small>Location: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>
      </div>
    `);
    layer.addLayer(marker);
  });

  return layer;
}

function createPredictedLayer(features) {
  // Predicted dots are no longer clustered: clustering hides the per-cell
  // colour distribution, and the heatmap underneath already conveys density.
  // The checkbox labelled "Cluster Observed Markers" applies only to FIRMS
  // detections now.
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });
  const gridM = _gridSizeMeters();

  features.forEach(f => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;

    const color = URGENCY_COLORS[p.urgency_level] || URGENCY_COLORS.NONE;
    const frac  = URGENCY_DOT_FRAC[p.urgency_level] ?? URGENCY_DOT_FRAC.NONE;
    const baseM = _dotRadiusMeters(frac, gridM);
    const px = _clampPxForFrac(frac);
    const radiusM = _clampedRadiusMeters(lat, baseM, px.min, px.max);

    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer: renderer,
      fillColor: color, color: "#fff", weight: 1, fillOpacity: 0.85,
    });
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const fireDate   = p.predicted_fire_date;
    const daysUntil  = p.days_until_fire;
    const confidence = (p.confidence != null) ? (p.confidence * 100).toFixed(0) : "—";
    const rawPred    = (p.raw_prediction != null) ? Number(p.raw_prediction).toFixed(2) : null;
    const histCount  = p.historical_fire_count_30d;

    let html = `
      <div class="popup" style="min-width:220px;">
        <b style="color:${color};">🔮 Fire Prediction</b><br>
        <div class="popup-block">
          <b>Predicted: ${fireDate}</b><br>
          <small>In ${daysUntil} day${daysUntil !== 1 ? "s" : ""}` +
          (rawPred ? ` (raw=${rawPred})` : "") + `</small>
        </div>
        <small>Urgency: <b>${p.urgency_level}</b> (calibrated)</small><br>
        <small>Confidence (rounding proxy): ${confidence}%</small><br>`;
    if (histCount != null) {
      html += `<small>Historical fires (30d, FIRMS): <b>${histCount}</b></small><br>`;
    }
    if (p.fire_days_per_year != null) {
      // Annualized empirical fire-day rate over the full FIRMS record.
      // Provides crucial context: a CRITICAL prediction in a cell with
      // <1 fire-day/year history should be read with skepticism.
      const rate = Number(p.fire_days_per_year);
      html += `<small>Historical rate: <b>${rate.toFixed(1)}</b> fire-days/year</small><br>`;
    }
    if (p.nearest_urban_area && p.nearest_urban_distance_km != null) {
      // The nearest major urban area gives geographic context — useful for
      // sanity-checking ("this CRITICAL cell is 4 km from Chiang Mai") and
      // for spotting filter near-misses (cells that *just* escaped the
      // urban-exclusion radius).
      const km = Number(p.nearest_urban_distance_km).toFixed(1);
      html += `<small>Nearest city: ${p.nearest_urban_area} (${km} km)</small><br>`;
    }
    html += `<small>Cell: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small></div>`;
    marker.bindPopup(html);

    layer.addLayer(marker);
  });

  return layer;
}

// ===================================
// COLOR INTERPOLATION
//
// Maps a raw_prediction value (in days) to an RGB color via the calibrated
// urgency thresholds. The piecewise linear gradient between threshold colors
// is what gives the smooth tier-to-tier transitions you'd see in a contour
// plot (CRITICAL→HIGH→MEDIUM→LOW), and ensures a cell at any zoom level
// renders in its own urgency color at its center — no green tint just
// because the user zoomed in past where points cluster.
// ===================================
function _hexToRgb(hex) {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function _valueToColor(value, thresholds) {
  // Stops are anchored at the calibrated thresholds. Values below CRITICAL
  // clamp to red; values above LOW clamp to green. Between stops we lerp.
  const stops = [
    { v: 0,                    c: _hexToRgb(URGENCY_COLORS.CRITICAL) },
    { v: thresholds.CRITICAL,  c: _hexToRgb(URGENCY_COLORS.CRITICAL) },
    { v: thresholds.HIGH,      c: _hexToRgb(URGENCY_COLORS.HIGH)     },
    { v: thresholds.MEDIUM,    c: _hexToRgb(URGENCY_COLORS.MEDIUM)   },
    { v: thresholds.LOW,       c: _hexToRgb(URGENCY_COLORS.LOW)      },
  ];

  if (value <= stops[0].v) return stops[0].c;
  for (let i = 1; i < stops.length; i++) {
    if (value <= stops[i].v) {
      const a = stops[i - 1], b = stops[i];
      const span = b.v - a.v;
      const t = span > 1e-9 ? (value - a.v) / span : 0;
      return {
        r: Math.round(a.c.r + t * (b.c.r - a.c.r)),
        g: Math.round(a.c.g + t * (b.c.g - a.c.g)),
        b: Math.round(a.c.b + t * (b.c.b - a.c.b)),
      };
    }
  }
  return stops[stops.length - 1].c;
}

// ===================================
// HEATMAP LAYER — IDW interpolation, urgency-colored
//
// Renders the predictions as a continuous color surface where each cell's
// CENTER is its own tier color, and adjacent cells blend smoothly into a
// gradient. This is the contour-style look from the example image.
//
// Why IDW instead of L.heatLayer:
//   The previous Leaflet.heat layer was density-based — its color came from
//   the *number* of overlapping points at a pixel, then mapped through a
//   green→red gradient. When zoomed in, isolated points had low density
//   relative to the canvas max, so the gradient mapped them to the bottom
//   of the scale (green). That made even CRITICAL cells look green at high
//   zoom, which was wrong.
//
//   IDW (inverse-distance weighting) solves this by computing each pixel's
//   color from the actual `raw_prediction` value of nearby cells:
//
//     pixel_value = Σ (w_i × value_i)  /  Σ w_i,    w_i = 1 / (d_i² + 1)
//
//   Then pixel_value (in days) maps to a color via the calibrated urgency
//   thresholds. A CRITICAL cell's center always reads red because its
//   raw_prediction is low — independent of zoom or local point density.
//
// Implementation notes:
//   • L.GridLayer extension renders 256×256 tiles on demand and caches them.
//   • Each tile filters points to those within its bounds + a buffer so
//     the per-pixel inner loop only touches relevant cells.
//   • Internal grid is rendered at 1/STRIDE resolution and upscaled with
//     canvas bilinear smoothing — ~16× speedup and free Gaussian-ish blur.
//   • Alpha falls off quadratically from each point's center to the cutoff
//     so isolated cells render as soft circular blobs, not hard squares.
// ===================================
function createHeatmapLayer(features) {
  if (!features || features.length === 0) return null;

  // Calibrated thresholds with a sensible fallback if metadata is missing.
  const thresholds = state.thresholds || { CRITICAL: 1, HIGH: 2.5, MEDIUM: 4.5, LOW: 7 };
  const radius = Number(document.getElementById("heatRadius").value) || 50;

  // Pre-extract numeric points so createTile() never re-parses GeoJSON.
  const points = features.map(f => ({
    lat: f.geometry.coordinates[1],
    lon: f.geometry.coordinates[0],
    value: f.properties.raw_prediction != null
      ? Number(f.properties.raw_prediction)
      : Number(f.properties.days_until_fire),
  })).filter(p => isFinite(p.value));

  if (points.length === 0) return null;

  const STRIDE = 4;        // render at 1/STRIDE resolution then upscale
  const MAX_ALPHA = 200;   // peak per-pixel alpha (out of 255)
  const cutoff = radius;
  const cutoffSq = cutoff * cutoff;

  const InterpolationLayer = L.GridLayer.extend({
    createTile: function (coords) {
      const tile = document.createElement("canvas");
      const size = this.getTileSize();
      tile.width = size.x;
      tile.height = size.y;
      const ctx = tile.getContext("2d");

      // Tile geographic bounds → pixel projection
      const tileBounds = this._tileCoordsToBounds(coords);
      const nw = tileBounds.getNorthWest();
      const se = tileBounds.getSouthEast();
      const lonSpan = se.lng - nw.lng;
      const latSpan = nw.lat - se.lat;
      if (lonSpan <= 0 || latSpan <= 0) return tile;

      // Buffer (in degrees) corresponding to the cutoff in pixels — used to
      // pick up points just outside the tile that still influence its edges.
      const bufferLng = (radius / size.x) * lonSpan;
      const bufferLat = (radius / size.y) * latSpan;

      // Project nearby points into tile-local pixel coords
      const localPoints = [];
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (p.lat < se.lat - bufferLat || p.lat > nw.lat + bufferLat) continue;
        if (p.lon < nw.lng - bufferLng || p.lon > se.lng + bufferLng) continue;
        const px = ((p.lon - nw.lng) / lonSpan) * size.x;
        const py = ((nw.lat - p.lat) / latSpan) * size.y;
        localPoints.push({ px, py, value: p.value });
      }
      if (localPoints.length === 0) return tile;

      // Render at reduced resolution (W × H) then upscale to full tile.
      const W = Math.ceil(size.x / STRIDE);
      const H = Math.ceil(size.y / STRIDE);
      const lowRes = ctx.createImageData(W, H);
      const data = lowRes.data;

      for (let py = 0; py < H; py++) {
        for (let px = 0; px < W; px++) {
          const x = (px + 0.5) * STRIDE;
          const y = (py + 0.5) * STRIDE;

          let nearestValue = null;
          let nearestDistSq = Infinity;
          let blobAlpha = 0;   // tracked separately for soft circular edges

          for (let i = 0; i < localPoints.length; i++) {
            const lp = localPoints[i];
            const dx = lp.px - x;
            const dy = lp.py - y;
            const d2 = dx * dx + dy * dy;
            if (d2 > cutoffSq) continue;

            // Color comes from the NEAREST cell only — guarantees the surface
            // around a green marker stays green even when a yellow neighbour
            // is within the IDW cutoff. (IDW averaging caused tier-mismatches.)
            if (d2 < nearestDistSq) {
              nearestDistSq = d2;
              nearestValue = lp.value;
            }

            // Quadratic falloff for blob alpha — goes to 0 exactly at cutoff
            const f = 1 - Math.sqrt(d2) / cutoff;
            const wAlpha = f * f;
            if (wAlpha > blobAlpha) blobAlpha = wAlpha;
          }

          const idx = (py * W + px) * 4;
          if (nearestValue === null) {
            data[idx + 3] = 0;
            continue;
          }

          const color = _valueToColor(nearestValue, thresholds);
          data[idx]     = color.r;
          data[idx + 1] = color.g;
          data[idx + 2] = color.b;
          // Bias alpha slightly so blob centers read solid; cap at MAX_ALPHA
          data[idx + 3] = Math.round(Math.min(1, blobAlpha * 1.4) * MAX_ALPHA);
        }
      }

      // Upscale low-res → full tile using bilinear smoothing (free blur).
      const tmp = document.createElement("canvas");
      tmp.width = W;
      tmp.height = H;
      tmp.getContext("2d").putImageData(lowRes, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(tmp, 0, 0, size.x, size.y);

      return tile;
    },
  });

  return new InterpolationLayer({
    opacity: 0.85,
    keepBuffer: 2,   // pre-render a 2-tile margin to avoid edge gaps on pan
  });
}

// ===================================
// Stats / timeline
// ===================================
function updateStatistics(predicted) {
  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  predicted.forEach(f => {
    const u = f.properties.urgency_level;
    if (u in counts) counts[u]++;
  });
  document.getElementById("criticalCount").textContent = counts.CRITICAL;
  document.getElementById("highCount").textContent = counts.HIGH;
  document.getElementById("mediumCount").textContent = counts.MEDIUM;
  document.getElementById("lowCount").textContent = counts.LOW;
}

function updateTimeline(predicted, baseDate) {
  const timeline = document.getElementById("timeline");
  timeline.innerHTML = "";

  const dayCounts = {};
  for (let i = 0; i <= 7; i++) dayCounts[i] = 0;
  predicted.forEach(f => {
    const d = f.properties.days_until_fire;
    if (d >= 0 && d <= 7) dayCounts[d]++;
  });

  // Day buttons that map to empty days are hidden so the selector reflects
  // the model's actual predictive range. The "All" button is always shown.
  // If the currently-selected day went empty (e.g. user filtered by +4 then
  // a refresh produced 0 cells there), fall back to "All".
  let activeWentEmpty = false;
  document.querySelectorAll(".day-btn").forEach(btn => {
    const day = btn.dataset.day;
    if (day === "all") return;
    const count = dayCounts[Number(day)] || 0;
    if (count === 0) {
      btn.style.display = "none";
      if (state.selectedDay === day) activeWentEmpty = true;
    } else {
      btn.style.display = "";
    }
  });
  if (activeWentEmpty) {
    state.selectedDay = "all";
    document.querySelectorAll(".day-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.day === "all");
    });
  }

  // Timeline rows: skip days with no predictions so the panel doesn't
  // display a stack of "0 fires" placeholders for the prediction-horizon
  // tail the model can't reach.
  for (let i = 0; i <= 7; i++) {
    const count = dayCounts[i];
    if (count === 0) continue;
    const dateStr = dateAdd(baseDate, i);
    const label = i === 0 ? "Today" : i === 1 ? "Tomorrow" : `+${i} days`;

    const item = document.createElement("div");
    item.className = "timeline-item has-fires";
    item.innerHTML = `
      <div class="timeline-day">${label}</div>
      <div class="timeline-date">${dateStr}</div>
      <div class="timeline-count">${count} fire${count !== 1 ? "s" : ""}</div>
    `;
    item.style.cursor = "pointer";
    item.addEventListener("click", () => selectDay(String(i)));
    timeline.appendChild(item);
  }

  // If no day has any predictions, show a single explanatory row instead of
  // an empty panel — operators noticing this should know the model's view
  // is "no imminent fire risk" rather than "the dashboard is broken."
  if (timeline.childElementCount === 0) {
    const item = document.createElement("div");
    item.className = "timeline-item";
    item.innerHTML = `
      <div class="timeline-day" style="opacity:0.6;">No predictions</div>
      <div class="timeline-date">over the next 7 days</div>
      <div class="timeline-count" style="opacity:0.6;">—</div>
    `;
    timeline.appendChild(item);
  }
}

// ===================================
// Layer mgmt
//
// Layer add order matters: heatmap is added FIRST so it renders below the
// observed/predicted marker layers. That keeps the smooth gradient as the
// background while the dot markers stay clickable on top for popups.
// ===================================
const LAYER_RENDER_ORDER = ["heatmap", "observed", "predicted"];

function clearLayers() {
  for (const k of LAYER_RENDER_ORDER) {
    if (state.layers[k]) {
      map.removeLayer(state.layers[k]);
      state.layers[k] = null;
    }
  }
}
function addLayersToMap() {
  for (const k of LAYER_RENDER_ORDER) {
    if (state.layers[k]) map.addLayer(state.layers[k]);
  }
}

// Re-apply the [minPx, maxPx] pixel clamp to every dot whenever the user zooms.
// Heatmap re-renders natively on zoom — only the L.circle layers need this.
function _reclampAllMarkers() {
  for (const k of ["observed", "predicted"]) {
    const layer = state.layers[k];
    if (!layer || typeof layer.eachLayer !== "function") continue;
    layer.eachLayer(m => {
      if (m._baseRadiusM == null || typeof m.setRadius !== "function") return;
      const r = _clampedRadiusMeters(m._anchorLat, m._baseRadiusM, m._minPx, m._maxPx);
      m.setRadius(r);
    });
  }
}
map.on("zoomend", _reclampAllMarkers);

// ===================================
// Day selector
// ===================================
function selectDay(day) {
  state.selectedDay = day;
  document.querySelectorAll(".day-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.day === String(day));
  });
  displayData();
}

function bindEvents() {
  document.querySelectorAll(".day-btn").forEach(btn => {
    btn.addEventListener("click", () => selectDay(btn.dataset.day));
  });
  document.getElementById("showObserved").addEventListener("change", displayData);
  document.getElementById("clusterMarkers").addEventListener("change", displayData);
  document.getElementById("showCellPins").addEventListener("change", displayData);

  // The Smoothing-radius slider lives inside the prediction toggle group:
  // it only makes sense when the heatmap is rendered, so we hide it when
  // predictions are off.
  const predictedToggle = document.getElementById("showPredicted");
  const radiusGroup = document.getElementById("heatRadiusGroup");
  const radiusInput = document.getElementById("heatRadius");
  const radiusValue = document.getElementById("heatRadiusValue");

  const syncRadiusVisibility = () => {
    radiusGroup.hidden = !predictedToggle.checked;
  };
  syncRadiusVisibility();

  predictedToggle.addEventListener("change", () => {
    syncRadiusVisibility();
    displayData();
  });

  // Live label update on every drag tick; only rebuild the heatmap once the
  // user releases the slider so a single drag doesn't re-render dozens of
  // times (each render bakes ~1k canvas tiles).
  radiusInput.addEventListener("input", () => {
    radiusValue.textContent = `${radiusInput.value} px`;
  });
  radiusInput.addEventListener("change", displayData);
}

// ===================================
// Start
// ===================================
init();