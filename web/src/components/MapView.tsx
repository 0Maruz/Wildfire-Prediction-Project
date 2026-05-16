import L from "leaflet";
import "leaflet.markercluster";
import { useEffect, useMemo, useRef } from "react";
import {
  OBSERVED_DOT_FRAC,
  URGENCY_COLORS,
  URGENCY_DOT_FRAC,
} from "../constants";
import type {
  DisplayOptions,
  FireFeature,
  GistdaFeature,
  UrgencyLevel,
  UrgencyThresholds,
} from "../types";
import { createHeatmapLayer } from "../utils/heatmap";
import { LIVE_FIRE_COLOR } from "../utils/gistda";
import {
  AnchoredCircle,
  clampPxForFrac,
  clampedRadiusMeters,
  dotRadiusMeters,
  gridSizeMeters,
} from "../utils/markers";

interface MapViewProps {
  observed: FireFeature[];
  predictedAll: FireFeature[]; // all predicted features for grid-size detection
  predictedVisible: FireFeature[]; // current day-filter applied
  liveFires: GistdaFeature[];
  thresholds: UrgencyThresholds | null;
  options: DisplayOptions;
}

// Render order: heatmap below, then observed / predicted / live-fire dots on
// top so popups stay clickable. Live fires render last so they sit above
// the historical-observation layer when both are on.
const LAYER_ORDER = ["heatmap", "observed", "predicted", "livefire"] as const;
type LayerKey = (typeof LAYER_ORDER)[number];

export default function MapView(props: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<Record<LayerKey, L.Layer | null>>({
    heatmap: null,
    observed: null,
    predicted: null,
    livefire: null,
  });

  // Theme-aware tile layer — swaps the basemap when the user toggles light/dark
  // mode (ThemeToggle sets <html data-theme="light">). Kept as a ref so we can
  // remove/add it cleanly without recreating the whole map.
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  const tileUrlFor = (theme: "light" | "dark") =>
    theme === "light"
      ? "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

  const currentTheme = (): "light" | "dark" =>
    document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";

  // One-time map init.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, { center: [15.0, 101.0], zoom: 6 });
    tileLayerRef.current = L.tileLayer(tileUrlFor(currentTheme()), {
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    }).addTo(map);

    map.on("zoomend", () => reclampAllMarkers(layersRef.current, map));
    mapRef.current = map;

    // Watch for theme changes from ThemeToggle and swap tiles live.
    const observer = new MutationObserver(() => {
      if (!mapRef.current || !tileLayerRef.current) return;
      mapRef.current.removeLayer(tileLayerRef.current);
      tileLayerRef.current = L.tileLayer(tileUrlFor(currentTheme()), {
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
      }).addTo(mapRef.current);
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
      tileLayerRef.current = null;
    };
  }, []);

  // Stable grid-size for sizing dots (detected from the full set, not just visible).
  const gridM = useMemo(() => gridSizeMeters(props.predictedAll), [props.predictedAll]);

  // Rebuild layers whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    clearLayers(layersRef.current, map);

    if (props.options.showObserved) {
      layersRef.current.observed = buildObservedLayer(map, props.observed);
    }

    if (props.options.showPredicted) {
      layersRef.current.heatmap = createHeatmapLayer(
        props.predictedVisible,
        props.thresholds,
        props.options.heatRadius
      );
      if (props.options.showCellPins) {
        layersRef.current.predicted = buildPredictedLayer(map, props.predictedVisible, gridM);
      }
    }

    if (props.options.showLiveFires && props.liveFires.length > 0) {
      layersRef.current.livefire = buildLiveFireLayer(map, props.liveFires, gridM);
    }

    addLayers(layersRef.current, map);
  }, [
    props.observed,
    props.predictedVisible,
    props.liveFires,
    props.thresholds,
    props.options.showObserved,
    props.options.showPredicted,
    props.options.showCellPins,
    props.options.showLiveFires,
    props.options.heatRadius,
    gridM,
  ]);

  return <div id="map" ref={containerRef} />;
}

// ───────────────────── helpers ─────────────────────

function clearLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) {
    if (layers[k]) {
      map.removeLayer(layers[k]!);
      layers[k] = null;
    }
  }
}

function addLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) if (layers[k]) map.addLayer(layers[k]!);
}

function reclampAllMarkers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of ["observed", "predicted", "livefire"] as const) {
    const layer = layers[k];
    if (!layer) continue;
    const eachLayer = (layer as unknown as { eachLayer?: (cb: (m: L.Layer) => void) => void })
      .eachLayer;
    if (!eachLayer) continue;
    eachLayer.call(layer, (m) => {
      const marker = m as AnchoredCircle;
      if (marker._baseRadiusM == null || typeof marker.setRadius !== "function") return;
      marker.setRadius(
        clampedRadiusMeters(map, marker._anchorLat, marker._baseRadiusM, marker._minPx, marker._maxPx)
      );
    });
  }
}

function buildObservedLayer(map: L.Map, features: FireFeature[]): L.Layer {
  // Cluster mode removed from the UI — observed markers are always individual
  // dots now, matching the simplified frontend/index.html layout.
  const layer = L.layerGroup();

  const renderer = L.canvas({ padding: 0.5 });
  const gridM = gridSizeMeters(features);
  const baseM = dotRadiusMeters(OBSERVED_DOT_FRAC, gridM);
  const px = clampPxForFrac(OBSERVED_DOT_FRAC);

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: "#ff5722",
      color: "#fff",
      weight: 1,
      fillOpacity: 0.8,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;
    marker.bindPopup(
      `<div class="popup">
         <b style="color:#ff5722;">🔥 Observed Fire (FIRMS)</b><br>
         <small>Date: ${p.date ?? "—"}</small><br>
         <small>FIRMS detections: ${p.fire_count ?? "—"}</small><br>
         <small>Location: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>
       </div>`
    );
    (layer as unknown as { addLayer: (m: L.Layer) => void }).addLayer(marker);
  }
  return layer as unknown as L.Layer;
}

function buildPredictedLayer(map: L.Map, features: FireFeature[], gridM: number): L.Layer {
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const urgency = (p.urgency_level ?? "NONE") as UrgencyLevel;
    const color = URGENCY_COLORS[urgency] ?? URGENCY_COLORS.NONE;
    const frac = URGENCY_DOT_FRAC[urgency] ?? URGENCY_DOT_FRAC.NONE;
    const baseM = dotRadiusMeters(frac, gridM);
    const px = clampPxForFrac(frac);
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);

    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: color,
      color: "#fff",
      weight: 1,
      fillOpacity: 0.85,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const fireDate = p.predicted_fire_date ?? "—";
    const daysUntil = p.days_until_fire ?? 0;
    // raw_prediction is pseudo-days from the binary classifier:
    //   days = 1 + (1 - prob) * 6   (see train.py _prob_to_days_for_compat)
    // Invert to recover the operationally-meaningful probability.
    //   prob = 1 - (days - 1) / 6   clipped to [0, 1]
    const rawPred = p.raw_prediction;
    let probability: number | null = null;
    if (typeof rawPred === "number" && isFinite(rawPred)) {
      probability = Math.max(0, Math.min(1, 1 - (rawPred - 1) / 6));
    }
    const histCount = p.historical_fire_count_30d;

    // Risk-tier headline label (Thai) per urgency. Maps the calibrated
    // urgency tier to operator-language so the popup reads at a glance.
    const urgencyTh: Record<UrgencyLevel, string> = {
      CRITICAL: "🔴 เสี่ยงสูงมาก",
      HIGH:     "🟠 เสี่ยงสูง",
      MEDIUM:   "🟡 เสี่ยงปานกลาง",
      LOW:      "🟢 เสี่ยงต่ำ",
      NONE:     "⚪ ไม่ระบุ",
    };
    const urgencyLabel = urgencyTh[urgency];

    // Probability tier paraphrase — bind the % to ground-truth intuition.
    // Calibration is rough (model is not calibrated probability-wise) but
    // gives the operator a feel for what the number means in practice.
    let probInterpretation = "";
    if (probability !== null) {
      if (probability >= 0.7)       probInterpretation = "≈ ใน 10 cell ระดับนี้ ~7+ เกิดไฟใน 3 วัน";
      else if (probability >= 0.5)  probInterpretation = "≈ ใน 10 cell ระดับนี้ ~5 เกิดไฟใน 3 วัน";
      else if (probability >= 0.35) probInterpretation = "≈ ใน 10 cell ระดับนี้ ~3 เกิดไฟใน 3 วัน";
      else if (probability >= 0.2)  probInterpretation = "≈ ใน 10 cell ระดับนี้ ~2 เกิดไฟใน 3 วัน";
      else                          probInterpretation = "≈ ใน 10 cell ระดับนี้ &lt;1 เกิดไฟใน 3 วัน";
    }

    const probPct = probability !== null ? Math.round(probability * 100) : null;
    const barPct = probPct ?? 0;

    let html =
      `<div class="popup" style="min-width:240px;">
         <div style="font-weight:600; color:${color}; font-size:14px; margin-bottom:4px;">
           ${urgencyLabel}
         </div>`;
    if (probPct !== null) {
      html +=
        `<div style="margin:6px 0;">
           <div style="font-size:11px; color:#9aa0aa;">โอกาสเกิดไฟใน 3 วัน</div>
           <div style="font-size:26px; font-weight:700; line-height:1;">${probPct}%</div>
           <div style="margin-top:4px; height:6px; background:#22272e; border-radius:3px; overflow:hidden;">
             <div style="height:100%; width:${barPct}%; background:${color};"></div>
           </div>
           <div style="font-size:10px; color:#6c707a; margin-top:4px;">${probInterpretation}</div>
         </div>`;
    } else {
      html +=
        `<div class="popup-block">
           <b>Predicted: ${fireDate}</b><br>
           <small>In ${daysUntil} day${daysUntil !== 1 ? "s" : ""}</small>
         </div>`;
    }
    html += `<small>ทำนายวันที่: <b>${fireDate}</b></small><br>`;
    if (histCount != null)
      html += `<small>ไฟใน 30 วันที่ผ่านมา (FIRMS): <b>${histCount}</b></small><br>`;
    if (p.fire_days_per_year != null)
      html += `<small>อัตราเกิดไฟ: <b>${Number(p.fire_days_per_year).toFixed(1)}</b> วัน/ปี</small><br>`;
    if (p.nearest_urban_area && p.nearest_urban_distance_km != null) {
      const km = Number(p.nearest_urban_distance_km).toFixed(1);
      html += `<small>เมืองใกล้สุด: ${p.nearest_urban_area} (${km} km)</small><br>`;
    }
    html += `<small>พิกัด: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small></div>`;
    marker.bindPopup(html);
    layer.addLayer(marker);
  }
  return layer;
}

// Live GISTDA VIIRS hotspots — distinct cyan dots so they don't visually
// merge with the urgency-coloured prediction layer.
function buildLiveFireLayer(map: L.Map, features: GistdaFeature[], gridM: number): L.Layer {
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });
  const frac = 0.22;
  const baseM = dotRadiusMeters(frac, gridM);
  const px = clampPxForFrac(frac);

  for (const feat of features) {
    const a = feat.attributes ?? {};
    const lat = Number(a.latitude);
    const lon = Number(a.longitude);
    if (!isFinite(lat) || !isFinite(lon)) continue;

    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: LIVE_FIRE_COLOR,
      color: "#fff",
      weight: 1,
      fillOpacity: 0.88,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const dateMs = a.date;
    const dateStr = dateMs ? new Date(dateMs).toLocaleDateString() : "—";
    const timeStr = (a.time ?? "").trim() || "—";
    const conf = a.confident ?? "—";
    const lu = a.lu_name ?? "—";
    const prov = a.pv_tn ?? "—";
    const dist = a.ap_tn ?? "—";
    const sat = a.satellite ?? "VIIRS-NPP";

    marker.bindPopup(
      `<div class="popup">
         <b style="color:${LIVE_FIRE_COLOR};">🛰 Live Hotspot · GISTDA ${sat}</b><br>
         <div class="popup-block">
           <b>${dateStr} ${timeStr}</b><br>
           <small>${prov}${dist ? " · " + dist : ""}</small>
         </div>
         <small>Land use: ${lu}</small><br>
         <small>Confidence: ${conf}</small><br>
         <small>${lat.toFixed(3)}°N, ${lon.toFixed(3)}°E</small>
       </div>`
    );
    layer.addLayer(marker);
  }
  return layer;
}
