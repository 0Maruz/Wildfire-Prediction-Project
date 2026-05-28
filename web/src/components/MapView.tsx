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
import { fmtTr, useLang } from "../utils/i18n";
import { readProbability } from "../utils/probability";
import {
  AnchoredCircle,
  clampPxForFrac,
  clampedRadiusMeters,
  dotRadiusMeters,
  gridSizeMeters,
} from "../utils/markers";

type Translator = (key: string, fallback?: string) => string;

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
  const { t } = useLang();
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

    // Listen for "firewatch:flyto" events dispatched by AlertToasts so
    // clicking an alert pans the map to that hotspot.
    const onFlyTo = (e: Event) => {
      const detail = (e as CustomEvent<{ lat: number; lon: number }>).detail;
      if (!detail || !mapRef.current) return;
      mapRef.current.flyTo([detail.lat, detail.lon], 11, { duration: 1.2 });
    };
    window.addEventListener("firewatch:flyto", onFlyTo);

    return () => {
      window.removeEventListener("firewatch:flyto", onFlyTo);
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
        layersRef.current.predicted = buildPredictedLayer(map, props.predictedVisible, gridM, t);
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
    t,
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
      weight: 0.4,
      fillOpacity: 0.55,
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

function buildPredictedLayer(map: L.Map, features: FireFeature[], gridM: number, t: Translator): L.Layer {
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
      weight: 0.4,
      fillOpacity: 0.6,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const fireDate = p.predicted_fire_date ?? "—";
    const daysUntil = p.days_until_fire ?? 0;
    // Prefer the persisted `probability` field (new snapshots) and fall
    // back to inverting raw_prediction (legacy snapshots).
    const probability = readProbability(p);
    const histCount = p.historical_fire_count_30d;

    const urgencyMap: Record<UrgencyLevel, string> = {
      CRITICAL: t("map.urgency.critical"),
      HIGH:     t("map.urgency.high"),
      MEDIUM:   t("map.urgency.medium"),
      LOW:      t("map.urgency.low"),
      NONE:     t("map.urgency.none"),
    };
    const urgencyLabel = urgencyMap[urgency];

    let probInterpretation = "";
    if (probability !== null) {
      if (probability >= 0.7)       probInterpretation = t("map.prob.interp.veryhigh");
      else if (probability >= 0.5)  probInterpretation = t("map.prob.interp.high");
      else if (probability >= 0.35) probInterpretation = t("map.prob.interp.medium");
      else if (probability >= 0.2)  probInterpretation = t("map.prob.interp.low");
      else                          probInterpretation = t("map.prob.interp.verylow");
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
           <div style="font-size:11px; color:#9aa0aa;">${t("map.prob.title")}</div>
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
    html += `<small>${t("map.popup.predicted_date")}: <b>${fireDate}</b></small><br>`;
    if (histCount != null)
      html += `<small>${t("map.popup.fires_30d")}: <b>${histCount}</b></small><br>`;
    if (p.fire_days_per_year != null)
      html += `<small><b>${Number(p.fire_days_per_year).toFixed(1)}</b> ${t("map.popup.fires_per_year")}</small><br>`;
    if (p.nearest_urban_area && p.nearest_urban_distance_km != null) {
      const km = Number(p.nearest_urban_distance_km).toFixed(1);
      html += `<small>${t("map.popup.nearest_city")}: ${p.nearest_urban_area} (${km} km)</small><br>`;
    }
    html += `<small>${t("map.popup.coords")}: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>`;
    const weatherSlotId = `wx-${lat.toFixed(4)}-${lon.toFixed(4)}-${(p.base_date || "").replace(/-/g, "")}`;
    html +=
      `<hr style="border:0;border-top:1px dashed var(--border,#24252a);margin:6px 0">
       <div id="${weatherSlotId}" class="popup-weather-slot">
         <small style="color:#9aa0aa">${t("map.weather.loading")}</small>
       </div>`;
    html += `</div>`;
    marker.bindPopup(html);

    // Fetch + inject weather when popup opens (lazy).
    marker.on("popupopen", () => {
      const slot = document.getElementById(weatherSlotId);
      if (!slot || slot.dataset.loaded === "true") return;
      slot.dataset.loaded = "true";
      const targetDate = p.predicted_fire_date || p.base_date || undefined;
      import("../api").then(({ fetchCellWeather }) => fetchCellWeather(lat, lon, targetDate))
        .then((w) => {
          if (!w.available) {
            slot.innerHTML = `<small style="color:#9aa0aa">${fmtTr(t("map.weather.unavailable"), { reason: w.reason ?? "" })}</small>`;
            return;
          }
          const fwi = w.fire_weather_index ?? { level: "—", color: "#9aa0aa", emoji: "❔" };
          const tmin = w.temp_min_c ?? 0;
          const tmax = w.temp_max_c ?? 0;
          const p7 = w.precip_7d_mm ?? 0;
          const tempBar =
            `<svg width="160" height="14" style="border-radius:7px;display:block;margin-top:3px">
               <defs><linearGradient id="tg-${weatherSlotId}" x1="0" x2="1">
                 <stop offset="0%" stop-color="#3b82f6"/>
                 <stop offset="50%" stop-color="#eab308"/>
                 <stop offset="100%" stop-color="#ef4444"/>
               </linearGradient></defs>
               <rect width="160" height="14" fill="url(#tg-${weatherSlotId})" rx="7"/>
               <line x1="${Math.max(2, ((tmin - 15) / 30) * 160)}" y1="0" x2="${Math.max(2, ((tmin - 15) / 30) * 160)}" y2="14" stroke="#fff" stroke-width="2"/>
               <line x1="${Math.min(158, ((tmax - 15) / 30) * 160)}" y1="0" x2="${Math.min(158, ((tmax - 15) / 30) * 160)}" y2="14" stroke="#fff" stroke-width="2"/>
             </svg>`;
          slot.innerHTML =
            `<div style="font-size:11px;color:#9aa0aa;margin-bottom:2px">${fmtTr(t("map.weather.title"), { date: w.date })}</div>
             <div style="display:flex;align-items:baseline;gap:6px">
               <small>${t("map.weather.temp")}</small>
               <b>${tmin.toFixed(0)}°C – ${tmax.toFixed(0)}°C</b>
             </div>
             ${tempBar}
             <div style="margin-top:5px"><small>${fmtTr(t("map.weather.rain7d"), { value: p7.toFixed(1) })}</small></div>
             ${w.wind_max_kmh != null ? `<div><small>${fmtTr(t("map.weather.wind"), { value: w.wind_max_kmh.toFixed(1) })}</small></div>` : ""}
             <div style="margin-top:6px;display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:999px;background:${fwi.color}22;color:${fwi.color};border:1px solid ${fwi.color}66;font-size:11px;font-weight:700">
               ${fmtTr(t("map.weather.fwi"), { emoji: fwi.emoji, level: fwi.level })}
             </div>`;
        })
        .catch(() => {
          slot.innerHTML = `<small style="color:#9aa0aa">${t("map.weather.error")}</small>`;
        });
    });

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
      weight: 0.4,
      fillOpacity: 0.7,
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

