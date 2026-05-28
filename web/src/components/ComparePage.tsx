import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { FireFeature } from "../types";
import { fmtTr, useLang } from "../utils/i18n";
import { readProbability } from "../utils/probability";

interface Props {
  allFeatures: FireFeature[];
  observedFeatures: FireFeature[];
}

// ─────────────────────────────────────────────────────────────
// Compare page — "ทำนายไว้ vs เกิดจริง"
//
// Side-by-side honesty check on the model. Uses validation_status that
// risk_map.py already writes on every cell in the historical GeoJSON:
//
//   hit    = predicted high-urgency cell AND a real FIRMS hotspot landed
//            within ±1 day of the predicted window
//   miss   = predicted high-urgency cell but no FIRMS hotspot showed up
//   future = predicted but the ±1-day window hasn't closed yet
//
// Overlay map colour code:
//   🟢 green  = ทำนายถูก (hit — predicted AND occurred)
//   🟡 yellow = เตือนเกิน (false alarm — predicted but no fire)
//   🔴 red    = พลาด (fire occurred but never on the high-urgency watch-list)
//
// Stats: precision = hits / (hits + false alarms); recall = hits / (hits + misses)
//
// Window selector lets the operator pick a single past base_date snapshot to
// audit. "Latest closed window" is the default — most recent prediction
// where the ±1-day audit completed.
// ─────────────────────────────────────────────────────────────

interface Stats { hits: number; falseAlarms: number; misses: number; future: number;
                  precision: number; recall: number; f1: number; }

const PALETTE = {
  hit:   "#22c55e",
  alarm: "#eab308",
  miss:  "#ef4444",
  future:"#6c707a",
};

// Haversine distance in km between two (lat, lon) pairs.
function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

export default function ComparePage({ allFeatures, observedFeatures }: Props) {
  const { t } = useLang();
  const [baseDate, setBaseDate] = useState<string>("__latest__");
  const [view, setView] = useState<"split" | "overlay">("overlay");
  // Defaults: 25 km radius (~2× the 0.1° grid cell — covers neighbour-cell
  // misses and FIRMS pixel jitter) and ±1 day. Matches the sidebar
  // headline hit-rate so the two views read consistently. Slider can
  // tighten to 5 km for a strict audit.
  const [radiusKm, setRadiusKm] = useState<number>(30);
  const [dayWindow, setDayWindow] = useState<number>(1);
  // Audit tier filter — default is HIGH+MEDIUM only since LOW is the
  // model's "background" tier (bottom 60% by probability) and almost all
  // over-alarms come from there. Operators only act on HIGH/MEDIUM, so
  // showing audit numbers on those tiers reflects the system's real
  // operational performance. Toggle to "ALL" for the full distribution.
  const [tierFilter, setTierFilter] = useState<"HIGH_MED" | "HIGH" | "ALL">("HIGH_MED");
  // Clip observed FIRMS to "predicted area" — when ON, observations that
  // are >coverageKm km from any prediction this snapshot don't count as
  // misses. This makes recall reflect coverage *inside the model's
  // attention zone* (which is where it could even theoretically be a
  // miss). Default ON so the headline recall isn't dragged down by fires
  // in regions the model never tried to forecast (cross-border, novel
  // burns, etc.).
  const [clipToCoverage, setClipToCoverage] = useState<boolean>(true);
  const coverageKm = 50; // observations >50 km from any prediction are out-of-scope
  const leftMapRef = useRef<HTMLDivElement | null>(null);
  const rightMapRef = useRef<HTMLDivElement | null>(null);
  const overlayMapRef = useRef<HTMLDivElement | null>(null);

  // Available base_dates with closed validation windows (have hit/miss data,
  // not 100% future). Show newest first.
  const availableDates = useMemo(() => {
    const dates = new Map<string, { hit: number; miss: number; future: number }>();
    for (const f of allFeatures) {
      const bd = f.properties.base_date;
      const st = f.properties.validation_status;
      if (!bd) continue;
      let entry = dates.get(bd);
      if (!entry) { entry = { hit: 0, miss: 0, future: 0 }; dates.set(bd, entry); }
      if (st === "hit") entry.hit += 1;
      else if (st === "miss") entry.miss += 1;
      else if (st === "future") entry.future += 1;
    }
    return Array.from(dates.entries())
      .map(([d, c]) => ({ date: d, ...c, decided: c.hit + c.miss }))
      .sort((a, b) => b.date.localeCompare(a.date));
  }, [allFeatures]);

  // Resolve the actual base date to display
  const resolvedDate = useMemo(() => {
    if (baseDate !== "__latest__") return baseDate;
    const firstWithData = availableDates.find((d) => d.decided > 0);
    return firstWithData?.date ?? availableDates[0]?.date ?? null;
  }, [baseDate, availableDates]);

  // Filter to this snapshot's predicted features, then to the audit-tier
  // filter so the precision/recall numbers reflect the operationally-
  // relevant tier (default HIGH+MEDIUM). LOW being included always tanks
  // precision because LOW is the model's bottom-60% probability bucket.
  const snapshot = useMemo(() => {
    if (!resolvedDate) return [];
    const base = allFeatures.filter((f) => f.properties.base_date === resolvedDate);
    if (tierFilter === "ALL") return base;
    if (tierFilter === "HIGH") {
      return base.filter((f) => f.properties.urgency_level === "HIGH");
    }
    // HIGH_MED
    return base.filter((f) =>
      f.properties.urgency_level === "HIGH" ||
      f.properties.urgency_level === "MEDIUM"
    );
  }, [allFeatures, resolvedDate, tierFilter]);

  // Real FIRMS observed fires inside the wider audit window. We widen by
  // dayWindow so radius-based haversine matching has all the candidates.
  const observedInWindow = useMemo(() => {
    if (!resolvedDate) return observedFeatures;
    const base = new Date(resolvedDate);
    const lo = new Date(base.getTime() - (1 + dayWindow) * 86400_000);
    const hi = new Date(base.getTime() + (8 + dayWindow) * 86400_000);
    return observedFeatures.filter((f) => {
      const d = f.properties.date;
      if (!d) return false;
      const dt = new Date(d);
      return dt >= lo && dt <= hi;
    });
  }, [resolvedDate, observedFeatures, dayWindow]);

  // Re-match every prediction in the snapshot against observed FIRMS via
  // haversine distance + day window. This overrides the grid-cell-exact
  // validation_status from risk_map.py so the operator can dial radius /
  // day window live.
  //
  // Fallback: when observed data doesn't temporally cover this snapshot's
  // prediction dates, use the stored validation_status that risk_map.py
  // computed against the full historical FIRMS parquet.
  //
  // The simple `observedInWindow.length === 0` check is insufficient — the
  // 9-day window can include today's observations even for a week-old
  // snapshot (e.g. May 26 obs falls inside the May 18/22/23 windows).
  // Those obs are 2-8 days from the predicted fire dates, so live matching
  // produces 0 hits → all predictions show as false alarms. The coverage
  // ratio check below detects this and switches to stored data instead.
  type Match = { feature: FireFeature; status: "hit" | "alarm" | "future" };
  const matched = useMemo<{ rows: Match[]; missedObserved: FireFeature[]; isStoredFallback: boolean }>(() => {
    if (!resolvedDate) return { rows: [], missedObserved: [], isStoredFallback: false };

    // Latest observed timestamp across ALL obs (used globally for "future" classification).
    const latestObservedTs = observedFeatures.reduce((acc, f) => {
      const d = f.properties.date;
      if (!d) return acc;
      const ts = new Date(d).getTime();
      return ts > acc ? ts : acc;
    }, 0);

    // Timestamps of obs features inside this snapshot's window.
    const obsTs = observedInWindow
      .map(o => { const d = o.properties.date; return d ? new Date(d).getTime() : NaN; })
      .filter(v => Number.isFinite(v));

    // Non-future predictions: target date is within the observation horizon.
    const nonFutureSnap = snapshot.filter(pred => {
      const targetStr = pred.properties.predicted_fire_date;
      if (!targetStr) return false;
      const ts = new Date(targetStr).getTime();
      return Number.isFinite(ts) && ts < latestObservedTs + 86400_000;
    });

    // Coverage ratio: fraction of non-future predictions with at least one
    // obs within dayWindow days. < 0.5 means the obs data doesn't cover
    // the bulk of prediction dates — stored fallback is more accurate.
    const coveredCount = nonFutureSnap.filter(pred => {
      const t = new Date(pred.properties.predicted_fire_date!).getTime();
      return obsTs.some(ot => Math.abs(ot - t) / 86400_000 <= dayWindow);
    }).length;
    const coverageRatio = nonFutureSnap.length > 0 ? coveredCount / nonFutureSnap.length : 0;
    const isStoredFallback = snapshot.length > 0 && (obsTs.length === 0 || coverageRatio < 0.5);

    // Stored-fallback path: use validation_status computed by risk_map.py
    // against the full FIRMS parquet (more accurate for historical snapshots).
    if (isStoredFallback) {
      const rows: Match[] = snapshot.map((pred) => {
        const st = pred.properties.validation_status;
        if (st === "hit")    return { feature: pred, status: "hit" as const };
        if (st === "future") return { feature: pred, status: "future" as const };
        // stored "miss" = prediction fired but no fire detected = false alarm
        return { feature: pred, status: "alarm" as const };
      });
      return { rows, missedObserved: [], isStoredFallback: true };
    }

    const rows: Match[] = [];
    const matchedObs = new Set<FireFeature>();

    for (const pred of snapshot) {
      const [lon, lat] = pred.geometry.coordinates;
      const targetStr = pred.properties.predicted_fire_date;
      const target = targetStr ? new Date(targetStr).getTime() : NaN;
      // "future" = predicted target is at or beyond the latest observation day
      // (>= not > so "next day after latest obs" is also future, not alarm)
      if (!Number.isFinite(target) || target >= latestObservedTs + 86400_000) {
        rows.push({ feature: pred, status: "future" });
        continue;
      }
      let hit = false;
      for (const obs of observedInWindow) {
        const [olon, olat] = obs.geometry.coordinates;
        const od = obs.properties.date;
        if (!od) continue;
        const ot = new Date(od).getTime();
        const dayDelta = Math.abs(ot - target) / 86400_000;
        if (dayDelta > dayWindow) continue;
        if (haversineKm(lat, lon, olat, olon) <= radiusKm) {
          hit = true;
          matchedObs.add(obs);
          break;
        }
      }
      rows.push({ feature: pred, status: hit ? "hit" : "alarm" });
    }

    // Missed = observations no prediction claimed. Optionally clip to the
    // predicted area: drop observations that are >coverageKm km from EVERY
    // prediction this snapshot, since the model never even tried to forecast
    // those regions (cross-border fires, novel-area burns, etc.). This keeps
    // recall honest about coverage WITHIN the model's attention zone.
    let missedObserved = observedInWindow.filter((o) => !matchedObs.has(o));
    if (clipToCoverage && snapshot.length > 0) {
      const predCoords = snapshot.map((p) => p.geometry.coordinates);
      missedObserved = missedObserved.filter((o) => {
        const [olon, olat] = o.geometry.coordinates;
        for (const [plon, plat] of predCoords) {
          if (Math.abs(plat - olat) > coverageKm / 80) continue;
          if (haversineKm(olat, olon, plat, plon) <= coverageKm) return true;
        }
        return false;
      });
    }
    return { rows, missedObserved, isStoredFallback: false };
  }, [snapshot, observedInWindow, observedFeatures, resolvedDate, radiusKm, dayWindow, clipToCoverage]);

  const stats: Stats = useMemo(() => {
    let hits = 0, falseAlarms = 0, future = 0;
    for (const m of matched.rows) {
      if (m.status === "hit") hits += 1;
      else if (m.status === "alarm") falseAlarms += 1;
      else future += 1;
    }
    const misses = matched.missedObserved.length;
    const decided = hits + falseAlarms;
    const recallDenom = hits + misses;
    const precision = decided ? hits / decided : 0;
    // recall is only meaningful when we have live obs data (not stored fallback)
    const recall = (!matched.isStoredFallback && recallDenom) ? hits / recallDenom : 0;
    const f1 = (precision + recall) ? (2 * precision * recall) / (precision + recall) : 0;
    return { hits, falseAlarms, misses, future, precision, recall, f1 };
  }, [matched]);

  // ── Build Leaflet maps ──
  useEffect(() => {
    const ctxMaps: L.Map[] = [];
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    const tileUrl = isLight
      ? "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

    function attach(container: HTMLDivElement | null, mode: "predicted" | "actual" | "overlay") {
      if (!container) return null;
      const map = L.map(container, { center: [15.0, 101.0], zoom: 6 });
      L.tileLayer(tileUrl, { attribution: "© OpenStreetMap" }).addTo(map);
      ctxMaps.push(map);

      const statusByFeature = new Map(matched.rows.map((m) => [m.feature, m.status]));
      const missedSet = new Set(matched.missedObserved);

      if (mode === "predicted" || mode === "overlay") {
        for (const f of snapshot) {
          const [lon, lat] = f.geometry.coordinates;
          const u = f.properties.urgency_level;
          const st = statusByFeature.get(f) ?? "future";
          let color = PALETTE.future, label = t("compare.label.future");
          if (st === "hit") { color = PALETTE.hit; label = t("compare.label.hit"); }
          else if (st === "alarm") { color = PALETTE.alarm; label = t("compare.label.alarm"); }
          const radius = u === "CRITICAL" ? 7 : u === "HIGH" ? 6 : u === "MEDIUM" ? 5 : 4;
          L.circleMarker([lat, lon], {
            radius, fillColor: color, color: "#fff", weight: 1, fillOpacity: 0.85,
          }).bindPopup(
            `<div style="font-size:12px"><b>${label}</b><br>` +
            `${f.properties.province ?? "—"}<br>` +
            `urgency: ${u}, prob: ${typeof f.properties.raw_prediction === "number"
              ? ((readProbability(f.properties) ?? 0) * 100).toFixed(0) + "%"
              : "—"}</div>`
          ).addTo(map);
        }
      }
      if (mode === "actual" || mode === "overlay") {
        for (const f of observedInWindow) {
          const [lon, lat] = f.geometry.coordinates;
          const isMissed = mode === "overlay" && missedSet.has(f);
          if (mode === "actual") {
            L.circleMarker([lat, lon], {
              radius: 4, fillColor: "#ef4444", color: "#fff",
              weight: 0.5, fillOpacity: 0.9,
            }).bindPopup(`<small>🛰 FIRMS · ${f.properties.date ?? "—"}</small>`).addTo(map);
          } else if (isMissed) {
            L.circleMarker([lat, lon], {
              radius: 6, fillColor: PALETTE.miss, color: "#fff",
              weight: 1.5, fillOpacity: 0.95,
            }).bindPopup(`<div style="font-size:12px"><b>${t("compare.label.miss")}</b><br>${f.properties.date ?? "—"}</div>`).addTo(map);
          }
        }
      }
      return map;
    }

    if (view === "overlay") {
      attach(overlayMapRef.current, "overlay");
    } else {
      attach(leftMapRef.current, "predicted");
      attach(rightMapRef.current, "actual");
    }

    return () => {
      ctxMaps.forEach((m) => m.remove());
    };
  }, [view, snapshot, observedInWindow, matched, t]);

  const fmt = (v: number, d = 1) => (v * 100).toFixed(d) + "%";

  return (
    <div className="notify-page">
      <header className="notify-page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1>{t("page.compare.title", "🆚 Predicted vs Actual")}</h1>
          <p className="notify-page-subtitle">
            {t("page.compare.subtitle", "Model honesty audit — overlay past predictions onto real FIRMS observations")}
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          <select
            value={baseDate}
            onChange={(e) => setBaseDate(e.target.value)}
            style={{ padding: "7px 10px", background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", fontSize: 12, minWidth: 200 }}
          >
            <option value="__latest__">{t("compare.basedate.latest")}</option>
            {availableDates.map((d) => (
              <option key={d.date} value={d.date}>
                {d.date} — {d.hit} ✓ / {d.miss} ⚠
                {d.future > 0 ? ` (… ${d.future})` : ""}
              </option>
            ))}
          </select>
          <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 5, overflow: "hidden" }}>
            <button type="button" onClick={() => setView("overlay")} className={`action-btn ${view === "overlay" ? "primary" : ""}`} style={{ borderRadius: 0, fontSize: 11 }}>
              {t("compare.view.overlay")}
            </button>
            <button type="button" onClick={() => setView("split")} className={`action-btn ${view === "split" ? "primary" : ""}`} style={{ borderRadius: 0, fontSize: 11 }}>
              {t("compare.view.split")}
            </button>
          </div>
        </div>
      </header>

      {/* Stats strip */}
      <div className="alert-counts" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.hit }}>
          <div className="alert-count-label" style={{ color: PALETTE.hit }}>{t("compare.label.hit")}</div>
          <div className="alert-count-value">{stats.hits}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.alarm }}>
          <div className="alert-count-label" style={{ color: PALETTE.alarm }}>{t("compare.label.alarm")}</div>
          <div className="alert-count-value">{stats.falseAlarms}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.miss }}>
          <div className="alert-count-label" style={{ color: PALETTE.miss }}>{t("compare.label.miss")}</div>
          <div className="alert-count-value">{stats.misses}</div>
        </div>
        <div className="alert-count-card" style={{ borderLeftColor: PALETTE.future }}>
          <div className="alert-count-label" style={{ color: PALETTE.future }}>{t("compare.label.future")}</div>
          <div className="alert-count-value">{stats.future}</div>
        </div>
      </div>

      {/* Methodology callout — clarify why these numbers differ from AccuracyHero */}
      <div style={{
        padding: "10px 14px", marginBottom: 12,
        background: "rgba(59, 130, 246, 0.08)",
        border: "1px solid rgba(59, 130, 246, 0.25)",
        borderRadius: 6, fontSize: 12, color: "var(--text-2)",
        lineHeight: 1.55,
      }}>
        {t("compare.methodology")}
      </div>

      {/* Stored-fallback notice: shown when no live FIRMS data covers this snapshot's window */}
      {matched.isStoredFallback && (
        <div style={{
          padding: "10px 14px", marginBottom: 12,
          background: "rgba(234, 179, 8, 0.08)",
          border: "1px solid rgba(234, 179, 8, 0.30)",
          borderRadius: 6, fontSize: 12, color: "var(--text-2)",
          lineHeight: 1.55, display: "flex", gap: 8, alignItems: "flex-start",
        }}>
          <span style={{ fontSize: 15, flexShrink: 0 }}>⚠️</span>
          <span>
            <b style={{ color: "var(--text)" }}>Using stored validation data</b> — live FIRMS observations don't cover this snapshot's date range.
            Stats come from <code style={{ fontSize: 11, background: "var(--surface-2)", padding: "1px 5px", borderRadius: 3 }}>validation_status</code> written by
            {" "}<code style={{ fontSize: 11, background: "var(--surface-2)", padding: "1px 5px", borderRadius: 3 }}>risk_map.py</code> against the full FIRMS parquet.
            Radius / day-window sliders have no effect here. Recall is not available (no observed coordinates to count misses).
          </span>
        </div>
      )}

      {/* Matching controls — radius + day window (greyed out in stored-fallback mode) */}
      <div style={{
        padding: "10px 14px", marginBottom: 12,
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        opacity: matched.isStoredFallback ? 0.45 : 1,
        pointerEvents: matched.isStoredFallback ? "none" : undefined,
      }}>
        <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: "var(--text-3)", letterSpacing: "0.06em" }}>
            {t("compare.matching.title")}
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, flex: "1 1 280px" }}>
            <span style={{ minWidth: 150 }}>
              {t("compare.matching.radius")} <b style={{ color: "var(--accent)" }}>{radiusKm} km</b>
            </span>
            <input
              type="range" min={5} max={50} step={5}
              value={radiusKm}
              onChange={(e) => setRadiusKm(Number(e.target.value))}
              style={{ flex: 1, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: 10, color: "var(--text-3)", minWidth: 130 }}>
              {t("compare.matching.radius.hint")}
            </span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, flex: "1 1 240px" }}>
            <span style={{ minWidth: 130 }}>
              {t("compare.matching.window")} <b style={{ color: "var(--accent)" }}>{dayWindow}</b>
            </span>
            <input
              type="range" min={0} max={3} step={1}
              value={dayWindow}
              onChange={(e) => setDayWindow(Number(e.target.value))}
              style={{ flex: 1, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: 10, color: "var(--text-3)", minWidth: 110 }}>
              {t("compare.matching.window.hint")}
            </span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, flex: "0 0 auto" }}>
            <span style={{ minWidth: 90, fontWeight: 600 }}>{t("compare.tier.label")}</span>
            <select
              value={tierFilter}
              onChange={(e) => setTierFilter(e.target.value as "HIGH_MED" | "HIGH" | "ALL")}
              style={{ padding: "5px 8px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", fontSize: 12 }}
            >
              <option value="HIGH_MED">{t("compare.tier.high_med")}</option>
              <option value="HIGH">{t("compare.tier.high")}</option>
              <option value="ALL">{t("compare.tier.all")}</option>
            </select>
          </label>
          <label
            style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, flex: "0 0 auto", cursor: "pointer" }}
            title={t("compare.clip.tooltip", `Drop FIRMS observations that are >${coverageKm} km from every prediction this snapshot (model never tried to cover those regions, so they shouldn't count as misses).`)}
          >
            <input
              type="checkbox"
              checked={clipToCoverage}
              onChange={(e) => setClipToCoverage(e.target.checked)}
              style={{ accentColor: "var(--accent)" }}
            />
            <span>{t("compare.clip.label", `Clip to predicted area (≤${coverageKm} km)`)}</span>
          </label>
        </div>
      </div>

      {/* Precision/Recall/F1 */}
      <section className="report-section" style={{ marginBottom: 14 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>{t("compare.precision")}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: stats.hits + stats.falseAlarms > 0 ? PALETTE.hit : "var(--text-3)", marginTop: 4 }}>
              {stats.hits + stats.falseAlarms > 0 ? fmt(stats.precision) : "—"}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {stats.hits + stats.falseAlarms > 0
                ? fmtTr(t("compare.precision.exp"), { pct: fmt(stats.precision, 0) })
                : "No decided predictions yet"}
            </div>
          </div>
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6, opacity: matched.isStoredFallback ? 0.45 : 1 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>
              {t("compare.recall")}
              {matched.isStoredFallback && <span style={{ marginLeft: 6, fontWeight: 400, textTransform: "none", fontSize: 9 }}>(N/A — stored)</span>}
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, color: matched.isStoredFallback ? "var(--text-3)" : PALETTE.hit, marginTop: 4 }}>
              {matched.isStoredFallback ? "—" : fmt(stats.recall)}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {matched.isStoredFallback
                ? "Need live FIRMS obs to count missed fires"
                : fmtTr(t("compare.recall.exp"), { pct: fmt(stats.recall, 0) })}
            </div>
          </div>
          <div style={{ padding: "12px 14px", background: "var(--surface-2)", borderRadius: 6, opacity: matched.isStoredFallback ? 0.45 : 1 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase" }}>{t("compare.f1")}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: matched.isStoredFallback ? "var(--text-3)" : "var(--accent)", marginTop: 4 }}>
              {matched.isStoredFallback ? "—" : fmt(stats.f1)}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>
              {matched.isStoredFallback ? "Requires recall" : t("compare.f1.exp")}
            </div>
          </div>
        </div>
      </section>

      {/* Maps */}
      {view === "overlay" ? (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 6, fontSize: 11, flexWrap: "wrap" }}>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.hit, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.hit")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.alarm, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.alarm")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.miss, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.miss")}</span>
            <span><span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 6, background: PALETTE.future, verticalAlign: "middle", marginRight: 4 }}/>{t("compare.legend.future")}</span>
          </div>
          <div ref={overlayMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{t("compare.map.predicted")}</h3>
            <div ref={leftMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
          </div>
          <div>
            <h3 style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{t("compare.map.actual")}</h3>
            <div ref={rightMapRef} style={{ height: 520, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }} />
          </div>
        </div>
      )}

      <p className="report-section-hint" style={{ marginTop: 12, fontSize: 11 }}>
        {t("compare.audit.line")}
      </p>
      {stats.hits + stats.falseAlarms === 0 && stats.future > 0 && (
        <div style={{
          marginTop: 12, padding: "10px 14px",
          background: "rgba(234, 179, 8, 0.10)",
          border: "1px solid rgba(234, 179, 8, 0.35)",
          borderRadius: 6, fontSize: 12, color: "var(--text-2)",
          lineHeight: 1.55,
        }}>
          {fmtTr(t("compare.pending.warn"), { date: resolvedDate ?? "" })}
        </div>
      )}
      {stats.hits + stats.falseAlarms + stats.future === 0 && (
        <div style={{
          marginTop: 12, padding: "10px 14px",
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: 6, fontSize: 12, color: "var(--text-3)",
          lineHeight: 1.55,
        }}>
          {fmtTr(t("compare.no_status"), { cmd: "cd src && python risk_map.py" })}
        </div>
      )}
    </div>
  );
}
