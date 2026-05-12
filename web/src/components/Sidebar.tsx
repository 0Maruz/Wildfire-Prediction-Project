import type {
  DaySelection,
  DisplayOptions,
  FireFeature,
  GeoJsonMetadata,
  LiveFireMeta,
  UrgencyLevel,
  UrgencyThresholds,
  ValidationMetrics,
} from "../types";
import { computeFreshness, dateAdd } from "../utils/dates";
import AccuracyHero from "./AccuracyHero";

interface SidebarProps {
  // Header / base date
  activeBaseDate: string;
  allBaseDates: string[];
  selectedBaseDate: string; // "latest" or an ISO date
  onBaseDateChange: (v: string) => void;

  // Province
  provinces: string[];
  selectedProvince: string;
  onProvinceChange: (v: string) => void;

  // Day selector
  selectedDay: DaySelection;
  onDayChange: (d: DaySelection) => void;
  predicted: FireFeature[]; // for the snapshot — used by timeline + landcover + urgency
  visibleCount: number;
  daySelectorMessage: string;

  // Urgency / metrics / hit-rate
  thresholds: UrgencyThresholds | null;
  metrics: ValidationMetrics | null;
  metadata: GeoJsonMetadata | null;

  // Display options + export
  options: DisplayOptions;
  onOptionsChange: (o: Partial<DisplayOptions>) => void;
  onExportCsv: () => void;

  // Live-fire status (GISTDA)
  liveFireMeta: LiveFireMeta;

  // Info modal
  onShowInfoModal: () => void;
}

export default function Sidebar(p: SidebarProps) {
  const fresh = computeFreshness(p.activeBaseDate);
  const isQuantileFallback = !!p.thresholds && Number(p.thresholds.CRITICAL) > 0;

  // Day counts for timeline + day selector hide-empty logic.
  const dayCounts: Record<number, number> = {};
  for (let i = 0; i <= 7; i++) dayCounts[i] = 0;
  for (const f of p.predicted) {
    const d = f.properties.days_until_fire;
    if (d != null && d >= 0 && d <= 7) dayCounts[d]++;
  }

  const urgencyCounts: Record<UrgencyLevel, number> = {
    CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, NONE: 0,
  };
  let forest = 0, mixed = 0, open = 0;
  for (const f of p.predicted) {
    const u = f.properties.urgency_level;
    if (u && u in urgencyCounts) urgencyCounts[u]++;
    const c = f.properties.tree_cover_pct_2000;
    if (c != null) {
      if (c >= 50) forest++;
      else if (c >= 10) mixed++;
      else open++;
    }
  }

  const fmtRange = (v: number | undefined) =>
    v == null ? "≤ —" : `≤ ${Number(v).toFixed(1)} d`;

  return (
    <div id="sidebar">
      <div className="header">
        <h1>🔥 Fire Date Predictor</h1>
        <p className="subtitle">NASA FIRMS · real-data wildfire forecasting</p>
      </div>

      {/* Base Date */}
      <div className="info-card">
        <div className="info-label">
          Base Date
          {fresh && (
            <span className={`freshness-badge ${fresh.cls}`}>{fresh.label}</span>
          )}
        </div>
        <div className="info-value">{p.activeBaseDate || "Loading..."}</div>
        {fresh && fresh.msg && <div className="info-sub">{fresh.msg}</div>}

        <div className="date-picker-row">
          <label className="date-picker-label" htmlFor="baseDatePicker">
            View predictions from:
          </label>
          <select
            id="baseDatePicker"
            className="date-picker"
            value={p.selectedBaseDate}
            onChange={(e) => p.onBaseDateChange(e.target.value)}
          >
            <option value="latest">
              Latest ({p.allBaseDates[p.allBaseDates.length - 1] ?? "—"})
            </option>
            {[...p.allBaseDates].reverse().map((d) => {
              const today = new Date();
              today.setHours(0, 0, 0, 0);
              const dObj = new Date(d);
              dObj.setHours(0, 0, 0, 0);
              const lag = Math.round((today.getTime() - dObj.getTime()) / 86400000);
              const isLatest =
                d === p.allBaseDates[p.allBaseDates.length - 1];
              const suffix = isLatest ? "newest" : `${lag} d behind today`;
              return (
                <option key={d} value={d}>
                  {d} ({suffix})
                </option>
              );
            })}
          </select>
        </div>

        <div className="date-picker-row">
          <label className="date-picker-label" htmlFor="provinceFilter">
            Filter by province:
          </label>
          <select
            id="provinceFilter"
            className="date-picker"
            value={p.selectedProvince}
            onChange={(e) => p.onProvinceChange(e.target.value)}
          >
            <option value="all">All provinces ({p.provinces.length})</option>
            {p.provinces.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Hero Accuracy — most-trusted number near the top, with details modal */}
      <AccuracyHero metrics={p.metrics} onShowDetails={p.onShowInfoModal} />

      {/* Day Selector */}
      <div className="section">
        <h3>📆 Day Selector</h3>
        <div className="day-selector">
          <button
            className={`day-btn ${p.selectedDay === "all" ? "active" : ""}`}
            onClick={() => p.onDayChange("all")}
          >
            All
          </button>
          {([0, 1, 2, 3, 4, 5, 6, 7] as const).map((d) => {
            const count = dayCounts[d];
            const visible = count > 0;
            const label = d === 0 ? "Today" : `+${d}`;
            const key = String(d) as DaySelection;
            return (
              <button
                key={d}
                className={`day-btn ${p.selectedDay === key ? "active" : ""}`}
                onClick={() => p.onDayChange(key)}
                style={{ display: visible ? "" : "none" }}
              >
                {label}
              </button>
            );
          })}
        </div>
        <div className="day-selector-info">{p.daySelectorMessage}</div>
      </div>

      {/* Timeline */}
      <div className="section">
        <h3>📅 7-Day Fire Timeline</h3>
        <div id="timeline">
          {(() => {
            const items: JSX.Element[] = [];
            for (let i = 0; i <= 7; i++) {
              const count = dayCounts[i];
              if (count === 0) continue;
              const dateStr = dateAdd(p.activeBaseDate, i);
              const label =
                i === 0 ? "Today" : i === 1 ? "Tomorrow" : `+${i} days`;
              items.push(
                <div
                  key={i}
                  className="timeline-item has-fires"
                  onClick={() => p.onDayChange(String(i) as DaySelection)}
                >
                  <div className="timeline-day">{label}</div>
                  <div className="timeline-date">{dateStr}</div>
                  <div className="timeline-count">
                    {count} fire{count !== 1 ? "s" : ""}
                  </div>
                </div>
              );
            }
            if (items.length === 0)
              items.push(
                <div key="none" className="timeline-item">
                  <div className="timeline-day" style={{ opacity: 0.6 }}>
                    No predictions
                  </div>
                  <div className="timeline-date">over the next 7 days</div>
                  <div className="timeline-count" style={{ opacity: 0.6 }}>—</div>
                </div>
              );
            return items;
          })()}
        </div>
      </div>

      {/* Urgency */}
      <div className="section">
        <h3>⚡ Urgency Summary</h3>
        <div className="urgency-cards">
          {(["critical", "high", "medium", "low"] as const).map((tier) => {
            const key = tier.toUpperCase() as UrgencyLevel;
            const upper = key as keyof UrgencyThresholds;
            const tip = {
              critical: "Fire predicted within 24 hours of the base date. Triage first.",
              high: "Fire predicted within ~2 days. Watch closely.",
              medium: "Fire predicted within ~4 days. Plan ahead.",
              low: "Fire predicted within the full 7-day horizon. Background risk.",
            }[tier];
            return (
              <div key={tier} className={`urgency-card ${tier}`} title={tip}>
                <div className="urgency-label">{tier.toUpperCase()}</div>
                <div className="urgency-count">{urgencyCounts[key]}</div>
                <div className="urgency-desc">
                  {fmtRange(p.thresholds?.[upper])}
                </div>
              </div>
            );
          })}
        </div>
        <div className="threshold-note">
          {p.thresholds == null
            ? "No calibrated thresholds in metadata — falling back to legacy 0/2/4/7 cutoffs."
            : isQuantileFallback ? (
              <>
                <strong>Quantile mode:</strong> model output too narrow for fixed
                cutoffs, so tiers are 25/50/75 percentile ranks of this snapshot's
                raw predictions — relative ordering, not absolute risk.
              </>
            ) : (
              "Fixed-domain cutoffs (CRITICAL=0d, HIGH≤2d, MEDIUM≤4d, LOW≤7d)."
            )}
        </div>
      </div>

      {/* Land cover + export */}
      <div className="section">
        <h3>🌳 Land Cover (Hansen GFC)</h3>
        <div className="landcover-grid">
          <div className="landcover-row">
            <span className="landcover-bar forest" />
            <span className="landcover-label">Forest (≥50%)</span>
            <span className="landcover-count">{forest || "—"}</span>
          </div>
          <div className="landcover-row">
            <span className="landcover-bar mixed" />
            <span className="landcover-label">Mixed (10–50%)</span>
            <span className="landcover-count">{mixed || "—"}</span>
          </div>
          <div className="landcover-row">
            <span className="landcover-bar open" />
            <span className="landcover-label">Open / Grassland (&lt;10%)</span>
            <span className="landcover-count">{open || "—"}</span>
          </div>
        </div>
        <div className="landcover-note">
          Tree cover bucket per cell from Hansen GFC v1.11. Open cells correlate
          with agricultural-burn signals; forest cells suggest wildfire risk.
        </div>
        <button className="export-btn" onClick={p.onExportCsv}>
          ⬇ Export visible cells (CSV)
        </button>
      </div>

      {/* Hit-rate */}
      <HitRate metadata={p.metadata} activeBaseDate={p.activeBaseDate} />

      {/* Display Options — simplified: only Observed FIRMS + Live GISTDA toggles.
          Predictions + cell pins are always on (matches frontend/index.html v=8). */}
      <DisplaySection
        options={p.options}
        liveFireMeta={p.liveFireMeta}
        onChange={p.onOptionsChange}
      />

      <div className="info-footer">
        <small>
          Data sources (real only):<br />
          • NASA FIRMS VIIRS NRT<br />
          • Open-Meteo ERA5 (if enabled)<br />
          No synthetic data is used.
        </small>
      </div>
    </div>
  );
}

// ───────────────────── HitRate sub-component ─────────────────────

function HitRate({
  metadata,
  activeBaseDate,
}: {
  metadata: GeoJsonMetadata | null;
  activeBaseDate: string;
}) {
  const perSnap = metadata?.validation_summary?.per_snapshot ?? {};
  const bucket = perSnap[activeBaseDate];

  let value = "—";
  let suffix = "no past predictions yet";
  let detail = "For this snapshot · ±1 day window";

  if (!bucket) {
    suffix = "no validation data";
    detail = "Older snapshots may have been pruned by retention policy.";
  } else {
    const hits = bucket.hits ?? 0;
    const misses = bucket.misses ?? 0;
    const future = bucket.future ?? 0;
    const validatable = hits + misses;
    if (validatable === 0) {
      suffix = `${future} pending`;
      detail = `All predicted dates are in the future; check back after ${
        future > 0 ? "FIRMS catches up" : "the next refresh"
      }.`;
    } else {
      const pct = (hits / validatable) * 100;
      value = `${pct.toFixed(0)}%`;
      suffix = `${hits} / ${validatable} hits`;
      detail =
        `For ${activeBaseDate} · ±1 day window` +
        (future > 0 ? ` · ${future} still pending` : "");
    }
  }

  return (
    <div className="section">
      <h3>🎯 Hit Rate vs FIRMS</h3>
      <div
        className="hitrate-card"
        title="Out of the predicted cells whose target date has passed, what fraction actually burned within ±1 day of when the model said. Live measurement of dashboard skill on real ground truth."
      >
        <div className="hitrate-headline">
          <span id="hitrateValue">{value}</span>
          <span className="hitrate-suffix">{suffix}</span>
        </div>
        <div className="hitrate-detail">{detail}</div>
      </div>
      <div className="metrics-note">
        Each historical prediction is checked against the actual NASA FIRMS
        detections within ±1 day of its predicted fire date. "Hit" = the cell
        did burn near the predicted time.
      </div>
    </div>
  );
}

// ───────────────────── Display options sub-component ─────────────────────

function DisplaySection({
  options,
  liveFireMeta,
  onChange,
}: {
  options: DisplayOptions;
  liveFireMeta: LiveFireMeta;
  onChange: (o: Partial<DisplayOptions>) => void;
}) {
  // Status line below the live-fire toggle: mirrors the legacy
  // #liveFireStatus div, with a class-driven colour for idle / loading /
  // ok / error states.
  let statusClass = "live-fire-status";
  let statusText = "";
  if (liveFireMeta.status === "loading") {
    statusClass += " loading";
    statusText = "⟳ Fetching live hotspots…";
  } else if (liveFireMeta.status === "ok") {
    statusClass += " ok";
    const t = liveFireMeta.lastFetch ? liveFireMeta.lastFetch.toLocaleTimeString() : "—";
    statusText = `${liveFireMeta.count} hotspot${liveFireMeta.count !== 1 ? "s" : ""} · refreshed ${t}`;
  } else if (liveFireMeta.status === "error") {
    statusClass += " error";
    statusText = `Could not load: ${liveFireMeta.error ?? "network error"}`;
  }

  return (
    <div className="section">
      <h3>🎨 Display Options</h3>

      <div className="option-group">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={options.showObserved}
            onChange={(e) => onChange({ showObserved: e.target.checked })}
          />
          <span>Show Observed Fires (FIRMS)</span>
        </label>
      </div>

      <div className="option-group">
        <label className="toggle-label" style={{ borderColor: "rgba(6,182,212,0.25)" }}>
          <input
            type="checkbox"
            checked={options.showLiveFires}
            onChange={(e) => onChange({ showLiveFires: e.target.checked })}
          />
          <span style={{ color: "#06b6d4" }}>Live Fires — GISTDA VIIRS (now)</span>
        </label>
        {statusText && <div className={statusClass}>{statusText}</div>}
      </div>
    </div>
  );
}
