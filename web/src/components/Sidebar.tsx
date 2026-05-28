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
import { fmtTr, useLang } from "../utils/i18n";
import AccuracyHero from "./AccuracyHero";
import ActionToolbar from "./ActionToolbar";
import DataSourcesPanel from "./DataSourcesPanel";
import RetrospectivePanel from "./RetrospectivePanel";

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

  // ALL predicted features across base_dates — for retrospective panel
  predictedAll: FireFeature[];

  // Display options + export
  options: DisplayOptions;
  onOptionsChange: (o: Partial<DisplayOptions>) => void;
  onExportCsv: () => void;

  // Live-fire status (GISTDA)
  liveFireMeta: LiveFireMeta;

  // Info modal
  onShowInfoModal: () => void;
  onShowAlertSettings: () => void;
}

export default function Sidebar(p: SidebarProps) {
  const { lang, t } = useLang();
  const fresh = computeFreshness(p.activeBaseDate);
  const isQuantileFallback = !!p.thresholds && Number(p.thresholds.CRITICAL) > 0;
  const dateLocale = lang === "th" ? "th-TH" : "en-GB";
  // Day-offset → human label. Uses the i18n dictionary for the first three
  // offsets and "+N days" for the rest, with locale-appropriate date format.
  const dayLabel = (d: number): string => {
    if (d === 0) return t("sidebar.daypicker.today");
    if (d === 1) return t("sidebar.daypicker.tomorrow");
    if (d === 2) return t("sidebar.daypicker.dayafter");
    return fmtTr(t("sidebar.daypicker.inNdays"), { n: d });
  };
  const fmtShortDate = (dateStr: string): string => {
    try {
      return new Date(dateStr).toLocaleDateString(dateLocale, { day: "numeric", month: "short" });
    } catch { return dateStr; }
  };

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

      {/* Layer toggles — top of sidebar for quick access */}
      <LayerPills
        options={p.options}
        liveFireMeta={p.liveFireMeta}
        onChange={p.onOptionsChange}
      />

      {/* Quick stats — at-a-glance Critical / High / Total */}
      <QuickStats urgencyCounts={urgencyCounts} visibleCount={p.visibleCount} />

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

      {/* Action toolbar — production actions: export, share, threshold, details */}
      <ActionToolbar
        predicted={p.predicted}
        metrics={p.metrics}
        onShowDetails={p.onShowInfoModal}
        onShowAlertSettings={p.onShowAlertSettings}
        baseDate={p.activeBaseDate}
      />

      {/* Hero Accuracy — multi-card grid; "model is good" metrics first */}
      <AccuracyHero metrics={p.metrics} onShowDetails={p.onShowInfoModal} />

      {/* Day Selector */}
      <div className="section">
        <h3>📆 {t("sidebar.daypicker.label")}</h3>
        <div className="day-selector">
          <button
            className={`day-btn ${p.selectedDay === "all" ? "active" : ""}`}
            onClick={() => p.onDayChange("all")}
            title={t("sidebar.daypicker.label")}
          >
            {t("sidebar.daypicker.all")}
          </button>
          {([0, 1, 2, 3, 4, 5, 6, 7] as const).map((d) => {
            const count = dayCounts[d];
            const hasData = count > 0;
            const dateStr = dateAdd(p.activeBaseDate, d);
            const dateLabel = (() => {
              if (d <= 2) return dayLabel(d);
              return fmtShortDate(dateStr);
            })();
            const fullDate = fmtShortDate(dateStr);
            const key = String(d) as DaySelection;
            const ptsSuffix = t("sidebar.daypicker.points_suffix");
            return (
              <button
                key={d}
                className={`day-btn ${p.selectedDay === key ? "active" : ""}`}
                onClick={() => hasData && p.onDayChange(key)}
                disabled={!hasData}
                style={{
                  flexDirection: "column",
                  padding: "6px 4px",
                  gap: 1,
                  opacity: hasData ? 1 : 0.35,
                  cursor: hasData ? "pointer" : "default",
                }}
                title={
                  hasData
                    ? `${fullDate} · ${count} ${ptsSuffix}`
                    : `${fullDate} · no predictions for this day`
                }
              >
                <span style={{ fontSize: 11, fontWeight: 700 }}>{dateLabel}</span>
                <span style={{ fontSize: 9, opacity: 0.7 }}>
                  {hasData ? `${count} ${ptsSuffix}` : "—"}
                </span>
              </button>
            );
          })}
        </div>
        <div className="day-selector-info">
          {p.daySelectorMessage}
          <br/>
          <span style={{ fontSize: 10, color: "var(--text-3)" }}>
            {t("sidebar.range.label")}: {p.activeBaseDate} {t("sidebar.range.to")}: {dateAdd(p.activeBaseDate, 7)}
          </span>
        </div>
      </div>

      {/* Timeline */}
      <div className="section">
        <h3>📅 {t("sidebar.fires7d.title")}</h3>
        <div id="timeline">
          {(() => {
            const items: JSX.Element[] = [];
            const ptsSuffix = t("sidebar.daypicker.points_suffix");
            for (let i = 0; i <= 7; i++) {
              const count = dayCounts[i];
              if (count === 0) continue;
              const dateStr = dateAdd(p.activeBaseDate, i);
              const label = dayLabel(i);
              const shortDate = fmtShortDate(dateStr);
              items.push(
                <div
                  key={i}
                  className="timeline-item has-fires"
                  onClick={() => p.onDayChange(String(i) as DaySelection)}
                  title={`${dateStr} · ${count} ${ptsSuffix}`}
                >
                  <div className="timeline-day">{label}</div>
                  <div className="timeline-date">{shortDate}</div>
                  <div className="timeline-count">
                    {count} {ptsSuffix}
                  </div>
                </div>
              );
            }
            if (items.length === 0)
              items.push(
                <div key="none" className="timeline-item">
                  <div className="timeline-day" style={{ opacity: 0.6 }}>
                    {t("sidebar.fires7d.empty")}
                  </div>
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

      {/* Past predictions retrospective — strongest trust signal */}
      <RetrospectivePanel allFeatures={p.predictedAll} />

      {/* Data sources / provenance */}
      <DataSourcesPanel />

      <div className="info-footer">
        <small style={{ display: "block", lineHeight: 1.6 }}>
          <b>🔥 Thailand Wildfire Imminence Predictor v0.5</b>
          <br />
          Open-source · MIT License
          <br />
          <a
            href="https://github.com/0Maruz/Science-Project-version-3"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)" }}
          >
            🐙 GitHub source
          </a>
          {" · "}
          <a href="docs/MODEL_CARD.md" style={{ color: "var(--accent)" }}>
            📋 Model Card
          </a>
          {" · "}
          <a href="docs/METHODOLOGY.md" style={{ color: "var(--accent)" }}>
            📐 Methodology
          </a>
          <br />
          <span style={{ color: "var(--text-3)", fontSize: 10 }}>
            Data: NASA FIRMS + ECMWF ERA5 · No synthetic values · Calibrated (Platt sigmoid)
            <br />
            ⚠ Research-grade — not for life-safety decisions
          </span>
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
  let detail = "For this snapshot · 15 km / ±1 day · HIGH+MEDIUM tiers";

  // Read methodology config off the validation summary if risk_map persisted
  // it; fall back to the defaults the backend now uses.
  const vs = metadata?.validation_summary as unknown as {
    match_radius_km?: number; match_day_window?: number;
    operational_tiers?: string[];
  } | undefined;
  const radiusKm = vs?.match_radius_km ?? 15;
  const dayWindow = vs?.match_day_window ?? 1;
  const tiers = vs?.operational_tiers?.join("+") ?? "HIGH+MEDIUM";

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
        `${activeBaseDate} · ${tiers} · ${radiusKm} km / ±${dayWindow} day` +
        (future > 0 ? ` · ${future} still pending` : "");
    }
  }

  return (
    <div className="section">
      <h3>🎯 Hit Rate vs FIRMS</h3>
      <div
        className="hitrate-card"
        title={`Of the past predictions (${tiers} tiers only) whose target date has passed, what fraction had a real FIRMS detection within ${radiusKm} km and ±${dayWindow} day. LOW-tier predictions are excluded from the headline since they are background-rank cells, not operational alerts (see Compare page for the full-distribution audit).`}
      >
        <div className="hitrate-headline">
          <span id="hitrateValue">{value}</span>
          <span className="hitrate-suffix">{suffix}</span>
        </div>
        <div className="hitrate-detail">{detail}</div>
      </div>
      <div className="metrics-note">
        Each historical HIGH/MEDIUM prediction is checked against FIRMS
        detections within {radiusKm} km and ±{dayWindow} day of its predicted
        fire date. "Hit" = a real fire was detected near the predicted location
        and time. LOW-tier cells are tracked in the Compare page audit.
      </div>
    </div>
  );
}

// ───────────────────── Layer pill toggles ─────────────────────

function LayerPills({
  options,
  liveFireMeta,
  onChange,
}: {
  options: DisplayOptions;
  liveFireMeta: LiveFireMeta;
  onChange: (o: Partial<DisplayOptions>) => void;
}) {
  const { t } = useLang();
  const liveCount = liveFireMeta.status === "ok" ? liveFireMeta.count : null;
  const liveError = liveFireMeta.status === "error";

  return (
    <div className="layer-pills">
      <button
        type="button"
        className={`layer-pill predicted${options.showPredicted ? " active" : ""}`}
        onClick={() => onChange({ showPredicted: !options.showPredicted })}
        title={t("sidebar.display.predicted")}
      >
        <span className="layer-pill-dot" style={{ background: "#f97316" }} />
        {t("sidebar.display.predicted")}
      </button>

      <button
        type="button"
        className={`layer-pill observed${options.showObserved ? " active" : ""}`}
        onClick={() => onChange({ showObserved: !options.showObserved })}
        title={t("sidebar.display.observed")}
      >
        <span className="layer-pill-dot" style={{ background: "#ff6b35" }} />
        {t("sidebar.display.observed")}
      </button>

      <button
        type="button"
        className={`layer-pill live${options.showLiveFires ? " active" : ""}${liveError ? " error-state" : ""}`}
        onClick={() => onChange({ showLiveFires: !options.showLiveFires })}
        title={
          liveFireMeta.status === "error"
            ? fmtTr(t("sidebar.live.error"), { msg: liveFireMeta.error ?? "network error" })
            : t("sidebar.display.live")
        }
      >
        <span className="layer-pill-dot" style={{ background: "#06b6d4" }} />
        {t("sidebar.display.live")}
        {liveCount !== null && liveCount > 0 && (
          <span className="layer-pill-count">· {liveCount}</span>
        )}
      </button>

    </div>
  );
}

// ───────────────────── Quick stats strip ─────────────────────

function QuickStats({
  urgencyCounts,
  visibleCount,
}: {
  urgencyCounts: Record<UrgencyLevel, number>;
  visibleCount: number;
}) {
  return (
    <div className="quick-stats-strip">
      <div className="quick-stat critical" title="CRITICAL urgency — fire predicted within 24 h">
        <div className="quick-stat-value">{urgencyCounts.CRITICAL}</div>
        <div className="quick-stat-label">Critical</div>
      </div>
      <div className="quick-stat high" title="HIGH urgency — fire predicted within ~2 days">
        <div className="quick-stat-value">{urgencyCounts.HIGH}</div>
        <div className="quick-stat-label">High</div>
      </div>
      <div className="quick-stat medium" title="MEDIUM urgency — fire predicted within ~4 days">
        <div className="quick-stat-value">{urgencyCounts.MEDIUM}</div>
        <div className="quick-stat-label">Medium</div>
      </div>
      <div className="quick-stat total" title="Total visible predictions">
        <div className="quick-stat-value">{visibleCount}</div>
        <div className="quick-stat-label">Visible</div>
      </div>
    </div>
  );
}
