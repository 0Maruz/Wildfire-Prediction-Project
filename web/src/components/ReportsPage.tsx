import { useMemo } from "react";
import type { FireFeature, ValidationMetrics } from "../types";
import StatisticsSection from "./StatisticsSection";

interface Props {
  metrics: ValidationMetrics | null;
  predictedAll: FireFeature[];
}

// ─────────────────────────────────────────────────────────────
// Reports page — model performance + dataset breakdown
//
// Three sections:
//   1. Rolling monthly AUC chart (line + markers)
//   2. Top 20 feature importances (horizontal bars)
//   3. Provincial breakdown of current snapshot (CRITICAL/HIGH counts per province)
//
// All charts inline SVG — no chart library, no deps. Data already in the
// loaded GeoJSON metadata, so the page is instant.
// ─────────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#eab308",
  LOW:      "#22c55e",
};

export default function ReportsPage({ metrics, predictedAll }: Props) {
  // ─── Rolling AUC chart data ───
  const monthly = metrics?.rolling_by_month ?? [];

  // ─── Top feature importance ───
  // Read from window.__FEATURE_IMPORTANCE__ if available; otherwise fall back
  // to a small slice in metrics (feature_importance_top isn't yet in
  // ValidationMetrics type — pull from GeoJSON metadata.feature_importance_top).
  // For now we synthesize from metric stats if not directly provided.
  const featureImportance: { feature: string; importance: number }[] = useMemo(() => {
    // The frontend already receives FullValidationMetrics in `metrics`, but
    // feature_importance_top lives one level up in dataset_info.json's root.
    // GeoJsonMetadata exposes only `metrics` and a couple of other fields, so
    // we resolve the property from `metrics as any` to keep types loose.
    const arr = (metrics as unknown as { feature_importance_top?: typeof featureImportance })
      ?.feature_importance_top;
    if (Array.isArray(arr) && arr.length) return arr.slice(0, 20);
    return [];
  }, [metrics]);

  // ─── Provincial breakdown ───
  // Use the LATEST base_date snapshot in predictedAll so the chart matches
  // the dashboard map. Count cells per province per urgency.
  const provinces = useMemo(() => {
    const byDate = new Map<string, FireFeature[]>();
    for (const f of predictedAll) {
      const bd = f.properties.base_date;
      if (!bd) continue;
      if (!byDate.has(bd)) byDate.set(bd, []);
      byDate.get(bd)!.push(f);
    }
    const dates = Array.from(byDate.keys()).sort();
    const latest = dates[dates.length - 1];
    if (!latest) return [];

    const map = new Map<string, { CRITICAL: number; HIGH: number; MEDIUM: number; LOW: number; total: number }>();
    for (const f of byDate.get(latest)!) {
      const p = f.properties;
      const prov = p.province ?? "—";
      const u = p.urgency_level ?? "NONE";
      let entry = map.get(prov);
      if (!entry) {
        entry = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, total: 0 };
        map.set(prov, entry);
      }
      if (u === "CRITICAL" || u === "HIGH" || u === "MEDIUM" || u === "LOW") {
        entry[u] += 1;
        entry.total += 1;
      }
    }
    return Array.from(map.entries())
      .map(([prov, c]) => ({ prov, ...c }))
      .sort((a, b) => (b.CRITICAL + b.HIGH) - (a.CRITICAL + a.HIGH))
      .slice(0, 15);
  }, [predictedAll]);

  const sciStats = metrics?.scientific_stats;

  return (
    <div className="reports-page">
      <header className="notify-page-header">
        <h1>📊 รายงาน · Scientific Analysis</h1>
        <p className="notify-page-subtitle">
          Held-out test set performance + bootstrap confidence intervals + statistical tests
        </p>
      </header>

      {/* Scientific stats first — the main "proof" section */}
      {sciStats ? (
        <StatisticsSection stats={sciStats} />
      ) : (
        <section className="report-section">
          <p style={{ color: "var(--text-3)" }}>
            ยังไม่มี scientific stats · รัน <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>.venv/bin/python scripts/scientific_stats.py</code> เพื่อสร้าง
          </p>
        </section>
      )}

      {/* Section 1: Rolling AUC */}
      <section className="report-section">
        <div className="report-section-head">
          <h2>📈 Model stability — Rolling AUC ตลอด {monthly.length} เดือน</h2>
          <p className="report-section-hint">
            แต่ละจุด = AUC ของโมเดลใน 1 เดือน · เส้นแดงทแยง = baseline 0.5 (สุ่ม)
          </p>
        </div>
        {monthly.length > 0 ? (
          <RollingAucChart data={monthly} />
        ) : (
          <EmptyReport msg="ยังไม่มีข้อมูล rolling eval — รัน scripts/rolling_eval.py" />
        )}
        {metrics?.stability_auc_mean != null && (
          <div className="report-stats-grid">
            <Stat label="AUC mean" value={metrics.stability_auc_mean?.toFixed(3)} />
            <Stat label="AUC std" value={metrics.stability_auc_std?.toFixed(3)} />
            <Stat label="AUC min" value={metrics.stability_auc_min?.toFixed(3)} />
            <Stat label="AUC max" value={metrics.stability_auc_max?.toFixed(3)} />
          </div>
        )}
      </section>

      {/* Section 2: Feature importance */}
      <section className="report-section">
        <div className="report-section-head">
          <h2>🌿 Feature importance — top {Math.min(featureImportance.length, 20)}</h2>
          <p className="report-section-hint">
            Features ที่โมเดล LightGBM ใช้บ่อยที่สุดในการ split — บอกว่าตัวไหนสำคัญ
          </p>
        </div>
        {featureImportance.length > 0 ? (
          <FeatureImportanceChart data={featureImportance} />
        ) : (
          <EmptyReport msg="ยังไม่มี feature importance ใน metadata.metrics — เทรนใหม่ครั้งหน้าจะรวมให้" />
        )}
      </section>

      {/* Section 3: Provincial breakdown */}
      <section className="report-section">
        <div className="report-section-head">
          <h2>🇹🇭 Cells ต่อจังหวัด — top 15 (snapshot ล่าสุด)</h2>
          <p className="report-section-hint">
            จำนวน cells ที่ถูก flag ในแต่ละจังหวัด · stacked โดย urgency
          </p>
        </div>
        {provinces.length > 0 ? (
          <ProvinceChart data={provinces} />
        ) : (
          <EmptyReport msg="ไม่มี prediction ล่าสุด — รัน risk_map.py" />
        )}
      </section>

      <footer className="report-footer">
        <p style={{ fontSize: 11, color: "var(--text-3)" }}>
          ดูข้อมูลดิบ:{" "}
          <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>outputs/metadata/rolling_eval.json</code>{" · "}
          <code style={{ background: "var(--surface-2)", padding: "1px 4px", borderRadius: 3 }}>outputs/metadata/dataset_info.json</code>
        </p>
      </footer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Inline SVG charts (no chart library)
// ─────────────────────────────────────────────────────────────

interface RollingPoint {
  month: string;
  auc: number;
  positive_rate: number;
  n: number;
}

function RollingAucChart({ data }: { data: RollingPoint[] }) {
  const W = 800;
  const H = 240;
  const PAD_L = 50;
  const PAD_R = 20;
  const PAD_T = 20;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const n = data.length;
  const xToPx = (i: number) => PAD_L + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yToPx = (auc: number) => PAD_T + (1 - (auc - 0.5) / 0.5) * plotH;

  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xToPx(i).toFixed(1)} ${yToPx(d.auc).toFixed(1)}`)
    .join(" ");

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: 900, background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Rolling monthly AUC chart"
      >
        {/* Y-axis grid + labels */}
        {[0.5, 0.65, 0.75, 0.85, 1.0].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L} y1={yToPx(t)} x2={W - PAD_R} y2={yToPx(t)}
              stroke="var(--border-soft)" strokeWidth={1}
            />
            <text x={PAD_L - 6} y={yToPx(t) + 3} fill="var(--text-3)" fontSize="10" textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {/* Baseline 0.5 */}
        <line
          x1={PAD_L} y1={yToPx(0.5)} x2={W - PAD_R} y2={yToPx(0.5)}
          stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3"
        />
        <text x={W - PAD_R - 4} y={yToPx(0.5) - 4} fill="#ef4444" fontSize="9" textAnchor="end">
          random = 0.5
        </text>
        {/* Line */}
        <path d={path} fill="none" stroke="#22c55e" strokeWidth={2} />
        {/* Markers */}
        {data.map((d, i) => (
          <g key={d.month}>
            <circle
              cx={xToPx(i)}
              cy={yToPx(d.auc)}
              r={4}
              fill={d.auc >= 0.85 ? "#22c55e" : d.auc >= 0.70 ? "#eab308" : "#ef4444"}
              stroke="var(--surface)"
              strokeWidth={1.5}
            >
              <title>{`${d.month}: AUC ${d.auc.toFixed(3)} · positives ${(d.positive_rate * 100).toFixed(2)}% · n=${d.n.toLocaleString()}`}</title>
            </circle>
            {/* Month label every ~3rd */}
            {(i % Math.max(1, Math.floor(n / 8)) === 0 || i === n - 1) && (
              <text
                x={xToPx(i)} y={H - PAD_B + 14}
                fill="var(--text-3)" fontSize="9" textAnchor="middle"
                transform={`rotate(-25 ${xToPx(i)} ${H - PAD_B + 14})`}
              >
                {d.month}
              </text>
            )}
          </g>
        ))}
        {/* Axis labels */}
        <text x={PAD_L + plotW / 2} y={H - 6} fill="var(--text-3)" fontSize="11" textAnchor="middle">Month</text>
        <text
          x={14} y={PAD_T + plotH / 2}
          fill="var(--text-3)" fontSize="11" textAnchor="middle"
          transform={`rotate(-90 14 ${PAD_T + plotH / 2})`}
        >
          ROC-AUC
        </text>
      </svg>
    </div>
  );
}

function FeatureImportanceChart({
  data,
}: {
  data: { feature: string; importance: number }[];
}) {
  const W = 800;
  const ROW_H = 22;
  const PAD_L = 220;
  const PAD_R = 60;
  const PAD_T = 6;
  const H = data.length * ROW_H + PAD_T * 2;
  const plotW = W - PAD_L - PAD_R;
  const max = Math.max(...data.map((d) => d.importance), 1);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Feature importance chart"
      >
        {data.map((d, i) => {
          const y = PAD_T + i * ROW_H;
          const barW = (d.importance / max) * plotW;
          const color = i < 3 ? "#22c55e" : i < 10 ? "#84cc16" : "#3b82f6";
          return (
            <g key={d.feature}>
              <text
                x={PAD_L - 8} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11" textAnchor="end"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.feature.length > 30 ? d.feature.slice(0, 30) + "…" : d.feature}
              </text>
              <rect
                x={PAD_L} y={y + 3}
                width={barW} height={ROW_H - 6}
                fill={color}
                fillOpacity={0.85}
                rx={2}
              >
                <title>{`${d.feature}: ${d.importance.toFixed(1)}`}</title>
              </rect>
              <text
                x={PAD_L + barW + 6} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.importance.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

interface ProvincePoint {
  prov: string;
  CRITICAL: number;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
  total: number;
}

function ProvinceChart({ data }: { data: ProvincePoint[] }) {
  const W = 800;
  const ROW_H = 26;
  const PAD_L = 160;
  const PAD_R = 40;
  const PAD_T = 10;
  const H = data.length * ROW_H + PAD_T * 2;
  const plotW = W - PAD_L - PAD_R;
  const max = Math.max(...data.map((d) => d.total), 1);

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", background: "var(--surface-2)", borderRadius: 6 }}
        role="img"
        aria-label="Provinces by alert count"
      >
        {data.map((d, i) => {
          const y = PAD_T + i * ROW_H;
          let xOffset = PAD_L;
          const segments: { tier: keyof typeof TIER_COLORS; value: number }[] = [
            { tier: "CRITICAL", value: d.CRITICAL },
            { tier: "HIGH",     value: d.HIGH },
            { tier: "MEDIUM",   value: d.MEDIUM },
            { tier: "LOW",      value: d.LOW },
          ];
          return (
            <g key={d.prov}>
              <text
                x={PAD_L - 8} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11" textAnchor="end"
              >
                {d.prov.length > 22 ? d.prov.slice(0, 22) + "…" : d.prov}
              </text>
              {segments.map((s) => {
                if (s.value === 0) return null;
                const segW = (s.value / max) * plotW;
                const seg = (
                  <rect
                    key={s.tier}
                    x={xOffset} y={y + 4}
                    width={segW} height={ROW_H - 8}
                    fill={TIER_COLORS[s.tier]}
                    rx={2}
                  >
                    <title>{`${d.prov} ${s.tier}: ${s.value}`}</title>
                  </rect>
                );
                xOffset += segW;
                return seg;
              })}
              <text
                x={xOffset + 6} y={y + ROW_H / 2 + 4}
                fill="var(--text-2)" fontSize="11"
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {d.total}
              </text>
            </g>
          );
        })}
      </svg>
      {/* Legend */}
      <div style={{ display: "flex", gap: 12, marginTop: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((t) => (
          <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-2)" }}>
            <span style={{ width: 10, height: 10, background: TIER_COLORS[t], borderRadius: 2 }} />
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>
        {value ?? "—"}
      </div>
    </div>
  );
}

function EmptyReport({ msg }: { msg: string }) {
  return (
    <div style={{
      padding: 28,
      background: "var(--surface-2)",
      borderRadius: 6,
      color: "var(--text-3)",
      fontSize: 12,
      textAlign: "center",
    }}>
      {msg}
    </div>
  );
}
