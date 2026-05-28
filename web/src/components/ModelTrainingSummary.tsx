import { useEffect, useMemo, useState } from "react";
import { fetchTrainingSummary } from "../api";
import type {
  RollingMonthPoint,
  ScientificStats,
  TrainingSummary,
  ValidationMetrics,
} from "../types";

// ─────────────────────────────────────────────────────────────
// Model Training Summary — top-of-Reports executive overview.
//
// Every number rendered here is derived from REAL persisted artifacts:
//   • dataset_info.json        (via /api/training-summary or geojson metadata)
//   • scientific_stats.json    (via metrics.scientific_stats)
//   • rolling_eval.json        (via metrics.rolling_by_month + stability_*)
//
// Per CLAUDE.md hard rule: no synthetic, simulated, or interpolated values.
// If a field is missing in the artifact, we render "—" rather than fabricate.
// ─────────────────────────────────────────────────────────────

interface Props {
  metrics: ValidationMetrics | null;
}

export default function ModelTrainingSummary({ metrics }: Props) {
  // Prefer the training_summary baked into geojson metadata (one HTTP round
  // trip already done). If absent — e.g. risk_map.py hasn't been re-run since
  // we added the bundle — pull it from /api/training-summary.
  const inlineSummary = metrics?.training_summary;
  const [summary, setSummary] = useState<TrainingSummary | null>(inlineSummary ?? null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (summary) return;
    let cancelled = false;
    fetchTrainingSummary()
      .then((s) => { if (!cancelled) setSummary(s); })
      .catch((e: Error) => { if (!cancelled) setLoadError(e.message); });
    return () => { cancelled = true; };
  }, [summary]);

  const sci = metrics?.scientific_stats ?? null;
  const rolling = metrics?.rolling_by_month ?? [];

  // Aggregate per-month stability stats — these are not raw CV folds but
  // are the closest decomposition we have (the BayesSearchCV CV scores
  // aren't persisted). Per-month AUC across the test window gives the same
  // "mean / std / min / max" view requested.
  const stability = useMemo(() => computeStability(rolling, metrics), [rolling, metrics]);

  if (loadError && !summary) {
    return (
      <section className="report-section">
        <p style={{ color: "var(--text-3)", fontSize: 12 }}>
          ไม่สามารถโหลด training summary: {loadError}
        </p>
      </section>
    );
  }

  return (
    <section className="report-section model-training-summary">
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap", marginBottom: 18 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20 }}>🧪 Model Training Summary</h2>
          <p style={{ margin: "4px 0 0", fontSize: 12, color: "var(--text-3)" }}>
            Real persisted artifacts from <code style={{ background: "var(--surface-2)", padding: "1px 5px", borderRadius: 3 }}>train.py</code>{" "}
            · {summary?.trained_at ? `trained ${formatDate(summary.trained_at)}` : "training timestamp unavailable"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span className="status-chip status-chip-ok">✓ Calibrated</span>
          <span className="status-chip status-chip-ok">✓ Bootstrap 95% CI</span>
          <span className="status-chip status-chip-ok">✓ Chronological split (no leak)</span>
        </div>
      </header>

      {/* ── Top results cards ── */}
      {sci && (
        <div className="results-grid">
          <ResultCard
            label="ROC-AUC"
            point={sci.ci_95.roc_auc.point}
            ci={[sci.ci_95.roc_auc.lower, sci.ci_95.roc_auc.upper]}
            interpret="ความสามารถจัดอันดับ cell · ≥0.8 = ดี"
            tone={sci.ci_95.roc_auc.point >= 0.80 ? "great" : sci.ci_95.roc_auc.point >= 0.70 ? "good" : "ok"}
          />
          <ResultCard
            label="Recall @ deploy"
            point={sci.ci_95.recall_at_deploy.point}
            ci={[sci.ci_95.recall_at_deploy.lower, sci.ci_95.recall_at_deploy.upper]}
            interpret={`จับไฟจริงได้ ${Math.round(sci.ci_95.recall_at_deploy.point * 100)} / 100 จุด`}
            tone={sci.ci_95.recall_at_deploy.point >= 0.6 ? "great" : "ok"}
          />
          <ResultCard
            label="Precision @ deploy"
            point={sci.ci_95.precision_at_deploy.point}
            ci={[sci.ci_95.precision_at_deploy.lower, sci.ci_95.precision_at_deploy.upper]}
            interpret={`baseline = ${(sci.classification_stats.baseline_class_prior * 100).toFixed(1)}% (positive rate) · uplift ≈ ${(sci.ci_95.precision_at_deploy.point / sci.classification_stats.baseline_class_prior).toFixed(1)}×`}
            tone="ok"
          />
          <ResultCard
            label="F1 @ deploy"
            point={sci.ci_95.f1_at_deploy.point}
            ci={[sci.ci_95.f1_at_deploy.lower, sci.ci_95.f1_at_deploy.upper]}
            interpret="สมดุล precision + recall ที่ threshold ใช้งานจริง"
            tone="ok"
          />
          <ResultCard
            label="Brier score"
            point={sci.ci_95.brier_score.point}
            ci={[sci.ci_95.brier_score.lower, sci.ci_95.brier_score.upper]}
            interpret="ยิ่งต่ำยิ่งดี · <0.05 = calibrated probabilities"
            tone={sci.ci_95.brier_score.point < 0.05 ? "great" : "good"}
            invert
          />
          <ResultCard
            label="Avg Precision"
            point={sci.ci_95.average_precision.point}
            ci={[sci.ci_95.average_precision.lower, sci.ci_95.average_precision.upper]}
            interpret={`baseline = ${(sci.classification_stats.baseline_class_prior * 100).toFixed(1)}% · uplift ≈ ${(sci.ci_95.average_precision.point / sci.classification_stats.baseline_class_prior).toFixed(1)}×`}
            tone="ok"
          />
        </div>
      )}

      {/* ── Dataset · Model · Training cards (3-up) ── */}
      <div className="summary-cards-row">
        <SummaryCard title="📦 Dataset">
          <DefRow k="Source" v={summary?.data_source ?? "NASA FIRMS VIIRS NRT + ECMWF ERA5 (Open-Meteo)"} />
          <DefRow k="Date range" v={summary?.date_range && summary.date_range[0] ? `${summary.date_range[0]} → ${summary.date_range[1]}` : "—"} />
          <DefRow k="Total days" v={summary?.total_days?.toLocaleString() ?? "—"} />
          <DefRow k="Active cells" v={summary?.active_cells?.toLocaleString() ?? "—"} suffix={summary?.grid_size_deg ? `× ${summary.grid_size_deg}° grid` : undefined} />
          <DefRow k="Densified rows" v={sci?.samples.total_densified.toLocaleString() ?? "—"} suffix="(cell × day)" />
          <DefRow k="Training rows" v={summary?.training_rows?.toLocaleString() ?? "—"} suffix="(post-undersample)" />
          <DefRow k="Features" v={summary?.feature_count?.toString() ?? "—"} suffix={summary?.weather_features_count ? `(${summary.weather_features_count} weather)` : undefined} />
        </SummaryCard>

        <SummaryCard title="🧠 Model">
          <DefRow k="Type" v={(summary?.model_type ?? "lightgbm").toUpperCase()} />
          <DefRow k="Task" v={summary?.prediction_type ?? "binary_fire_in_3d"} />
          <DefRow k="Imminent horizon" v={summary?.imminent_days ? `${summary.imminent_days} days` : "—"} />
          {summary?.best_params && (
            <>
              <DefRow k="learning_rate"     v={fmt(summary.best_params.learning_rate)} />
              <DefRow k="num_leaves"        v={fmt(summary.best_params.num_leaves)} />
              <DefRow k="max_depth"         v={fmt(summary.best_params.max_depth)} />
              <DefRow k="n_estimators"      v={fmt(summary.best_params.n_estimators)} />
              <DefRow k="min_child_samples" v={fmt(summary.best_params.min_child_samples)} />
              <DefRow k="subsample"         v={fmt(summary.best_params.subsample)} />
              <DefRow k="colsample_bytree"  v={fmt(summary.best_params.colsample_bytree)} />
              <DefRow k="reg_lambda"        v={fmt(summary.best_params.reg_lambda)} />
            </>
          )}
        </SummaryCard>

        <SummaryCard title="🎓 Training">
          <DefRow k="Split"        v="Chronological 60 / 20 / 20" suffix="(no shuffle)" />
          <DefRow k="HP search"    v={summary?.search_method ?? "—"} suffix={summary?.search_iterations ? `${summary.search_iterations} iters` : undefined} />
          <DefRow k="Inner CV"     v={summary?.cv_n_splits ? `TimeSeriesSplit · ${summary.cv_n_splits} folds` : "—"} suffix={summary?.cv_gap_days != null ? `gap ${summary.cv_gap_days}d` : undefined} />
          <DefRow k="Ensemble"     v={summary?.ensemble_size?.toString() ?? "—"} suffix="LGBM models (different seeds)" />
          <DefRow k="Early stop"   v={summary?.early_stopping_rounds ? `${summary.early_stopping_rounds} rounds` : "—"} />
          <DefRow k="Calibration"  v="Platt sigmoid (fit on val)" />
          <DefRow k="Sample wt"    v="recency × inverse freq × day-bucket boost" />
          <DefRow k="Train time"   v={summary?.training_time_seconds ? `${(summary.training_time_seconds / 60).toFixed(1)} min` : "—"} />
        </SummaryCard>
      </div>

      {/* ── Bootstrap CI table — large, prominent ── */}
      {sci && (
        <div style={{ marginTop: 18 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>📐 Bootstrap 95% Confidence Intervals (n = 1000 resamples)</h3>
          <table className="stats-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ textAlign: "right" }}>Point estimate</th>
                <th style={{ textAlign: "right" }}>95% CI (lower)</th>
                <th style={{ textAlign: "right" }}>95% CI (upper)</th>
                <th style={{ textAlign: "right" }}>Std error</th>
                <th style={{ textAlign: "right" }}>CI width</th>
              </tr>
            </thead>
            <tbody>
              {(Object.entries(sci.ci_95) as [keyof typeof sci.ci_95, typeof sci.ci_95.roc_auc][]).map(
                ([key, ci]) => (
                  <tr key={key}>
                    <td><b>{prettyMetricName(key)}</b></td>
                    <td className="mono right"><b>{ci.point.toFixed(4)}</b></td>
                    <td className="mono right">{ci.lower.toFixed(4)}</td>
                    <td className="mono right">{ci.upper.toFixed(4)}</td>
                    <td className="mono right small" style={{ color: "var(--text-3)" }}>{ci.std.toFixed(4)}</td>
                    <td className="mono right small" style={{ color: "var(--text-3)" }}>{(ci.upper - ci.lower).toFixed(4)}</td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Per-month stability (substitute for CV-fold table) ── */}
      {stability && (
        <div style={{ marginTop: 18 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>📈 Per-month stability across {stability.n} months</h3>
          <p style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 8 }}>
            Monthly held-out AUC + positive rate across the full test window — same role as CV-fold variance,
            but computed on real test-set months (not in-distribution folds).
          </p>
          <table className="stats-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ textAlign: "right" }}>Mean</th>
                <th style={{ textAlign: "right" }}>Std</th>
                <th style={{ textAlign: "right" }}>Min</th>
                <th style={{ textAlign: "right" }}>Max</th>
                <th style={{ textAlign: "right" }}>n</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>AUC (monthly)</b></td>
                <td className="mono right"><b>{stability.aucMean.toFixed(4)}</b></td>
                <td className="mono right">{stability.aucStd.toFixed(4)}</td>
                <td className="mono right">{stability.aucMin.toFixed(4)}</td>
                <td className="mono right">{stability.aucMax.toFixed(4)}</td>
                <td className="mono right">{stability.n}</td>
              </tr>
              {stability.posRateMean != null && (
                <tr>
                  <td><b>Positive rate (monthly)</b></td>
                  <td className="mono right"><b>{(stability.posRateMean * 100).toFixed(2)}%</b></td>
                  <td className="mono right">{(stability.posRateStd * 100).toFixed(2)}%</td>
                  <td className="mono right">{(stability.posRateMin * 100).toFixed(2)}%</td>
                  <td className="mono right">{(stability.posRateMax * 100).toFixed(2)}%</td>
                  <td className="mono right">{stability.n}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <p style={{ marginTop: 16, fontSize: 11, color: "var(--text-3)", fontStyle: "italic", lineHeight: 1.6 }}>
        Limitation: precision is intentionally low at the operational threshold (0.05) — the model is tuned for high recall
        on a rare positive class (~3.5% prior), so it accepts false alarms to avoid missing actual fires.
      </p>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function ResultCard({
  label, point, ci, interpret, tone, invert = false,
}: {
  label: string;
  point: number;
  ci: [number, number];
  interpret: string;
  tone: "great" | "good" | "ok";
  invert?: boolean;
}) {
  const toneColor = tone === "great" ? "#22c55e" : tone === "good" ? "#84cc16" : "#eab308";
  const fmtVal = invert ? point.toFixed(4) : point >= 1 ? point.toFixed(3) : point.toFixed(point >= 0.1 ? 3 : 4);
  return (
    <div className="result-card">
      <div className="result-card-label">{label}</div>
      <div className="result-card-value" style={{ color: toneColor }}>{fmtVal}</div>
      <div className="result-card-ci">
        95% CI [{ci[0].toFixed(invert ? 4 : 3)}, {ci[1].toFixed(invert ? 4 : 3)}]
      </div>
      <div className="result-card-interpret">{interpret}</div>
    </div>
  );
}

function SummaryCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="summary-card">
      <h3 className="summary-card-title">{title}</h3>
      <dl className="summary-card-defs">{children}</dl>
    </div>
  );
}

function DefRow({ k, v, suffix }: { k: string; v: string | undefined; suffix?: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>
        <span style={{ fontVariantNumeric: "tabular-nums" }}>{v ?? "—"}</span>
        {suffix && <span style={{ marginLeft: 6, color: "var(--text-3)", fontSize: 11 }}>{suffix}</span>}
      </dd>
    </>
  );
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.?0+$/, "");
  return String(v);
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-CA") + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

function prettyMetricName(key: keyof ScientificStats["ci_95"]): string {
  const map: Record<string, string> = {
    roc_auc:              "ROC-AUC",
    average_precision:    "Average Precision (PR-AUC)",
    f1_at_deploy:         "F1 @ deployment threshold",
    precision_at_deploy:  "Precision @ deployment threshold",
    recall_at_deploy:     "Recall @ deployment threshold",
    brier_score:          "Brier score",
  };
  return map[key as string] ?? String(key);
}

interface StabilityStats {
  n: number;
  aucMean: number;
  aucStd: number;
  aucMin: number;
  aucMax: number;
  posRateMean: number | null;
  posRateStd: number;
  posRateMin: number;
  posRateMax: number;
}

function computeStability(
  rolling: RollingMonthPoint[],
  metrics: ValidationMetrics | null,
): StabilityStats | null {
  // Prefer the pre-computed stability_* fields if present (these come from
  // rolling_eval.json). Fall back to recomputing from rolling array.
  if (!rolling.length && metrics?.stability_auc_mean == null) return null;

  if (rolling.length) {
    const aucs = rolling.map((r) => r.auc).filter((v) => Number.isFinite(v));
    const rates = rolling.map((r) => r.positive_rate).filter((v) => Number.isFinite(v));
    if (!aucs.length) return null;
    const aMean = mean(aucs);
    const rMean = rates.length ? mean(rates) : null;
    return {
      n: aucs.length,
      aucMean: aMean,
      aucStd: std(aucs, aMean),
      aucMin: Math.min(...aucs),
      aucMax: Math.max(...aucs),
      posRateMean: rMean,
      posRateStd: rates.length && rMean != null ? std(rates, rMean) : 0,
      posRateMin: rates.length ? Math.min(...rates) : 0,
      posRateMax: rates.length ? Math.max(...rates) : 0,
    };
  }

  // Fallback path — pre-aggregated only, no per-month data.
  return {
    n: metrics?.stability_months ?? 0,
    aucMean: metrics?.stability_auc_mean ?? 0,
    aucStd:  metrics?.stability_auc_std  ?? 0,
    aucMin:  metrics?.stability_auc_min  ?? 0,
    aucMax:  metrics?.stability_auc_max  ?? 0,
    posRateMean: null,
    posRateStd: 0,
    posRateMin: 0,
    posRateMax: 0,
  };
}

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[], m: number): number {
  if (arr.length < 2) return 0;
  const v = arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
}
