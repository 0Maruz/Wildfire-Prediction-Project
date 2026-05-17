import type { ScientificStats, BootstrapCI } from "../types";

interface Props {
  stats: ScientificStats;
}

// ─────────────────────────────────────────────────────────────
// Scientific-statistics section for the Reports page.
//
// Lays out (in order, easiest → most technical):
//   1. Dataset breakdown table (sample sizes per split)
//   2. 95% Bootstrap CIs for headline metrics (point ± uncertainty)
//   3. Confusion matrix at deployment threshold (2x2 heat-style)
//   4. Classification stats (sensitivity, specificity, κ, MCC, Brier)
//   5. ROC curve + PR curve (SVG line charts)
//
// All charts inline SVG. Designed for a Thai science-project audience —
// metric labels in Thai, with English in parentheses.
// ─────────────────────────────────────────────────────────────

export default function StatisticsSection({ stats }: Props) {
  return (
    <>
      <DatasetSplitTable samples={stats.samples} />
      <BootstrapCITable ci={stats.ci_95} />
      <ConfusionMatrixCard cm={stats.confusion_matrix} cs={stats.classification_stats} />
      <ClassificationStatsCard cs={stats.classification_stats} />
      <CurvesRow roc={stats.roc_curve} pr={stats.pr_curve} prior={stats.classification_stats.baseline_class_prior} />
    </>
  );
}

// ── Dataset Split ───────────────────────────────────────────
function DatasetSplitTable({ samples }: { samples: ScientificStats["samples"] }) {
  const rows = [
    { label: "Training",   data: samples.train },
    { label: "Validation", data: samples.val },
    { label: "Test",       data: samples.test },
  ];
  return (
    <section className="report-section">
      <div className="report-section-head">
        <h2>📋 Dataset breakdown (chronological 60/20/20 split)</h2>
        <p className="report-section-hint">
          จำนวนข้อมูล train/val/test · ไม่มีการ shuffle (เรียงตามวันที่)
        </p>
      </div>
      <table className="stats-table">
        <thead>
          <tr>
            <th>Split</th>
            <th style={{ textAlign: "right" }}>N (rows)</th>
            <th style={{ textAlign: "right" }}>Positives</th>
            <th style={{ textAlign: "right" }}>Positive rate</th>
            <th>Date range</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td><b>{r.label}</b></td>
              <td className="mono right">{r.data.n.toLocaleString()}</td>
              <td className="mono right">{r.data.positives.toLocaleString()}</td>
              <td className="mono right">{(r.data.positive_rate * 100).toFixed(2)}%</td>
              <td className="mono small">{r.data.date_range[0]} → {r.data.date_range[1]}</td>
            </tr>
          ))}
          <tr style={{ borderTop: "2px solid var(--border)" }}>
            <td><b>Total densified</b></td>
            <td className="mono right"><b>{samples.total_densified.toLocaleString()}</b></td>
            <td className="mono right" colSpan={3} style={{ color: "var(--text-3)" }}>
              cell-day rows (active cells × full date range)
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}

// ── 95% Confidence Intervals ─────────────────────────────────
function BootstrapCITable({ ci }: { ci: ScientificStats["ci_95"] }) {
  const rows: { label: string; labelEn: string; ci: BootstrapCI; isPct: boolean }[] = [
    { label: "ROC-AUC", labelEn: "Area under ROC curve", ci: ci.roc_auc, isPct: false },
    { label: "Average Precision", labelEn: "Area under PR curve", ci: ci.average_precision, isPct: false },
    { label: "F1 score", labelEn: "@deployment threshold", ci: ci.f1_at_deploy, isPct: false },
    { label: "Precision", labelEn: "@deployment threshold", ci: ci.precision_at_deploy, isPct: false },
    { label: "Recall (Sensitivity)", labelEn: "@deployment threshold", ci: ci.recall_at_deploy, isPct: false },
    { label: "Brier score", labelEn: "ค่ายิ่งต่ำยิ่งดี", ci: ci.brier_score, isPct: false },
  ];
  return (
    <section className="report-section">
      <div className="report-section-head">
        <h2>🎯 95% Confidence Intervals (bootstrap, n=1000)</h2>
        <p className="report-section-hint">
          ช่วงความเชื่อมั่น 95% ของ metric แต่ละตัว · resample ข้อมูล test แบบสุ่ม 1000 ครั้ง
        </p>
      </div>
      <table className="stats-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th style={{ textAlign: "right" }}>Point estimate</th>
            <th style={{ textAlign: "right" }}>95% CI (lower)</th>
            <th style={{ textAlign: "right" }}>95% CI (upper)</th>
            <th style={{ textAlign: "right" }}>± std</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td>
                <b>{r.label}</b>
                <div className="metric-sub">{r.labelEn}</div>
              </td>
              <td className="mono right"><b>{r.ci.point.toFixed(4)}</b></td>
              <td className="mono right">{r.ci.lower.toFixed(4)}</td>
              <td className="mono right">{r.ci.upper.toFixed(4)}</td>
              <td className="mono right">{r.ci.std.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="report-section-hint" style={{ marginTop: 8 }}>
        🧪 <b>วิธีการคำนวณ:</b> สุ่มเลือก rows จาก test set (มี replacement) เป็นชุดใหม่ 1000 ครั้ง · คำนวณ metric ทุกครั้ง · เอา quantile 2.5% และ 97.5% เป็นช่วง 95% CI
      </p>
    </section>
  );
}

// ── Confusion Matrix ─────────────────────────────────────────
function ConfusionMatrixCard({
  cm, cs,
}: {
  cm: ScientificStats["confusion_matrix"];
  cs: ScientificStats["classification_stats"];
}) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const cell = (count: number, color: string, label: string, sublabel: string) => {
    const pct = total ? (count / total) * 100 : 0;
    const intensity = total ? Math.min(0.85, 0.15 + (count / total) * 1.5) : 0.15;
    return (
      <div
        className="cm-cell"
        style={{ background: `${color}${Math.round(intensity * 255).toString(16).padStart(2, "0")}` }}
      >
        <div className="cm-cell-label">{label}</div>
        <div className="cm-cell-count">{count.toLocaleString()}</div>
        <div className="cm-cell-pct">{pct.toFixed(2)}%</div>
        <div className="cm-cell-sub">{sublabel}</div>
      </div>
    );
  };
  return (
    <section className="report-section">
      <div className="report-section-head">
        <h2>🔢 Confusion Matrix (@deployment threshold)</h2>
        <p className="report-section-hint">
          จับคู่ predicted vs actual บน test set · เซลล์เข้ม = นับเยอะ
        </p>
      </div>
      <div className="cm-wrap">
        <div className="cm-side-label" aria-hidden="true">Actual</div>
        <div className="cm-grid">
          <div></div>
          <div className="cm-col-label">Predicted: No fire</div>
          <div className="cm-col-label">Predicted: Fire</div>

          <div className="cm-row-label">No fire</div>
          {cell(cm.tn, "#22c55e", "True Negative", "ตอบไม่เกิด → ไม่เกิดจริง")}
          {cell(cm.fp, "#eab308", "False Positive", "ตอบเกิด → ไม่เกิด (false alarm)")}

          <div className="cm-row-label">Fire</div>
          {cell(cm.fn, "#ef4444", "False Negative", "ตอบไม่เกิด → เกิดจริง (พลาด!)")}
          {cell(cm.tp, "#22c55e", "True Positive", "ตอบเกิด → เกิดจริง")}
        </div>
      </div>
      <div className="cm-summary">
        <span>Sensitivity = TP/(TP+FN) = <b>{(cs.sensitivity * 100).toFixed(2)}%</b></span>
        <span>Specificity = TN/(TN+FP) = <b>{(cs.specificity * 100).toFixed(2)}%</b></span>
        <span>PPV = TP/(TP+FP) = <b>{(cs.ppv * 100).toFixed(2)}%</b></span>
        <span>NPV = TN/(TN+FN) = <b>{(cs.npv * 100).toFixed(2)}%</b></span>
      </div>
    </section>
  );
}

// ── Classification stats (extra) ─────────────────────────────
function ClassificationStatsCard({ cs }: { cs: ScientificStats["classification_stats"] }) {
  const interp = (label: string, value: number, good: number, ok: number, badLabel: string, fmt: (v: number) => string) => {
    const status = value >= good ? "good" : value >= ok ? "ok" : "bad";
    return { label, value: fmt(value), status, badLabel };
  };
  const items = [
    interp("Cohen's κ (kappa)", cs.cohen_kappa, 0.6, 0.4, "เดามากกว่าโมเดล", (v) => v.toFixed(4)),
    interp("Matthews MCC", cs.matthews_corr_coef, 0.5, 0.3, "ไม่ค่อย discriminative", (v) => v.toFixed(4)),
    interp("Brier Skill Score", cs.brier_skill_score, 0.2, 0.0, "แย่กว่าเดา prior", (v) => v.toFixed(4)),
    { label: "Log loss", value: cs.log_loss.toFixed(4), status: "ok", badLabel: "compare with -log(prior)" },
    { label: "False alarm rate", value: (cs.false_positive_rate * 100).toFixed(2) + "%", status: "ok", badLabel: "operator burden" },
    { label: "Miss rate", value: (cs.false_negative_rate * 100).toFixed(2) + "%", status: "ok", badLabel: "missed fires" },
  ];
  return (
    <section className="report-section">
      <div className="report-section-head">
        <h2>📐 Additional Classification Statistics</h2>
        <p className="report-section-hint">
          เมตริกขั้นสูงสำหรับ binary classification — รวม class-imbalance-robust scores
        </p>
      </div>
      <div className="cs-grid">
        {items.map((it) => (
          <div key={it.label} className={`cs-item cs-${it.status}`}>
            <div className="cs-label">{it.label}</div>
            <div className="cs-value">{it.value}</div>
            <div className="cs-hint">{it.badLabel}</div>
          </div>
        ))}
      </div>
      <details className="cs-details">
        <summary>💡 ความหมายของแต่ละค่า (กดดู)</summary>
        <ul>
          <li><b>Cohen's κ</b> — ความสอดคล้องระหว่าง predicted กับ actual <i>เหนือกว่าการเดามั่ว</i>. -1 = ตรงข้าม, 0 = เท่ามั่ว, 1 = สมบูรณ์. &gt;0.4 = ปานกลาง, &gt;0.6 = ดี</li>
          <li><b>Matthews MCC</b> — single-score metric ที่ robust ต่อ class imbalance. -1 ถึง +1. คิดทั้ง 4 cell ของ confusion matrix</li>
          <li><b>Brier Skill Score</b> — เปรียบเทียบกับ baseline ที่ทำนาย class prior. &gt;0 = ดีกว่า baseline, &lt;0 = แย่กว่า baseline</li>
          <li><b>Log loss</b> — ลงโทษ over-confident wrong predictions รุนแรง. ยิ่งต่ำยิ่งดี</li>
          <li><b>False alarm rate</b> (FPR) = FP/(FP+TN) — จาก cell ที่ไม่มีไฟจริง โมเดลเตือนกี่%</li>
          <li><b>Miss rate</b> (FNR) = FN/(FN+TP) — จาก cell ที่มีไฟจริง โมเดลพลาดกี่%</li>
        </ul>
      </details>
    </section>
  );
}

// ── ROC + PR curves ──────────────────────────────────────────
function CurvesRow({
  roc, pr, prior,
}: {
  roc: ScientificStats["roc_curve"];
  pr: ScientificStats["pr_curve"];
  prior: number;
}) {
  return (
    <section className="report-section">
      <div className="report-section-head">
        <h2>📈 ROC + Precision-Recall curves</h2>
        <p className="report-section-hint">
          กราฟ classic ของ binary classifier · hover ดูค่า threshold
        </p>
      </div>
      <div className="curves-grid">
        <div className="curve-card">
          <h3>ROC curve</h3>
          <RocSvg points={roc} />
          <p className="curve-hint">
            <b>X = False Positive Rate</b> (เตือนผิด) · <b>Y = True Positive Rate</b> (จับถูก) ·
            <span style={{ color: "#ef4444" }}> ทแยงแดง</span> = random baseline · กราฟใกล้มุมซ้ายบน = ดี
          </p>
        </div>
        <div className="curve-card">
          <h3>Precision-Recall curve</h3>
          <PrSvg points={pr} prior={prior} />
          <p className="curve-hint">
            <b>X = Recall</b> · <b>Y = Precision</b> ·
            <span style={{ color: "#eab308" }}> เส้นเหลือง</span> = baseline (positive rate = {(prior * 100).toFixed(2)}%) ·
            ใกล้มุมขวาบน = ดี
          </p>
        </div>
      </div>
    </section>
  );
}

function RocSvg({ points }: { points: { x: number; y: number; t: number }[] }) {
  const W = 320, H = 280, PAD = 38;
  const plotW = W - PAD - 12, plotH = H - PAD - 24;
  const xToPx = (x: number) => PAD + x * plotW;
  const yToPx = (y: number) => H - PAD - y * plotH;
  const path = points.length
    ? points.map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x).toFixed(1)} ${yToPx(p.y).toFixed(1)}`).join(" ")
    : "";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 380, background: "var(--surface-2)", borderRadius: 6 }}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={xToPx(t)} y1={yToPx(0)} x2={xToPx(t)} y2={yToPx(1)} stroke="var(--border-soft)" strokeWidth={1} />
          <line x1={xToPx(0)} y1={yToPx(t)} x2={xToPx(1)} y2={yToPx(t)} stroke="var(--border-soft)" strokeWidth={1} />
          <text x={xToPx(t)} y={H - 8} fill="var(--text-3)" fontSize="9" textAnchor="middle">{t.toFixed(2)}</text>
          <text x={PAD - 4} y={yToPx(t) + 3} fill="var(--text-3)" fontSize="9" textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      {/* Random baseline */}
      <line x1={xToPx(0)} y1={yToPx(0)} x2={xToPx(1)} y2={yToPx(1)} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3" />
      {/* ROC line */}
      <path d={path} fill="none" stroke="#22c55e" strokeWidth={2} />
      {/* Area shading */}
      {path && <path d={`${path} L ${xToPx(1).toFixed(1)} ${yToPx(0).toFixed(1)} L ${xToPx(0).toFixed(1)} ${yToPx(0).toFixed(1)} Z`} fill="#22c55e" fillOpacity={0.12} />}
      <text x={W / 2} y={H - 22} fill="var(--text-3)" fontSize="10" textAnchor="middle">FPR (1 - Specificity)</text>
      <text x={10} y={H / 2 - 8} fill="var(--text-3)" fontSize="10" textAnchor="middle" transform={`rotate(-90 10 ${H / 2 - 8})`}>TPR (Sensitivity)</text>
    </svg>
  );
}

function PrSvg({ points, prior }: { points: { x: number; y: number; t: number }[]; prior: number }) {
  const W = 320, H = 280, PAD = 38;
  const plotW = W - PAD - 12, plotH = H - PAD - 24;
  const xToPx = (x: number) => PAD + x * plotW;
  const yToPx = (y: number) => H - PAD - y * plotH;
  const path = points.length
    ? points.map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x).toFixed(1)} ${yToPx(p.y).toFixed(1)}`).join(" ")
    : "";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 380, background: "var(--surface-2)", borderRadius: 6 }}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={xToPx(t)} y1={yToPx(0)} x2={xToPx(t)} y2={yToPx(1)} stroke="var(--border-soft)" strokeWidth={1} />
          <line x1={xToPx(0)} y1={yToPx(t)} x2={xToPx(1)} y2={yToPx(t)} stroke="var(--border-soft)" strokeWidth={1} />
          <text x={xToPx(t)} y={H - 8} fill="var(--text-3)" fontSize="9" textAnchor="middle">{t.toFixed(2)}</text>
          <text x={PAD - 4} y={yToPx(t) + 3} fill="var(--text-3)" fontSize="9" textAnchor="end">{t.toFixed(2)}</text>
        </g>
      ))}
      {/* No-skill baseline = positive rate */}
      <line x1={xToPx(0)} y1={yToPx(prior)} x2={xToPx(1)} y2={yToPx(prior)} stroke="#eab308" strokeWidth={1.5} strokeDasharray="4 3" />
      <path d={path} fill="none" stroke="#3b82f6" strokeWidth={2} />
      <text x={W / 2} y={H - 22} fill="var(--text-3)" fontSize="10" textAnchor="middle">Recall</text>
      <text x={10} y={H / 2 - 8} fill="var(--text-3)" fontSize="10" textAnchor="middle" transform={`rotate(-90 10 ${H / 2 - 8})`}>Precision</text>
    </svg>
  );
}
