import { useEffect } from "react";
import type {
  FireFeature,
  GeoJsonMetadata,
  GistdaFeature,
  UrgencyThresholds,
  ValidationMetrics,
} from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  activeBaseDate: string;
  predicted: FireFeature[]; // current snapshot (post-province filter)
  observed: FireFeature[]; // all observed features
  liveFires: GistdaFeature[];
  metrics: ValidationMetrics | null;
  thresholds: UrgencyThresholds | null;
  metadata: GeoJsonMetadata | null;
  selectedProvince: string;
  selectedDay: string;
}

// Pop-up with the detail an operator might want before trusting a number on
// the dashboard: training metrics, snapshot composition, urgency thresholds,
// data freshness, etc. Reads everything from already-loaded state — no
// network call.
export default function InfoModal(props: Props) {
  useEffect(() => {
    if (!props.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") props.onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [props.open, props.onClose]);

  if (!props.open) return null;

  const m = props.metrics ?? {};
  const acc = m.accuracy_within_1day;
  const mae = m.mae_days;
  const rmse = m.rmse_days;
  const r2 = m.r2;

  // Per-tier counts on the current snapshot
  const tiers = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
  for (const f of props.predicted) {
    const u = f.properties.urgency_level;
    if (u && u in tiers) tiers[u]++;
  }

  // Per-day counts
  const dayCounts: Record<number, number> = {};
  for (let i = 0; i <= 7; i++) dayCounts[i] = 0;
  for (const f of props.predicted) {
    const d = f.properties.days_until_fire;
    if (d != null && d >= 0 && d <= 7) dayCounts[d]++;
  }

  // Province distribution
  const byProvince: Record<string, number> = {};
  for (const f of props.predicted) {
    const p = (f.properties.province ?? "").trim() || "(unassigned)";
    byProvince[p] = (byProvince[p] ?? 0) + 1;
  }
  const topProvinces = Object.entries(byProvince)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const fmt = (v: number | undefined, d = 3) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—";
  const fmtPct = (v: number | undefined) =>
    typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";

  return (
    <div className="info-modal-backdrop" role="dialog" aria-modal="true" onClick={props.onClose}>
      <div className="info-modal" onClick={(e) => e.stopPropagation()}>
        <div className="info-modal-header">
          <h2>Model & Snapshot Details</h2>
          <button className="info-modal-close" onClick={props.onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="info-modal-body">
          <section>
            <h3>Held-out test accuracy</h3>
            <div className="info-grid">
              <div><span>±1 day acc</span><b>{fmtPct(acc)}</b></div>
              <div><span>MAE</span><b>{fmt(mae, 2)} d</b></div>
              <div><span>RMSE</span><b>{fmt(rmse, 2)} d</b></div>
              <div><span>R²</span><b>{fmt(r2, 3)}</b></div>
            </div>
            <p className="info-note">
              From the held-out final 20% of the chronologically-split training
              data. The model never saw these rows during tuning or selection.
            </p>
          </section>

          <section>
            <h3>Current snapshot · {props.activeBaseDate}</h3>
            <div className="info-grid">
              <div><span>Predicted cells</span><b>{props.predicted.length}</b></div>
              <div><span>Observed (FIRMS)</span><b>{props.observed.length}</b></div>
              <div><span>Live (GISTDA)</span><b>{props.liveFires.length || "—"}</b></div>
              <div><span>Province filter</span><b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b></div>
            </div>
          </section>

          <section>
            <h3>Urgency distribution</h3>
            <div className="info-grid">
              <div><span style={{ color: "#dc2626" }}>CRITICAL</span><b>{tiers.CRITICAL}</b></div>
              <div><span style={{ color: "#ea580c" }}>HIGH</span><b>{tiers.HIGH}</b></div>
              <div><span style={{ color: "#f59e0b" }}>MEDIUM</span><b>{tiers.MEDIUM}</b></div>
              <div><span style={{ color: "#10b981" }}>LOW</span><b>{tiers.LOW}</b></div>
            </div>
            {props.thresholds && (
              <p className="info-note">
                Cutoffs: ≤{props.thresholds.CRITICAL.toFixed(1)}d CRITICAL,
                ≤{props.thresholds.HIGH.toFixed(1)}d HIGH,
                ≤{props.thresholds.MEDIUM.toFixed(1)}d MEDIUM,
                ≤{props.thresholds.LOW.toFixed(1)}d LOW.
              </p>
            )}
          </section>

          <section>
            <h3>Predicted day distribution</h3>
            <div className="info-bars">
              {Array.from({ length: 8 }, (_, i) => {
                const count = dayCounts[i];
                const max = Math.max(...Object.values(dayCounts), 1);
                const pct = (count / max) * 100;
                const label = i === 0 ? "Today" : `+${i}d`;
                return (
                  <div key={i} className="info-bar-row">
                    <span className="info-bar-label">{label}</span>
                    <div className="info-bar-track"><div className="info-bar-fill" style={{ width: `${pct}%` }} /></div>
                    <span className="info-bar-count">{count}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {topProvinces.length > 0 && (
            <section>
              <h3>Top 5 provinces by predicted cells</h3>
              <table className="info-table">
                <tbody>
                  {topProvinces.map(([name, count]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td><b>{count}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section>
            <h3>Glossary — what these numbers mean</h3>
            <dl className="info-dl">
              <dt>±1 day accuracy</dt>
              <dd>The share of test-set rows where the model's predicted fire date landed within ±1 day of the real fire date. The single most operator-meaningful number: <b>higher = better</b>. 50% means about half the predictions are practically correct.</dd>

              <dt>MAE (mean absolute error, days)</dt>
              <dd>Average distance between predicted and real fire dates, in days. <b>Lower = better</b>. MAE = 1.5 means predictions are off by 1½ days on average. Robust to outliers.</dd>

              <dt>RMSE (root mean squared error, days)</dt>
              <dd>Like MAE but squares the errors before averaging, so big misses count more. <b>Lower = better</b>. If RMSE is much higher than MAE, a few large errors dominate the picture.</dd>

              <dt>R² (coefficient of determination)</dt>
              <dd>How much variance in the real fire dates the model explains. <b>0 = no better than predicting the mean, 1 = perfect, negative = worse than predicting the mean</b>. Wildfire date is inherently noisy, so even R² ≈ 0.3 is useful.</dd>

              <dt>Skill check</dt>
              <dd>"Passed" means the model beats the predict-mean baseline by ≥5% on test MAE. "Failed" means the model is barely better than guessing the average — investigate before trusting predictions.</dd>

              <dt>Validation MAE vs Test MAE</dt>
              <dd>Validation is the slice used during hyperparameter tuning; test is the held-out final 20% the model has never seen. If validation is dramatically worse than test, the validation window was unusually hard (often: off-peak fire season).</dd>

              <dt>Urgency tiers (CRITICAL / HIGH / MEDIUM / LOW)</dt>
              <dd>Buckets of <code>days_until_fire</code>: CRITICAL = fire today, HIGH ≤ 2 days, MEDIUM ≤ 4 days, LOW ≤ 7 days. Used for at-a-glance prioritisation; thresholds are listed under "Urgency distribution" above.</dd>

              <dt>days_until_fire (the model's actual output)</dt>
              <dd>An integer 0–7 returned per grid cell — "the model thinks this cell will burn N days from the base date". 0 = today, 7 = end of the prediction horizon.</dd>

              <dt>raw_prediction</dt>
              <dd>The continuous version of <code>days_until_fire</code> before flooring to an integer. Floor (not round) is used so a value of 0.96 lands on day 0 ("within 24h"), matching how operators read the bucket label.</dd>

              <dt>Confidence (rounding proxy)</dt>
              <dd>How close the raw prediction landed to the centre of its day-bucket: 1.0 = dead-centre, 0 = at the edge. <b>Not a calibrated probability</b> — don't read 0.80 as "80% chance the fire happens".</dd>

              <dt>Historical fires (last 30 days, FIRMS)</dt>
              <dd>Literal count of NASA FIRMS hotspot detections in the cell over the last 30 days. Used as a sanity-check: a CRITICAL prediction in a cell with 0 recent fires deserves scepticism.</dd>

              <dt>Hit rate vs FIRMS</dt>
              <dd>For each historical prediction, did the predicted cell actually burn within ±1 day of the predicted date? Higher = the dashboard is calling real events, not noise.</dd>
            </dl>
          </section>

          <section>
            <h3>Data sources</h3>
            <ul className="info-list">
              <li><b>NASA FIRMS VIIRS NRT</b> — hotspot detections + FRP / brightness, used for training labels and historical aggregates.</li>
              <li><b>GISTDA NRT VIIRS + MODIS</b> — independent hotspot feed from Thailand's national space agency, optional live overlay.</li>
              <li><b>Open-Meteo ERA5</b> — daily reanalysis (temp / precip / wind), used as features only when the weather cache is present.</li>
              <li><b>Hansen GFC v1.11</b> — per-cell tree cover baseline (2000) + recent loss %, distinguishes wildfire risk from agricultural-burn signal.</li>
            </ul>
            <p className="info-note">
              All values shown anywhere in the dashboard come from real sources — no synthetic, interpolated, or simulated data.
            </p>
          </section>

          <section>
            <h3>Selected filter</h3>
            <p className="info-note">
              Day filter: <b>{props.selectedDay === "all" ? "all" : `+${props.selectedDay}d`}</b> · Province: <b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
