import type { ValidationMetrics } from "../types";

interface Props {
  metrics: ValidationMetrics | null;
  onShowDetails: () => void;
}

// Hero block displayed near the top of the sidebar — the headline accuracy
// number deserves more visual weight than the small 2×2 stats card buried
// later in the scroll. ±1-day accuracy is the most operator-meaningful
// number (was the day burned within ±1 day of the prediction?), so it gets
// the largest font; MAE / RMSE sit underneath as supporting detail.
export default function AccuracyHero({ metrics, onShowDetails }: Props) {
  const acc = metrics?.accuracy_within_1day;
  const mae = metrics?.mae_days;
  const rmse = metrics?.rmse_days;

  const accStr = typeof acc === "number" ? `${(acc * 100).toFixed(1)}%` : "—";
  const maeStr = typeof mae === "number" ? mae.toFixed(2) : "—";
  const rmseStr = typeof rmse === "number" ? rmse.toFixed(2) : "—";

  return (
    <div className="accuracy-hero">
      <div className="accuracy-hero-top">
        <div className="accuracy-hero-label">Model Accuracy (held-out test)</div>
        <button
          className="info-btn"
          type="button"
          onClick={onShowDetails}
          title="View detailed model info + dataset stats"
          aria-label="Model details"
        >
          ⓘ
        </button>
      </div>
      <div className="accuracy-hero-value">{accStr}</div>
      <div className="accuracy-hero-sub">predictions within ±1 day of the real fire</div>
      <div className="accuracy-hero-grid">
        <div className="accuracy-hero-stat" title="Mean Absolute Error in days — how far off, on average, the model's prediction is. Lower is better.">
          <div className="accuracy-hero-stat-label">MAE</div>
          <div className="accuracy-hero-stat-value">{maeStr}<span> d</span></div>
        </div>
        <div className="accuracy-hero-stat" title="Root Mean Squared Error in days — like MAE but penalises big misses more. Lower is better.">
          <div className="accuracy-hero-stat-label">RMSE</div>
          <div className="accuracy-hero-stat-value">{rmseStr}<span> d</span></div>
        </div>
      </div>
      <div className="accuracy-hero-hint">
        Big number = share of test cases where the model's predicted fire date
        was within <b>±1 day</b> of the real date. <b>MAE</b> is the average
        miss in days; <b>RMSE</b> punishes big misses more. Tap <b>ⓘ</b> above
        for full definitions.
      </div>
    </div>
  );
}
