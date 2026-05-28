// Recover the real fire probability from the pseudo-days value train.py
// persists into `raw_prediction`.
//
// train.py uses a PIECEWISE-LINEAR mapping from probability → pseudo-days
// (see _prob_to_days_for_compat in src/train.py):
//
//      prob     pseudo-days
//      1.00          0.0
//      0.70          0.5
//      0.50          1.5
//      0.35          2.5
//      0.20          4.0
//      0.10          5.5
//      0.00          7.0
//
// The previous frontend code computed prob = 1 − (days − 1) / 6, which is
// the wrong inverse — it assumes a single linear segment from prob=1→day=1
// through prob=0→day=7 and produces 100% for any days ≤ 1, hiding the
// model's actual confidence.
//
// This helper inverts the real piecewise mapping so popups read the
// probability the model actually emitted.

const DAYS_ANCHORS  = [0.0, 0.5, 1.5, 2.5, 4.0, 5.5, 7.0];
const PROB_ANCHORS  = [1.0, 0.70, 0.50, 0.35, 0.20, 0.10, 0.00];

export function rawPredToProb(rawPred: number | undefined | null): number | null {
  if (typeof rawPred !== "number" || !isFinite(rawPred)) return null;
  if (rawPred <= DAYS_ANCHORS[0]) return PROB_ANCHORS[0];
  const last = DAYS_ANCHORS.length - 1;
  if (rawPred >= DAYS_ANCHORS[last]) return PROB_ANCHORS[last];
  for (let i = 1; i < DAYS_ANCHORS.length; i++) {
    if (rawPred <= DAYS_ANCHORS[i]) {
      const t = (rawPred - DAYS_ANCHORS[i - 1]) / (DAYS_ANCHORS[i] - DAYS_ANCHORS[i - 1]);
      const p = PROB_ANCHORS[i - 1] + t * (PROB_ANCHORS[i] - PROB_ANCHORS[i - 1]);
      return Math.max(0, Math.min(1, p));
    }
  }
  return 0;
}

// Newer snapshots (risk_map.py from May 2026+) write the real probability
// straight into feature properties. Prefer it; fall back to the piecewise
// inverse for older snapshots that only have raw_prediction.
export function readProbability(props: {
  probability?: number;
  raw_prediction?: number;
} | null | undefined): number | null {
  if (!props) return null;
  if (typeof props.probability === "number" && isFinite(props.probability)) {
    return Math.max(0, Math.min(1, props.probability));
  }
  return rawPredToProb(props.raw_prediction);
}
