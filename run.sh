#!/usr/bin/env bash
# =====================================================================
# 🔥 Fire Date Predictor — one-shot runner
#
# Real data only. No simulated values, no fake fallbacks.
#
# Default behaviour:
#   1. Train on whatever real FIRMS data is already in data/raw + data/firms/
#   2. Refresh outputs/riskmap/fire_dates_all.geojson (real model output)
#   3. Serve the FastAPI (which also serves the built React SPA at /) on
#                                      http://localhost:8000/
#
# Optional flags fetch the latest real hotspots and/or real ERA5 weather
# before training. Skip training entirely with --no-train.
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FETCH_FIRMS=0
FETCH_WEATHER=0
FETCH_GISTDA=0
DO_TRAIN=1
PREDICT_ONLY=0
OPEN_BROWSER=0
WEATHER_ARGS=()
TRAIN_ARGS=()
API_PORT=8000

usage() {
  cat <<EOF
Usage: $0 [flags] [-- <extra args forwarded to train.py>]

Real-data pipeline orchestrator. Runs train.py (which triggers risk_map.run()
inside it) and then serves both the dashboard and the FastAPI.

Flags:
  --fresh                Fetch latest NASA FIRMS VIIRS NRT hotspots first
                         (requires FIRMS_API_KEY in .env). Falls back to the
                         cached data if the fetch fails.
  --weather              Run fetch_weather.py to pull real ERA5 daily
                         aggregates from Open-Meteo (no API key) before
                         training so weather features are included.
  --gistda               Run fetch_gistda_hotspots.py (GISTDA ArcGIS REST,
                         no key) → data/gistda/gistda_hotspots.parquet before
                         training. Not yet merged into model features — cache only.
  --weather-arg <ARG>    Pass an extra arg to fetch_weather.py (repeatable).
                         e.g. --weather-arg --limit-cells --weather-arg 50
  --no-train             Skip training; just start the servers using whatever
                         is already in outputs/.
  --predict-only         Run train.py --predict-only: rebuild full_features from
                         latest FIRMS (+ weather cache if present), refresh GeoJSON,
                         leave the existing model .pkl unchanged. Overrides --no-train
                         for the train step. Pair with --fresh / --weather for daily ops.
  --open                 Open the dashboard in the default browser when ready.
  --quick                Shortcut: forward --quick to train.py (~8–15 min run,
                         LightGBM only). Useful for iteration.
  --fast                 Forward --fast to train.py: LightGBM only, 35×3 CV
                         (faster than default two-booster 100×5, good overnight-lite).
  --api-port  N          Port for FastAPI (also serves the SPA, default 8000).
  -h, --help             Show this message.

Anything after a literal "--" is forwarded verbatim to train.py, e.g.:
  $0 --fresh -- --n-iter 30 --only lightgbm,xgboost

Real data sources:
  • NASA FIRMS VIIRS NRT  (always)
  • Open-Meteo ERA5        (only if --weather is used or cache exists)
  • GISTDA hotspots        (optional --gistda; cache only until wired to training)
EOF
}

# ── Parse flags ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)        FETCH_FIRMS=1; shift ;;
    --weather)      FETCH_WEATHER=1; shift ;;
    --gistda)       FETCH_GISTDA=1; shift ;;
    --weather-arg)  WEATHER_ARGS+=("$2"); shift 2 ;;
    --no-train)     DO_TRAIN=0; shift ;;
    --predict-only) PREDICT_ONLY=1; shift ;;
    --open)         OPEN_BROWSER=1; shift ;;
    --quick)        TRAIN_ARGS+=("--quick"); shift ;;
    --fast)         TRAIN_ARGS+=("--fast"); shift ;;
    --api-port)     API_PORT="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; TRAIN_ARGS+=("$@"); break ;;
    *)              echo "Unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ── Activate virtualenv if present ──────────────────────────────────────
if [[ -d ".venv" ]]; then
  echo "→ Activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Sanity-check Python deps without installing anything (no surprise side effects)
if ! python -c "import fastapi, uvicorn, pandas, sklearn, joblib" >/dev/null 2>&1; then
  cat <<'EOF' >&2
❌ Missing Python deps. Run once:
     python -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt
EOF
  exit 1
fi

# ── Pre-flight: how stale is the FIRMS cache? ────────────────────────────
# If the most recent hotspot date in data/firms/firms_all.{parquet,csv} is
# more than STALE_DAYS old, auto-promote --fresh so we pull new data before
# training. This keeps daily runs from silently using week-old hotspots.
STALE_DAYS="${STALE_DAYS:-2}"
STALENESS=$(python <<PY 2>/dev/null || echo "unknown"
import os, sys
sys.path.insert(0, "src")
from datetime import date
import pandas as pd
from io_utils import resolve_existing
p = resolve_existing(os.getenv("FIRMS_PATH", "data/firms/firms_all.parquet"))
if not p:
    print("missing"); sys.exit(0)
df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
if "acq_datetime" not in df.columns or df.empty:
    print("missing"); sys.exit(0)
latest = pd.to_datetime(df["acq_datetime"]).max().date()
lag = (date.today() - latest).days
print(f"{latest}|{lag}")
PY
)

if [[ "$STALENESS" == "missing" ]]; then
  echo "→ No FIRMS cache found — promoting --fresh so we fetch initial data."
  FETCH_FIRMS=1
elif [[ "$STALENESS" == "unknown" ]]; then
  echo "  ⚠️  Could not determine FIRMS cache freshness; continuing without auto-refresh."
else
  LATEST_DATE="${STALENESS%|*}"
  LAG_DAYS="${STALENESS#*|}"
  echo "→ FIRMS cache: latest=${LATEST_DATE} (${LAG_DAYS} day(s) behind today)"
  if [[ "${LAG_DAYS}" -gt "${STALE_DAYS}" && $FETCH_FIRMS -eq 0 ]]; then
    echo "  ⚠️  Data is >${STALE_DAYS} days stale → promoting --fresh."
    echo "      Set STALE_DAYS=999 to disable, or run with --fresh explicitly."
    FETCH_FIRMS=1
  fi
fi

# ── Stage 1: optional FIRMS fetch (real NASA data) ──────────────────────
if [[ $FETCH_FIRMS -eq 1 ]]; then
  echo "→ Fetching latest FIRMS VIIRS NRT hotspots …"
  if ! (cd src && python fetch_firms.py); then
    echo "  ⚠️  FIRMS fetch failed (likely missing/exhausted FIRMS_API_KEY)."
    echo "      Continuing with whatever real data is already cached."
  fi
fi

# ── Stage 2: optional ERA5 weather fetch (real Open-Meteo data) ─────────
if [[ $FETCH_WEATHER -eq 1 ]]; then
  echo "→ Fetching real ERA5 daily weather (Open-Meteo Archive) …"
  (cd src && python fetch_weather.py "${WEATHER_ARGS[@]}")
fi

if [[ $FETCH_GISTDA -eq 1 ]]; then
  echo "→ Fetching GISTDA hotspots (no API key) …"
  if ! (cd src && python fetch_gistda_hotspots.py); then
    echo "  ⚠️  GISTDA fetch failed — continuing (see docs/DATA_APIS_TH.md)."
  fi
fi

# ── Stage 3: full train OR predict-only refresh (features + risk map) ───
if [[ $PREDICT_ONLY -eq 1 ]]; then
  echo "→ predict-only: refresh features + risk map (no model tuning) …"
  # Memory guard: feature engineering on the full 4.4M-row densified frame
  # peaks ~18 GB in pandas, which OOM-kills predict-only on a 22 GB laptop
  # with a browser/IDE open. Default the predict window to the last 180 days
  # (~1.6 M rows, peak ~7 GB) unless the user explicitly overrode it.
  EXTRA_TRAIN_ARGS=()
  if ! printf '%s\n' "${TRAIN_ARGS[@]}" | grep -q -- "--max-history-days"; then
    if [[ -z "${MAX_TRAIN_HISTORY_DAYS:-}" || "${MAX_TRAIN_HISTORY_DAYS:-0}" == "0" ]]; then
      EXTRA_TRAIN_ARGS+=("--max-history-days" "180")
      echo "   (memory guard: --max-history-days 180  · override with --max-history-days N)"
    fi
  fi
  (cd src && python train.py --predict-only "${EXTRA_TRAIN_ARGS[@]}" "${TRAIN_ARGS[@]}")
elif [[ $DO_TRAIN -eq 1 ]]; then
  echo "→ Training on real FIRMS data (+ ERA5 weather if cached) …"
  (cd src && python train.py "${TRAIN_ARGS[@]}")
fi

# ── Sanity check artefacts before serving ───────────────────────────────
if [[ ! -f "outputs/models/lgbm_fire_date_model.pkl" ]]; then
  cat <<'EOF' >&2
❌ No trained model at outputs/models/lgbm_fire_date_model.pkl
   Run without --no-train, or place a trained artefact there first.
EOF
  exit 1
fi
if [[ ! -f "outputs/riskmap/fire_dates_all.geojson" ]]; then
  echo "→ GeoJSON missing; running risk_map.py to generate it from real data …"
  (cd src && python risk_map.py)
fi

# ── Stage 4: serve the FastAPI (which serves the built React SPA at /) ─────
# The legacy /frontend/ static-server step is gone; the React app lives in
# /web and is built into /web/dist, which FastAPI mounts at root via
# src/api.py's StaticFiles fallback. One process serves the dashboard +
# the API on the same origin.
API_URL="http://localhost:${API_PORT}/"

cleanup() {
  echo
  echo "→ Shutting down API …"
}
trap cleanup EXIT INT TERM

if [[ ! -d "web/dist" ]]; then
  echo "→ Building React dashboard (web/) for the first time …"
  (cd web && npm install --silent && npm run build)
fi

if [[ $OPEN_BROWSER -eq 1 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    (sleep 1 && xdg-open "${API_URL}") &
  elif command -v open >/dev/null 2>&1; then
    (sleep 1 && open "${API_URL}") &
  fi
fi

cat <<EOF

✅ Ready.
   Dashboard:  ${API_URL}
   API:        ${API_URL}
   API docs:   ${API_URL}docs

   All values shown are derived from real data sources only:
     • NASA FIRMS VIIRS NRT hotspots
     • Open-Meteo ERA5 reanalysis (if cached)
     • Calendar derived from real dates
   No synthetic, simulated, or interpolated values.

   Press Ctrl+C to stop both servers.

EOF

# Foreground FastAPI — Ctrl+C tears down the HTTP server via the trap.
(cd src && exec python -m uvicorn api:app --host 0.0.0.0 --port "${API_PORT}")
