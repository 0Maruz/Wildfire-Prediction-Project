# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wildfire **date** prediction system for Thailand (BBOX `96,4,107,22`). Given satellite hotspot history from NASA FIRMS (real data only — no simulation), a tree-ensemble regressor predicts `days_until_fire` (1–7) per spatial grid cell. Predictions are rendered on a Leaflet map with urgency tiers (CRITICAL/HIGH/MEDIUM/LOW).

This was previously a binary risk classifier; the current system is regression-based and predicts *when* a fire will occur. **Do not reintroduce classification semantics.**

### Hard rule: real data only

Every feature must come from a real, measurable source. The pipeline currently consumes:

- **NASA FIRMS VIIRS NRT** (always on) — hotspot detections; powers fire/FRP/brightness/confidence and all spatial-neighbour features.
- **Open-Meteo Archive API** (optional, no key) — real ECMWF ERA5 daily reanalysis (temp_max/min, precip_sum, wind_max, et0). Activated by running `python fetch_weather.py`, which caches to `data/weather/weather_cache.parquet`.
- **Calendar** — derived from each row's real `date`.

No synthetic, simulated, randomly-generated, or interpolated values anywhere. If a real source isn't available for a given column, the column is simply not added to `FEATURES`. Don't introduce fake fallbacks or fabricated defaults — when something is missing, it's missing.

## Common commands

All Python entry points use bare imports of each other (`from features import FEATURES`), so they must be run from inside `src/`:

```bash
# 1. Pull latest VIIRS NRT hotspots from NASA FIRMS into data/firms/firms_all.parquet
cd src && python fetch_firms.py [--days 1-10]

# 1b. (OPTIONAL) Pull real ERA5 weather for every active FIRMS cell into
#      data/weather/weather_cache.parquet. Requires no API key. Skip this and the
#      training pipeline silently runs without weather features.
cd src && python fetch_weather.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit-cells N]

# 2. Train the model (runs the full data → features → tune → save pipeline)
cd src && python train.py [--n-iter 20] [--n-splits 5] [--only lightgbm,xgboost]
#                          [--quick]            ← 10 iter × 3 splits, LGBM only (~3 min)
#                          [--min-confidence 30] ← VIIRS conf floor (default 30)
# Default candidate set is `lightgbm,xgboost`; random_forest is excluded because
# it has lost on val MAE while consuming ~95 % of tuning time. Pass
# `--only random_forest,lightgbm,xgboost` to reinstate.

# `training.py` is a thin shim around `train.main()` for backwards compatibility.

# 3. Regenerate the GeoJSON map without retraining
cd src && python risk_map.py

# 4. Serve the API (FastAPI on :8000)
cd src && uvicorn api:app --reload

# OR: end-to-end orchestrator at the project root (trains, then serves dashboard
# on :8080 and FastAPI on :8000). Flags: --fresh (fetch FIRMS first), --weather,
# --no-train, --open. Anything after `--` is forwarded to train.py.
./run.sh [--fresh] [--weather] [--no-train] [-- --n-iter 30 --only lightgbm]
```

Dependencies: `pip install -r requirements.txt` into `.venv/`. Requires a `.env` file with `FIRMS_API_KEY` (see `.env.example` for the full set). All scripts use `load_dotenv()`.

There is no test suite, linter, or build step.

## Architecture

### Module layout (in `src/`)

| File | Role |
|---|---|
| `fetch_firms.py` | Fetches VIIRS NRT hotspots from NASA FIRMS with retry/backoff; writes accumulative `data/firms/firms_all.parquet`. |
| `fetch_weather.py` | **Optional.** Fetches real ECMWF ERA5 daily aggregates from Open-Meteo Archive (no key) for every active FIRMS cell, caches to `data/weather/weather_cache.parquet`. Idempotent — only fetches missing (cell, date) tuples. |
| `fetch_treecover.py` | **Optional, run-once.** Streams Hansen Global Forest Change v1.11 `treecover2000` + `lossyear` rasters via HTTP range reads, aggregates 30 m source pixels into the project's 0.1° grid, writes per-cell `tree_cover_pct_2000` + `tree_loss_pct_recent` to `data/static/tree_cover_per_cell.parquet`. Adds vegetation context that the model otherwise lacks. |
| `io_utils.py` | Format-agnostic table I/O. `read_table` / `write_table` / `resolve_existing` dispatch on file extension (`.csv` ↔ `.parquet`); `list_tables` resolves dirs/globs and prefers Parquet when both extensions exist for the same basename. **Use these helpers** instead of `pd.read_csv` / `pd.to_csv` so files stay swappable. |
| `data_loader.py` | Pure I/O: loads raw + FIRMS hotspot tables (CSV or Parquet via `io_utils`), cleans, snaps to grid, aggregates to daily cell-day, **densifies active cells** over the date range, drops hotspots inside curated urban areas (`filter_urban_hotspots`), and (if `weather_path` is supplied) left-joins the weather cache. |
| `urban_areas.py` | Curated list of ~40 Thai urban centres with hand-tuned exclusion radii. `classify_urban(lats, lons, buffer_km)` returns `(is_urban, nearest_dist_km, nearest_name)` for any batch of cells. Used at training time (drop urban hotspots) and at inference time (drop urban predictions, annotate "nearest city"). |
| `features.py` | Lag/rolling/calendar + **3×3 spatial-neighbour** feature engineering, label generation, dry-streak counter, expanding-window fire rate, urban-distance feature, and burn-season-distance feature. Owns `FEATURES_CORE` (always-on) and `FEATURES_WEATHER` (only used when ERA5 columns are present). Use `resolve_features(df)` to get the deployed-model contract. |
| `model.py` | Candidate factory (RandomForest, LightGBM, XGBoost), `RandomizedSearchCV` tuner using `TimeSeriesSplit`, evaluation (`MAE`, `RMSE`, `R²`, `acc±1`). |
| `train.py` | Orchestrator: load → features → label → chronological 60/20/20 split → tune candidates (default `lightgbm,xgboost`) → pick best val MAE → held-out test eval + predict-mean baseline + prediction-distribution sanity check → refit on train+val → persist (current model + timestamped snapshot in `outputs/models/history/`) → trigger `risk_map.run()`. |
| `risk_map.py` | Loads the trained model + densified feature CSV, predicts for the latest base date, attaches fixed-domain urgency tier + `historical_fire_count_30d` (real FIRMS) + raw model output, and appends to `fire_dates_all.geojson`. Top-level GeoJSON metadata carries thresholds + held-out test metrics for the frontend. |
| `api.py` | FastAPI on top of the same artifacts — `/predictions/today`, `/predictions/timeline`, `/predictions/day/{n}`, `/predict/location`, `/metrics`, `/geojson`. Reads urgency thresholds from `dataset_info.json`. |

### Pipeline

```
data/raw/*.parquet ─┐
                    ├─► data_loader ─► features ─► train ─► outputs/models/*.pkl
data/firms/        ─┘    (densify)    (lag/roll/   (RF, LGBM, XGB)
firms_all.parquet                      calendar)         │
                                                         ▼
                                            outputs/features/full_features.parquet
                                            outputs/metadata/dataset_info.json
                                                       │
                                                       ▼
                                                 risk_map.py
                                                       │
                                                       ▼
                                       outputs/riskmap/fire_dates_all.geojson
                                                       │
                                       ┌───────────────┴───────────────┐
                                       ▼                               ▼
                                    api.py                      frontend/app.js
                                  (FastAPI)               (fetches geojson directly)
```

### Two parallel data sources, intentionally merged

`data_loader.load_and_prepare()` ingests **two separate hotspot sources** and concatenates them:
- `RAW_DIR` (`data/raw/*.parquet` or `*.csv`) — historical bulk archive
- `FIRMS_PATH` (`data/firms/firms_all.parquet` or `.csv`) — NRT data accumulated by `fetch_firms.py`

Both formats are read transparently via `io_utils.list_tables` / `read_table` (Parquet preferred when both extensions exist for the same basename). Files are gridded to `GRID_SIZE` (default 0.1°) cells via `(coord / GRID).round() * GRID`, then aggregated to one row per `(lat_grid, lon_grid, date)`.

### Densification (important)

After aggregation, `data_loader.densify_active_cells()` expands the sparse fire-only frame into a **dense (active-cell × every-day-in-range)** grid, with no-fire days filled with zeros. This is what makes lag features (`fire_lag_1` = literally yesterday) and rolling windows (`fire_sum_7d` = last 7 calendar days) correct. Without densification, "yesterday" would mean "the previous day this cell happened to burn", which is a major source of bias in the original system. **Inactive cells (zero fires ever)** are excluded — they have no signal to learn from.

### Label semantics

`features.make_label_days_until_fire()` walks each cell's densified history and labels each row with the number of days until the next fire (1–7). Rows with no fire in the next 7 days get `-1` and are dropped from training. This means **the model only learns from cells that did go on to burn**; at inference time it still emits a 0–7 value for every cell, which the urgency mapping turns into CRITICAL/HIGH/MEDIUM/LOW.

### Feature contract — single source of truth

`features.py` exposes two tuples and a resolver:

- `FEATURES_CORE` (~57 columns) — always present:
  - Lags at 1/2/3/7/14/30 days, rolls at 3/7/14/30 days, active-day counts, FRP trend.
  - Current-day signals (`fire_count_today`, `frp_sum_today`, `bright_mean_today`, `confidence_mean_today`).
  - Cyclic month/DOY, burn-season flag, `days_from_burn_peak` (distance from DOY 75, Thailand's peak burn day).
  - **3×3 spatial-neighbour** fire/FRP aggregates with their own lags & rolls.
  - **Tier-1 cell features** (added 2026-05): `distance_to_nearest_city_km` (static, from `urban_areas.classify_urban`), `fire_days_per_year_so_far` (expanding window, no leakage — uses only data before each row's date), `days_since_last_fire` (causal dry-streak counter, resets on each fire-day).
  - **Vegetation features** (when `data/static/tree_cover_per_cell.parquet` exists from `fetch_treecover.py`): `tree_cover_pct_2000` (Hansen GFC baseline canopy density), `tree_loss_pct_recent` (% pixels in cell that lost forest 2018-2023). Distinguishes "what's there to burn" — dense forest cells behave very differently from grassland given identical fire history.
  - `lat_grid`, `lon_grid` — coarse spatial encoding.
- `FEATURES_WEATHER` (~30 columns) — only emitted when ERA5 weather columns are present in the daily frame. Per-variable today + lags 1/3/7 + rolls 3/7 over `temp_max` / `temp_min` / `precip_sum` / `wind_max` / `et0`. Auto-dropped at training time when coverage falls below `MIN_WEATHER_COVERAGE` (default 20 %, configurable via env) — a sparse weather cache is more harmful than absent weather, since lag/roll features collapse to NaN→0 noise.
- `resolve_features(df)` — returns the actually-present subset given a feature dataframe. **Use this** (or `dataset_info.json["features"]` after training) instead of hardcoding a list.

`api.py` and `risk_map.py` resolve the feature list at runtime by reading `outputs/metadata/dataset_info.json["features"]` first (matches the deployed model exactly), and fall back to `resolve_features(df)`. **Don't re-introduce hardcoded feature lists in those files.**

### Spatial neighbours (real FIRMS, not synthetic)

`features.add_neighbor_features()` sums `fire_count` and `frp_sum` across the 8 cells surrounding each `(lat_grid, lon_grid)` for the same date. Implementation shifts the source frame's coords by negative offsets so each row's `(lat_grid, lon_grid)` becomes the *target* cell whose neighbour it is, then merges. Pure aggregation — no smoothing, no interpolation, zero when a neighbour cell has no detection.

Spatial features must be computed *before* `add_temporal_features()` so the lags/rolls of `neighbor_fire_today` / `neighbor_frp_today` exist downstream.

### Urgency thresholds — fixed-domain, consistent across surfaces

`train.py` persists fixed-domain cutoffs (`CRITICAL=0, HIGH=2, MEDIUM=4, LOW=7`) into `dataset_info.json["urgency_thresholds"]`. `api.py`, `risk_map.py`, and the frontend all read from this single source. **Do not re-introduce quantile/percentile recalibration in any consumer** — earlier versions of `risk_map.py` did, which made the dashboard's tier counts look balanced (~25 % each) at the cost of (a) hiding bunched-prediction pathologies, and (b) drifting away from the API's view of the same data. The fixed cutoffs let `NONE` actually appear (predictions > 7d) and surface model bunching honestly.

`features.urgency_from_thresholds(days, thresholds)` is the canonical mapping. `features.calibrate_urgency_thresholds(...)` and `get_urgency()` remain in the codebase as backwards-compat helpers but are no longer wired into the pipeline.

### Model selection

`train.chronological_split()` produces a 3-way **60 / 20 / 20** train / val / test split sorted by date — train tunes, val selects, test is held out and never seen during tuning or selection.

`model.select_best()` runs `RandomizedSearchCV` (default 20 iterations, 5 folds) for each candidate against a `TimeSeriesSplit` — never a random shuffle, since rows are temporally ordered. Selection criterion is best **validation** MAE. After selection, the winner is evaluated **once** on the held-out test set (`test_metrics` in `dataset_info.json`), then refit on train+val and saved to `outputs/models/lgbm_fire_date_model.pkl`. The filename is historical (kept stable for `api.py` / `risk_map.py`) — the actual model class can be RandomForest, LightGBM, or XGBoost; `dataset_info.json["best_model"]` tells you which. The deployed artefact is **not** re-evaluated after the train+val refit, to avoid leakage.

The default candidate set is **`lightgbm,xgboost`** — RandomForest is excluded because (a) it consistently loses on val MAE and (b) it consumes ~95 % of total tuning wall-clock time. Pass `--only random_forest,lightgbm,xgboost` to restore the three-way contest.

`--quick` is the iteration default: `--n-iter 10 --n-splits 3 --only lightgbm` (~3 minutes end-to-end). Use it when prototyping feature changes; switch back to the full search before deploying.

If you change the candidate pool or feature list, **delete `outputs/models/*.pkl` before retraining** to avoid loading a stale artifact.

### Training-time safeguards

`train.py` runs three sanity checks during STEP 6 and persists their results into `dataset_info.json` so downstream consumers can flag misbehaving models:

1. **Predict-mean baseline** — `mean(y_train)` is broadcast across the test set and scored. If `mae_improvement_over_baseline_pct < 5 %`, a `⚠️  MODEL SKILL CHECK FAILED` warning fires. A model that barely beats "predict the prior" is essentially useless and shouldn't be deployed without investigating feature signal / target framing.
2. **Prediction distribution check** — min/p25/median/p75/max/std of test predictions. If `IQR < 0.5 days`, predictions are flagged as `predictions_bunched: true`. Bunched output collapses every cell into the same urgency tier on the dashboard regardless of the threshold scheme.
3. **Versioned snapshot** — every training run also dumps `outputs/models/history/{UTC_TIMESTAMP}_{best_name}.pkl` alongside the canonical `lgbm_fire_date_model.pkl`, so older artefacts can be A/B'd or rolled back without retraining.

### Urban exclusion (training-time and inference-time)

`urban_areas.THAI_URBAN_AREAS` is a hand-curated list of ~40 Thai cities with hand-tuned exclusion radii. The same list is consulted in two places:

- **Training (`data_loader.filter_urban_hotspots`)** — drops raw FIRMS detections that fall inside any city's radius before gridding, so the model never learns "wildfire patterns" from garbage burning, industrial heat, or rooftop hotspots. Default-on; controlled via `URBAN_FILTER_ENABLED` / `URBAN_BUFFER_KM` env vars.
- **Inference (`risk_map.py`)** — drops cells whose centre falls inside the same exclusion zones from the dashboard, plus annotates every kept cell with `nearest_urban_area` / `nearest_urban_distance_km` for the popup tooltip.

The training-time filter typically removes only 0.1 % of hotspots (~370 of 437k for Thailand 2025), but those few rows account for a disproportionate share of false-positive predictions in the dashboard — most cells inside major cities have hundreds of detections per year that a model otherwise treats as a strong "fire-prone" signal.

### "confidence" is a rounding proxy, not a probability

Both `api.py` (`_rounding_confidence`) and `risk_map.py` (`prediction_confidence`) compute `1 - |raw_pred - rounded_pred|`. This is **not** a calibrated likelihood — it only reflects how close the regressor's continuous output landed to a whole number. Don't treat it as a probability in downstream UI or aggregations, and don't add fake calibration without changing the underlying model. The frontend tooltip labels it "rounding proxy" for this reason.

### Frontend day selector

`frontend/app.js` exposes a Day 1–7 selector that filters predicted markers by the model's clipped `days_until_fire` integer — pure filtering of real outputs, no smoothing or interpolation. Clicking a row in the timeline also triggers the same filter. The validation-metrics panel reads `metadata.metrics` written into the GeoJSON by `risk_map.append_geojson` (sourced from `dataset_info.json["model"]["test_metrics"]`).

### Frontend marker rendering — meters with pixel clamps

Markers use `L.circle` (radius in **meters**) so they scale naturally with zoom. Because a 0.4-fraction CRITICAL dot at grid 0.1° works out to ~2.2 km, raw meters-based scaling produces unusable extremes: ~1 px at zoom 6 (invisible) and ~900 px at zoom 14 (covers the whole map). `_clampedRadiusMeters(lat, baseM, minPx, maxPx)` plus a `map.on("zoomend", _reclampAllMarkers)` listener keep every dot within `[3 px, 28 px]` (CRITICAL — others scaled proportionally to their `URGENCY_DOT_FRAC`). Mid-zoom levels get the natural meters-based scaling; only the extremes are clamped.

### Frontend heatmap — nearest-cell IDW

The "smooth surface" layer is an IDW grid built per Leaflet tile. Per-pixel **alpha** still falls off with distance for soft circular edges, but per-pixel **colour** comes from the *nearest* cell's `raw_prediction`, not a weighted average — averaging caused tier-mismatches around tier boundaries (e.g. green LOW marker surrounded by yellow MEDIUM smear because the IDW interpolated through the tier cutoff). With nearest-cell colouring, the dashboard always renders the marker's tier colour through that cell's full surface footprint.

### Frontend reads GeoJSON directly, not the API

`frontend/app.js` fetches `../outputs/riskmap/fire_dates_all.geojson` via a **relative path** — it does not hit `api.py`. To view the dashboard, the frontend must be served such that `../outputs/...` resolves (e.g. serve the project root, then open `/frontend/index.html`). The FastAPI server exists for programmatic access but is not what the dashboard depends on.

### GeoJSON is appended, not overwritten

`risk_map.append_geojson` reads the existing `fire_dates_all.geojson`, strips out predictions whose `base_date` matches the current run (preserving `source: "observed"` features), and appends the new observed + predicted features. Historical predictions for prior base dates accumulate. Delete the file to reset.

### Path resolution

All entry points resolve paths via `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` so they work regardless of cwd. `train.py` additionally reads `RAW_DIR` / `FIRMS_PATH` / `OUTPUT_DIR` from `.env`, and **resolves any relative env values against `BASE_DIR` via `_resolve()`** — so `RAW_DIR=./data/raw` in `.env.example` works whether you launch from the project root or from `src/`. If you set absolute paths in `.env`, they pass through unchanged.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `FIRMS_API_KEY` | (required) | NASA FIRMS map key (see `.env.example`). |
| `RAW_DIR` | `./data/raw` | Bulk archive of historical FIRMS exports. |
| `FIRMS_PATH` | `./data/firms/firms_all.parquet` | NRT cache populated by `fetch_firms.py`. |
| `WEATHER_DIR` | `./data/weather` | Cache dir for `fetch_weather.py`. |
| `OUTPUT_DIR` | `./outputs` | Models, features, metadata, GeoJSON. |
| `GRID_SIZE` | `0.1` | Lat/lon snap resolution in degrees. |
| `URBAN_FILTER_ENABLED` | `true` | Drop urban hotspots at training and inference. |
| `URBAN_BUFFER_KM` | `0` | Extra km beyond each city's hand-tuned radius. |
| `COUNTRY_FILTER_ENABLED` | `true` | Drop predicted cells outside Thailand's land border at risk_map time. |
| `STALE_WARN_DAYS` | `5` | Warn (and persist `data_is_stale=true`) when latest FIRMS observation is older than this. |
| `MIN_WEATHER_COVERAGE` | `0.20` | Below this share of non-NaN weather rows, training auto-drops weather columns. |
| `MIN_HISTORICAL_FIRES_FOR_DISPLAY` | `1` | risk_map.py filter: minimum fires in last 30d. |
| `MIN_LONG_HISTORICAL_FIRES` | `3` | risk_map.py filter: minimum fires in last 90d. |
| `MIN_FIRE_DAYS_PER_YEAR` | `1.0` | risk_map.py filter: annualized fire-day rate floor. |

### Gotchas

- **`fetch_firms.py` HTTP 400 across all datasets** = bad / rate-limited `MAP_KEY`. Check status via `https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=…`; FIRMS resets the daily transaction limit roughly every 24h. Training does not require a successful fetch — it can run on whatever is already in `data/raw/` + `data/firms/firms_all.parquet`.
- **`uvicorn api:app` startup `RuntimeError: Model not found`** = no trained artifact yet. Run `train.py` to completion first; the API loads `outputs/models/lgbm_fire_date_model.pkl` at startup and refuses to serve without it.
- **Stale model after feature changes**: if you add/remove anything in `FEATURES_CORE`/`FEATURES_WEATHER`, or you start/stop running `fetch_weather.py`, delete `outputs/models/*.pkl` before retraining. `api.py` and `risk_map.py` resolve the feature list from `dataset_info.json` to stay in sync, but a leftover `.pkl` from a different feature contract will silently mispredict.
- **Weather cache lag**: Open-Meteo's archive endpoint trails real-time by ~5 days (ERA5T preliminary release). `fetch_weather.py` automatically caps `end_date` at `today - 5d` — for the most recent days, weather columns will be NaN and `features.add_temporal_features` fills NaN with 0 only at model-input time. The cache itself preserves the genuine missing-data signal; do not impute.
- **Sparse weather → auto-skipped**: if `MIN_WEATHER_COVERAGE` (default 20 %) is not met, `train.py` drops the weather columns from the daily frame *before* feature engineering, so the model trains on `FEATURES_CORE` only. With a partial cache (e.g. 8 % coverage after a rate-limited fetch), this is **better** than training with the columns — sparse weather lags collapse into a 30-column block of zero-noise. To reinstate weather features after a partial fetch, complete the cache and retrain (or set `MIN_WEATHER_COVERAGE=0` to override).
- **Skill check failed**: if `train.py` logs `⚠️  MODEL SKILL CHECK FAILED`, the model beats the predict-mean baseline by less than 5 %. `dataset_info.json["model"]["skill_check_passed"]` is `false`. Don't ship this artefact — investigate feature signal, target framing, or data quality first.
- **Predictions bunched**: similarly, `predictions_bunched: true` in metadata means the test-set IQR is < 0.5 days. Every cell will land in the same urgency tier on the dashboard. Either richer features are needed, or the regression target is too hard for the available signal.
- **Open-Meteo 429s**: the archive endpoint is rate-limited per hour (~5,000 location-calls) and per day (~10,000). `fetch_weather.py` is idempotent — completed (cell, date) tuples in the cache are skipped on the next run, so just wait an hour and re-run. The optimised default (`--workers 10 --batch-size 1`) is conservative; pushing `--batch-size 100` cuts wall-clock at the cost of more 429s — useful only when the quota is freshly reset.
