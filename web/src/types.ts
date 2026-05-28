// Mirrors what risk_map.append_geojson writes. The backend is the authority;
// these types only document what the frontend reads.

export type UrgencyLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE";

export interface UrgencyThresholds {
  CRITICAL: number;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
}

export interface ValidationMetrics {
  // Task discriminator — "binary_fire_in_3d" or undefined (legacy regression).
  task?: string;
  imminent_days?: number;
  // Legacy regression metrics — meaningful when task is undefined.
  mae_days?: number;
  rmse_days?: number;
  r2?: number;
  accuracy_within_1day?: number;
  accuracy_exact?: number;
  // Binary classification metrics — meaningful when task = "binary_fire_in_3d".
  roc_auc?: number;
  average_precision?: number;
  binary_accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  best_f1?: number;
  best_threshold?: number;
  precision_at_best_thr?: number;
  recall_at_best_thr?: number;
  precision_at_top_5pct?: number;
  precision_at_top_10pct?: number;
  precision_at_top_20pct?: number;
  // Baseline + uplift framing (test_positive_rate is what a random
  // ranker would achieve on precision@K).
  test_positive_rate?: number;
  uplift_at_top_5pct?: number;
  uplift_at_top_10pct?: number;
  uplift_at_top_20pct?: number;
  // Calibration metrics (added by Phase 0.5 post-calibration).
  // ECE = expected calibration error on test (lower = better).
  // < 0.05 = trustworthy probability; > 0.15 = treat as rank-only.
  ece?: number;
  ece_val_before_calibration?: number;
  ece_val_after_calibration?: number;
  calibration_method?: string;
  // Locked deployment threshold + its metrics. Headline dashboard numbers
  // should come from these (they're the ones an operator actually sees).
  deployment_threshold?: number;
  deployment_precision?: number;
  deployment_recall?: number;
  deployment_f1?: number;
  deployment_accuracy?: number;
  // Reliability bins for the calibration curve plot.
  reliability_bins?: ReliabilityBin[];
  // Where the metrics were evaluated. "full_distribution_test_window" = real
  // class balance (positive ~3-5%); legacy runs evaluated on undersampled.
  evaluated_on?: string;
  // Stability across rolling monthly evaluations (from rolling_eval.json).
  // Tells the operator whether the model's AUC drifts over time.
  stability_months?: number;
  stability_valid_months?: number;
  stability_auc_mean?: number;
  stability_auc_std?: number;
  stability_auc_min?: number;
  stability_auc_max?: number;
  rolling_by_month?: RollingMonthPoint[];
  feature_importance_top?: { feature: string; importance: number }[];
  // ── Scientific statistics (from scripts/scientific_stats.py) ──
  scientific_stats?: ScientificStats;
  // ── Training summary (real persisted values from dataset_info.json) ──
  training_summary?: TrainingSummary;
}

export interface TrainingSummary {
  trained_at?: string;
  data_source?: string;
  date_range?: [string | null, string | null];
  total_days?: number;
  active_cells?: number;
  grid_size_deg?: number;
  training_rows?: number;
  feature_count?: number;
  weather_features_count?: number;
  prediction_type?: string;
  imminent_days?: number;
  training_time_seconds?: number;
  model_type?: string;
  search_method?: string;
  search_iterations?: number;
  cv_n_splits?: number;
  cv_gap_days?: number;
  ensemble_size?: number;
  early_stopping_rounds?: number;
  best_params?: Record<string, number | string | boolean | null>;
}

export interface BootstrapCI {
  point: number;
  lower: number;
  upper: number;
  std: number;
  n_boot: number;
  confidence: number;
}

export interface ScientificStats {
  samples: {
    total_densified: number;
    train: SampleSplit;
    val: SampleSplit;
    test: SampleSplit;
  };
  ci_95: {
    roc_auc: BootstrapCI;
    average_precision: BootstrapCI;
    f1_at_deploy: BootstrapCI;
    precision_at_deploy: BootstrapCI;
    recall_at_deploy: BootstrapCI;
    brier_score: BootstrapCI;
  };
  confusion_matrix: {
    tn: number; fp: number; fn: number; tp: number;
    matrix: number[][];
    row_labels: string[];
    col_labels: string[];
  };
  classification_stats: {
    sensitivity: number;
    specificity: number;
    ppv: number;
    npv: number;
    false_positive_rate: number;
    false_negative_rate: number;
    cohen_kappa: number;
    matthews_corr_coef: number;
    log_loss: number;
    brier_score: number;
    brier_skill_score: number;
    baseline_class_prior: number;
  };
  roc_curve: CurvePoint[];
  pr_curve: CurvePoint[];
}

export interface SampleSplit {
  n: number;
  positives: number;
  positive_rate: number;
  date_range: [string, string];
}

export interface CurvePoint {
  x: number;
  y: number;
  t: number;  // threshold at this point
}

export interface RollingMonthPoint {
  month: string;          // "2025-04"
  auc: number;
  positive_rate: number;
  n: number;
}

export interface ReliabilityBin {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  actual_rate: number;
  count: number;
}

export interface SnapshotHitRate {
  hits?: number;
  misses?: number;
  future?: number;
}

export interface GeoJsonMetadata {
  urgency_thresholds?: UrgencyThresholds;
  metrics?: ValidationMetrics;
  validation_summary?: {
    per_snapshot?: Record<string, SnapshotHitRate>;
  };
}

export interface PredictionProperties {
  source: "predicted" | "observed";
  base_date?: string;
  predicted_fire_date?: string;
  days_until_fire?: number;
  raw_prediction?: number;
  // Real probability (0..1) recovered from the binary classifier — only
  // written by newer risk_map.py runs. Older snapshots omit it and the
  // frontend falls back to inverting raw_prediction.
  probability?: number;
  urgency_level?: UrgencyLevel;
  confidence?: number;
  province?: string;
  historical_fire_count_30d?: number;
  fire_days_per_year?: number;
  tree_cover_pct_2000?: number;
  tree_loss_pct_recent?: number;
  nearest_urban_area?: string;
  nearest_urban_distance_km?: number;
  // observed
  date?: string;
  fire_count?: number;
  // Retrospective validation (set by risk_map.py on past snapshots once the
  // ±1-day window closes):
  //   "hit"    — predicted cell did burn within the window
  //   "miss"   — no fire observed in the window
  //   "future" — prediction window hasn't closed yet
  validation_status?: "hit" | "miss" | "future";
}

export interface FireFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: PredictionProperties;
}

export interface FireGeoJson {
  type: "FeatureCollection";
  features: FireFeature[];
  metadata?: GeoJsonMetadata;
}

// ───────────── Notify / Alert Dispatch ─────────────

export type NotifyChannel = "sms" | "line" | "email" | "all";
export type NotifyPriority = "normal" | "urgent" | "emergency";

export interface NotifyRequest {
  channel: NotifyChannel;
  recipients: string[];
  message: string;
  zone_ids: string[];
  priority: NotifyPriority;
  template?: string;
}

export interface NotifyResponse {
  status: "queued";
  id: string;
  timestamp: string;
}

export interface NotifyLogRecord {
  id: string;
  timestamp: string;
  channel: NotifyChannel;
  recipients_count: number;
  recipients_preview: string[];
  zone_ids_count: number;
  zone_ids_preview: string[];
  priority: NotifyPriority;
  template: string | null;
  message_preview: string;
  status: "queued" | "sent" | "failed";
}

export type AlertPageRoute = "dashboard" | "notify" | "live" | "reports" | "compare" | "analytics";

export type DaySelection = "all" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7";

export interface DisplayOptions {
  showObserved: boolean;
  showLiveFires: boolean;
  showPredicted: boolean;
  showCellPins: boolean;
  heatRadius: number;
}

export type LiveFireStatus = "idle" | "loading" | "ok" | "error";

export interface LiveFireMeta {
  status: LiveFireStatus;
  count: number;
  lastFetch: Date | null;
  error: string | null;
}

// GISTDA ArcGIS feature attributes — typed loosely because the upstream
// schema can vary between the NPP and MODIS endpoints (some fields like
// `satellite` only show up on one).
export interface GistdaFeature {
  attributes: {
    latitude?: number | string;
    longitude?: number | string;
    confident?: string | number;
    lu_name?: string;
    pv_tn?: string;
    ap_tn?: string;
    date?: number;
    time?: string;
    satellite?: string;
  };
}
