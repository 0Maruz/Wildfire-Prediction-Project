import type {
  FireGeoJson,
  NotifyLogRecord,
  NotifyRequest,
  NotifyResponse,
  RollingMonthPoint,
  TrainingSummary,
} from "./types";

// In dev the Vite proxy forwards /geojson → http://localhost:8000.
// In prod FastAPI serves the SPA from the same origin, so /geojson is
// already on the right host. VITE_API_BASE lets a deployer override
// (e.g. point the static SPA at a separate API host) without rebuilding.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchGeoJson(): Promise<FireGeoJson> {
  const res = await fetch(`${API_BASE}/geojson`);
  if (!res.ok) throw new Error(`GeoJSON fetch failed: HTTP ${res.status}`);
  return res.json();
}

export interface RollingEvalResponse {
  summary: { auc_mean?: number; auc_std?: number; auc_min?: number; auc_max?: number; valid_months?: number } | null;
  months: RollingMonthPoint[];
}

export async function fetchRollingEval(): Promise<RollingEvalResponse> {
  const res = await fetch(`${API_BASE}/api/rolling-eval`);
  if (!res.ok) throw new Error(`Rolling-eval fetch failed: HTTP ${res.status}`);
  return res.json();
}

export async function fetchTrainingSummary(): Promise<TrainingSummary> {
  const res = await fetch(`${API_BASE}/api/training-summary`);
  if (!res.ok) throw new Error(`Training summary fetch failed: HTTP ${res.status}`);
  return res.json();
}

// ── Live status (used by the header pulse badge) ──
export type LiveStatus = "live" | "stale" | "offline";

export async function fetchHealth(timeoutMs = 4000): Promise<LiveStatus> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return "offline";
    const body = await res.json().catch(() => ({}));
    // /health returns { data_stale_days, ... }. If FIRMS data > 5 days old
    // we surface "stale" so operators see a yellow badge.
    if (typeof body?.data_stale_days === "number" && body.data_stale_days > 5) {
      return "stale";
    }
    return "live";
  } catch {
    return "offline";
  }
}

// ── Notify dispatch + log ──
export async function postNotify(req: NotifyRequest): Promise<NotifyResponse> {
  const res = await fetch(`${API_BASE}/api/notify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Notify POST failed: HTTP ${res.status} ${body}`);
  }
  return res.json();
}

export async function fetchNotifyLog(limit = 100): Promise<NotifyLogRecord[]> {
  const res = await fetch(`${API_BASE}/api/notify/log?limit=${limit}`);
  if (!res.ok) throw new Error(`Notify log fetch failed: HTTP ${res.status}`);
  const body = await res.json();
  return body.records ?? [];
}

// ── Per-cell weather (ERA5) + Fire Weather Index ──
export interface CellWeather {
  available: boolean;
  reason?: string;
  source?: string;
  lat_grid?: number;
  lon_grid?: number;
  date?: string;
  temp_min_c?: number | null;
  temp_max_c?: number | null;
  precip_sum_mm?: number | null;
  wind_max_kmh?: number | null;
  et0_mm?: number | null;
  precip_7d_mm?: number | null;
  fire_weather_index?: { level: string; color: string; emoji: string };
}

export async function fetchCellWeather(
  lat: number, lon: number, date?: string,
): Promise<CellWeather> {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon) });
  if (date) params.set("date", date);
  const res = await fetch(`${API_BASE}/api/cell_weather?${params}`);
  if (!res.ok) return { available: false, reason: `HTTP ${res.status}` };
  return res.json();
}
