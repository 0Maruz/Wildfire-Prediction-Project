import type {
  FireGeoJson,
  NotifyLogRecord,
  NotifyRequest,
  NotifyResponse,
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
