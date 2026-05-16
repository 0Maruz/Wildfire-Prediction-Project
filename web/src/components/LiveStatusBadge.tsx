import { useEffect, useState } from "react";
import { fetchHealth, type LiveStatus } from "../api";

// Header badge showing real-time API health. Polls /health every 60s with a
// 4-second timeout — if the backend doesn't respond we degrade to "offline"
// gracefully. Pulse animation only on "live" so a frozen page doesn't lie.
//
// "stale" appears when FIRMS data is > 5 days old (the backend reports
// data_stale_days in /health).

const POLL_INTERVAL_MS = 60_000;

export default function LiveStatusBadge() {
  const [status, setStatus] = useState<LiveStatus>("offline");
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const s = await fetchHealth();
      if (cancelled) return;
      setStatus(s);
      setLastCheck(new Date());
    };
    tick();
    const id = window.setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const label =
    status === "live" ? "LIVE" :
    status === "stale" ? "STALE" : "OFFLINE";
  const description =
    status === "live" ? "API ตอบรับ + ข้อมูลสด" :
    status === "stale" ? "API ตอบรับ แต่ข้อมูลเก่าเกิน 5 วัน" :
    "ติดต่อ API ไม่ได้";

  return (
    <div
      className={`live-status ${status}`}
      title={`${description}\nLast check: ${lastCheck?.toLocaleTimeString() ?? "—"}`}
      aria-live="polite"
    >
      <span className={`live-dot ${status}`} aria-hidden="true" />
      <span className="live-label">{label}</span>
    </div>
  );
}
