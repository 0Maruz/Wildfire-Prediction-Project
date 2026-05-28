import { useState } from "react";
import type { FireFeature, ValidationMetrics } from "../types";
import { readProbability } from "../utils/probability";

interface Props {
  predicted: FireFeature[];
  metrics: ValidationMetrics | null;
  onShowDetails: () => void;
  onShowAlertSettings: () => void;
  baseDate: string;
}

// Top-of-sidebar action bar — quick operator actions:
//   • Export the currently-visible cells as CSV (operator field-day handoff)
//   • Share the current view as a copy-paste URL
//   • Open the alert-settings modal to play with thresholds
//   • Open the technical details modal
export default function ActionToolbar({
  predicted, metrics, onShowDetails, onShowAlertSettings, baseDate,
}: Props) {
  const [shareStatus, setShareStatus] = useState<string | null>(null);

  const handleExport = () => {
    if (!predicted.length) return;
    const headers = [
      "lat", "lon", "predicted_fire_date", "days_until_fire", "probability_pct",
      "urgency_level", "historical_fire_count_30d", "nearest_urban_area",
      "nearest_urban_distance_km", "province",
    ];
    const rows = predicted.map((f) => {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const prob = readProbability(p);
      return [
        lat.toFixed(4),
        lon.toFixed(4),
        p.predicted_fire_date ?? "",
        p.days_until_fire ?? "",
        prob != null ? (prob * 100).toFixed(1) : "",
        p.urgency_level ?? "",
        p.historical_fire_count_30d ?? "",
        p.nearest_urban_area ?? "",
        p.nearest_urban_distance_km != null ? Number(p.nearest_urban_distance_km).toFixed(1) : "",
        p.province ?? "",
      ].map((v) => {
        const s = String(v);
        return s.includes(",") ? `"${s}"` : s;
      }).join(",");
    });
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fire_predictions_${baseDate}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleShare = async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setShareStatus("✓ คัดลอกแล้ว");
      setTimeout(() => setShareStatus(null), 2000);
    } catch {
      setShareStatus("❌ คัดลอกไม่ได้");
      setTimeout(() => setShareStatus(null), 2000);
    }
  };

  const isCalibrated = metrics?.calibration_method != null;

  return (
    <div className="action-toolbar">
      <button
        className="action-btn"
        type="button"
        onClick={handleExport}
        disabled={!predicted.length}
        title={predicted.length ? `Export ${predicted.length} cells as CSV` : "ไม่มี cells ให้ export"}
      >
        <span className="action-btn-icon">📥</span>
        <span>Export CSV</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={handleShare}
        title="คัดลอก URL ปัจจุบัน"
      >
        <span className="action-btn-icon">🔗</span>
        <span>{shareStatus ?? "Share"}</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={onShowAlertSettings}
        title="ปรับ alert threshold + ดูผล"
        disabled={!isCalibrated}
      >
        <span className="action-btn-icon">⚙️</span>
        <span>Threshold</span>
      </button>
      <button
        className="action-btn"
        type="button"
        onClick={onShowDetails}
        title="ดู technical metrics + dataset details"
      >
        <span className="action-btn-icon">📊</span>
        <span>Details</span>
      </button>
    </div>
  );
}
