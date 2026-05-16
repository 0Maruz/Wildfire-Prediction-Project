import { useEffect, useMemo, useState } from "react";
import { fetchNotifyLog, postNotify } from "../api";
import type {
  FireFeature,
  NotifyChannel,
  NotifyLogRecord,
  NotifyPriority,
  UrgencyLevel,
} from "../types";

interface Props {
  predictedAll: FireFeature[];
}

// ─────────────────────────────────────────────────────────────
// Alert Dispatch page
//
// Operator workflow:
//   1. Lands on /web/#notify → table auto-loads CRITICAL + HIGH cells
//   2. Selects rows (single or multi) → right drawer opens with the cells
//   3. Picks channel + template + recipients → preview renders
//   4. Clicks "ส่งการแจ้งเตือน" → confirmation modal → POST /api/notify
//   5. Log entry appears at the bottom, row gets ✅ status
//
// Notes
//   • Backend is a stub — actual SMS/LINE/Email delivery would require
//     per-channel provider integration. We acknowledge this honestly in
//     the UI ("Stub mode: messages queued, not delivered").
//   • All "active alerts" are derived live from the GeoJSON predictedAll
//     prop, filtered to the LATEST base_date that has any rows.
// ─────────────────────────────────────────────────────────────

const URGENCY_RANK: Record<UrgencyLevel, number> = {
  CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, NONE: 4,
};

const URGENCY_COLOR: Record<UrgencyLevel, string> = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#eab308",
  LOW:      "#22c55e",
  NONE:     "#6b7280",
};

interface AlertRow {
  id: string;            // base_date|lat|lon
  lat: number;
  lon: number;
  province: string;
  urgency: UrgencyLevel;
  daysUntilFire: number;
  fireCount30d: number;
  baseDate: string;
  predictedDate: string;
  probability: number | null;
}

const TEMPLATES = [
  {
    id: "critical",
    label: "Critical (วิกฤต)",
    body:
      "🔥 [CRITICAL] พื้นที่ {province} ({lat},{lon}) มีความเสี่ยงไฟป่าสูงมาก " +
      "ทำนายจะเกิดภายใน {days} วัน กรุณาเตรียมพร้อมรับมือทันที",
  },
  {
    id: "high",
    label: "High Alert (เสี่ยงสูง)",
    body:
      "⚠️ [HIGH] ตรวจพบความเสี่ยงไฟป่าระดับสูงที่ {province} ภายใน {days} วัน " +
      "โปรดติดตามสถานการณ์",
  },
  {
    id: "summary",
    label: "Daily Summary (รายงานประจำวัน)",
    body:
      "📊 รายงานประจำวัน: CRITICAL {critical_count} จุด, HIGH {high_count} จุด, " +
      "อัปเดต {date}",
  },
] as const;

const CHANNELS: { id: NotifyChannel; label: string; emoji: string }[] = [
  { id: "sms",   label: "SMS",          emoji: "💬" },
  { id: "line",  label: "LINE Notify",  emoji: "🟢" },
  { id: "email", label: "Email",        emoji: "✉️" },
  { id: "all",   label: "ทั้งหมด",       emoji: "📡" },
];

const PRIORITIES: { id: NotifyPriority; label: string; color: string }[] = [
  { id: "normal",    label: "Normal",    color: "#22c55e" },
  { id: "urgent",    label: "Urgent",    color: "#f97316" },
  { id: "emergency", label: "Emergency", color: "#ef4444" },
];

function renderTemplate(
  body: string,
  ctx: { row?: AlertRow; rows: AlertRow[]; today: string }
): string {
  const first = ctx.row ?? ctx.rows[0];
  const criticalCount = ctx.rows.filter((r) => r.urgency === "CRITICAL").length;
  const highCount = ctx.rows.filter((r) => r.urgency === "HIGH").length;
  return body
    .replace(/{province}/g, first?.province ?? "—")
    .replace(/{lat}/g, first?.lat?.toFixed(3) ?? "—")
    .replace(/{lon}/g, first?.lon?.toFixed(3) ?? "—")
    .replace(/{days}/g, String(first?.daysUntilFire ?? "—"))
    .replace(/{critical_count}/g, String(criticalCount))
    .replace(/{high_count}/g, String(highCount))
    .replace(/{date}/g, ctx.today);
}

function extractAlertRows(features: FireFeature[]): AlertRow[] {
  // Use only the latest base_date that has any rows so the table reflects
  // the operator's current decision window — older snapshots are visible
  // in the dashboard's Past Predictions panel, not here.
  const byDate = new Map<string, FireFeature[]>();
  for (const f of features) {
    const bd = f.properties.base_date;
    if (!bd) continue;
    if (!byDate.has(bd)) byDate.set(bd, []);
    byDate.get(bd)!.push(f);
  }
  const dates = Array.from(byDate.keys()).sort();
  const latest = dates[dates.length - 1];
  if (!latest) return [];
  const rows: AlertRow[] = [];
  for (const f of byDate.get(latest)!) {
    const p = f.properties;
    if (p.source !== "predicted") continue;
    const [lon, lat] = f.geometry.coordinates;
    const prob = typeof p.raw_prediction === "number"
      ? Math.max(0, Math.min(1, 1 - (p.raw_prediction - 1) / 6))
      : null;
    rows.push({
      id: `${p.base_date}|${lat.toFixed(4)}|${lon.toFixed(4)}`,
      lat,
      lon,
      province: p.province ?? "—",
      urgency: (p.urgency_level ?? "NONE") as UrgencyLevel,
      daysUntilFire: p.days_until_fire ?? 0,
      fireCount30d: p.historical_fire_count_30d ?? 0,
      baseDate: latest,
      predictedDate: p.predicted_fire_date ?? "—",
      probability: prob,
    });
  }
  return rows;
}

type SortKey = "urgency" | "days" | "fire30" | "province" | "prob";

export default function NotifyPage({ predictedAll }: Props) {
  const allRows = useMemo(() => extractAlertRows(predictedAll), [predictedAll]);

  // ── Filters / sort ──
  const [urgencyFilter, setUrgencyFilter] = useState<UrgencyLevel | "ALL">("ALL");
  const [provinceFilter, setProvinceFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("urgency");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  // ── Selection ──
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // ── Drawer / send state ──
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [channel, setChannel] = useState<NotifyChannel>("all");
  const [priority, setPriority] = useState<NotifyPriority>("urgent");
  const [templateId, setTemplateId] = useState<string>("critical");
  const [recipientInput, setRecipientInput] = useState("");
  const [recipients, setRecipients] = useState<string[]>([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Log ──
  const [log, setLog] = useState<NotifyLogRecord[]>([]);
  const [logLoading, setLogLoading] = useState(false);
  const [logError, setLogError] = useState<string | null>(null);
  const [logExpanded, setLogExpanded] = useState(true);

  // ── Date filter for log ──
  const [logFromDate, setLogFromDate] = useState("");
  const [logToDate, setLogToDate] = useState("");

  // Auto-default urgency filter to CRITICAL+HIGH on first load (most useful)
  useEffect(() => {
    setUrgencyFilter("ALL");
  }, []);

  // Fetch log on mount + after each send
  const reloadLog = async () => {
    setLogLoading(true);
    setLogError(null);
    try {
      const records = await fetchNotifyLog(100);
      setLog(records);
    } catch (e) {
      setLogError(e instanceof Error ? e.message : String(e));
    } finally {
      setLogLoading(false);
    }
  };
  useEffect(() => {
    reloadLog();
  }, []);

  // Provinces dropdown options (sorted, unique)
  const provinces = useMemo(() => {
    const s = new Set<string>();
    for (const r of allRows) if (r.province && r.province !== "—") s.add(r.province);
    return Array.from(s).sort();
  }, [allRows]);

  // Filtered + sorted rows
  const rows = useMemo(() => {
    let r = allRows;
    if (urgencyFilter !== "ALL") {
      r = r.filter((row) => row.urgency === urgencyFilter);
    }
    if (provinceFilter !== "ALL") {
      r = r.filter((row) => row.province === provinceFilter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      r = r.filter((row) =>
        row.province.toLowerCase().includes(q) ||
        `${row.lat},${row.lon}`.includes(q)
      );
    }
    r = [...r].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "urgency": cmp = URGENCY_RANK[a.urgency] - URGENCY_RANK[b.urgency]; break;
        case "days":    cmp = a.daysUntilFire - b.daysUntilFire; break;
        case "fire30":  cmp = a.fireCount30d - b.fireCount30d; break;
        case "province": cmp = a.province.localeCompare(b.province); break;
        case "prob":    cmp = (a.probability ?? 0) - (b.probability ?? 0); break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return r;
  }, [allRows, urgencyFilter, provinceFilter, search, sortKey, sortDir]);

  // ── Counts header ──
  const counts = useMemo(() => {
    const c = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, NONE: 0 };
    for (const r of allRows) c[r.urgency] += 1;
    return c;
  }, [allRows]);

  // ── Selected rows for drawer ──
  const selectedRows = useMemo(
    () => rows.filter((r) => selected.has(r.id)),
    [rows, selected]
  );

  // ── Open drawer when clicking a single row ──
  const openDrawerForRow = (row: AlertRow) => {
    setSelected(new Set([row.id]));
    // Auto-pick template based on urgency
    if (row.urgency === "CRITICAL") setTemplateId("critical");
    else if (row.urgency === "HIGH") setTemplateId("high");
    setDrawerOpen(true);
  };

  // ── Recipients tag input ──
  const addRecipient = () => {
    const v = recipientInput.trim();
    if (!v) return;
    if (!recipients.includes(v)) setRecipients([...recipients, v]);
    setRecipientInput("");
  };
  const removeRecipient = (r: string) => {
    setRecipients(recipients.filter((x) => x !== r));
  };
  const onRecipientKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addRecipient();
    } else if (e.key === "Backspace" && !recipientInput && recipients.length > 0) {
      setRecipients(recipients.slice(0, -1));
    }
  };

  // ── Send flow ──
  const today = new Date().toISOString().slice(0, 10);
  const tmpl = TEMPLATES.find((t) => t.id === templateId) ?? TEMPLATES[0];
  const previewMessage =
    selectedRows.length === 1
      ? renderTemplate(tmpl.body, { row: selectedRows[0], rows: selectedRows, today })
      : renderTemplate(tmpl.body, { rows: selectedRows.length ? selectedRows : allRows, today });

  const canSend =
    !sending &&
    recipients.length > 0 &&
    previewMessage.length > 0 &&
    (selectedRows.length > 0 || templateId === "summary");

  const performSend = async () => {
    setSending(true);
    setError(null);
    try {
      const zoneIds = (selectedRows.length ? selectedRows : []).map((r) => r.id);
      await postNotify({
        channel,
        recipients,
        message: previewMessage,
        zone_ids: zoneIds,
        priority,
        template: tmpl.id,
      });
      setConfirmOpen(false);
      setDrawerOpen(false);
      setSelected(new Set());
      setRecipients([]);
      await reloadLog();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  };

  // ── CSV export ──
  const exportLogCsv = () => {
    if (!log.length) return;
    const filtered = log.filter((r) => {
      const t = r.timestamp;
      if (logFromDate && t < logFromDate) return false;
      if (logToDate && t > `${logToDate}T23:59:59`) return false;
      return true;
    });
    const headers = ["timestamp", "channel", "recipients_count", "zone_ids_count", "priority", "template", "status", "message_preview"];
    const csv = [
      headers.join(","),
      ...filtered.map((r) =>
        headers
          .map((h) => {
            const v = (r as any)[h];
            const s = String(v ?? "");
            return s.includes(",") || s.includes("\n") ? `"${s.replace(/"/g, '""')}"` : s;
          })
          .join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `notify_log_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // ── Keyboard: Escape closes drawer/modal ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (confirmOpen) setConfirmOpen(false);
        else if (drawerOpen) setDrawerOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [confirmOpen, drawerOpen]);

  // ── Sort header click ──
  const onSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  return (
    <div className="notify-page">
      <header className="notify-page-header">
        <div>
          <h1>🔔 Alert Dispatch</h1>
          <p className="notify-page-subtitle">
            ส่งการแจ้งเตือนให้ stakeholders เมื่อ cell มีความเสี่ยงสูง.
            Backend เป็น stub mode — ข้อความถูก queue แต่ไม่ได้ส่งจริง (ต้องเชื่อม SMS/LINE/Email provider)
          </p>
        </div>
      </header>

      {/* Counts strip */}
      <div className="alert-counts">
        {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as UrgencyLevel[]).map((u) => (
          <button
            key={u}
            type="button"
            className={`alert-count-card ${urgencyFilter === u ? "active" : ""}`}
            onClick={() => setUrgencyFilter(urgencyFilter === u ? "ALL" : u)}
            style={{ borderLeftColor: URGENCY_COLOR[u] }}
          >
            <div className="alert-count-label" style={{ color: URGENCY_COLOR[u] }}>{u}</div>
            <div className="alert-count-value">{counts[u]}</div>
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="alert-filters">
        <select
          value={urgencyFilter}
          onChange={(e) => setUrgencyFilter(e.target.value as UrgencyLevel | "ALL")}
          aria-label="Urgency filter"
        >
          <option value="ALL">ทุกระดับ</option>
          <option value="CRITICAL">CRITICAL</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </select>
        <select
          value={provinceFilter}
          onChange={(e) => setProvinceFilter(e.target.value)}
          aria-label="Province filter"
        >
          <option value="ALL">ทุกจังหวัด</option>
          {provinces.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <input
          type="search"
          placeholder="ค้นหาจังหวัด/พิกัด..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search"
        />
        <div className="alert-filter-summary">
          {rows.length} / {allRows.length} cells · เลือก {selected.size}
        </div>
      </div>

      {/* Table */}
      {rows.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🌲</div>
          <div className="empty-title">
            {allRows.length === 0
              ? "ไม่มี prediction ล่าสุด"
              : "ไม่มี cell ที่ตรงกับเงื่อนไข"}
          </div>
          <div className="empty-hint">
            {allRows.length === 0
              ? "รัน ./run.sh --fresh --predict-only เพื่อสร้าง prediction"
              : "ลองรีเซ็ต filters หรือเปลี่ยน urgency"}
          </div>
        </div>
      ) : (
        <div className="alert-table-wrapper">
          <table className="alert-table">
            <thead>
              <tr>
                <th style={{ width: 32 }}>
                  <input
                    type="checkbox"
                    checked={rows.length > 0 && rows.every((r) => selected.has(r.id))}
                    onChange={(e) => {
                      if (e.target.checked) setSelected(new Set([...selected, ...rows.map((r) => r.id)]));
                      else {
                        const next = new Set(selected);
                        for (const r of rows) next.delete(r.id);
                        setSelected(next);
                      }
                    }}
                    aria-label="Select all"
                  />
                </th>
                <Th label="Urgency" sortKey="urgency" current={sortKey} dir={sortDir} onSort={onSort} />
                <Th label="Province" sortKey="province" current={sortKey} dir={sortDir} onSort={onSort} />
                <th>Location</th>
                <Th label="Days" sortKey="days" current={sortKey} dir={sortDir} onSort={onSort} />
                <Th label="Prob" sortKey="prob" current={sortKey} dir={sortDir} onSort={onSort} />
                <Th label="Fires 30d" sortKey="fire30" current={sortKey} dir={sortDir} onSort={onSort} />
                <th>Predicted date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 500).map((r) => {
                const isSel = selected.has(r.id);
                return (
                  <tr key={r.id} className={isSel ? "selected" : ""}>
                    <td data-label="">
                      <input
                        type="checkbox"
                        checked={isSel}
                        onChange={(e) => {
                          const next = new Set(selected);
                          if (e.target.checked) next.add(r.id);
                          else next.delete(r.id);
                          setSelected(next);
                        }}
                        aria-label={`Select ${r.province}`}
                      />
                    </td>
                    <td data-label="Urgency">
                      <span
                        className="urgency-pill"
                        style={{ background: URGENCY_COLOR[r.urgency] + "22", color: URGENCY_COLOR[r.urgency] }}
                      >
                        {r.urgency}
                      </span>
                    </td>
                    <td data-label="Province">{r.province}</td>
                    <td className="mono" data-label="Location">{r.lat.toFixed(3)}, {r.lon.toFixed(3)}</td>
                    <td className="mono" data-label="Days">{r.daysUntilFire}</td>
                    <td className="mono" data-label="Prob">
                      {r.probability != null ? `${(r.probability * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td className="mono" data-label="Fires 30d">{r.fireCount30d}</td>
                    <td className="mono" data-label="Predicted">{r.predictedDate}</td>
                    <td data-label="">
                      <button
                        type="button"
                        className="action-btn"
                        style={{ padding: "4px 8px", fontSize: 11 }}
                        onClick={() => openDrawerForRow(r)}
                      >
                        Notify →
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {rows.length > 500 && (
            <div className="alert-table-overflow">
              แสดง 500 จาก {rows.length} rows · ใช้ filter เพื่อจำกัด
            </div>
          )}
        </div>
      )}

      {/* Bulk action bar */}
      {selected.size > 0 && (
        <div className="bulk-action-bar">
          <span>เลือก <b>{selected.size}</b> cells</span>
          <button
            type="button"
            className="action-btn primary"
            onClick={() => setDrawerOpen(true)}
          >
            🔔 ส่งแจ้งเตือน ({selected.size})
          </button>
          <button
            type="button"
            className="action-btn"
            onClick={() => setSelected(new Set())}
          >
            ยกเลิก
          </button>
        </div>
      )}

      {/* Send drawer */}
      <div
        className={`send-drawer ${drawerOpen ? "open" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label="Send alert"
      >
        <div className="send-drawer-header">
          <h2>ส่งการแจ้งเตือน</h2>
          <button
            type="button"
            className="send-drawer-close"
            onClick={() => setDrawerOpen(false)}
            aria-label="Close"
          >×</button>
        </div>

        <div className="send-drawer-body">
          <div className="form-group">
            <label>ช่องทาง</label>
            <div className="channel-toggle">
              {CHANNELS.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={`channel-btn ${channel === c.id ? "active" : ""}`}
                  onClick={() => setChannel(c.id)}
                >
                  <span>{c.emoji}</span> {c.label}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="recipients-input">
              ผู้รับ ({recipients.length}) · พิมพ์ชื่อ/เบอร์ แล้ว Enter
            </label>
            <div className="recipient-input-wrap">
              {recipients.map((r) => (
                <span key={r} className="recipient-tag">
                  {r}
                  <button type="button" onClick={() => removeRecipient(r)} aria-label={`Remove ${r}`}>×</button>
                </span>
              ))}
              <input
                id="recipients-input"
                type="text"
                value={recipientInput}
                onChange={(e) => setRecipientInput(e.target.value)}
                onKeyDown={onRecipientKeyDown}
                placeholder={recipients.length === 0 ? "เช่น +66891234567, agent_a, ..." : ""}
              />
            </div>
            {recipients.length === 0 && (
              <div className="form-hint">ต้องมีอย่างน้อย 1 ผู้รับ</div>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="template-select">เทมเพลตข้อความ</label>
            <select
              id="template-select"
              value={templateId}
              onChange={(e) => setTemplateId(e.target.value)}
            >
              {TEMPLATES.map((t) => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>ระดับความเร่งด่วน</label>
            <div className="priority-group">
              {PRIORITIES.map((p) => (
                <label
                  key={p.id}
                  className={`priority-radio ${priority === p.id ? "active" : ""}`}
                  style={{ borderColor: priority === p.id ? p.color : undefined }}
                >
                  <input
                    type="radio"
                    name="priority"
                    value={p.id}
                    checked={priority === p.id}
                    onChange={() => setPriority(p.id)}
                  />
                  <span style={{ color: priority === p.id ? p.color : undefined }}>
                    {p.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label>Preview</label>
            <div className="message-preview">
              {previewMessage}
            </div>
            <div className="form-hint">
              จะส่งไปยัง <b>{recipients.length}</b> ผู้รับ ·{" "}
              {selectedRows.length > 0 ? `${selectedRows.length} zones` : "summary mode"}
            </div>
          </div>

          {error && (
            <div className="error-banner" role="alert">⚠️ {error}</div>
          )}
        </div>

        <div className="send-drawer-footer">
          <button
            type="button"
            className="action-btn"
            onClick={() => setDrawerOpen(false)}
            disabled={sending}
          >
            ยกเลิก
          </button>
          <button
            type="button"
            className="action-btn primary"
            disabled={!canSend}
            onClick={() => setConfirmOpen(true)}
            style={{
              background:
                priority === "emergency" ? "#ef4444" :
                priority === "urgent" ? "#f97316" : undefined,
              color: priority !== "normal" ? "#fff" : undefined,
              borderColor:
                priority === "emergency" ? "#ef4444" :
                priority === "urgent" ? "#f97316" : undefined,
            }}
          >
            {sending ? "กำลังส่ง..." : "🔔 ส่งการแจ้งเตือน"}
          </button>
        </div>
      </div>
      {drawerOpen && (
        <div className="drawer-backdrop" onClick={() => setDrawerOpen(false)} />
      )}

      {/* Confirmation modal */}
      {confirmOpen && (
        <div className="info-modal-backdrop" role="dialog" aria-modal="true" onClick={() => setConfirmOpen(false)}>
          <div className="info-modal" style={{ maxWidth: 480 }} onClick={(e) => e.stopPropagation()}>
            <div className="info-modal-header">
              <h2>ยืนยันการส่งการแจ้งเตือน</h2>
              <button className="info-modal-close" onClick={() => setConfirmOpen(false)} aria-label="Close">×</button>
            </div>
            <div className="info-modal-body">
              <div className="info-grid">
                <div><span>ช่องทาง</span><b>{CHANNELS.find((c) => c.id === channel)?.label}</b></div>
                <div><span>ผู้รับ</span><b>{recipients.length} คน</b></div>
                <div><span>Zones</span><b>{selectedRows.length}</b></div>
                <div><span>ระดับ</span><b style={{ color: PRIORITIES.find((p) => p.id === priority)?.color }}>{priority.toUpperCase()}</b></div>
              </div>
              <h4 style={{ marginTop: 14, marginBottom: 6 }}>Preview</h4>
              <div className="message-preview">{previewMessage}</div>
              {error && (
                <div className="error-banner" role="alert" style={{ marginTop: 10 }}>⚠️ {error}</div>
              )}
              <div style={{ display: "flex", gap: 8, marginTop: 18 }}>
                <button
                  type="button"
                  className="action-btn"
                  onClick={() => setConfirmOpen(false)}
                  disabled={sending}
                  style={{ flex: 1 }}
                >
                  ยกเลิก
                </button>
                <button
                  type="button"
                  className="action-btn primary"
                  onClick={performSend}
                  disabled={sending}
                  style={{ flex: 2 }}
                >
                  {sending ? "กำลังส่ง..." : "ยืนยันส่ง"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Call Log */}
      <div className={`call-log ${logExpanded ? "open" : ""}`}>
        <div className="call-log-header">
          <button
            type="button"
            className="call-log-toggle"
            onClick={() => setLogExpanded(!logExpanded)}
          >
            <span>{logExpanded ? "▼" : "▶"}</span>
            <h3>📜 Dispatch Log ({log.length})</h3>
          </button>
          <div className="call-log-actions">
            <input
              type="date"
              value={logFromDate}
              onChange={(e) => setLogFromDate(e.target.value)}
              aria-label="From date"
              style={{ width: 130 }}
            />
            <span style={{ color: "var(--text-3)", fontSize: 11 }}>→</span>
            <input
              type="date"
              value={logToDate}
              onChange={(e) => setLogToDate(e.target.value)}
              aria-label="To date"
              style={{ width: 130 }}
            />
            <button
              type="button"
              className="action-btn"
              onClick={reloadLog}
              disabled={logLoading}
            >
              {logLoading ? "..." : "↻ Refresh"}
            </button>
            <button
              type="button"
              className="action-btn"
              onClick={exportLogCsv}
              disabled={!log.length}
            >
              📥 CSV
            </button>
          </div>
        </div>
        {logExpanded && (
          <div className="call-log-body">
            {logError ? (
              <div className="error-banner" role="alert">⚠️ {logError}</div>
            ) : log.length === 0 ? (
              <div className="empty-state" style={{ padding: 20 }}>
                <div className="empty-icon">📭</div>
                <div className="empty-title">ยังไม่มีการแจ้งเตือน</div>
                <div className="empty-hint">เลือก cell แล้วกด Notify → log จะปรากฏที่นี่</div>
              </div>
            ) : (
              <table className="log-table">
                <thead>
                  <tr>
                    <th>เวลา</th>
                    <th>ช่องทาง</th>
                    <th>ระดับ</th>
                    <th>ผู้รับ</th>
                    <th>Zones</th>
                    <th>เทมเพลต</th>
                    <th>สถานะ</th>
                    <th>ข้อความ</th>
                  </tr>
                </thead>
                <tbody>
                  {log
                    .filter((r) => {
                      const t = r.timestamp;
                      if (logFromDate && t < logFromDate) return false;
                      if (logToDate && t > `${logToDate}T23:59:59`) return false;
                      return true;
                    })
                    .map((r) => (
                      <tr key={r.id}>
                        <td className="mono">{new Date(r.timestamp).toLocaleString()}</td>
                        <td>{r.channel}</td>
                        <td>
                          <span
                            className="urgency-pill"
                            style={{
                              background: PRIORITIES.find((p) => p.id === r.priority)?.color + "22",
                              color: PRIORITIES.find((p) => p.id === r.priority)?.color,
                            }}
                          >
                            {r.priority}
                          </span>
                        </td>
                        <td className="mono">{r.recipients_count}</td>
                        <td className="mono">{r.zone_ids_count}</td>
                        <td>{r.template ?? "—"}</td>
                        <td>
                          <span className="status-badge good" style={{ padding: "2px 6px", fontSize: 10 }}>
                            ✓ {r.status}
                          </span>
                        </td>
                        <td className="log-message">{r.message_preview}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface ThProps {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  dir: "asc" | "desc";
  onSort: (k: SortKey) => void;
}
function Th({ label, sortKey, current, dir, onSort }: ThProps) {
  const isCur = sortKey === current;
  return (
    <th
      className={`sortable ${isCur ? "sorted" : ""}`}
      onClick={() => onSort(sortKey)}
      role="button"
      tabIndex={0}
      aria-sort={isCur ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      {label}
      {isCur && <span className="sort-arrow">{dir === "asc" ? "▲" : "▼"}</span>}
    </th>
  );
}
