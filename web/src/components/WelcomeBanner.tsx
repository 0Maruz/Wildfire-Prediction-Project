import { useEffect, useState } from "react";

// Lightweight first-visit welcome / legend banner.
//
// Shows once per browser (localStorage flag). Dismissable with X. Explains
// the two main marker colours and points the new operator to the docs.
// Avoids loading a heavyweight onboarding library (Intro.js, Shepherd) —
// for a 5-line banner that's overkill.
const STORAGE_KEY = "firewatch.welcomeDismissed.v2";

export default function WelcomeBanner() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      if (!window.localStorage.getItem(STORAGE_KEY)) setShow(true);
    } catch { /* storage disabled */ }
  }, []);

  const dismiss = () => {
    setShow(false);
    try { window.localStorage.setItem(STORAGE_KEY, "1"); } catch { /* */ }
  };

  if (!show) return null;
  return (
    <div className="welcome-banner" role="dialog" aria-label="Welcome">
      <div className="welcome-banner-icon" aria-hidden="true">👋</div>
      <div className="welcome-banner-body">
        <div className="welcome-banner-title">ยินดีต้อนรับสู่ FireWatch Thailand</div>
        <div className="welcome-banner-text">
          <span><span className="welcome-dot" style={{ background: "#f97316" }} /> สีส้ม = จุดที่ระบบ <b>ทำนาย</b>ว่าจะเกิดไฟ</span>
          <span><span className="welcome-dot" style={{ background: "#06b6d4" }} /> สีฟ้า = ไฟที่ <b>ดาวเทียมยืนยันแล้ว</b> (live)</span>
          <span style={{ color: "var(--text-3)" }}>💡 คลิกจุดบนแผนที่เพื่อดูรายละเอียด · สลับแท็บ <b>🔥 ไฟล่าสุด</b> หรือ <b>🆚 เทียบจริง</b> ได้ที่ด้านบน</span>
        </div>
      </div>
      <button
        type="button"
        className="welcome-banner-close"
        onClick={dismiss}
        aria-label="ปิดข้อความต้อนรับ"
        title="ไม่แสดงอีก"
      >×</button>
    </div>
  );
}
