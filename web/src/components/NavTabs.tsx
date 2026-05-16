import type { AlertPageRoute } from "../types";

interface Props {
  active: AlertPageRoute;
  onNavigate: (r: AlertPageRoute) => void;
  criticalCount: number;
}

// Top tab navigation between Dashboard / Notify / Reports. Notify tab carries
// a red badge showing how many CRITICAL alerts currently demand operator
// action — auto-updates from the parent's GeoJSON state.
interface Tab {
  route: AlertPageRoute;
  icon: string;
  label: string;
  labelEn: string;
}

const TABS: Tab[] = [
  { route: "dashboard", icon: "🗺", label: "แผนที่", labelEn: "Map" },
  { route: "notify",    icon: "🔔", label: "แจ้งเตือน", labelEn: "Notify" },
  { route: "reports",   icon: "📊", label: "รายงาน", labelEn: "Reports" },
];

export default function NavTabs({ active, onNavigate, criticalCount }: Props) {
  return (
    <nav className="nav-tabs" aria-label="หน้าหลัก">
      {TABS.map((t) => {
        const isActive = active === t.route;
        const showBadge = t.route === "notify" && criticalCount > 0;
        return (
          <button
            key={t.route}
            type="button"
            className={`nav-tab${isActive ? " active" : ""}`}
            onClick={() => onNavigate(t.route)}
            aria-current={isActive ? "page" : undefined}
            aria-label={`${t.label} (${t.labelEn})`}
          >
            <span className="nav-tab-icon" aria-hidden="true">{t.icon}</span>
            <span className="nav-tab-label">{t.label}</span>
            {showBadge && (
              <span className="nav-tab-badge" aria-label={`${criticalCount} critical alerts`}>
                {criticalCount > 99 ? "99+" : criticalCount}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
