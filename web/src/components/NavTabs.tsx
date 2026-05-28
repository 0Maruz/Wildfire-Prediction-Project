import type { AlertPageRoute } from "../types";
import { useLang } from "../utils/i18n";

interface Props {
  active: AlertPageRoute;
  onNavigate: (r: AlertPageRoute) => void;
  criticalCount: number;
}

interface Tab {
  route: AlertPageRoute;
  icon: string;
  i18nKey: string;
  fallback: string;
}

const TABS: Tab[] = [
  { route: "dashboard", icon: "🗺", i18nKey: "nav.map",       fallback: "Map" },
  { route: "live",      icon: "🔥", i18nKey: "nav.live",      fallback: "Live Fires" },
  { route: "analytics", icon: "📈", i18nKey: "nav.analytics", fallback: "Analytics" },
  { route: "compare",   icon: "🆚", i18nKey: "nav.compare",   fallback: "Compare" },
  { route: "notify",    icon: "🔔", i18nKey: "nav.notify",    fallback: "Alerts" },
  { route: "reports",   icon: "📊", i18nKey: "nav.reports",   fallback: "Reports" },
];

export default function NavTabs({ active, onNavigate, criticalCount }: Props) {
  const { t } = useLang();
  return (
    <nav className="nav-tabs" aria-label="primary">
      {TABS.map((tab) => {
        const isActive = active === tab.route;
        const showBadge = tab.route === "notify" && criticalCount > 0;
        // Strip leading emoji from t() since the tab renders its own icon span.
        const fullLabel = t(tab.i18nKey, tab.fallback);
        const label = fullLabel.replace(/^[\p{Extended_Pictographic}\s]+/u, "");
        return (
          <button
            key={tab.route}
            type="button"
            className={`nav-tab${isActive ? " active" : ""}`}
            onClick={() => onNavigate(tab.route)}
            aria-current={isActive ? "page" : undefined}
            aria-label={fullLabel}
          >
            <span className="nav-tab-icon" aria-hidden="true">{tab.icon}</span>
            <span className="nav-tab-label">{label}</span>
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
