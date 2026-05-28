import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGeoJson } from "./api";
import AlertSettings from "./components/AlertSettings";
import AlertToasts from "./components/AlertToasts";
import ComparePage from "./components/ComparePage";
import HotspotAnalyticsPage from "./components/HotspotAnalyticsPage";
import InfoModal from "./components/InfoModal";
import LiveFiresPage from "./components/LiveFiresPage";
import LiveStatusBadge from "./components/LiveStatusBadge";
import MapView from "./components/MapView";
import NavTabs from "./components/NavTabs";
import NotifyPage from "./components/NotifyPage";
import ReportsPage from "./components/ReportsPage";
import Sidebar from "./components/Sidebar";
import ThemeToggle from "./components/ThemeToggle";
import WelcomeBanner from "./components/WelcomeBanner";
import { useFireAlerts, useFireSseStream } from "./utils/fireAlerts";
import { useHashRoute } from "./utils/hashRoute";
import { usePersistedState } from "./utils/usePersistedState";
import type {
  AlertPageRoute,
  DaySelection,
  DisplayOptions,
  FireGeoJson,
  GistdaFeature,
  LiveFireMeta,
} from "./types";
import { exportCellsCsv } from "./utils/csv";
import { dateAdd } from "./utils/dates";
import { fetchLiveFires, LIVE_REFRESH_MS } from "./utils/gistda";
import { isInThailandBbox } from "./constants";
import LanguageToggle from "./components/LanguageToggle";
import { fmtTr, LanguageContext, makeT, useLang, type LangCtx as LangCtxValue, type Lang } from "./utils/i18n";

const DEFAULT_OPTIONS: DisplayOptions = {
  showObserved: false,
  // ON by default so real-time fire alerts (useFireAlerts) work out of
  // the box — operator opens the dashboard and starts getting alerts
  // within minutes of new GISTDA detections.
  showLiveFires: true,
  // showPredicted + showCellPins are always-on now (Sidebar doesn't surface
  // toggles for them). Keep heatRadius hard-coded at 33 to match the new
  // frontend default.
  showPredicted: true,
  showCellPins: true,
  heatRadius: 33,
};

export default function App() {
  const [lang, setLang] = usePersistedState<Lang>("firewatch:lang", "en");
  const langCtx = useMemo<LangCtxValue>(() => ({
    lang, setLang, t: makeT(lang),
  }), [lang, setLang]);

  return (
    <LanguageContext.Provider value={langCtx}>
      <AppInner />
    </LanguageContext.Provider>
  );
}

function AppInner() {
  const { t } = useLang();
  const [geojson, setGeojson] = useState<FireGeoJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedBaseDate, setSelectedBaseDate] = useState<string>("latest");
  const [selectedProvince, setSelectedProvince] = useState<string>("all");
  const [selectedDay, setSelectedDay] = useState<DaySelection>("all");
  const [options, setOptions] = useState<DisplayOptions>(DEFAULT_OPTIONS);

  // GISTDA live fires — fetched directly from the public ArcGIS endpoint
  // when the toggle is on. Auto-refreshes every 30 min while active.
  const [liveFires, setLiveFires] = useState<GistdaFeature[]>([]);
  const [liveFireMeta, setLiveFireMeta] = useState<LiveFireMeta>({
    status: "idle",
    count: 0,
    lastFetch: null,
    error: null,
  });
  const liveFireTimerRef = useRef<number | null>(null);
  const liveFireAbortRef = useRef<AbortController | null>(null);

  // Info modal visibility
  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [alertSettingsOpen, setAlertSettingsOpen] = useState(false);
  // Mobile sidebar drawer (desktop ignores this — sidebar always visible)
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Desktop sidebar collapse — persists across visits via localStorage so a
  // returning user keeps the layout they chose. Mobile uses sidebarOpen instead.
  const [sidebarCollapsed, setSidebarCollapsed] = usePersistedState<boolean>(
    "firewatch.sidebarCollapsed", false,
  );
  // Ctrl/Cmd+B toggles desktop sidebar collapse (VS Code convention)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        if (window.innerWidth >= 768) {
          setSidebarCollapsed((v) => !v);
        } else {
          setSidebarOpen((v) => !v);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setSidebarCollapsed]);

  // Hash-based routing between Dashboard / Notify / Reports
  const [route, navigate] = useHashRoute();
  // Close mobile drawer whenever the route changes
  useEffect(() => { setSidebarOpen(false); }, [route]);

  // ── Real-time fire alerts ──
  // MUST be called before any early-return below — React's Rules of Hooks
  // require hooks to run in the same order every render. Putting this after
  // `if (!geojson) return …` made the first render skip the hook entirely
  // and crashed the page with a "Rendered more hooks than during the
  // previous render" white-screen on data load.
  const fireAlerts = useFireAlerts(liveFires);

  // ── Server-Sent Events stream from FastAPI ──
  // Backend polls GISTDA every 60s and pushes new fires; we react below
  // (after refreshLiveFires is declared) by triggering an immediate refresh
  // so the alert toast fires within seconds instead of waiting 5 min for
  // the next browser-side poll cycle.
  const sse = useFireSseStream();
  const lastSseFireRef = useRef<number>(0);

  const refreshLiveFires = useCallback(async () => {
    // Abort any in-flight fetch so a quick toggle on/off doesn't pile up requests.
    liveFireAbortRef.current?.abort();
    const controller = new AbortController();
    liveFireAbortRef.current = controller;
    setLiveFireMeta((m) => ({ ...m, status: "loading", error: null }));
    try {
      const feats = await fetchLiveFires(controller.signal);
      const inThailand = feats.filter((f) => {
        const lat = Number(f.attributes?.latitude);
        const lon = Number(f.attributes?.longitude);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
        return isInThailandBbox(lat, lon);
      });
      setLiveFires(inThailand);
      setLiveFireMeta({
        status: "ok",
        count: inThailand.length,
        lastFetch: new Date(),
        error: null,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      const msg = err instanceof Error ? err.message : String(err);
      setLiveFires([]);
      setLiveFireMeta((m) => ({ ...m, status: "error", count: 0, error: msg }));
    }
  }, []);

  // SSE → immediate refresh: when the backend pushes a new fire event,
  // re-fetch GISTDA so useFireAlerts diffs the new payload + toasts fire
  // within ~1-2 seconds of detection (vs 5 min polling worst case).
  useEffect(() => {
    if (sse.lastFireAt && sse.lastFireAt !== lastSseFireRef.current) {
      lastSseFireRef.current = sse.lastFireAt;
      refreshLiveFires();
    }
  }, [sse.lastFireAt, refreshLiveFires]);

  // Toggle drives fetch + auto-refresh timer lifecycle.
  useEffect(() => {
    if (!options.showLiveFires) {
      if (liveFireTimerRef.current != null) {
        window.clearTimeout(liveFireTimerRef.current);
        liveFireTimerRef.current = null;
      }
      liveFireAbortRef.current?.abort();
      setLiveFires([]);
      setLiveFireMeta({ status: "idle", count: 0, lastFetch: null, error: null });
      return;
    }
    let cancelled = false;
    const tick = async () => {
      await refreshLiveFires();
      if (cancelled || !options.showLiveFires) return;
      liveFireTimerRef.current = window.setTimeout(tick, LIVE_REFRESH_MS);
    };
    tick();
    return () => {
      cancelled = true;
      if (liveFireTimerRef.current != null) {
        window.clearTimeout(liveFireTimerRef.current);
        liveFireTimerRef.current = null;
      }
    };
  }, [options.showLiveFires, refreshLiveFires]);

  // Load GeoJSON once on mount.
  useEffect(() => {
    fetchGeoJson()
      .then(setGeojson)
      .catch((e: Error) => {
        console.error(e);
        setError(
          "Failed to load fire prediction data. Run train.py + risk_map.py first."
        );
      });
  }, []);

  const derived = useMemo(() => {
    if (!geojson) return null;
    // Two-stage Thailand filter:
    //   1. Bbox — fast first pass, drops obvious foreign cells.
    //   2. Province presence — for PREDICTED features only, also require a
    //      non-empty `province` field. risk_map.py only sets that when
    //      find_province() returns a Thai province for the cell (inside the
    //      77-province polygon). Bbox-alone leaks into S.Myanmar / N.Laos
    //      etc; province filter drops those.
    //   Observed FIRMS features keep just the bbox check — they're real
    //   detections and the user might still want to see cross-border ones
    //   for context (the Live Fires page has its own TH-only toggle).
    const rawFeatures = (geojson.features ?? []).filter((f) => {
      const [lon, lat] = f.geometry.coordinates;
      return isInThailandBbox(lat, lon);
    });
    const features = rawFeatures.filter((f) => {
      const prov = (f.properties.province ?? "").trim();
      // Predicted cells: require province annotation (drops S.Myanmar/N.Laos leaks
      // that fall inside the BBOX but outside Thailand's polygon).
      if (f.properties.source === "predicted") return prov.length > 0;
      // Observed FIRMS: bbox is sufficient — real hotspot coordinates are already
      // inside Thailand. Province annotation is present on newer snapshots
      // (risk_map.py ≥2026-05-26) but absent on older ones. Dropping province-less
      // observations would blank out the Compare page and Live Fires observed layer.
      return true;
    });

    const observed = features.filter((f) => f.properties.source === "observed");
    const predictedAll = features.filter(
      (f) => f.properties.source === "predicted"
    );

    const allBaseDates = [
      ...new Set(predictedAll.map((f) => f.properties.base_date).filter(Boolean) as string[]),
    ].sort();
    const latestBaseDate = allBaseDates[allBaseDates.length - 1] ?? "N/A";

    let activeBaseDate = selectedBaseDate;
    if (
      activeBaseDate === "latest" ||
      !allBaseDates.includes(activeBaseDate)
    ) {
      activeBaseDate = latestBaseDate;
    }

    const snapshotPredicted = predictedAll.filter(
      (f) => f.properties.base_date === activeBaseDate
    );

    const provinceSet = new Set(
      snapshotPredicted
        .map((f) => (f.properties.province ?? "").trim())
        .filter(Boolean)
    );
    const provinces = [...provinceSet].sort();

    let provinceFiltered = snapshotPredicted;
    let resolvedProvince = selectedProvince;
    if (selectedProvince !== "all") {
      if (provinceSet.has(selectedProvince)) {
        provinceFiltered = snapshotPredicted.filter(
          (f) => f.properties.province === selectedProvince
        );
      } else {
        // Snapshot doesn't contain the picked province — report this so the
        // App effect can reset state (next render).
        resolvedProvince = "all";
      }
    }

    const dayFiltered =
      selectedDay === "all"
        ? provinceFiltered
        : provinceFiltered.filter(
            (f) => f.properties.days_until_fire === Number(selectedDay)
          );

    // Day-selector status message — translated via the active language.
    const daySelectorMessage =
      selectedDay === "all"
        ? fmtTr(t("sidebar.showing.all"), { n: provinceFiltered.length })
        : fmtTr(t("sidebar.showing.day"), {
            n: dayFiltered.length,
            date: dateAdd(activeBaseDate, Number(selectedDay)),
          });

    return {
      observed,
      predictedAll,                  // for grid-size detection
      snapshotPredicted: provinceFiltered, // for sidebar urgency / timeline / landcover
      visiblePredicted: dayFiltered,       // map + CSV export
      allBaseDates,
      latestBaseDate,
      activeBaseDate,
      provinces,
      resolvedProvince,
      daySelectorMessage,
    };
  }, [geojson, selectedBaseDate, selectedProvince, selectedDay, t]);

  // If the active province got reset because the snapshot dropped it, sync
  // the controlled state so the dropdown reflects "all".
  useEffect(() => {
    if (derived && derived.resolvedProvince !== selectedProvince) {
      setSelectedProvince(derived.resolvedProvince);
    }
  }, [derived, selectedProvince]);

  // If the active day went empty in this snapshot, fall back to "all".
  useEffect(() => {
    if (!derived || selectedDay === "all") return;
    const count = derived.snapshotPredicted.filter(
      (f) => f.properties.days_until_fire === Number(selectedDay)
    ).length;
    if (count === 0) setSelectedDay("all");
  }, [derived, selectedDay]);

  const onExportCsv = () => {
    if (!derived || !derived.visiblePredicted.length) {
      alert("Nothing to export — current filter has no cells.");
      return;
    }
    exportCellsCsv(
      derived.visiblePredicted,
      derived.activeBaseDate,
      selectedProvince,
      selectedDay
    );
  };

  const onBaseDateChange = (v: string) => {
    setSelectedBaseDate(v);
    // Reset day filter so the new snapshot's full prediction set is visible.
    setSelectedDay("all");
  };

  if (error) {
    return (
      <div style={{ padding: "40px", color: "#f87171" }}>
        <h2>Failed to load</h2>
        <p>{error}</p>
      </div>
    );
  }

  if (!geojson || !derived) {
    return (
      <div style={{ padding: "40px", color: "#a0a3aa" }}>
        Loading fire prediction data…
      </div>
    );
  }

  const meta = geojson.metadata ?? {};

  // Critical count for nav badge — across the LATEST base_date snapshot only
  const criticalCount = derived.snapshotPredicted.filter(
    (f) => f.properties.urgency_level === "CRITICAL"
  ).length;

  // Toast JSX — uses already-computed fireAlerts state. Not a hook, just JSX.
  const fireToasts = (
    <AlertToasts
      alerts={fireAlerts.activeAlerts}
      onDismiss={fireAlerts.dismissAlert}
      onDismissAll={fireAlerts.dismissAll}
      onFlyTo={(lat, lon) => {
        if (route !== "dashboard") navigate("dashboard");
        window.dispatchEvent(new CustomEvent("firewatch:flyto", { detail: { lat, lon } }));
      }}
    />
  );
  const welcomeBanner = <WelcomeBanner />;

  // ── Render route-specific content ──
  if (route === "live") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          criticalCount={criticalCount}
          showSidebarToggle={false}
        />
        <LiveFiresPage
          liveFires={liveFires}
          observed={derived.observed}
          liveFireMeta={liveFireMeta}
          onRefresh={refreshLiveFires}
          onNavigateToMap={(lat, lon) => {
            navigate("dashboard");
            // Defer to next tick so MapView mounts before flyTo fires
            setTimeout(() => {
              window.dispatchEvent(
                new CustomEvent("firewatch:flyto", { detail: { lat, lon } })
              );
            }, 50);
          }}
        />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "notify") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          criticalCount={criticalCount}
          showSidebarToggle={false}
        />
        <NotifyPage predictedAll={derived.predictedAll} />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "analytics") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          criticalCount={criticalCount}
          showSidebarToggle={false}
        />
        <HotspotAnalyticsPage
          liveFires={liveFires}
          liveCount={liveFireMeta.count}
        />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "compare") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          criticalCount={criticalCount}
          showSidebarToggle={false}
        />
        <ComparePage
          allFeatures={derived.predictedAll}
          observedFeatures={derived.observed}
        />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  if (route === "reports") {
    return (
      <>
        <TopBar
          route={route}
          onNavigate={navigate}
          criticalCount={criticalCount}
          showSidebarToggle={false}
        />
        <ReportsPage
          metrics={meta.metrics ?? null}
          predictedAll={derived.predictedAll}
        />
        {welcomeBanner}
        {fireToasts}
      </>
    );
  }

  // Dashboard (default)
  return (
    <>
      <TopBar
        route={route}
        onNavigate={navigate}
        criticalCount={criticalCount}
        showSidebarToggle={true}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebarCollapsed={() => setSidebarCollapsed((v) => !v)}
      />
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <div
        id="sidebar-wrap"
        className={`${sidebarOpen ? "open" : ""}${sidebarCollapsed ? " desktop-collapsed" : ""}`}
      >
      <Sidebar
        activeBaseDate={derived.activeBaseDate}
        allBaseDates={derived.allBaseDates}
        selectedBaseDate={selectedBaseDate}
        onBaseDateChange={onBaseDateChange}
        provinces={derived.provinces}
        selectedProvince={selectedProvince}
        onProvinceChange={setSelectedProvince}
        selectedDay={selectedDay}
        onDayChange={setSelectedDay}
        predicted={derived.snapshotPredicted}
        predictedAll={derived.predictedAll}
        visibleCount={derived.visiblePredicted.length}
        daySelectorMessage={derived.daySelectorMessage}
        thresholds={meta.urgency_thresholds ?? null}
        metrics={meta.metrics ?? null}
        metadata={meta}
        options={options}
        onOptionsChange={(o) => setOptions((prev) => ({ ...prev, ...o }))}
        onExportCsv={onExportCsv}
        liveFireMeta={liveFireMeta}
        onShowInfoModal={() => setInfoModalOpen(true)}
        onShowAlertSettings={() => setAlertSettingsOpen(true)}
      />
      </div>

      <MapView
        observed={derived.observed}
        predictedAll={derived.predictedAll}
        predictedVisible={derived.visiblePredicted}
        liveFires={liveFires}
        thresholds={meta.urgency_thresholds ?? null}
        options={options}
      />

      <Legend showLiveFire={liveFireMeta.status === "ok" && liveFireMeta.count > 0} />

      <InfoModal
        open={infoModalOpen}
        onClose={() => setInfoModalOpen(false)}
        activeBaseDate={derived.activeBaseDate}
        predicted={derived.snapshotPredicted}
        observed={derived.observed}
        liveFires={liveFires}
        metrics={meta.metrics ?? null}
        thresholds={meta.urgency_thresholds ?? null}
        metadata={meta}
        selectedProvince={selectedProvince}
        selectedDay={selectedDay}
      />

      <AlertSettings
        open={alertSettingsOpen}
        onClose={() => setAlertSettingsOpen(false)}
        predicted={derived.snapshotPredicted}
        metrics={meta.metrics ?? null}
      />

      {welcomeBanner}

      {fireToasts}
    </>
  );
}

function TopBar({
  route, onNavigate, criticalCount,
  showSidebarToggle = false, sidebarOpen = false, onToggleSidebar,
  sidebarCollapsed = false, onToggleSidebarCollapsed,
}: {
  route: AlertPageRoute;
  onNavigate: (r: AlertPageRoute) => void;
  criticalCount: number;
  showSidebarToggle?: boolean;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  sidebarCollapsed?: boolean;
  onToggleSidebarCollapsed?: () => void;
}) {
  const { t } = useLang();
  return (
    <header className="top-bar">
      <div className="top-bar-brand">
        {/* Mobile: hamburger ☰ that opens off-canvas drawer */}
        {showSidebarToggle && (
          <button
            type="button"
            className="hamburger-btn mobile-only"
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? t("sidebar.close") : t("sidebar.open")}
            aria-expanded={sidebarOpen}
          >
            <span>{sidebarOpen ? "✕" : "☰"}</span>
          </button>
        )}
        {/* Desktop: chevron that collapses sidebar to 0 width */}
        {showSidebarToggle && onToggleSidebarCollapsed && (
          <button
            type="button"
            className="hamburger-btn desktop-only"
            onClick={onToggleSidebarCollapsed}
            aria-label={sidebarCollapsed ? t("sidebar.expand") : t("sidebar.collapse")}
            aria-expanded={!sidebarCollapsed}
            title={sidebarCollapsed ? t("sidebar.expand.hint") : t("sidebar.collapse.hint")}
          >
            <span style={{ display: "inline-block", transition: "transform 0.2s" }}>
              {sidebarCollapsed ? "›" : "‹"}
            </span>
          </button>
        )}
        <span className="top-bar-logo" aria-hidden="true">🔥</span>
        <span className="top-bar-title">FireWatch Thailand</span>
      </div>
      <NavTabs active={route} onNavigate={onNavigate} criticalCount={criticalCount} />
      <div className="top-bar-status">
        <LanguageToggle />
        <ThemeToggle />
        <LiveStatusBadge />
      </div>
    </header>
  );
}

function Legend({ showLiveFire }: { showLiveFire: boolean }) {
  return (
    <div className="legend">
      <div className="legend-title">Fire Urgency</div>
      <div className="legend-items">
        <div className="legend-item">
          <div className="legend-color critical" />
          <span>Critical</span>
        </div>
        <div className="legend-item">
          <div className="legend-color high" />
          <span>High</span>
        </div>
        <div className="legend-item">
          <div className="legend-color medium" />
          <span>Medium</span>
        </div>
        <div className="legend-item">
          <div className="legend-color low" />
          <span>Low</span>
        </div>
        <div className="legend-item">
          <div className="legend-color observed" />
          <span>Observed (FIRMS)</span>
        </div>
        {showLiveFire && (
          <div className="legend-item">
            <div className="legend-color live-fire" />
            <span>Live VIIRS (GISTDA)</span>
          </div>
        )}
      </div>
    </div>
  );
}
