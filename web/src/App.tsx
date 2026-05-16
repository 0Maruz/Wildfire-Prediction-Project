import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGeoJson } from "./api";
import AlertSettings from "./components/AlertSettings";
import InfoModal from "./components/InfoModal";
import LiveStatusBadge from "./components/LiveStatusBadge";
import MapView from "./components/MapView";
import NavTabs from "./components/NavTabs";
import NotifyPage from "./components/NotifyPage";
import ReportsPage from "./components/ReportsPage";
import Sidebar from "./components/Sidebar";
import ThemeToggle from "./components/ThemeToggle";
import { useHashRoute } from "./utils/hashRoute";
import type {
  DaySelection,
  DisplayOptions,
  FireGeoJson,
  GistdaFeature,
  LiveFireMeta,
} from "./types";
import { exportCellsCsv } from "./utils/csv";
import { dateAdd } from "./utils/dates";
import { fetchLiveFires, LIVE_REFRESH_MS } from "./utils/gistda";

const DEFAULT_OPTIONS: DisplayOptions = {
  showObserved: false,
  showLiveFires: false,
  // showPredicted + showCellPins are always-on now (Sidebar doesn't surface
  // toggles for them). Keep heatRadius hard-coded at 33 to match the new
  // frontend default.
  showPredicted: true,
  showCellPins: true,
  heatRadius: 33,
};

export default function App() {
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

  // Hash-based routing between Dashboard / Notify / Reports
  const [route, navigate] = useHashRoute();
  // Close mobile drawer whenever the route changes
  useEffect(() => { setSidebarOpen(false); }, [route]);

  const refreshLiveFires = useCallback(async () => {
    // Abort any in-flight fetch so a quick toggle on/off doesn't pile up requests.
    liveFireAbortRef.current?.abort();
    const controller = new AbortController();
    liveFireAbortRef.current = controller;
    setLiveFireMeta((m) => ({ ...m, status: "loading", error: null }));
    try {
      const feats = await fetchLiveFires(controller.signal);
      setLiveFires(feats);
      setLiveFireMeta({
        status: "ok",
        count: feats.length,
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
    const features = geojson.features ?? [];

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

    // Day-selector status message — same wording as the original frontend.
    const daySelectorMessage =
      selectedDay === "all"
        ? `Showing all ${provinceFiltered.length} predicted cells.`
        : `Showing ${dayFiltered.length} cells predicted to fire on ${dateAdd(
            activeBaseDate,
            Number(selectedDay)
          )} (Day +${selectedDay}).`;

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
  }, [geojson, selectedBaseDate, selectedProvince, selectedDay]);

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

  // ── Render route-specific content ──
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
      />
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <div id="sidebar-wrap" className={sidebarOpen ? "open" : ""}>
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
    </>
  );
}

function TopBar({
  route, onNavigate, criticalCount,
  showSidebarToggle = false, sidebarOpen = false, onToggleSidebar,
}: {
  route: "dashboard" | "notify" | "reports";
  onNavigate: (r: "dashboard" | "notify" | "reports") => void;
  criticalCount: number;
  showSidebarToggle?: boolean;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}) {
  return (
    <header className="top-bar">
      <div className="top-bar-brand">
        {showSidebarToggle && (
          <button
            type="button"
            className="hamburger-btn"
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            aria-expanded={sidebarOpen}
          >
            <span>{sidebarOpen ? "✕" : "☰"}</span>
          </button>
        )}
        <span className="top-bar-logo" aria-hidden="true">🔥</span>
        <span className="top-bar-title">FireWatch Thailand</span>
      </div>
      <NavTabs active={route} onNavigate={onNavigate} criticalCount={criticalCount} />
      <div className="top-bar-status">
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
