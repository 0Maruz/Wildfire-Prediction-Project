const map = L.map("map", { center: [13.5, 101], zoom: 6 });

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
).addTo(map);

let pointLayer = null;
let heatLayer = null;
let uncertaintyLayer = null;
let geojsonAll = [];

const legendRisk = document.getElementById("legend-risk");
const legendUnc  = document.getElementById("legend-uncertainty");

/* ================================
   INIT
================================ */
fetch("../outputs/riskmap/fire_risk_all.geojson")
  .then(r => r.json())
  .then(g => {
    geojsonAll = g.features;
    initDate();
  });

function initDate() {
  const predictedDates = geojsonAll
    .filter(f => f.properties.source === "predicted")
    .map(f => f.properties.date)
    .sort();

  const latestPred = predictedDates.at(-1);

  document.getElementById("datePicker").value = latestPred;
  loadData();
}

/* ================================
   UI EVENTS
================================ */
document.querySelectorAll("input").forEach(el =>
  el.addEventListener("change", loadData)
);

/* ================================
   CORE
================================ */
function loadData() {
  const selectedDate = document.getElementById("datePicker").value;
  const source = document.querySelector("input[name='source']:checked").value;

  let features = [];

  if (source === "predicted") {
    features = geojsonAll.filter(f =>
      f.properties.source === "predicted" &&
      f.properties.date === selectedDate
    );
    updateHeader(selectedDate, "predicted");
  }

  if (source === "observed") {
    const obsDates = geojsonAll
      .filter(f => f.properties.source === "observed")
      .map(f => f.properties.date)
      .filter(d => d <= selectedDate)
      .sort();

    const latestObs = obsDates.at(-1);

    features = geojsonAll.filter(f =>
      f.properties.source === "observed" &&
      f.properties.date === latestObs
    );

    updateHeader(latestObs, "observed");
  }

  buildLayers(features, source);
}

/* ================================
   LAYERS
================================ */
function buildLayers(features, source) {
  clearLayers();

  pointLayer = L.markerClusterGroup();
  let heatData = [];
  let uncData = [];
  let count = { LOW: 0, MEDIUM: 0, HIGH: 0 };

  features.forEach(f => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;

    // 🔥 OBSERVED
    if (p.source === "observed") {
      const m = L.circleMarker([lat, lon], {
        radius: 5,
        fillColor: "#ff5722",
        color: "#000",
        weight: 0.5,
        fillOpacity: 0.9
      }).bindPopup(`
        <b>Observed Fire</b><br>
        Date: ${p.date}
      `);

      pointLayer.addLayer(m);
      return;
    }

    // 🤖 PREDICTED
    if (p.source === "predicted") {
      count[p.risk_level]++;

      const m = L.circleMarker([lat, lon], {
        radius: 6,
        fillColor:
          p.risk_level === "HIGH" ? "#e74c3c" :
          p.risk_level === "MEDIUM" ? "#f39c12" : "#2ecc71",
        color: "#000",
        weight: 0.5,
        fillOpacity: 0.9
      }).bindPopup(`
        <b>Predicted Fire Risk</b><br>
        Risk: ${(p.fire_risk * 100).toFixed(1)}%<br>
        Level: ${p.risk_level}<br>
        Uncertainty: ${(p.uncertainty * 100).toFixed(0)}%
      `);

      pointLayer.addLayer(m);
      heatData.push([lat, lon, p.fire_risk]);
      uncData.push([lat, lon, p.uncertainty]);
    }
  });

  heatLayer = L.heatLayer(heatData, { radius: 35, blur: 30 });
  uncertaintyLayer = L.heatLayer(uncData, {
    radius: 50,
    blur: 40,
    gradient: { 0.0: "rgba(255,255,255,0.1)", 1.0: "#fff" }
  });

  // counter
  document.getElementById("lowCount").innerText = source === "predicted" ? count.LOW : "-";
  document.getElementById("medCount").innerText = source === "predicted" ? count.MEDIUM : "-";
  document.getElementById("highCount").innerText = source === "predicted" ? count.HIGH : "-";

  updateView();
}

/* ================================
   VIEW
================================ */
function updateView() {
  const mode = document.querySelector("input[name='mode']:checked").value;
  const showUnc = document.getElementById("uncertaintyToggle").checked;

  map.addLayer(pointLayer);

  if (mode === "heat" && heatLayer) {
    map.addLayer(heatLayer);
    legendRisk.style.display = "block";
  } else {
    legendRisk.style.display = "none";
  }

  if (showUnc && uncertaintyLayer) {
    map.addLayer(uncertaintyLayer);
    legendUnc.style.display = "block";
  } else {
    legendUnc.style.display = "none";
  }
}

/* ================================
   UTILS
================================ */
function clearLayers() {
  if (pointLayer) map.removeLayer(pointLayer);
  if (heatLayer) map.removeLayer(heatLayer);
  if (uncertaintyLayer) map.removeLayer(uncertaintyLayer);
}

function updateHeader(date, source) {
  document.getElementById("prediction-info").innerText =
    source === "observed"
      ? `👁 Observed on: ${date}`
      : `🔮 Predicted for: ${date}`;
}
