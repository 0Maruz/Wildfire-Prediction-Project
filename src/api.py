# =====================================
# FASTAPI BACKEND - FIRE RISK
# =====================================

import os
import json
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# =====================================
# CONFIG
# =====================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "lgbm_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "outputs", "features", "full_features.csv")
META_PATH = os.path.join(BASE_DIR, "outputs", "metadata", "dataset_info.json")
RISKMAP_DIR = os.path.join(BASE_DIR, "outputs", "riskmap")

FEATURES = [
    "fire_3d",
    "frp_3d",
    "frp_max",
    "fire_days_7d",
    "fire_yesterday",
    "frp_trend",
    "bright_mean",
    "confidence_mean",
]

# =====================================
# LOAD ASSETS (ON STARTUP)
# =====================================

app = FastAPI(title="🔥 Fire Risk API", version="1.0")

@app.on_event("startup")
def load_assets():
    global model, df

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])

    print("✅ Model & data loaded")

# =====================================
# ROUTES
# =====================================

@app.get("/")
def root():
    return {"status": "Fire Risk API running"}

# -----------------------------
# DATASET METADATA
# -----------------------------
@app.get("/metadata")
def metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# TODAY RISK (JSON)
# -----------------------------
@app.get("/risk/today")
def risk_today():
    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    X = today[FEATURES].fillna(0)
    today["fire_risk"] = model.predict_proba(X)[:, 1]

    return {
        "date": str(latest_date),
        "total_grids": len(today),
        "data": today[
            ["lat_grid", "lon_grid", "fire_risk"]
        ].to_dict(orient="records")
    }

# -----------------------------
# RISK MAP (HTML)
# -----------------------------
@app.get("/risk/map")
def risk_map():
    html_files = sorted(
        [f for f in os.listdir(RISKMAP_DIR) if f.endswith(".html")]
    )

    if not html_files:
        return JSONResponse(
            {"error": "Risk map not generated yet"},
            status_code=404
        )

    latest_map = html_files[-1]
    return FileResponse(
        os.path.join(RISKMAP_DIR, latest_map),
        media_type="text/html"
    )

# -----------------------------
# AUTO GENERATE RISK MAP
# -----------------------------
@app.post("/risk/generate")
def generate_risk_map():
    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    if today.empty:
        return JSONResponse(
            {"error": "No data for latest date"},
            status_code=400
        )

    # predict
    X = today[FEATURES].fillna(0)
    today["fire_risk"] = model.predict_proba(X)[:, 1]

    today["risk_level"] = pd.cut(
        today["fire_risk"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    # save files
    csv_path, geo_path = risk_map.save_outputs(today, latest_date)
    html_path = risk_map.build_folium_map(today, latest_date)

    return {
        "status": "success",
        "date": str(latest_date),
        "total_grids": len(today),
        "outputs": {
            "csv": csv_path,
            "geojson": geo_path,
            "html": html_path
        }
    }
