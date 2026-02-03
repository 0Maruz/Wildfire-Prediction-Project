# =========================================================
# FIRE RISK MAP GENERATION (OBSERVED + PREDICTED + UNCERTAINTY)
# =========================================================

import os
import json
import joblib
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "lgbm_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "outputs", "features", "full_features.csv")

RISKMAP_DIR  = os.path.join(BASE_DIR, "outputs", "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_risk_all.geojson")
LATEST_PATH  = os.path.join(RISKMAP_DIR, "latest.json")

os.makedirs(RISKMAP_DIR, exist_ok=True)

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

# =========================================================
# LOAD MODEL & DATA
# =========================================================

def load_assets():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return model, df


# =========================================================
# BUILD OBSERVED (LATEST REAL DAY)
# =========================================================

def build_observed(df):
    observed_date = df["date"].max()
    obs = df[df["date"] == observed_date].copy()

    return obs, observed_date


# =========================================================
# BUILD PREDICTED (DAY + 1)
# =========================================================

def build_predicted(df, model, base_date):
    base = df[df["date"] == base_date].copy()

    X = base[FEATURES].fillna(0)
    base["fire_risk"] = model.predict_proba(X)[:, 1]

    base["risk_level"] = pd.cut(
        base["fire_risk"],
        bins=[0.0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True
    )

    # 🔮 Predict next day
    predicted_date = base_date + timedelta(days=1)
    base["date"] = predicted_date

    # 🌫️ Uncertainty proxy
    base["uncertainty"] = 1 - abs(base["fire_risk"] - 0.5) * 2

    return base, predicted_date


# =========================================================
# APPEND TO SINGLE GEOJSON
# =========================================================

def append_geojson(observed, predicted, predicted_date):
    date_str = predicted_date.strftime("%Y-%m-%d")

    geojson = {"type": "FeatureCollection", "features": []}

    if os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Corrupted GeoJSON → recreate")

    # Remove previous same predicted date
    geojson["features"] = [
        f for f in geojson["features"]
        if f["properties"].get("date") != date_str
    ]

    # ---------- OBSERVED ----------
    for _, r in observed.iterrows():
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])]
            },
            "properties": {
                "date": observed["date"].iloc[0].strftime("%Y-%m-%d"),
                "source": "observed",
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"])
            }
        })

    # ---------- PREDICTED ----------
    for _, r in predicted.iterrows():
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])]
            },
            "properties": {
                "date": date_str,
                "source": "predicted",
                "fire_risk": float(r["fire_risk"]),
                "risk_level": str(r["risk_level"]),
                "uncertainty": float(r["uncertainty"]),
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"])
            }
        })

    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "predicted_date": date_str,
                "base_observed_date": observed["date"].iloc[0].strftime("%Y-%m-%d")
            },
            f,
            indent=2
        )


# =========================================================
# PIPELINE
# =========================================================

def run():
    print("🔄 Loading assets...")
    model, df = load_assets()

    print("📍 Building observed layer...")
    observed, obs_date = build_observed(df)

    print("🔮 Predicting Day + 1...")
    predicted, pred_date = build_predicted(df, model, obs_date)

    append_geojson(observed, predicted, pred_date)

    print("\n✅ FIRE RISK MAP UPDATED")
    print("Observed date :", obs_date)
    print("Predicted date:", pred_date)
    print("GeoJSON       :", GEOJSON_PATH)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()
