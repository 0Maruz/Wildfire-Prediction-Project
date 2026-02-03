# =========================================================
# FIRE RISK TRAINING PIPELINE (FINAL CLEAN VERSION)
# =========================================================

import json
import joblib
import os
import glob
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# =========================================================
# 0) LOAD ENV & CONFIG
# =========================================================
load_dotenv()

GRID = float(os.getenv("GRID", 0.1))
RAW_DIR = os.getenv("RAW_DIR")
FIRMS_PATH = os.getenv("FIRMS_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 500))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.05))
NUM_LEAVES = int(os.getenv("NUM_LEAVES", 31))
MIN_CHILD_SAMPLES = int(os.getenv("MIN_CHILD_SAMPLES", 50))

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(RAW_DIR), f"RAW_DIR not found: {RAW_DIR}"
assert os.path.exists(FIRMS_PATH), f"FIRMS_PATH not found: {FIRMS_PATH}"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# =========================================================
# 1) LOAD & CLEAN RAW SATELLITE DATA
# =========================================================
raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)

raw["acq_datetime"] = pd.to_datetime(
    raw["acq_date"].astype(str) + " " +
    raw["acq_time"].astype(str).str.zfill(4),
    errors="coerce"
)
raw["date"] = raw["acq_datetime"].dt.date

raw.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

raw["bright_main"] = np.nan
if "bright_ti4" in raw.columns:
    raw["bright_main"] = raw["bright_ti4"]
if "bright" in raw.columns:
    raw["bright_main"] = raw["bright_main"].fillna(raw["bright"])

use_cols = ["lat", "lon", "date", "frp", "bright_main", "confidence"]
raw = raw[use_cols].copy()

for c in ["frp", "bright_main", "confidence"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw.dropna(subset=["lat", "lon", "date"], inplace=True)

raw["lat_grid"] = (raw["lat"] / GRID).round() * GRID
raw["lon_grid"] = (raw["lon"] / GRID).round() * GRID

# =========================================================
# 2) DAILY AGGREGATION (RAW)
# =========================================================
daily_raw = raw.groupby(
    ["lat_grid", "lon_grid", "date"],
    as_index=False
).agg(
    fire_count=("frp", "count"),
    frp_sum=("frp", "sum"),
    frp_max=("frp", "max"),
    bright_mean=("bright_main", "mean"),
    confidence_mean=("confidence", "mean"),
)

# =========================================================
# 3) LOAD & AGGREGATE FIRMS DATA
# =========================================================
firms = pd.read_csv(FIRMS_PATH)

firms["acq_datetime"] = pd.to_datetime(
    firms["acq_date"].astype(str) + " " +
    firms["acq_time"].astype(str).str.zfill(4),
    errors="coerce"
)
firms["date"] = firms["acq_datetime"].dt.date

firms.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

firms["bright_main"] = pd.to_numeric(firms["bright_ti4"], errors="coerce")
firms["confidence"] = pd.to_numeric(firms["confidence"], errors="coerce")

firms["lat_grid"] = (firms["lat"] / GRID).round() * GRID
firms["lon_grid"] = (firms["lon"] / GRID).round() * GRID

daily_firms = firms.groupby(
    ["lat_grid", "lon_grid", "date"],
    as_index=False
).agg(
    fire_count=("frp", "count"),
    frp_sum=("frp", "sum"),
    frp_max=("frp", "max"),
    bright_mean=("bright_main", "mean"),
    confidence_mean=("confidence", "mean"),
)

# =========================================================
# 4) MERGE & SORT
# =========================================================
df = pd.concat([daily_raw, daily_firms], ignore_index=True)
df.fillna(0, inplace=True)
df.sort_values(["lat_grid", "lon_grid", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# =========================================================
# 5) TEMPORAL FEATURES
# =========================================================
grp = df.groupby(["lat_grid", "lon_grid"])

df["fire_3d"] = grp["fire_count"].rolling(3, min_periods=1).sum().reset_index(drop=True)
df["frp_3d"] = grp["frp_sum"].rolling(3, min_periods=1).sum().reset_index(drop=True)

df["fire_days_7d"] = grp["fire_count"].rolling(7, min_periods=1)\
    .apply(lambda x: (x > 0).sum()).reset_index(drop=True)

df["fire_yesterday"] = grp["fire_count"].shift(1).fillna(0)
df["frp_trend"] = grp["frp_sum"].diff().fillna(0)

# =========================================================
# 6) LABEL (NEXT DAY FIRE)
# =========================================================
df["label_next_day"] = grp["fire_count"].shift(-1).fillna(0).gt(0).astype(int)

# =========================================================
# 7) TRAIN MODEL
# =========================================================
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

X = df[FEATURES]
y = df["label_next_day"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

model = LGBMClassifier(
    objective="binary",
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    num_leaves=NUM_LEAVES,
    min_child_samples=MIN_CHILD_SAMPLES,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    force_row_wise=True
)

model.fit(X_train, y_train)

# =========================================================
# 8) EVALUATION
# =========================================================
y_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_proba)

print(f"ROC AUC: {auc:.4f}")

def topk_hit(y_true, y_score, k):
    cutoff = np.percentile(y_score, 100 - k)
    return y_true[y_score >= cutoff].mean()

for k in [5, 10, 20]:
    print(f"Top {k}% hit rate:", round(topk_hit(y_val.values, y_proba, k), 3))

# =========================================================
# 9) FEATURE IMPORTANCE
# =========================================================
imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(imp)

# =========================================================
# 10) DATASET METADATA (BACKEND READY)
# =========================================================
print("\nDATASET INFO")
print("Latest date   :", df["date"].max())
print("Earliest date :", df["date"].min())
print("Total days    :", df["date"].nunique())
print("Total grids   :", df[["lat_grid", "lon_grid"]].drop_duplicates().shape[0])

# =========================================================
# 11) SAVE MODEL, FEATURES, METADATA
# =========================================================

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")
FEATURE_PATH = os.path.join(FEATURE_DIR, "full_features.csv")
META_PATH = os.path.join(META_DIR, "dataset_info.json")

# save model
joblib.dump(model, MODEL_PATH)

# save full feature dataset (for risk_map)
df.to_csv(FEATURE_PATH, index=False)

# save metadata (backend friendly)
metadata = {
    "latest_date": str(df["date"].max()),
    "earliest_date": str(df["date"].min()),
    "total_days": int(df["date"].nunique()),
    "total_grids": int(df[["lat_grid", "lon_grid"]].drop_duplicates().shape[0]),
    "features": FEATURES,
    "model": {
        "type": "LightGBM",
        "roc_auc": round(float(auc), 4),
        "params": {
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD_SAMPLES,
        }
    }
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n💾 OUTPUT SAVED")
print("Model   :", MODEL_PATH)
print("Features:", FEATURE_PATH)
print("Meta    :", META_PATH)
