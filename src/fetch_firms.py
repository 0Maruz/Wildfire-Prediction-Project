import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# ===============================
# LOAD ENV
# ===============================
load_dotenv()
FIRMS_API_KEY = os.getenv("FIRMS_API_KEY")
if not FIRMS_API_KEY:
    raise RuntimeError("❌ FIRMS_API_KEY not found in .env")

# ===============================
# CONFIG
# ===============================
TH_BBOX = "96,4,107,22"   # Thailand
DATASETS = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

DATA_DIR = os.getenv("DATA_DIR", "./data")
OUT_FILE = os.getenv("FIRMS_OUT", "./data/firms/firms_all.csv")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)


BASE_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp",
    "confidence",
]

# ===============================
# FETCH TODAY (NRT)
# ===============================
def fetch_firms_today() -> pd.DataFrame:
    all_dfs = []

    for dataset in DATASETS:
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{FIRMS_API_KEY}/{dataset}/{TH_BBOX}/1"
        )

        print(f"📡 Fetching {dataset} (today)")
        try:
            res = requests.get(url, timeout=30)
        except Exception as e:
            print("❌ Request failed:", e)
            continue

        if res.status_code != 200:
            print(f"❌ HTTP {res.status_code}")
            continue

        df = pd.read_csv(StringIO(res.text))
        if df.empty:
            print(f"⚠️ No data from {dataset}")
            continue

        if not set(BASE_COLUMNS).issubset(df.columns):
            print(f"⚠️ Schema mismatch from {dataset}")
            print("📄 Columns:", df.columns.tolist())
            continue

        df = df[BASE_COLUMNS].copy()
        df["dataset"] = dataset
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# ===============================
# CLEAN + FEATURE ENGINEERING
# ===============================
def clean_firms(df: pd.DataFrame) -> pd.DataFrame:
    # acq_datetime (SAFE)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " +
        df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        errors="coerce"
    )

    # confidence → numeric (NO WARNING)
    df["confidence"] = pd.to_numeric(
        df["confidence"]
        .map({"l": 0, "n": 50, "h": 100})
        .fillna(df["confidence"]),
        errors="coerce"
    )

    # numeric columns (ML safe)
    numeric_cols = [
        "latitude", "longitude",
        "bright_ti4", "bright_ti5",
        "scan", "track",
        "frp", "confidence"
    ]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    df.dropna(subset=["acq_datetime"], inplace=True)

    return df


# ===============================
# UPDATE (ACCUMULATIVE)
# ===============================
def update_firms():
    os.makedirs(DATA_DIR, exist_ok=True)

    new_df = fetch_firms_today()
    if new_df.empty:
        print("⚠️ No new data today")
        return

    new_df = clean_firms(new_df)

    # load old data safely
    if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
        try:
            old_df = pd.read_csv(OUT_FILE)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    combined = pd.concat([old_df, new_df], ignore_index=True)

    # 🔒 FORCE acq_datetime (กัน CSV เก่า)
    combined["acq_datetime"] = pd.to_datetime(
        combined["acq_datetime"],
        errors="coerce"
    )

    # remove duplicates
    combined.drop_duplicates(
        subset=["latitude", "longitude", "acq_datetime"],
        inplace=True
    )

    combined.sort_values("acq_datetime", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved {len(combined)} records → {OUT_FILE}")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("🚀 Updating FIRMS (VIIRS NRT, accumulative)")
    update_firms()
