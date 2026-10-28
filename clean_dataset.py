
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("starter_kit/train_telemetry.csv")
OUTPUT_FILE = Path("starter_kit/train_telemetry_cleaned.csv")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE.resolve()}\n"
        "Check that train_telemetry.csv is inside the starter_kit folder."
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "timestamp",
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
    "target_demand_kw",
]

missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Keep only the columns required by the hackathon dataset schema.
df = df[required].copy()

# Parse timestamps and numeric columns.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

numeric_columns = [
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
    "target_demand_kw",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with invalid timestamps or missing/non-numeric values.
df = df.dropna(subset=required).copy()

# Remove exact duplicate records.
df = df.drop_duplicates().copy()

# Add numeric time features for simpler model training.
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Retain only numeric model inputs and the regression target.
output_columns = [
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
    "hour",
    "day",
    "month",
    "target_demand_kw",
]

clean = df[output_columns].replace([np.inf, -np.inf], np.nan)
clean = clean.dropna().copy()

# Basic sanity checks.
if clean.empty:
    raise ValueError("No usable rows remain after cleaning.")

if (clean["target_demand_kw"] < 0).any():
    raise ValueError("Negative target demand values found; inspect the source data.")

clean.to_csv(OUTPUT_FILE, index=False)

print("Dataset cleaning completed.")
print("Original rows:", len(pd.read_csv(INPUT_FILE)))
print("Cleaned rows:", len(clean))
print("Columns:", list(clean.columns))
print("Missing values:", int(clean.isna().sum().sum()))
print("Exact duplicate rows remaining:", int(clean.duplicated().sum()))
print("Saved to:", OUTPUT_FILE.resolve())
print("\nPreview:")
print(clean.head().to_string(index=False))
