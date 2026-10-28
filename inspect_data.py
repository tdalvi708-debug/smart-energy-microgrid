
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project ke andar official training dataset ka path
DATA_PATH = Path("starter_kit/train_telemetry.csv")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

# Demand distribution
plt.figure(figsize=(10, 5))
plt.hist(df["target_demand_kw"], bins=30, edgecolor="black")
plt.title("Historical Energy Demand Distribution")
plt.xlabel("Energy Demand (kW)")
plt.ylabel("Number of Readings")
plt.tight_layout()
plt.savefig("demand_distribution.png", dpi=150)
plt.close()

# First 168 readings = up to 7 days of hourly data
sample = df.sort_values("timestamp").head(168)

plt.figure(figsize=(12, 5))
plt.plot(
    sample["timestamp"],
    sample["target_demand_kw"],
    linewidth=1.5,
)
plt.title("Historical Energy Demand — First 168 Hourly Readings")
plt.xlabel("Timestamp")
plt.ylabel("Energy Demand (kW)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("demand_timeseries.png", dpi=150)
plt.close()

print("\nGraphs saved:")
print("1. demand_distribution.png")
print("2. demand_timeseries.png")
