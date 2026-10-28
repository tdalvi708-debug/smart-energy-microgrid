
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "sample_test_input.csv"
OUTPUT_PATH = BASE_DIR / "future_weather_demo.csv"

# Read historical weather observations for a demo scenario.
df = pd.read_csv(SOURCE_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

weather_columns = [
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
]

df = df.dropna(subset=["timestamp"] + weather_columns)

if df.empty:
    raise ValueError("No valid weather observations found.")

# Use the last available observation as the forecast start reference.
last_timestamp = df["timestamp"].max()
last_hour = df.loc[df["timestamp"].idxmax(), "timestamp"]

# Generate the next 24 hourly timestamps after the reference time.
future_timestamps = pd.date_range(
    start=last_hour + pd.Timedelta(hours=1),
    periods=24,
    freq="h",
)

# Create a clearly labelled DEMO scenario using historical
# weather observations. These are NOT real future weather forecasts.
weather = df[weather_columns].tail(24).reset_index(drop=True)

if len(weather) < 24:
    weather = pd.concat(
        [weather] * (24 // len(weather) + 1),
        ignore_index=True,
    ).head(24)

future_weather = pd.DataFrame({
    "timestamp": future_timestamps,
    "ambient_temp_c": weather["ambient_temp_c"].to_numpy(),
    "humidity_pct": weather["humidity_pct"].to_numpy(),
    "solar_irradiance_wm2": weather["solar_irradiance_wm2"].to_numpy(),
})

future_weather.to_csv(OUTPUT_PATH, index=False)

print("Future weather DEMO input created.")
print("Reference timestamp:", last_timestamp)
print("Forecast window:", future_timestamps[0], "to", future_timestamps[-1])
print("Rows:", len(future_weather))
print("Saved to:", OUTPUT_PATH)
print()
print(future_weather.head().to_string(index=False))
print()
print("WARNING: Weather values are reused historical observations,")
print("not real or externally forecast future weather.")
