
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from grid_controller import get_grid_status


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "demand_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"


def load_model():
    """Load the trained model and its expected feature names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python train_model.py first."
        )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}. "
            "Run python train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


def prepare_features(input_df):
    """Validate weather inputs and create training-time features."""
    required_columns = [
        "timestamp",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    missing = [
        column for column in required_columns
        if column not in input_df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = input_df[required_columns].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    numeric_columns = [
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.isnull().any().any():
        raise ValueError(
            "Input contains missing or invalid timestamps/weather values."
        )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.day_of_year

    return df


def forecast_demand(input_df):
    """Predict energy demand and calculate grid decisions."""
    model, features = load_model()
    prepared_df = prepare_features(input_df)

    predictions = model.predict(prepared_df[features])

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")

    results = []

    for timestamp, demand in zip(
        prepared_df["timestamp"], predictions
    ):
        status = get_grid_status(timestamp, demand)
        results.append(status)

    return pd.DataFrame(results)


def forecast_next_24_hours(start_time, weather_data):
    """
    Generate a 24-hour forecast.

    weather_data must contain 24 rows with weather values for
    each corresponding hour.
    """
    weather_columns = [
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    missing = [
        column for column in weather_columns
        if column not in weather_data.columns
    ]

    if missing:
        raise ValueError(f"Missing weather columns: {missing}")

    if len(weather_data) != 24:
        raise ValueError(
            f"Expected 24 hourly weather rows, received {len(weather_data)}."
        )

    start_time = pd.Timestamp(start_time)

    forecast_input = weather_data[weather_columns].copy()
    forecast_input.insert(
        0,
        "timestamp",
        pd.date_range(start=start_time, periods=24, freq="h"),
    )

    return forecast_demand(forecast_input)



if __name__ == "__main__":
    print("SMART ENERGY 24-HOUR DEMO FORECAST")
    print("=" * 60)

    input_path = BASE_DIR / "future_weather_demo.csv"
    output_path = BASE_DIR / "forecast_results.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path.name} not found. "
            "Run python create_future_weather.py first."
        )

    weather_df = pd.read_csv(input_path)
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"], errors="raise"
    )

    if len(weather_df) != 24:
        raise ValueError(
            f"Expected 24 rows, found {len(weather_df)}."
        )

    # These timestamps are future relative to the historical dataset,
    # but not necessarily future relative to today's real date.
    forecast = forecast_demand(weather_df)

    forecast.to_csv(output_path, index=False)

    print("\nFirst 5 predicted results:")
    print(forecast.head().to_string(index=False))

    print("\nGrid mode counts:")
    print(forecast["grid_mode"].value_counts().to_string())

    print(
        "\nCritical overload alerts:",
        int(forecast["critical_overload_alert"].sum()),
    )

    print("\nSaved forecast to:", output_path)
    print(
        "\nNOTE: This is a demo forecast using reused historical "
        "weather inputs, not a live future forecast."
    )
