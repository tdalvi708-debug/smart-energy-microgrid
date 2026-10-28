
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort

from pathlib import Path
from grid_controller import get_grid_status


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "demand_model.onnx"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"

_onnx_session = None
_input_name = None
_features = None


def load_model():
    """Load and cache the ONNX model and feature names."""
    global _onnx_session, _input_name, _features

    if _onnx_session is not None:
        return _onnx_session, _input_name, _features

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    features = joblib.load(FEATURE_PATH)
    input_name = session.get_inputs()[0].name

    _onnx_session = session
    _input_name = input_name
    _features = features

    return _onnx_session, _input_name, _features


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
    """Predict demand using ONNX and calculate grid decisions."""
    session, input_name, features = load_model()
    prepared_df = prepare_features(input_df)

    model_input = prepared_df[features].to_numpy(dtype=np.float32)

    predictions = np.asarray(
        session.run(None, {input_name: model_input})[0]
    ).reshape(-1)

    if len(predictions) != len(prepared_df):
        raise ValueError("Prediction count does not match input rows.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid predictions.")

    results = []

    for timestamp, demand in zip(
        prepared_df["timestamp"], predictions
    ):
        results.append(get_grid_status(timestamp, float(demand)))

    return pd.DataFrame(results)


def forecast_next_24_hours(start_time, weather_data):
    """Generate a 24-hour forecast from 24 hourly weather rows."""
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
            "Copy the demo weather CSV into the project folder."
        )

    weather_df = pd.read_csv(input_path)
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"], errors="raise"
    )

    if len(weather_df) != 24:
        raise ValueError(
            f"Expected 24 rows, found {len(weather_df)}."
        )

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
