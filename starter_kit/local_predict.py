"""
Smart Energy Microgrid - Local Demand Prediction
Uses the saved RandomForest model locally; no Zoho API, endpoint key,
OAuth token, internet connection, or paid QuickML quota is required.

Run from the project root:
    python -u starter_kit\local_predict.py
"""
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_FILE = SCRIPT_DIR / "sample_test_input.csv"
MODEL_FILE = PROJECT_DIR / "models" / "demand_model.joblib"
FEATURES_FILE = PROJECT_DIR / "models" / "feature_names.joblib"
# Separate filenames prevent mixing old Zoho predictions with local predictions.
CHECKPOINT_FILE = SCRIPT_DIR / "sample_predictions_local_checkpoint.csv"
OUTPUT_FILE = SCRIPT_DIR / "sample_predictions_local_output.csv"


def prepare_features(input_df, feature_names):
    if input_df.empty:
        raise ValueError("Input CSV has no data rows.")

    required = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
    missing = [c for c in required if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    original = input_df.copy()
    data = input_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        bad_rows = (data.index[data["timestamp"].isna()] + 2).tolist()[:10]
        raise ValueError(f"Invalid timestamp(s) in CSV line(s): {bad_rows}")

    numeric_columns = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = data[numeric_columns].isna().any(axis=1)
    if invalid.any():
        bad_rows = (data.index[invalid] + 2).tolist()[:10]
        raise ValueError(f"Missing/non-numeric weather values in CSV line(s): {bad_rows}")
    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite or non-finite weather values.")

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.day_of_week
    data["month"] = data["timestamp"].dt.month
    data["day_of_year"] = data["timestamp"].dt.day_of_year

    missing_features = [name for name in feature_names if name not in data.columns]
    if missing_features:
        raise ValueError(f"Model requires features that could not be created: {missing_features}")

    X = data[list(feature_names)].copy()
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Prepared model features contain non-finite values.")
    return original, X, data["timestamp"].reset_index(drop=True)


def load_checkpoint(total_rows, timestamps):
    saved_predictions = {}
    if not CHECKPOINT_FILE.exists():
        return saved_predictions
    try:
        saved = pd.read_csv(CHECKPOINT_FILE)
        required = {"input_index", "timestamp", "predicted_demand_kw"}
        if not required.issubset(saved.columns):
            print("Local checkpoint format is invalid; starting a fresh local run.", flush=True)
            return saved_predictions
        for _, row in saved.iterrows():
            try:
                idx = int(row["input_index"])
                value = float(row["predicted_demand_kw"])
                saved_time = pd.to_datetime(row["timestamp"], errors="coerce")
                if (0 <= idx < total_rows and pd.notna(saved_time)
                        and saved_time == timestamps.iloc[idx] and np.isfinite(value)):
                    saved_predictions[idx] = value
            except (TypeError, ValueError, OverflowError):
                continue
        print(f"Local checkpoint loaded: {len(saved_predictions)}/{total_rows} rows", flush=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Could not load local checkpoint ({type(exc).__name__}); starting a fresh run.", flush=True)
    return saved_predictions


def save_checkpoint(predictions, timestamps):
    rows = [
        {"input_index": idx, "timestamp": timestamps.iloc[idx],
         "predicted_demand_kw": value, "prediction_status": "LOCAL_SUCCESS"}
        for idx, value in sorted(predictions.items())
    ]
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    pd.DataFrame(rows, columns=[
        "input_index", "timestamp", "predicted_demand_kw", "prediction_status"
    ]).to_csv(temp_file, index=False)
    temp_file.replace(CHECKPOINT_FILE)


def main():
    print("=" * 66, flush=True)
    print("SMART ENERGY MICROGRID - LOCAL DEMAND PREDICTION", flush=True)
    print("=" * 66, flush=True)
    print("Mode: LOCAL MODEL (no Zoho API calls)", flush=True)

    for path, label in [
        (INPUT_FILE, "Input CSV"), (MODEL_FILE, "Trained model"),
        (FEATURES_FILE, "Feature names")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    input_df = pd.read_csv(INPUT_FILE)
    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    if not hasattr(model, "predict"):
        raise TypeError("The saved model does not have a predict() method.")
    if not isinstance(feature_names, (list, tuple, np.ndarray, pd.Index)):
        raise TypeError("feature_names.joblib must contain a list of feature names.")

    feature_names = list(feature_names)
    original, X, timestamps = prepare_features(input_df, feature_names)
    total = len(original)

    print(f"Input file: {INPUT_FILE}", flush=True)
    print(f"Model file: {MODEL_FILE}", flush=True)
    print(f"Model type: {type(model).__name__}", flush=True)
    print(f"Input rows: {total}", flush=True)
    print(f"Model features: {feature_names}", flush=True)
    print(f"Local checkpoint: {CHECKPOINT_FILE}", flush=True)

    predictions = load_checkpoint(total, timestamps)
    pending = [idx for idx in range(total) if idx not in predictions]
    print(f"Rows needing prediction: {len(pending)}", flush=True)

    # Small batches avoid unnecessary memory use. Each batch is checkpointed.
    batch_size = 100
    for start in range(0, len(pending), batch_size):
        batch_indices = pending[start:start + batch_size]
        batch_values = np.asarray(model.predict(X.iloc[batch_indices]), dtype=float).reshape(-1)
        if len(batch_values) != len(batch_indices):
            raise RuntimeError(
                f"Model returned {len(batch_values)} predictions for {len(batch_indices)} rows."
            )
        for idx, value in zip(batch_indices, batch_values):
            if not np.isfinite(value):
                print(f"WARNING: Non-finite prediction at row {idx + 1}; it will be left incomplete.", flush=True)
                continue
            predictions[idx] = float(value)
            print(f"[{idx + 1}/{total}] Local prediction: {value:.4f} kW", flush=True)
        save_checkpoint(predictions, timestamps)
        print(f"Progress saved: {len(predictions)}/{total}", flush=True)

    output = original.copy()
    output["predicted_demand_kw"] = [predictions.get(idx, np.nan) for idx in range(total)]
    output["prediction_status"] = [
        "LOCAL_SUCCESS" if idx in predictions else "LOCAL_FAILED" for idx in range(total)
    ]
    output.to_csv(OUTPUT_FILE, index=False)

    successful = len(predictions)
    failed = total - successful
    print("\n" + "=" * 66, flush=True)
    print("LOCAL PREDICTION RUN FINISHED", flush=True)
    print("=" * 66, flush=True)
    print(f"Successful predictions: {successful}/{total}", flush=True)
    print(f"Rows without predictions: {failed}", flush=True)
    print(f"Output CSV: {OUTPUT_FILE}", flush=True)
    print(f"Checkpoint CSV: {CHECKPOINT_FILE}", flush=True)
    if failed:
        print("Some rows are incomplete; rerun this script to resume from the local checkpoint.", flush=True)
        return 2
    print("All rows predicted locally. No Zoho API quota was used.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\nStopped by user. Any saved local checkpoint is preserved.", flush=True)
        exit_code = 130
    except Exception as exc:
        print(f"\nLOCAL PREDICTION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        exit_code = 1

    if sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    raise SystemExit(exit_code)
