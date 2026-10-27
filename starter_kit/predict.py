
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# ZOHO QUICKML CONFIGURATION
# ============================================================

QUICKML_ENDPOINT_URL = (
    "https://api.catalyst.zoho.in/quickml/v1/"
    "project/103516000000013050/endpoints/predict"
)

ORG_ID = "60093284709"
ENVIRONMENT = "Development"

# Maximum time to wait for one API request.
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = (2, 4, 8)

# Progress/checkpoint files are stored next to this script.
CHECKPOINT_FILE = Path(__file__).with_name("sample_predictions_checkpoint.csv")
OUTPUT_FILE = Path(__file__).with_name("sample_predictions_output.csv")
PRINT_EACH_ROW = True


# ============================================================
# CREDENTIALS
# ============================================================

def get_credentials():
    """Read credentials from environment variables."""

    key = os.getenv("QUICKML_ENDPOINT_KEY", "").strip()
    token = os.getenv("ZOHO_ACCESS_TOKEN", "").strip()

    if not key:
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY is missing.\n"
            "Set the actual QuickML endpoint key in this CMD window."
        )

    if not token:
        raise RuntimeError(
            "ZOHO_ACCESS_TOKEN is missing.\n"
            "Set a valid Zoho OAuth access token in this CMD window."
        )

    if key.lower().startswith(("http://", "https://")):
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY contains a URL, not an endpoint key."
        )

    placeholders = ("YOUR_", "PASTE_", "YAHAN_")

    if any(value in key.upper() for value in placeholders):
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY still contains placeholder text."
        )

    if any(value in token.upper() for value in placeholders):
        raise RuntimeError(
            "ZOHO_ACCESS_TOKEN still contains placeholder text."
        )

    return key, token


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Validate input data and generate time-related features."""

    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if input_df.empty:
        return input_df.copy()

    required_columns = [
        "timestamp",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required input columns: {missing_columns}"
        )

    data = input_df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = data["timestamp"].isna()

    if invalid_timestamps.any():
        rows = data.index[invalid_timestamps].tolist()[:10]

        raise ValueError(
            f"Invalid timestamps found at input indices: {rows}"
        )

    numeric_columns = [
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    invalid_values = data[numeric_columns].isna().any(axis=1)

    if invalid_values.any():
        rows = data.index[invalid_values].tolist()[:10]

        raise ValueError(
            f"Missing or non-numeric weather values at indices: {rows}"
        )

    numeric_values = data[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Input contains infinite or non-finite weather values."
        )

    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month

    return data


# ============================================================
# API RESPONSE VALIDATION
# ============================================================

def parse_prediction(response_body: dict, timestamp) -> float:
    """Extract one numeric prediction from a QuickML response."""

    if not isinstance(response_body, dict):
        raise RuntimeError(
            f"Unexpected response format for timestamp {timestamp}."
        )

    if "result" not in response_body:
        response_fields = list(response_body.keys())

        raise RuntimeError(
            "QuickML response does not contain the 'result' field. "
            f"Available response fields: {response_fields}"
        )

    value = response_body["result"]

    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(
                f"Expected one prediction for {timestamp}, "
                f"but received {len(value)} values."
            )

        value = value[0]

    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"QuickML returned a non-numeric prediction "
            f"for {timestamp}."
        ) from exc

    if not np.isfinite(prediction):
        raise RuntimeError(
            f"QuickML returned an invalid numeric prediction "
            f"for {timestamp}."
        )

    return prediction


# ============================================================
# API ERROR HANDLING
# ============================================================

def describe_http_error(response: requests.Response) -> str:
    """Return a helpful HTTP error description."""

    status = response.status_code

    messages = {
        400: (
            "Bad request. Check the JSON payload and required "
            "feature names."
        ),
        401: (
            "Authentication failed. The OAuth access token may be "
            "expired, invalid, or issued for the wrong account/region."
        ),
        403: (
            "Access forbidden. Check OAuth scopes, endpoint permissions, "
            "organization ID, and environment."
        ),
        404: (
            "Endpoint not found. Check the published endpoint URL and "
            "project ID."
        ),
        408: (
            "The request timed out. Try again later."
        ),
        429: (
            "The API rate limit may have been reached. Check the "
            "QuickML endpoint quota."
        ),
        500: (
            "Zoho returned an internal server error."
        ),
        502: (
            "Zoho returned a gateway error."
        ),
        503: (
            "The API service may be temporarily unavailable."
        ),
        504: (
            "The API gateway timed out while waiting for the service."
        ),
    }

    message = messages.get(
        status,
        "QuickML returned an unsuccessful HTTP response.",
    )

    # Do not display response bodies because they can sometimes
    # contain diagnostic information that should not be shared.
    return f"{message}\nHTTP status: {status}"


# ============================================================
# QUICKML PREDICTION
# ============================================================

def predict(input_df: pd.DataFrame) -> np.ndarray:
    """Predict demand with retries and save each successful row as a checkpoint."""
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if input_df.empty:
        print("Input dataset is empty. No requests will be sent.", flush=True)
        return np.array([], dtype=float)

    data = prepare_input(input_df).reset_index(drop=True)
    key, token = get_credentials()
    headers = {
        "X-QUICKML-ENDPOINT-KEY": key,
        "Authorization": f"Zoho-oauthtoken {token}",
        "CATALYST-ORG": ORG_ID,
        "Environment": ENVIRONMENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    total_rows = len(data)
    predictions = [None] * total_rows

    # Resume only when the checkpoint matches the current input row count
    # and timestamps. This prevents silently reusing results for changed input.
    if CHECKPOINT_FILE.exists():
        try:
            saved = pd.read_csv(CHECKPOINT_FILE)
            required = {"input_index", "timestamp", "predicted_demand_kw"}
            if required.issubset(saved.columns):
                valid_checkpoint = True
                for _, saved_row in saved.iterrows():
                    idx = int(saved_row["input_index"])
                    if idx < 0 or idx >= total_rows:
                        valid_checkpoint = False
                        break
                    saved_time = pd.to_datetime(saved_row["timestamp"], errors="coerce")
                    if pd.isna(saved_time) or saved_time != data.iloc[idx]["timestamp"]:
                        valid_checkpoint = False
                        break
                if valid_checkpoint:
                    for _, saved_row in saved.iterrows():
                        idx = int(saved_row["input_index"])
                        val = float(saved_row["predicted_demand_kw"])
                        if np.isfinite(val):
                            predictions[idx] = val
                    print(f"Checkpoint loaded: {sum(v is not None for v in predictions)}/{total_rows} rows", flush=True)
                else:
                    print("Checkpoint does not match this input; starting a fresh run.", flush=True)
            else:
                print("Checkpoint format is old or invalid; starting a fresh run.", flush=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Could not reuse checkpoint ({type(exc).__name__}); starting a fresh run.", flush=True)

    def save_checkpoint():
        rows = []
        for i, value in enumerate(predictions):
            if value is not None:
                rows.append({
                    "input_index": i,
                    "timestamp": data.iloc[i]["timestamp"],
                    "predicted_demand_kw": float(value),
                })
        pd.DataFrame(rows, columns=["input_index", "timestamp", "predicted_demand_kw"]).to_csv(
            CHECKPOINT_FILE, index=False
        )

    print(f"\nInput validation successful: {total_rows} rows", flush=True)
    print(f"Connection timeout: {CONNECT_TIMEOUT_SECONDS}s", flush=True)
    print(f"Response timeout: {READ_TIMEOUT_SECONDS}s", flush=True)
    print(f"Checkpoint file: {CHECKPOINT_FILE}", flush=True)

    with requests.Session() as session:
        for position, (_, row) in enumerate(data.iterrows(), start=1):
            index = position - 1
            timestamp = row["timestamp"]

            if predictions[index] is not None:
                print(f"[{position}/{total_rows}] Already saved; skipping.", flush=True)
                continue

            features = {
                "ambient_temp_c": float(row["ambient_temp_c"]),
                "humidity_pct": float(row["humidity_pct"]),
                "solar_irradiance_wm2": float(row["solar_irradiance_wm2"]),
                "hour": int(row["hour"]),
                "day": int(row["day"]),
                "month": int(row["month"]),
            }

            for attempt in range(1, MAX_RETRIES + 1):
                if PRINT_EACH_ROW:
                    print(f"[{position}/{total_rows}] Attempt {attempt}/{MAX_RETRIES}: {timestamp}", flush=True)
                try:
                    response = session.post(
                        QUICKML_ENDPOINT_URL,
                        headers=headers,
                        json={"data": features},
                        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                    )
                except requests.exceptions.RequestException as exc:
                    if attempt >= MAX_RETRIES:
                        save_checkpoint()
                        raise RuntimeError(
                            f"Request failed after {MAX_RETRIES} attempts at row {position} ({timestamp}). "
                            f"Error: {type(exc).__name__}. Completed predictions are checkpointed."
                        ) from exc
                    delay = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
                    print(f"Temporary network/timeout error ({type(exc).__name__}); retrying in {delay}s.", flush=True)
                    time.sleep(delay)
                    continue

                print(f"    HTTP response: {response.status_code}", flush=True)

                if not response.ok:
                    if response.status_code in (400, 401, 403, 404):
                        save_checkpoint()
                        raise RuntimeError(
                            f"Non-retryable API error at row {position}: HTTP {response.status_code}. "
                            f"{describe_http_error(response)}. Completed rows remain in {CHECKPOINT_FILE.name}."
                        )
                    if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
                        print(f"Temporary HTTP {response.status_code}; retrying in {delay}s.", flush=True)
                        time.sleep(delay)
                        continue
                    save_checkpoint()
                    raise RuntimeError(
                        f"QuickML failed at row {position}: HTTP {response.status_code}. "
                        f"{describe_http_error(response)}. Completed rows remain in {CHECKPOINT_FILE.name}."
                    )

                try:
                    body = response.json()
                    prediction = parse_prediction(body, timestamp)
                except (ValueError, RuntimeError) as exc:
                    save_checkpoint()
                    raise RuntimeError(f"Invalid QuickML response at row {position}: {exc}") from exc

                predictions[index] = prediction
                save_checkpoint()  # Save after every successful row.
                print(f"    Prediction: {prediction:.4f} kW", flush=True)
                print(f"    Progress: {sum(v is not None for v in predictions)}/{total_rows} completed; checkpoint saved.\n", flush=True)
                break

    if any(value is None for value in predictions):
        save_checkpoint()
        raise RuntimeError(f"Predictions are incomplete. Resume by running the script again.")

    return np.asarray(predictions, dtype=float)

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART ENERGY MICROGRID - QUICKML PREDICTION TEST")
    print("=" * 60)

    sample_path = Path(__file__).with_name(
        "sample_test_input.csv"
    )

    if not sample_path.exists():
        print(
            f"ERROR: Input CSV not found:\n{sample_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        sample_df = pd.read_csv(sample_path)

        print(f"Input CSV: {sample_path.name}")
        print(f"Input rows: {len(sample_df)}")

        result = predict(sample_df)

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Total predictions received: {len(result)}")
        print("First five predictions (kW):")
        print(result[:5])

        # Save the predictions for later inspection.
        output_df = sample_df.copy()
        output_df["predicted_demand_kw"] = result

        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nPredictions saved to: {OUTPUT_FILE}")

    except (RuntimeError, ValueError, TypeError, FileNotFoundError) as exc:
        print("\nPREDICTION FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nPrediction cancelled by user.",
            file=sys.stderr,
        )
        sys.exit(130)
