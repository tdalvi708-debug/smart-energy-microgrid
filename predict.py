
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# Published Zoho Catalyst QuickML endpoint URL.
QUICKML_ENDPOINT_URL = (
    "https://api.catalyst.zoho.in/quickml/v1/"
    "project/103516000000013050/endpoints/predict"
)

ORG_ID = "60093284709"
ENVIRONMENT = "Development"
TIMEOUT_SECONDS = 30


def get_credentials():
    """Read credentials from environment variables without printing them."""
    key = os.getenv("QUICKML_ENDPOINT_KEY", "").strip()
    token = os.getenv("ZOHO_ACCESS_TOKEN", "").strip()

    if not key or not token:
        raise RuntimeError(
            "Credentials missing. Set QUICKML_ENDPOINT_KEY and "
            "ZOHO_ACCESS_TOKEN in the current CMD window."
        )

    placeholders = (
        "YOUR_",
        "PASTE_",
        "YAHAN_",
    )

    if any(p in key.upper() for p in placeholders):
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY still contains placeholder text. "
            "Set the actual endpoint key."
        )

    if any(p in token.upper() for p in placeholders):
        raise RuntimeError(
            "ZOHO_ACCESS_TOKEN still contains placeholder text. "
            "Set the actual access token."
        )

    # An endpoint URL is not an endpoint key.
    if key.lower().startswith(("http://", "https://")):
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY contains a URL. Set the actual "
            "X-QUICKML-ENDPOINT-KEY value instead."
        )

    return key, token


def prepare_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Validate input columns and prepare time-derived model features."""
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required = [
        "timestamp",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    missing = [col for col in required if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    data = input_df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    if data["timestamp"].isna().any():
        bad_rows = data.index[data["timestamp"].isna()].tolist()[:10]
        raise ValueError(
            f"Invalid timestamps found at input row indices: {bad_rows}"
        )

    numeric_columns = [
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid = data[numeric_columns].isna().any(axis=1)
    if invalid.any():
        bad_rows = data.index[invalid].tolist()[:10]
        raise ValueError(
            f"Missing or invalid weather values at row indices: {bad_rows}"
        )

    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Weather inputs contain infinite values.")

    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month

    return data


def parse_prediction(body: dict, timestamp) -> float:
    """Extract and validate one numeric prediction from QuickML."""
    if not isinstance(body, dict):
        raise RuntimeError(
            f"Unexpected response type for {timestamp}: "
            f"{type(body).__name__}"
        )

    if "result" not in body:
        # Do not print request headers or credentials.
        safe_keys = list(body.keys())
        raise RuntimeError(
            f"QuickML response for {timestamp} has no 'result' field. "
            f"Response fields: {safe_keys}. "
            "Check the endpoint's documented response format."
        )

    value = body["result"]

    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(
                f"Expected one prediction for {timestamp}, "
                f"but received {len(value)} results."
            )
        value = value[0]

    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"QuickML returned a non-numeric prediction for {timestamp}."
        ) from exc

    if not np.isfinite(prediction):
        raise RuntimeError(
            f"QuickML returned a non-finite prediction for {timestamp}."
        )

    return prediction


def predict(input_df: pd.DataFrame) -> np.ndarray:
    """Get demand predictions from the published QuickML endpoint."""
    if input_df.empty:
        return np.array([], dtype=float)

    data = prepare_input(input_df)
    key, token = get_credentials()

    headers = {
        "X-QUICKML-ENDPOINT-KEY": key,
        "Authorization": f"Zoho-oauthtoken {token}",
        "CATALYST-ORG": ORG_ID,
        "Environment": ENVIRONMENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    predictions = []

    with requests.Session() as session:
        for position, (_, row) in enumerate(data.iterrows(), start=1):
            timestamp = row["timestamp"]

            features = {
                "ambient_temp_c": float(row["ambient_temp_c"]),
                "humidity_pct": float(row["humidity_pct"]),
                "solar_irradiance_wm2": float(
                    row["solar_irradiance_wm2"]
                ),
                "hour": int(row["hour"]),
                "day": int(row["day"]),
                "month": int(row["month"]),
            }

            try:
                response = session.post(
                    QUICKML_ENDPOINT_URL,
                    headers=headers,
                    json={"data": features},
                    timeout=TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Network request failed at row {position} "
                    f"({timestamp}). Check internet connectivity. "
                    f"Details: {type(exc).__name__}"
                ) from exc

            if not response.ok:
                # Print the response body to diagnose the HTTP error.
                # Never print request headers, keys, or tokens.
                body_text = response.text[:1500].strip()

                if response.status_code == 401:
                    reason = (
                        "Authentication failed. Check that the endpoint "
                        "key is the actual key, the OAuth access token is "
                        "valid, and the organization/environment headers "
                        "match the published endpoint configuration."
                    )
                elif response.status_code == 403:
                    reason = (
                        "Access forbidden. Check endpoint permissions, "
                        "OAuth scopes, organization, and environment."
                    )
                elif response.status_code == 404:
                    reason = (
                        "Endpoint not found. Verify the published endpoint "
                        "URL and project ID."
                    )
                elif response.status_code == 429:
                    reason = (
                        "Request limit reached. Check the endpoint quota "
                        "and rate limits."
                    )
                else:
                    reason = "QuickML returned an HTTP error."

                raise RuntimeError(
                    f"{reason}\n"
                    f"HTTP status: {response.status_code}\n"
                    f"Response body: {body_text or '[empty response]'}\n"
                    f"Failed at input row: {position}, {timestamp}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"QuickML returned a non-JSON response at row "
                    f"{position}. HTTP status: {response.status_code}"
                ) from exc

            predictions.append(parse_prediction(body, timestamp))

            if position % 100 == 0 or position == len(data):
                print(f"Predictions completed: {position}/{len(data)}")

    return np.asarray(predictions, dtype=float)


if __name__ == "__main__":
    sample_path = Path(__file__).with_name("sample_test_input.csv")

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Test input file not found: {sample_path}"
        )

    sample_df = pd.read_csv(sample_path)
    result = predict(sample_df)

    print(f"\nQuickML predictions received: {len(result)}")
    print("First five predictions (kW):")
    print(result[:5])
