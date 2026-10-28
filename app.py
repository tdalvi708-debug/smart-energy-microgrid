
from flask import Flask, render_template, jsonify, send_file, redirect, url_for
from pathlib import Path
import io
import traceback
import numpy as np
import pandas as pd

from battery_simulator import (
    simulate_battery,
    simulate_battery_baseline,
    validate_simulation,
    SOLAR_CAPACITY_KW,
    SOLAR_PERFORMANCE_RATIO,
)

try:
    from forecast_engine import forecast_demand
except ImportError:
    forecast_demand = None

try:
    from forecast_engine import forecast_next_24_hours
except ImportError:
    forecast_next_24_hours = None

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
WEATHER_FILE = BASE_DIR / "future_weather_demo.csv"
FORECAST_FILE = BASE_DIR / "forecast_results.csv"
BATTERY_FILE = BASE_DIR / "battery_simulation_results.csv"
BASELINE_FILE = BASE_DIR / "baseline_simulation_results.csv"


def run_forecast(weather: pd.DataFrame) -> pd.DataFrame:
    """Run whichever supported forecasting function is available."""
    if forecast_demand is not None:
        try:
            result = forecast_demand(weather.copy())
            if isinstance(result, pd.DataFrame):
                return result
        except TypeError:
            pass

    if forecast_next_24_hours is not None:
        start_time = pd.to_datetime(weather["timestamp"].iloc[0])
        result = forecast_next_24_hours(start_time, weather.copy())
        if isinstance(result, pd.DataFrame):
            return result

    raise RuntimeError(
        "Could not call the forecast function. Check forecast_engine.py "
        "for forecast_demand(weather) or "
        "forecast_next_24_hours(start_time, weather)."
    )


def find_column(df: pd.DataFrame, names, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:
        raise ValueError(
            f"Required column not found. Expected one of {names}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def numeric_series(df, names, default=0.0):
    col = find_column(df, names, required=False)

    if col is None:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def prepare_simulation():
    """Load the validated 1000-row simulation for the dashboard."""
    simulation_file = BASE_DIR / "starter_kit" / "full_battery_simulation.csv"

    if not simulation_file.exists():
        raise FileNotFoundError(
            f"Full simulation file not found: {simulation_file}"
        )

    battery = pd.read_csv(simulation_file)

    required = [
        "timestamp",
        "predicted_demand_kw",
        "renewable_generation_kw",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
        "interval_hours",
        "battery_soc_pct",
        "discharge_power_kw",
        "grid_backup_required_kw",
    ]

    missing = [column for column in required if column not in battery.columns]
    if missing:
        raise ValueError(f"Simulation CSV is missing columns: {missing}")

    battery["timestamp"] = pd.to_datetime(
        battery["timestamp"], errors="coerce"
    )

    battery = (
        battery.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if battery.empty:
        raise ValueError("The full simulation CSV contains no valid records.")

    numeric_columns = [
        "predicted_demand_kw",
        "renewable_generation_kw",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
        "interval_hours",
        "battery_soc_pct",
        "discharge_power_kw",
        "grid_backup_required_kw",
    ]

    for column in numeric_columns:
        battery[column] = pd.to_numeric(battery[column], errors="coerce")

    if battery[numeric_columns].isna().any().any():
        raise ValueError("The full simulation CSV contains missing numeric values.")

    if battery["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in the full simulation CSV.")

    # Rebuild the forecast input using only demand and weather inputs.
    forecast = battery[
        [
            "timestamp",
            "predicted_demand_kw",
            "ambient_temp_c",
            "humidity_pct",
            "solar_irradiance_wm2",
            "renewable_generation_kw",
        ]
    ].copy()

    # Preserve the already-computed intelligent battery simulation.
    # Do not run the intelligent battery dispatch a second time.
    battery.to_csv(BATTERY_FILE, index=False)
    forecast.to_csv(FORECAST_FILE, index=False)

    # Validate the original full simulation.
    validation = validate_simulation(battery)

    return forecast, battery, validation


def calculate_energy_metrics(battery: pd.DataFrame):
    """Calculate energy in kWh using power multiplied by interval duration."""
    hours = numeric_series(battery, ["interval_hours"], 1.0).clip(lower=0)

    demand = numeric_series(
        battery, ["predicted_demand_kw", "demand_kw"]
    )
    solar = numeric_series(
        battery, ["renewable_used_kw", "solar_used_kw"]
    )
    discharge = numeric_series(
        battery,
        ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"],
    )
    grid = numeric_series(
        battery,
        ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"],
    )

    total = float((demand * hours).sum())
    solar_energy = float((solar * hours).sum())
    battery_energy = float((discharge * hours).sum())
    grid_energy = float((grid * hours).sum())

    def percentage(value):
        return round(value / total * 100, 2) if total > 0 else 0.0

    return {
        "total_demand_kwh": round(total, 2),
        "solar_used_kwh": round(solar_energy, 2),
        "battery_used_kwh": round(battery_energy, 2),
        "grid_used_kwh": round(grid_energy, 2),
        "solar_contribution_pct": percentage(solar_energy),
        "battery_contribution_pct": percentage(battery_energy),
        "grid_dependency_pct": percentage(grid_energy),
    }


def safe_records(df: pd.DataFrame):
    """Convert a dataframe to JSON-safe records."""
    temp = df.copy().astype(object)
    temp = temp.where(pd.notna(temp), None)

    for col in temp.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            temp[col] = df[col].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ).where(df[col].notna(), None)

    return temp.to_dict(orient="records")



def build_dashboard_payload():
    forecast, battery, validation = prepare_simulation()

    # Generate baseline using the same 1000 forecast timestamps and inputs.
    baseline_input = forecast[
        [
            "timestamp",
            "predicted_demand_kw",
            "ambient_temp_c",
            "humidity_pct",
            "solar_irradiance_wm2",
            "renewable_generation_kw",
        ]
    ].copy()

    baseline = simulate_battery_baseline(baseline_input).copy()
    baseline["timestamp"] = pd.to_datetime(
        baseline["timestamp"], errors="coerce"
    )
    baseline.to_csv(BASELINE_FILE, index=False)

    if len(baseline) != len(battery):
        raise ValueError(
            f"Baseline has {len(baseline)} rows, but intelligent simulation "
            f"has {len(battery)} rows."
        )

    # Ensure baseline and intelligent simulation use matching timestamps.
    if not baseline["timestamp"].reset_index(drop=True).equals(
        pd.to_datetime(battery["timestamp"], errors="coerce").reset_index(drop=True)
    ):
        raise ValueError(
            "Baseline and intelligent simulation timestamps do not match."
        )

    energy = calculate_energy_metrics(battery)

    demand = numeric_series(
        forecast,
        ["predicted_demand_kw", "forecast_demand_kw", "demand_kw"],
    )
    solar = numeric_series(
        battery,
        ["renewable_generation_kw", "solar_generation_kw", "solar_kw"],
    )
    used_solar = numeric_series(
        battery, ["renewable_used_kw", "solar_used_kw"]
    )
    discharge = numeric_series(
        battery,
        ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"],
    )
    grid = numeric_series(
        battery,
        ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"],
    )
    soc = numeric_series(
        battery, ["battery_soc_pct", "battery_soc_after_pct", "soc_pct"]
    )
    charge = numeric_series(
        battery, ["charge_power_kw", "battery_charge_kw", "charge_kw"]
    )
    soc_before = numeric_series(
        battery, ["battery_soc_before_pct", "soc_before_pct"], 0.0
    )

    critical = battery.get(
        "critical_overload_alert",
        battery.get(
            "critical_alert",
            pd.Series(False, index=battery.index),
        ),
    )
    critical = pd.Series(critical, index=battery.index)
    critical_count = int(
        critical.fillna(False).astype(bool).sum()
    )

    baseline_grid = numeric_series(
        baseline,
        ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"],
    )
    baseline_discharge = numeric_series(
        baseline,
        ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"],
    )
    baseline_soc = numeric_series(
        baseline,
        ["battery_soc_pct", "battery_soc_after_pct", "soc_pct"],
    )

    hours = numeric_series(
        battery, ["interval_hours"], 1.0
    ).clip(lower=0)
    baseline_hours = numeric_series(
        baseline, ["interval_hours"], 1.0
    ).clip(lower=0)

    intelligent_peak_grid = float(grid.max()) if len(grid) else 0.0
    baseline_peak_grid = float(baseline_grid.max()) if len(baseline_grid) else 0.0

    peak_reduction_kw = baseline_peak_grid - intelligent_peak_grid
    peak_reduction_pct = (
        peak_reduction_kw / baseline_peak_grid * 100
        if baseline_peak_grid > 0
        else 0.0
    )

    intelligent_discharge_kwh = float((discharge * hours).sum())
    baseline_discharge_kwh = float(
        (baseline_discharge * baseline_hours).sum()
    )

    discharge_reduction_pct = (
        round(
            (1 - intelligent_discharge_kwh / baseline_discharge_kwh) * 100,
            2,
        )
        if baseline_discharge_kwh > 0
        else None
    )

    timestamps = pd.to_datetime(
        battery["timestamp"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M")

    forecasts = []
    batteries = []

    for i in range(len(battery)):
        ts = timestamps.iloc[i] if pd.notna(timestamps.iloc[i]) else str(i)

        forecasts.append({
            "timestamp": ts,
            "predicted_demand_kw": round(float(demand.iloc[i]), 3),
            "renewable_generation_kw": round(float(solar.iloc[i]), 3),
            "solar_used_kw": round(float(used_solar.iloc[i]), 3),
            "discharge_power_kw": round(float(discharge.iloc[i]), 3),
            "grid_backup_required_kw": round(float(grid.iloc[i]), 3),
            "battery_soc_pct": round(float(soc.iloc[i]), 3),
            "grid_mode": str(battery.iloc[i].get("grid_mode", "N/A")),
            "critical_alert": bool(critical.iloc[i]),
        })

        batteries.append({
            "timestamp": ts,
            "soc_before_pct": round(float(soc_before.iloc[i]), 3),
            "soc_after_pct": round(float(soc.iloc[i]), 3),
            "soc_percent": round(float(soc.iloc[i]), 3),
            "soc_pct": round(float(soc.iloc[i]), 3),
            "charge_power_kw": round(float(charge.iloc[i]), 3),
            "discharge_power_kw": round(float(discharge.iloc[i]), 3),
            "grid_backup_kw": round(float(grid.iloc[i]), 3),
            "grid_backup_required_kw": round(float(grid.iloc[i]), 3),
            "status": str(
                battery.iloc[i].get(
                    "battery_status",
                    battery.iloc[i].get("status", "N/A"),
                )
            ),
        })

    comparison = {
        "baseline_peak_grid_kw": round(baseline_peak_grid, 2),
        "intelligent_peak_grid_kw": round(intelligent_peak_grid, 2),
        "peak_grid_reduction_kw": round(peak_reduction_kw, 2),
        "peak_grid_reduction_pct": round(peak_reduction_pct, 2),
        "baseline_discharge_kwh": round(baseline_discharge_kwh, 2),
        "intelligent_discharge_kwh": round(intelligent_discharge_kwh, 2),
        "baseline_min_soc_pct": (
            round(float(baseline_soc.min()), 2)
            if len(baseline_soc) else None
        ),
        "intelligent_min_soc_pct": (
            round(float(soc.min()), 2) if len(soc) else None
        ),
        "discharge_reduction_pct": discharge_reduction_pct,
    }

    summary = {
        "count": int(len(forecast)),
        "avg_demand_kw": round(float(demand.mean()), 2) if len(demand) else 0.0,
        "peak_demand_kw": round(float(demand.max()), 2) if len(demand) else 0.0,
        "peak_solar_kw": round(float(solar.max()), 2) if len(solar) else 0.0,
        "peak_grid_kw": round(intelligent_peak_grid, 2),
        "latest_soc_pct": round(float(soc.iloc[-1]), 2) if len(soc) else 0.0,
        "min_soc_pct": round(float(soc.min()), 2) if len(soc) else 0.0,
        "max_soc_pct": round(float(soc.max()), 2) if len(soc) else 0.0,
        "critical_count": critical_count,
        "solar_capacity_kw": SOLAR_CAPACITY_KW,
        "solar_performance_ratio": SOLAR_PERFORMANCE_RATIO,
    }

    return {
        "success": True,
        "summary": summary,
        "energy": energy,
        "comparison": comparison,
        "validation": validation,
        "forecasts": forecasts,
        "batteries": batteries,
        "forecast_records": forecasts,
        "battery_records": batteries,
    }



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return redirect(url_for("index"))


@app.route("/run")
def run_again():
    return redirect(url_for("index", refresh="1"))


@app.route("/api/dashboard-data")
def api_dashboard_data():
    try:
        return jsonify(build_dashboard_payload())
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/download/report")
def download_report():
    if not BATTERY_FILE.exists():
        return "Run the simulation first before downloading the report.", 400

    battery = pd.read_csv(BATTERY_FILE)

    columns = [
        "timestamp",
        "predicted_demand_kw",
        "renewable_generation_kw",
        "renewable_used_kw",
        "charge_power_kw",
        "discharge_power_kw",
        "battery_soc_pct",
        "grid_backup_required_kw",
        "power_balance_error_kw",
        "grid_mode",
        "critical_overload_alert",
        "operational_status",
        "upcoming_peak_demand_kw",
        "target_reserve_soc_pct",
        "scheduling_strategy",
    ]

    available = [c for c in columns if c in battery.columns]

    output = io.BytesIO()
    battery[available].to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="smart_energy_simulation_report.csv",
    )


@app.route("/download/baseline")
def download_baseline():
    if not BASELINE_FILE.exists():
        return "Load the dashboard first to generate the baseline report.", 400

    return send_file(
        BASELINE_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="baseline_simulation_report.csv",
    )


if __name__ == "__main__":
    import os

    port = int(
        os.environ.get(
            "X_ZOHO_CATALYST_LISTEN_PORT",
            os.environ.get("PORT", "5000")
        )
    )

    app.run(host="0.0.0.0", port=port, debug=False)