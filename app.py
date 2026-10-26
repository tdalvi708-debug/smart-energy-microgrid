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
    """Call whichever supported forecast function exists in forecast_engine.py."""
    if forecast_demand is not None:
        try:
            result = forecast_demand(weather.copy())
            if isinstance(result, pd.DataFrame):
                return result
        except TypeError:
            # Try the alternative signature below.
            pass
    if forecast_next_24_hours is not None:
        start_time = pd.to_datetime(weather["timestamp"].iloc[0])
        result = forecast_next_24_hours(start_time, weather.copy())
        if isinstance(result, pd.DataFrame):
            return result
    raise RuntimeError(
        "Could not call the forecast function. Check forecast_engine.py: "
        "expected forecast_demand(weather) or forecast_next_24_hours(start_time, weather)."
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
    """Run the forecast and intelligent battery dispatch, saving output CSVs."""
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Weather demo file not found: {WEATHER_FILE}")

    weather = pd.read_csv(WEATHER_FILE)
    required = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather file is missing columns: {missing}")

    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    weather = weather.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if weather.empty:
        raise ValueError("No valid timestamps found in future_weather_demo.csv.")

    forecast = run_forecast(weather.copy()).copy()
    if forecast.empty:
        raise ValueError("The forecasting function returned no rows.")
    if "timestamp" not in forecast.columns:
        if len(forecast) != len(weather):
            raise ValueError("Forecast has no timestamp column and row count differs from weather data.")
        forecast["timestamp"] = weather["timestamp"].values

    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce")
    forecast = forecast.dropna(subset=["timestamp"]).copy()
    weather_cols = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
    forecast = forecast.drop(columns=[c for c in weather_cols[1:] if c in forecast.columns], errors="ignore")
    forecast = forecast.merge(weather[weather_cols], on="timestamp", how="left")
    if forecast["solar_irradiance_wm2"].isna().any():
        raise ValueError("Forecast timestamps do not match timestamps in the weather file.")

    demand_col = find_column(forecast, [
        "predicted_demand_kw", "forecast_demand_kw", "demand_prediction_kw",
        "target_demand_kw", "predicted_demand", "demand_kw"
    ])
    forecast["predicted_demand_kw"] = pd.to_numeric(forecast[demand_col], errors="coerce")
    if forecast["predicted_demand_kw"].isna().any():
        raise ValueError("Forecast contains missing or non-numeric demand values.")
    forecast["predicted_demand_kw"] = forecast["predicted_demand_kw"].clip(lower=0)

       
    # Solar generation is estimated from irradiance and the configured solar capacity.
    forecast["renewable_generation_kw"] = (
        pd.to_numeric(forecast["solar_irradiance_wm2"], errors="coerce")
        .fillna(0).clip(lower=0) / 1000.0 * SOLAR_CAPACITY_KW * SOLAR_PERFORMANCE_RATIO
    )

    # Run battery dispatch before saving the forecast CSV so dashboard
    # forecast rows can include the actual simulated battery state.
    battery = simulate_battery(forecast.copy()).copy()

    # Keep common aliases so both older and newer dashboard code can read the results.
    aliases = {
        "solar_generation_kw": "renewable_generation_kw",
        "solar_used_kw": "renewable_used_kw",
        "battery_charge_kw": "charge_power_kw",
        "battery_discharge_kw": "discharge_power_kw",
        "battery_soc_after_pct": "battery_soc_pct",
        "grid_backup_kw": "grid_backup_required_kw",
        "critical_alert": "critical_overload_alert",
    }
    for alias, source in aliases.items():
        if alias not in battery.columns and source in battery.columns:
            battery[alias] = battery[source]

    battery["timestamp"] = pd.to_datetime(battery["timestamp"], errors="coerce")
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce")

    # Merge only the battery values needed by the forecast dashboard.
    battery_columns = [
        "timestamp",
        "battery_soc_pct",
        "battery_status",
        "upcoming_peak_demand_kw",
        "target_reserve_soc_pct",
        "required_battery_power_kw",
        "available_battery_power_kw",
        "grid_backup_required_kw",
        "operational_status",
        "alert",
        "critical_overload_alert",
    ]
    battery_columns = [
        c for c in battery_columns if c in battery.columns
    ]

    forecast = forecast.drop(
        columns=[
            c for c in battery_columns
            if c != "timestamp" and c in forecast.columns
        ],
        errors="ignore",
    )

    forecast = forecast.merge(
        battery[battery_columns].drop_duplicates(subset=["timestamp"]),
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    if forecast["battery_soc_pct"].isna().any():
        raise ValueError(
            "Some forecast timestamps have no matching battery SOC values."
        )

    forecast.to_csv(FORECAST_FILE, index=False)
    battery.to_csv(BATTERY_FILE, index=False)

    return forecast, battery, validate_simulation(battery)




def calculate_energy_metrics(battery: pd.DataFrame):
    hours = numeric_series(battery, ["interval_hours"], 1.0).clip(lower=0)
    demand = numeric_series(battery, ["predicted_demand_kw", "demand_kw"])
    solar = numeric_series(battery, ["renewable_used_kw", "solar_used_kw"])
    discharge = numeric_series(battery, ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"])
    grid = numeric_series(battery, ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"])
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
    """Convert a dataframe to JSON-safe dictionaries (timestamps and NaN included)."""
    temp = df.copy().astype(object)
    temp = temp.where(pd.notna(temp), None)
    for col in temp.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            temp[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").where(df[col].notna(), None)
    return temp.to_dict(orient="records")


def build_dashboard_payload():
    forecast, battery, validation = prepare_simulation()
    baseline = simulate_battery_baseline(forecast.copy()).copy()
    baseline.to_csv(BASELINE_FILE, index=False)

    energy = calculate_energy_metrics(battery)
    demand = numeric_series(forecast, ["predicted_demand_kw", "forecast_demand_kw", "demand_kw"])
    solar = numeric_series(battery, ["renewable_generation_kw", "solar_generation_kw", "solar_kw"])
    used_solar = numeric_series(battery, ["renewable_used_kw", "solar_used_kw"])
    discharge = numeric_series(battery, ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"])
    grid = numeric_series(battery, ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"])
    soc = numeric_series(battery, ["battery_soc_pct", "battery_soc_after_pct", "soc_pct"])
    charge = numeric_series(battery, ["charge_power_kw", "battery_charge_kw", "charge_kw"])
    soc_before = numeric_series(battery, ["battery_soc_before_pct", "soc_before_pct"], 0.0)
    critical = battery.get("critical_overload_alert", battery.get("critical_alert", pd.Series(False, index=battery.index)))
    critical_count = int(pd.Series(critical).fillna(False).astype(bool).sum())

    baseline_grid = numeric_series(baseline, ["grid_backup_required_kw", "grid_backup_kw", "grid_kw"])
    baseline_discharge = numeric_series(baseline, ["discharge_power_kw", "battery_discharge_kw", "discharge_kw"])
    baseline_soc = numeric_series(baseline, ["battery_soc_pct", "battery_soc_after_pct", "soc_pct"])
    intelligent_peak_grid = float(grid.max()) if len(grid) else 0.0
    baseline_peak_grid = float(baseline_grid.max()) if len(baseline_grid) else 0.0
    peak_reduction_kw = baseline_peak_grid - intelligent_peak_grid
    peak_reduction_pct = (peak_reduction_kw / baseline_peak_grid * 100) if baseline_peak_grid > 0 else 0.0
    intelligent_discharge_kwh = float((discharge * numeric_series(battery, ["interval_hours"], 1.0)).sum())
    baseline_discharge_kwh = float((baseline_discharge * numeric_series(baseline, ["interval_hours"], 1.0)).sum())

    timestamps = pd.to_datetime(battery["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    forecasts = []
    batteries = []
    for i in range(len(battery)):
        ts = timestamps.iloc[i] if pd.notna(timestamps.iloc[i]) else str(i)
        forecasts.append({
            "timestamp": ts,
            "predicted_demand_kw": round(float(demand.iloc[i]) if i < len(demand) else 0, 3),
            "renewable_generation_kw": round(float(solar.iloc[i]), 3),
            "solar_used_kw": round(float(used_solar.iloc[i]), 3),
            "discharge_power_kw": round(float(discharge.iloc[i]), 3),
            "grid_backup_required_kw": round(float(grid.iloc[i]), 3),
            "battery_soc_pct": round(float(soc.iloc[i]), 3),
            "grid_mode": str(battery.iloc[i].get("grid_mode", "N/A")),
            "critical_alert": bool(critical.iloc[i]) if hasattr(critical, "iloc") else bool(critical),
        })
        batteries.append({
            "timestamp": ts,
            "soc_before_pct": round(float(soc_before.iloc[i]), 3),
            "soc_percent": round(float(soc.iloc[i]), 3),
            "soc_pct": round(float(soc.iloc[i]), 3),
            "soc": round(float(soc.iloc[i]), 3),
            "charge_power_kw": round(float(charge.iloc[i]), 3),
            "charging_kw": round(float(charge.iloc[i]), 3),
            "discharge_power_kw": round(float(discharge.iloc[i]), 3),
            "discharging_kw": round(float(discharge.iloc[i]), 3),
            "grid_backup_kw": round(float(grid.iloc[i]), 3),
            "status": str(battery.iloc[i].get("battery_status", battery.iloc[i].get("status", "N/A"))),
        })

    comparison = {
        "baseline_peak_grid_kw": round(baseline_peak_grid, 2),
        "intelligent_peak_grid_kw": round(intelligent_peak_grid, 2),
        "peak_grid_reduction_kw": round(peak_reduction_kw, 2),
        "peak_grid_reduction_pct": round(peak_reduction_pct, 2),
        "baseline_discharge_kwh": round(baseline_discharge_kwh, 2),
        "intelligent_discharge_kwh": round(intelligent_discharge_kwh, 2),
        "baseline_min_soc_pct": round(float(baseline_soc.min()), 2) if len(baseline_soc) else None,
        "intelligent_min_soc_pct": round(float(soc.min()), 2) if len(soc) else None,
        "discharge_reduction_pct": round((1 - intelligent_discharge_kwh / baseline_discharge_kwh) * 100, 2) if baseline_discharge_kwh > 0 else 0.0,
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
    # The frontend requests fresh simulation data when it loads.
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
        "timestamp", "predicted_demand_kw", "renewable_generation_kw", "renewable_used_kw",
        "charge_power_kw", "discharge_power_kw", "battery_soc_pct", "grid_backup_required_kw",
        "power_balance_error_kw", "grid_mode", "critical_overload_alert", "operational_status",
        "upcoming_peak_demand_kw", "target_reserve_soc_pct", "scheduling_strategy",
    ]
    available = [c for c in columns if c in battery.columns]
    output = io.BytesIO()
    battery[available].to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="smart_energy_simulation_report.csv")


@app.route("/download/baseline")
def download_baseline():
    if not BASELINE_FILE.exists():
        return "Load the dashboard first to generate the baseline report.", 400
    return send_file(BASELINE_FILE, mimetype="text/csv", as_attachment=True, download_name="baseline_simulation_report.csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
