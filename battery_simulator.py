
"""
Smart Energy Microgrid - Battery Simulation Engine
HackSphere 3.0

Features:
- Solar-first energy allocation
- Forecast-aware battery reserve protection
- Intelligent peak shaving
- Optional controlled grid charging
- Battery SOC and power limits
- Power-balance validation
- Dashboard-compatible output columns
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATTERY_CAPACITY_KWH = 500.0

INITIAL_SOC_PERCENT = 80.0
MIN_SOC_PERCENT = 20.0
MAX_SOC_PERCENT = 80.0

MAX_CHARGE_POWER_KW = 100.0
MAX_DISCHARGE_POWER_KW = 100.0

CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

SOLAR_CAPACITY_KW = 250.0
SOLAR_PERFORMANCE_RATIO = 0.80

PEAK_THRESHOLD_KW = 200.0
GRID_THRESHOLD_KW = 160.0

NORMAL_RESERVE_SOC_PERCENT = 20.0
FORECAST_RESERVE_SOC_PERCENT = 40.0
RESERVE_TRIGGER_PEAK_KW = PEAK_THRESHOLD_KW

# Grid charging is deliberately optional.
# Enable it only if grid charging is part of your project design.
GRID_CHARGING_ENABLED = False
GRID_CHARGE_TARGET_SOC_PERCENT = 40.0
GRID_CHARGE_MAX_KW = 30.0

LOOKAHEAD_HOURS = 12.0
EPSILON = 1e-9


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def _find_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            "Required column not found. Expected one of: "
            + ", ".join(candidates)
        )

    return None


def _numeric_column(df, candidates, default=0.0):
    col = _find_column(df, candidates, required=False)

    if col is None:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _interval_hours(df):
    if "interval_hours" in df.columns:
        return (
            pd.to_numeric(df["interval_hours"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(lower=0.0)
            .astype(float)
        )

    if "timestamp" in df.columns and len(df) > 1:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        differences = timestamps.diff().dt.total_seconds() / 3600.0
        positive = differences[differences > 0]

        default_hours = (
            float(positive.median()) if not positive.empty else 1.0
        )

        differences.iloc[0] = default_hours
        return differences.fillna(default_hours).clip(lower=0.0)

    return pd.Series(1.0, index=df.index, dtype=float)


def _solar_from_row(df):
    if "renewable_generation_kw" in df.columns:
        return _numeric_column(
            df, ["renewable_generation_kw"]
        ).clip(lower=0.0, upper=SOLAR_CAPACITY_KW)

    if "solar_generation_kw" in df.columns:
        return _numeric_column(
            df, ["solar_generation_kw"]
        ).clip(lower=0.0, upper=SOLAR_CAPACITY_KW)

    irradiance = _numeric_column(
        df, ["solar_irradiance_wm2"]
    ).clip(lower=0.0)

    return (
        irradiance / 1000.0
        * SOLAR_CAPACITY_KW
        * SOLAR_PERFORMANCE_RATIO
    ).clip(lower=0.0, upper=SOLAR_CAPACITY_KW)


def _available_discharge_power(soc_percent, interval_hours):
    if interval_hours <= EPSILON:
        return 0.0

    energy_above_minimum = max(
        0.0,
        (soc_percent - MIN_SOC_PERCENT)
        / 100.0
        * BATTERY_CAPACITY_KWH
        * DISCHARGE_EFFICIENCY,
    )

    return min(
        MAX_DISCHARGE_POWER_KW,
        energy_above_minimum / interval_hours,
    )


def _available_charge_power(soc_percent, interval_hours):
    if interval_hours <= EPSILON:
        return 0.0

    remaining_capacity = max(
        0.0,
        (MAX_SOC_PERCENT - soc_percent)
        / 100.0
        * BATTERY_CAPACITY_KWH,
    )

    return min(
        MAX_CHARGE_POWER_KW,
        remaining_capacity / (CHARGE_EFFICIENCY * interval_hours),
    )


def _future_peak(net_demand, hours, start_index):
    elapsed = 0.0
    peak = 0.0

    for j in range(start_index, len(net_demand)):
        peak = max(peak, float(net_demand[j]))
        elapsed += float(hours[j])

        if elapsed >= LOOKAHEAD_HOURS:
            break

    return peak


def _target_reserve_soc(future_peak_kw):
    if future_peak_kw >= RESERVE_TRIGGER_PEAK_KW:
        return float(np.clip(
            FORECAST_RESERVE_SOC_PERCENT,
            MIN_SOC_PERCENT,
            MAX_SOC_PERCENT,
        ))

    return float(np.clip(
        NORMAL_RESERVE_SOC_PERCENT,
        MIN_SOC_PERCENT,
        MAX_SOC_PERCENT,
    ))


def _prepare_energy_inputs(forecast_df):
    df = forecast_df.copy()

    demand_col = _find_column(
        df,
        [
            "predicted_demand_kw",
            "forecast_demand_kw",
            "demand_prediction_kw",
            "target_demand_kw",
            "predicted_demand",
            "demand_kw",
        ],
    )

    demand = (
        pd.to_numeric(df[demand_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )

    solar = _solar_from_row(df).to_numpy(dtype=float)
    solar_used = np.minimum(demand, solar)
    solar_surplus = np.maximum(solar - demand, 0.0)
    net_demand = np.maximum(demand - solar_used, 0.0)
    hours = _interval_hours(df).to_numpy(dtype=float)

    if not np.isfinite(hours).all():
        raise ValueError("Simulation interval contains invalid values.")

    return df, demand, solar, solar_used, solar_surplus, net_demand, hours


# ============================================================
# 3. BATTERY SIMULATION
# ============================================================

def _simulate(forecast_df, intelligent=True):
    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("forecast_df must be a pandas DataFrame.")

    if forecast_df.empty:
        raise ValueError("Forecast data is empty.")

    if not 0 <= MIN_SOC_PERCENT <= MAX_SOC_PERCENT <= 100:
        raise ValueError("Battery SOC limits are invalid.")

    if not MIN_SOC_PERCENT <= INITIAL_SOC_PERCENT <= MAX_SOC_PERCENT:
        raise ValueError("Initial SOC is outside permitted limits.")

    if not 0 < CHARGE_EFFICIENCY <= 1:
        raise ValueError("Charge efficiency must be between 0 and 1.")

    if not 0 < DISCHARGE_EFFICIENCY <= 1:
        raise ValueError("Discharge efficiency must be between 0 and 1.")

    if GRID_CHARGE_MAX_KW < 0:
        raise ValueError("Grid charging power cannot be negative.")

    (
        df, demand, solar, solar_used, solar_surplus,
        net_demand, hours
    ) = _prepare_energy_inputs(forecast_df)

    n = len(df)
    soc = INITIAL_SOC_PERCENT

    soc_before_values = []
    soc_after_values = []
    solar_charge_values = []
    grid_charge_values = []
    discharge_values = []
    grid_values = []
    balance_errors = []
    upcoming_peaks = []
    target_reserves = []
    available_powers = []
    grid_modes = []
    operational_statuses = []
    battery_statuses = []
    alerts = []
    critical_demand_alerts = []
    critical_grid_alerts = []
    any_critical_alerts = []

    for i in range(n):
        interval = max(0.0, float(hours[i]))
        current_demand = float(demand[i])
        solar_used_kw = float(solar_used[i])
        surplus_kw = float(solar_surplus[i])
        net_kw = float(net_demand[i])

        soc_before = soc
        future_peak = _future_peak(net_demand, hours, i)
        reserve_soc = _target_reserve_soc(future_peak)

        upcoming_peaks.append(future_peak)
        target_reserves.append(reserve_soc)

        solar_charge_kw = 0.0
        grid_charge_kw = 0.0
        discharge_kw = 0.0

        # A. Use solar surplus to charge the battery.
        if interval > EPSILON and surplus_kw > EPSILON:
            charge_limit = _available_charge_power(soc, interval)
            solar_charge_kw = min(surplus_kw, charge_limit)

            soc += (
                solar_charge_kw * interval * CHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH * 100.0
            )
            soc = min(soc, MAX_SOC_PERCENT)

        # B. Optional grid charging for an upcoming forecast peak.
        # Grid charging does not occur unless explicitly enabled.
        if (
            intelligent
            and GRID_CHARGING_ENABLED
            and interval > EPSILON
            and solar_charge_kw <= EPSILON
            and surplus_kw <= EPSILON
            and current_demand <= GRID_THRESHOLD_KW
            and future_peak >= RESERVE_TRIGGER_PEAK_KW
            and soc < min(
                GRID_CHARGE_TARGET_SOC_PERCENT,
                reserve_soc,
            )
        ):
            charge_limit = _available_charge_power(soc, interval)
            target_soc = min(
                GRID_CHARGE_TARGET_SOC_PERCENT,
                reserve_soc,
                MAX_SOC_PERCENT,
            )

            needed_kw = (
                max(0.0, target_soc - soc)
                / 100.0
                * BATTERY_CAPACITY_KWH
                / (CHARGE_EFFICIENCY * interval)
            )

            grid_charge_kw = min(
                GRID_CHARGE_MAX_KW,
                charge_limit,
                needed_kw,
            )

            soc += (
                grid_charge_kw * interval * CHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH * 100.0
            )
            soc = min(soc, target_soc, MAX_SOC_PERCENT)

                # C. FORECAST-AWARE BATTERY DISPATCH
        available_kw = _available_discharge_power(soc, interval)
        available_powers.append(available_kw)

        if interval > EPSILON:
            desired_discharge_kw = max(
                0.0,
                net_kw - GRID_THRESHOLD_KW,
            )

            if intelligent:
                reserve_floor = reserve_soc
            else:
                reserve_floor = MIN_SOC_PERCENT

            energy_above_reserve_kwh = max(
                0.0,
                (soc - reserve_floor)
                / 100.0
                * BATTERY_CAPACITY_KWH
                * DISCHARGE_EFFICIENCY,
            )

            # Estimate the remaining peak-shaving energy demand.
            future_peak_energy_kwh = 0.0
            elapsed_hours = 0.0

            for j in range(i, len(net_demand)):
                future_interval = max(
                    0.0, float(hours[j])
                )

                if future_interval <= EPSILON:
                    continue

                future_net_kw = max(
                    0.0, float(net_demand[j])
                )

                peak_excess_kw = max(
                    0.0,
                    future_net_kw - GRID_THRESHOLD_KW,
                )

                future_peak_energy_kwh += (
                    peak_excess_kw * future_interval
                )
                elapsed_hours += future_interval

                if elapsed_hours >= LOOKAHEAD_HOURS:
                    break

            if intelligent and future_peak_energy_kwh > EPSILON:
                # Share available energy across the forecast window.
                allocated_energy_kwh = (
                    energy_above_reserve_kwh
                    * desired_discharge_kw
                    * interval
                    / future_peak_energy_kwh
                )

                budgeted_power_kw = (
                    allocated_energy_kwh / interval
                )
            elif intelligent:
                budgeted_power_kw = 0.0
            else:
                budgeted_power_kw = desired_discharge_kw

            discharge_kw = min(
                desired_discharge_kw,
                budgeted_power_kw,
                available_kw,
                MAX_DISCHARGE_POWER_KW,
            )

            soc -= (
                discharge_kw
                * interval
                / DISCHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH
                * 100.0
            )

            soc = float(
                np.clip(
                    soc,
                    MIN_SOC_PERCENT,
                    MAX_SOC_PERCENT,
                )
            )

        # D. Grid supplies the remaining demand plus any grid charging.
        grid_kw = max(
            0.0,
            current_demand + grid_charge_kw
            - solar_used_kw - discharge_kw,
        )

        balance_error = (
            current_demand + grid_charge_kw
            - solar_used_kw - discharge_kw - grid_kw
        )

        # E. Alerts and status.
        critical_demand = current_demand >= PEAK_THRESHOLD_KW
        critical_grid = grid_kw > PEAK_THRESHOLD_KW
        any_critical = critical_demand or critical_grid

        grid_mode = (
            "High" if grid_kw > GRID_THRESHOLD_KW else "Normal"
        )

        if soc <= MIN_SOC_PERCENT + 1e-6:
            battery_status = "Low"
        elif soc < reserve_soc:
            battery_status = "Reserve"
        else:
            battery_status = "Healthy"

        if discharge_kw > EPSILON:
            operational_status = "Discharging"
        elif solar_charge_kw > EPSILON:
            operational_status = "Solar Charging"
        elif grid_charge_kw > EPSILON:
            operational_status = "Grid Charging"
        else:
            operational_status = "Idle"

        if critical_grid:
            alert_text = "CRITICAL GRID OVERLOAD"
        elif critical_demand:
            alert_text = "CRITICAL DEMAND"
        else:
            alert_text = "NORMAL"

        soc_before_values.append(soc_before)
        soc_after_values.append(soc)
        solar_charge_values.append(solar_charge_kw)
        grid_charge_values.append(grid_charge_kw)
        discharge_values.append(discharge_kw)
        grid_values.append(grid_kw)
        balance_errors.append(balance_error)

        grid_modes.append(grid_mode)
        operational_statuses.append(operational_status)
        battery_statuses.append(battery_status)
        alerts.append(alert_text)
        critical_demand_alerts.append(critical_demand)
        critical_grid_alerts.append(critical_grid)
        any_critical_alerts.append(any_critical)

    # F. Dashboard-compatible output.
    result = df.copy()

    result["predicted_demand_kw"] = demand
    result["renewable_generation_kw"] = solar
    result["renewable_used_kw"] = solar_used
    result["solar_used_kw"] = solar_used
    result["remaining_demand_kw"] = net_demand
    result["solar_surplus_kw"] = solar_surplus
    result["interval_hours"] = hours

    result["battery_soc_before_pct"] = soc_before_values
    result["battery_soc_after_pct"] = soc_after_values
    result["battery_soc_pct"] = soc_after_values

    result["solar_charge_kw"] = solar_charge_values
    result["grid_charge_kw"] = grid_charge_values
    result["charge_power_kw"] = (
        np.array(solar_charge_values) + np.array(grid_charge_values)
    )
    result["battery_charge_kw"] = result["charge_power_kw"]

    result["discharge_power_kw"] = discharge_values
    result["battery_discharge_kw"] = discharge_values

    result["grid_backup_required_kw"] = grid_values
    result["grid_backup_kw"] = grid_values
    result["grid_kw"] = grid_values
    result["power_balance_error_kw"] = balance_errors

    result["upcoming_peak_demand_kw"] = upcoming_peaks
    result["target_reserve_soc_pct"] = target_reserves
    result["required_battery_power_kw"] = np.maximum(
        0.0, net_demand - GRID_THRESHOLD_KW
    )
    result["available_battery_power_kw"] = available_powers

    result["grid_mode"] = grid_modes
    result["operational_status"] = operational_statuses
    result["battery_status"] = battery_statuses

    result["critical_demand_alert"] = critical_demand_alerts
    result["critical_grid_alert"] = critical_grid_alerts
    result["critical_overload_alert"] = critical_grid_alerts
    result["critical_alert"] = critical_grid_alerts
    result["any_critical_alert"] = any_critical_alerts
    result["alert"] = alerts

    result["scheduling_strategy"] = (
        "Intelligent Peak Shaving"
        if intelligent else "Baseline Threshold Control"
    )

    return result


# ============================================================
# 4. PUBLIC FUNCTIONS
# ============================================================

def simulate_battery(forecast_df):
    return _simulate(forecast_df, intelligent=True)


def simulate_battery_baseline(forecast_df):
    return _simulate(forecast_df, intelligent=False)


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_simulation(simulation_df):
    df = simulation_df.copy()

    if df.empty:
        return {
            "power_balance_pass": False,
            "max_power_balance_error_kw": 0.0,
            "soc_limits_pass": False,
            "soc_limit_pass": False,
            "min_soc_pct": 0.0,
            "max_soc_pct": 0.0,
            "charge_power_limit_pass": False,
            "discharge_power_limit_pass": False,
            "forecast_reserve_protection_pass": False,
        }

    soc = _numeric_column(
        df, ["battery_soc_pct", "battery_soc_after_pct"]
    )
    charge = _numeric_column(
        df, ["charge_power_kw", "battery_charge_kw"]
    )
    discharge = _numeric_column(
        df, ["discharge_power_kw", "battery_discharge_kw"]
    )
    balance = _numeric_column(df, ["power_balance_error_kw"])

    min_soc = float(soc.min())
    max_soc = float(soc.max())
    max_balance_error = float(balance.abs().max())
    max_charge = float(charge.max())
    max_discharge = float(discharge.max())

    soc_pass = (
        min_soc >= MIN_SOC_PERCENT - 1e-6
        and max_soc <= MAX_SOC_PERCENT + 1e-6
    )
    charge_pass = max_charge <= MAX_CHARGE_POWER_KW + 1e-6
    discharge_pass = max_discharge <= MAX_DISCHARGE_POWER_KW + 1e-6
    power_balance_pass = max_balance_error <= 1e-6

    reserve_pass = True

    if (
        "target_reserve_soc_pct" in df.columns
        and "battery_soc_before_pct" in df.columns
        and "battery_soc_pct" in df.columns
        and "discharge_power_kw" in df.columns
        and "scheduling_strategy" in df.columns
        and df["scheduling_strategy"].astype(str)
        .str.contains("Intelligent", case=False).any()
    ):
        intelligent_rows = (
            df["scheduling_strategy"].astype(str)
            .str.contains("Intelligent", case=False)
        )

        target = _numeric_column(df, ["target_reserve_soc_pct"])
        soc_after = _numeric_column(df, ["battery_soc_pct"])
        discharge_values = _numeric_column(df, ["discharge_power_kw"])

        # When discharging, the battery must not fall below its
        # predicted reserve. Rows without discharge must also respect
        # reserve if they started at or above that reserve.
        discharged = intelligent_rows & (discharge_values > EPSILON)
        reserve_pass = bool(
            (
                soc_after[discharged]
                >= target[discharged] - 1e-6
            ).all()
        )

    return {
        "power_balance_pass": power_balance_pass,
        "max_power_balance_error_kw": max_balance_error,
        "soc_limits_pass": soc_pass,
        "soc_limit_pass": soc_pass,
        "min_soc_pct": min_soc,
        "max_soc_pct": max_soc,
        "charge_power_limit_pass": charge_pass,
        "discharge_power_limit_pass": discharge_pass,
        "forecast_reserve_protection_pass": reserve_pass,
    }


# ============================================================
# 6. STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    forecast_file = base_dir / "forecast_results.csv"

    if not forecast_file.exists():
        print(
            "forecast_results.csv not found. "
            "Run the Flask dashboard first to generate a forecast."
        )
    else:
        forecast = pd.read_csv(forecast_file)

        smart = simulate_battery(forecast)
        baseline = simulate_battery_baseline(forecast)

        display_columns = [
            "timestamp",
            "predicted_demand_kw",
            "renewable_generation_kw",
            "upcoming_peak_demand_kw",
            "target_reserve_soc_pct",
            "solar_charge_kw",
            "grid_charge_kw",
            "discharge_power_kw",
            "battery_soc_pct",
            "grid_backup_required_kw",
            "critical_demand_alert",
            "critical_grid_alert",
        ]

        print("\nINTELLIGENT SIMULATION")
        print(smart[
            [c for c in display_columns if c in smart.columns]
        ].to_string(index=False))

        print("\nINTELLIGENT VALIDATION")
        print(validate_simulation(smart))

        print("\nBASELINE VALIDATION")
        print(validate_simulation(baseline))

        hours = _interval_hours(forecast).to_numpy()

        smart_grid_energy = float(
            (smart["grid_backup_required_kw"].to_numpy() * hours).sum()
        )
        baseline_grid_energy = float(
            (baseline["grid_backup_required_kw"].to_numpy() * hours).sum()
        )

        print(f"\nIntelligent grid energy (kWh): {smart_grid_energy:.2f}")
        print(f"Baseline grid energy (kWh): {baseline_grid_energy:.2f}")
        print(
            "Intelligent peak grid (kW): "
            f"{smart['grid_backup_required_kw'].max():.2f}"
        )
        print(
            "Baseline peak grid (kW): "
            f"{baseline['grid_backup_required_kw'].max():.2f}"
        )
        print(
            "Solar charging intervals:",
            int((smart["solar_charge_kw"] > EPSILON).sum()),
        )
        print(
            "Grid charging intervals:",
            int((smart["grid_charge_kw"] > EPSILON).sum()),
        )
        print(
            "Battery discharging intervals:",
            int((smart["discharge_power_kw"] > EPSILON).sum()),
        )
