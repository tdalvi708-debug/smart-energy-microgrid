
"""
Smart Energy Microgrid - Battery Simulation Engine
HackSphere 3.0

Features:
- Solar-first energy allocation
- Forecast-aware battery reserve
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

# Demand thresholds
PEAK_THRESHOLD_KW = 200.0
GRID_THRESHOLD_KW = 160.0

# Forecast-aware reserve settings
NORMAL_RESERVE_SOC_PERCENT = 20.0
FORECAST_RESERVE_SOC_PERCENT = 40.0

# Reserve is increased when a future peak reaches this threshold.
RESERVE_TRIGGER_PEAK_KW = PEAK_THRESHOLD_KW

# Optional grid charging. Disabled by default to avoid
# unintentionally increasing grid energy consumption.
GRID_CHARGING_ENABLED = False
GRID_CHARGE_TARGET_SOC_PERCENT = 40.0
GRID_CHARGE_MAX_KW = 30.0

# Look ahead for upcoming high-demand periods.
LOOKAHEAD_HOURS = 12.0

EPSILON = 1e-9


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def _find_column(df, candidates, required=True):
    """Return the first matching column name."""

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
    """Read a numeric column and safely fill invalid values."""

    col = _find_column(df, candidates, required=False)

    if col is None:
        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(default)
        .astype(float)
    )


def _interval_hours(df):
    """Determine the duration of each simulation interval."""

    if "interval_hours" in df.columns:
        hours = pd.to_numeric(
            df["interval_hours"],
            errors="coerce",
        ).fillna(1.0)

        return hours.clip(lower=0.0).astype(float)

    if "timestamp" in df.columns and len(df) > 1:
        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        differences = (
            timestamps.diff().dt.total_seconds() / 3600.0
        )

        positive = differences[differences > 0]

        default_hours = (
            float(positive.median())
            if not positive.empty
            else 1.0
        )

        differences.iloc[0] = default_hours

        return differences.fillna(
            default_hours
        ).clip(lower=0.0)

    return pd.Series(
        1.0,
        index=df.index,
        dtype=float,
    )


def _solar_from_row(df):
    """Read available solar generation or estimate it from irradiance."""

    if "renewable_generation_kw" in df.columns:
        return _numeric_column(
            df,
            ["renewable_generation_kw"],
        ).clip(lower=0.0)

    if "solar_generation_kw" in df.columns:
        return _numeric_column(
            df,
            ["solar_generation_kw"],
        ).clip(lower=0.0)

    irradiance = _numeric_column(
        df,
        ["solar_irradiance_wm2"],
    ).clip(lower=0.0)

    return (
        irradiance
        / 1000.0
        * SOLAR_CAPACITY_KW
        * SOLAR_PERFORMANCE_RATIO
    ).clip(
        lower=0.0,
        upper=SOLAR_CAPACITY_KW,
    )


def _available_discharge_power(soc_percent, interval_hours):
    """
    Calculate maximum output power allowed by the energy above
    the absolute minimum SOC.

    The intelligent strategy applies its higher forecast reserve
    separately before choosing the final discharge power.
    """

    if interval_hours <= EPSILON:
        return 0.0

    available_energy_kwh = max(
        0.0,
        (soc_percent - MIN_SOC_PERCENT)
        / 100.0
        * BATTERY_CAPACITY_KWH
        * DISCHARGE_EFFICIENCY,
    )

    return min(
        MAX_DISCHARGE_POWER_KW,
        available_energy_kwh / interval_hours,
    )


def _available_charge_power(soc_percent, interval_hours):
    """Calculate the maximum charge power allowed by battery capacity."""

    if interval_hours <= EPSILON:
        return 0.0

    remaining_capacity_kwh = max(
        0.0,
        (MAX_SOC_PERCENT - soc_percent)
        / 100.0
        * BATTERY_CAPACITY_KWH,
    )

    return min(
        MAX_CHARGE_POWER_KW,
        remaining_capacity_kwh
        / (CHARGE_EFFICIENCY * interval_hours),
    )


def _future_peak(net_demand, hours, start_index):
    """Find the maximum net demand within the look-ahead window."""

    elapsed = 0.0
    peak = 0.0

    for j in range(start_index, len(net_demand)):
        peak = max(
            peak,
            float(net_demand[j]),
        )

        elapsed += float(hours[j])

        if elapsed >= LOOKAHEAD_HOURS:
            break

    return peak


def _target_reserve_soc(future_peak_kw):
    """
    Protect 40% SOC if a high-demand period is forecast.

    Otherwise, use the normal 20% minimum reserve.

    This sets a target; it does not magically recharge a battery
    whose current SOC is already below that target.
    """

    if future_peak_kw >= RESERVE_TRIGGER_PEAK_KW:
        return min(
            MAX_SOC_PERCENT,
            max(
                MIN_SOC_PERCENT,
                FORECAST_RESERVE_SOC_PERCENT,
            ),
        )

    return min(
        MAX_SOC_PERCENT,
        max(
            MIN_SOC_PERCENT,
            NORMAL_RESERVE_SOC_PERCENT,
        ),
    )


# ============================================================
# 3. SOLAR-FIRST ENERGY ALLOCATION
# ============================================================

def _prepare_energy_inputs(forecast_df):
    """Prepare demand, solar, surplus, net demand, and interval arrays."""

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
        pd.to_numeric(
            df[demand_col],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )

    solar = _solar_from_row(df).to_numpy(dtype=float)

    # Solar meets demand first.
    solar_used = np.minimum(demand, solar)

    # Only excess solar can charge the battery.
    solar_surplus = np.maximum(
        solar - demand,
        0.0,
    )

    # Demand remaining after direct solar use.
    net_demand = np.maximum(
        demand - solar_used,
        0.0,
    )

    hours = _interval_hours(df).to_numpy(dtype=float)

    return (
        df,
        demand,
        solar,
        solar_used,
        solar_surplus,
        net_demand,
        hours,
    )


# ============================================================
# 4. SMART BATTERY SIMULATION
# ============================================================

def _simulate(forecast_df, intelligent=True):
    """
    Simulate intelligent peak shaving or baseline threshold control.

    Energy accounting:
        Grid + direct solar + battery discharge
        = demand + battery charging

    Solar used directly to serve demand is not counted twice.
    """

    if not isinstance(forecast_df, pd.DataFrame):
        raise TypeError(
            "forecast_df must be a pandas DataFrame."
        )

    if forecast_df.empty:
        raise ValueError("Forecast data is empty.")

    if not 0.0 <= MIN_SOC_PERCENT <= MAX_SOC_PERCENT <= 100.0:
        raise ValueError("Battery SOC limits are invalid.")

    if not MIN_SOC_PERCENT <= INITIAL_SOC_PERCENT <= MAX_SOC_PERCENT:
        raise ValueError("Initial SOC is outside the permitted limits.")

    if not 0.0 < CHARGE_EFFICIENCY <= 1.0:
        raise ValueError("Charge efficiency must be between 0 and 1.")

    if not 0.0 < DISCHARGE_EFFICIENCY <= 1.0:
        raise ValueError("Discharge efficiency must be between 0 and 1.")

    (
        df,
        demand,
        solar,
        solar_used,
        solar_surplus,
        net_demand,
        hours,
    ) = _prepare_energy_inputs(forecast_df)

    n = len(df)
    soc = INITIAL_SOC_PERCENT

    # Output arrays
    soc_before_values = []
    soc_after_values = []

    solar_charge_values = []
    grid_charge_values = []
    total_charge_values = []
    discharge_values = []

    grid_values = []
    balance_errors = []

    upcoming_peaks = []
    target_reserves = []
    required_battery_power = []
    available_battery_power = []

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
        current_solar = float(solar[i])
        solar_used_kw = float(solar_used[i])
        surplus_kw = float(solar_surplus[i])
        net_kw = float(net_demand[i])

        soc_before = soc

        # Forecast the maximum net demand in the look-ahead window.
        future_peak = _future_peak(
            net_demand,
            hours,
            i,
        )

        # Calculate the reserve target for this interval.
        reserve_soc = _target_reserve_soc(
            future_peak
        )

        upcoming_peaks.append(future_peak)
        target_reserves.append(reserve_soc)

        solar_charge_kw = 0.0
        grid_charge_kw = 0.0
        discharge_kw = 0.0

        # ----------------------------------------------------
        # A. SOLAR CHARGING
        # ----------------------------------------------------

        if interval > EPSILON:
            charge_limit = _available_charge_power(
                soc,
                interval,
            )

            solar_charge_kw = min(
                surplus_kw,
                charge_limit,
                MAX_CHARGE_POWER_KW,
            )

            soc += (
                solar_charge_kw
                * interval
                * CHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH
                * 100.0
            )

            soc = min(
                soc,
                MAX_SOC_PERCENT,
            )

        # ----------------------------------------------------
        # B. OPTIONAL CONTROLLED GRID CHARGING
        # ----------------------------------------------------

        # Disabled by default. It runs only when explicitly enabled,
        # there is no solar surplus, and demand is within the threshold.
        if (
            intelligent
            and GRID_CHARGING_ENABLED
            and interval > EPSILON
            and solar_charge_kw <= EPSILON
            and surplus_kw <= EPSILON
            and current_demand <= GRID_THRESHOLD_KW
            and soc < GRID_CHARGE_TARGET_SOC_PERCENT
        ):
            charge_limit = _available_charge_power(
                soc,
                interval,
            )

            needed_to_target_kw = (
                (
                    GRID_CHARGE_TARGET_SOC_PERCENT
                    - soc
                )
                / 100.0
                * BATTERY_CAPACITY_KWH
                / (
                    CHARGE_EFFICIENCY
                    * interval
                )
            )

            grid_charge_kw = min(
                GRID_CHARGE_MAX_KW,
                charge_limit,
                max(0.0, needed_to_target_kw),
            )

            soc += (
                grid_charge_kw
                * interval
                * CHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH
                * 100.0
            )

            soc = min(
                soc,
                GRID_CHARGE_TARGET_SOC_PERCENT,
                MAX_SOC_PERCENT,
            )

        # ----------------------------------------------------
        # C. BATTERY DISCHARGING
        # ----------------------------------------------------

        available_kw = _available_discharge_power(
            soc,
            interval,
        )

        available_battery_power.append(
            available_kw
        )

        if interval > EPSILON:
            desired_discharge_kw = max(
                0.0,
                net_kw - GRID_THRESHOLD_KW,
            )

            
            
            if intelligent:
                # Energy available above the forecast-aware reserve.
                reserve_energy_kwh = max(
                    0.0,
                    (soc - reserve_soc)
                    / 100.0
                    * BATTERY_CAPACITY_KWH,
                )

                # Maximum safe discharge power for this interval.
                reserve_safe_power_kw = (
                    reserve_energy_kwh
                    * DISCHARGE_EFFICIENCY
                    / interval
                )

                # Identify the remaining high-demand intervals.
                remaining_peak_hours = sum(
                    max(0.0, float(hours[j]))
                    for j in range(i, n)
                    if float(net_demand[j]) >= PEAK_THRESHOLD_KW
                )

                # Allocate stored energy across future peak hours.
                if remaining_peak_hours > EPSILON:
                    planned_discharge_kw = (
                        reserve_energy_kwh
                        * DISCHARGE_EFFICIENCY
                        / remaining_peak_hours
                    )
                else:
                    planned_discharge_kw = 0.0

                # Respect the grid target, energy reserve and
                # battery power limits.
                discharge_kw = min(
                    max(0.0, desired_discharge_kw),
                    planned_discharge_kw,
                    MAX_DISCHARGE_POWER_KW,
                    available_kw,
                    reserve_safe_power_kw,
                )


            else:
                # Baseline strategy retained for comparison.
                if net_kw >= PEAK_THRESHOLD_KW:
                    desired_discharge_kw = (
                        net_kw - GRID_THRESHOLD_KW
                    )
                else:
                    desired_discharge_kw = 0.0

                discharge_kw = min(
                    max(0.0, desired_discharge_kw),
                    MAX_DISCHARGE_POWER_KW,
                    available_kw,
                )

            # Convert delivered battery power into SOC reduction.
            soc -= (
                discharge_kw
                * interval
                / DISCHARGE_EFFICIENCY
                / BATTERY_CAPACITY_KWH
                * 100.0
            )

            # Apply absolute physical SOC limits.
            soc = max(
                MIN_SOC_PERCENT,
                soc,
            )

            soc = min(
                MAX_SOC_PERCENT,
                soc,
            )

        # ----------------------------------------------------
        # D. GRID POWER AND POWER BALANCE
        # ----------------------------------------------------

        # Grid supplies demand not covered by direct solar or battery,
        # plus power drawn from the grid for optional battery charging.
        grid_kw = max(
            0.0,
            current_demand
            + grid_charge_kw
            - solar_used_kw
            - discharge_kw,
        )

        balance_error = (
            current_demand
            + grid_charge_kw
            - solar_used_kw
            - discharge_kw
            - grid_kw
        )

        # ----------------------------------------------------
        # E. ALERTS AND OPERATIONAL STATUS
        # ----------------------------------------------------

        critical_demand = (
            current_demand >= PEAK_THRESHOLD_KW
        )

        critical_grid = (
            grid_kw > PEAK_THRESHOLD_KW
        )

        any_critical = (
            critical_demand or critical_grid
        )

        grid_mode = (
            "High"
            if grid_kw > GRID_THRESHOLD_KW
            else "Normal"
        )

        if soc <= MIN_SOC_PERCENT + 1e-6:
            battery_status = "Low"
        elif soc < 30.0:
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

        # ----------------------------------------------------
        # F. SAVE VALUES
        # ----------------------------------------------------

        soc_before_values.append(soc_before)
        soc_after_values.append(soc)

        solar_charge_values.append(solar_charge_kw)
        grid_charge_values.append(grid_charge_kw)

        total_charge_values.append(
            solar_charge_kw + grid_charge_kw
        )

        discharge_values.append(discharge_kw)
        grid_values.append(grid_kw)
        balance_errors.append(balance_error)

        required_battery_power.append(
            max(
                0.0,
                net_kw - GRID_THRESHOLD_KW,
            )
        )

        grid_modes.append(grid_mode)
        operational_statuses.append(
            operational_status
        )
        battery_statuses.append(battery_status)

        alerts.append(alert_text)
        critical_demand_alerts.append(
            critical_demand
        )
        critical_grid_alerts.append(
            critical_grid
        )
        any_critical_alerts.append(
            any_critical
        )

    # ========================================================
    # 5. OUTPUT DATAFRAME
    # ========================================================

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
    result["charge_power_kw"] = total_charge_values
    result["battery_charge_kw"] = total_charge_values

    result["discharge_power_kw"] = discharge_values
    result["battery_discharge_kw"] = discharge_values

    result["grid_backup_required_kw"] = grid_values
    result["grid_backup_kw"] = grid_values
    result["grid_kw"] = grid_values

    result["power_balance_error_kw"] = balance_errors

    result["upcoming_peak_demand_kw"] = upcoming_peaks
    result["target_reserve_soc_pct"] = target_reserves

    result["required_battery_power_kw"] = (
        required_battery_power
    )

    result["available_battery_power_kw"] = (
        available_battery_power
    )

    result["grid_mode"] = grid_modes
    result["operational_status"] = operational_statuses
    result["battery_status"] = battery_statuses

    result["critical_demand_alert"] = (
        critical_demand_alerts
    )

    result["critical_grid_alert"] = (
        critical_grid_alerts
    )

    # Legacy names retained for dashboard compatibility.
    result["critical_overload_alert"] = (
        critical_grid_alerts
    )
    result["critical_alert"] = (
        critical_grid_alerts
    )
    result["any_critical_alert"] = (
        any_critical_alerts
    )

    result["alert"] = alerts

    result["scheduling_strategy"] = (
        "Intelligent Peak Shaving"
        if intelligent
        else "Baseline Threshold Control"
    )

    return result


# ============================================================
# 6. PUBLIC FUNCTIONS
# ============================================================

def simulate_battery(forecast_df):
    """Run intelligent forecast-aware battery scheduling."""

    return _simulate(
        forecast_df,
        intelligent=True,
    )


def simulate_battery_baseline(forecast_df):
    """Run baseline threshold control for comparison."""

    return _simulate(
        forecast_df,
        intelligent=False,
    )


# ============================================================
# 7. VALIDATION
# ============================================================

def validate_simulation(simulation_df):
    """Validate SOC, charge/discharge limits, and power balance."""

    df = simulation_df.copy()

    soc = _numeric_column(
        df,
        ["battery_soc_pct", "battery_soc_after_pct"],
    )

    charge = _numeric_column(
        df,
        ["charge_power_kw", "battery_charge_kw"],
    )

    discharge = _numeric_column(
        df,
        ["discharge_power_kw", "battery_discharge_kw"],
    )

    balance = _numeric_column(
        df,
        ["power_balance_error_kw"],
    )

    min_soc = (
        float(soc.min())
        if not soc.empty
        else 0.0
    )

    max_soc = (
        float(soc.max())
        if not soc.empty
        else 0.0
    )

    max_balance_error = (
        float(balance.abs().max())
        if not balance.empty
        else 0.0
    )

    max_charge = (
        float(charge.max())
        if not charge.empty
        else 0.0
    )

    max_discharge = (
        float(discharge.max())
        if not discharge.empty
        else 0.0
    )

    soc_pass = (
        min_soc >= MIN_SOC_PERCENT - 1e-6
        and max_soc <= MAX_SOC_PERCENT + 1e-6
    )

    charge_pass = (
        max_charge <= MAX_CHARGE_POWER_KW + 1e-6
    )

    discharge_pass = (
        max_discharge <= MAX_DISCHARGE_POWER_KW + 1e-6
    )

    power_balance_pass = (
        max_balance_error <= 1e-6
    )

    # For the intelligent strategy, SOC must not fall below
    # its applicable reserve after a discharge. This check uses
    # the target stored for each interval.
    reserve_pass = True

    if (
        "target_reserve_soc_pct" in df.columns
        and "battery_soc_pct" in df.columns
        and "discharge_power_kw" in df.columns
    ):
        target = _numeric_column(
            df,
            ["target_reserve_soc_pct"],
        )

        actual_soc = _numeric_column(
            df,
            ["battery_soc_pct"],
        )

        discharge_values = _numeric_column(
            df,
            ["discharge_power_kw"],
        )

        # Reserve protection is expected only from the intelligent
        # strategy, not from the baseline comparison.
        if (
            "scheduling_strategy" in df.columns
            and df["scheduling_strategy"].astype(str).str
            .contains("Intelligent", case=False).any()
        ):
            discharged = discharge_values > EPSILON

            reserve_pass = bool(
                (
                    actual_soc[discharged]
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
# 8. STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent
    forecast_file = base_dir / "forecast_results.csv"

    if forecast_file.exists():
        forecast = pd.read_csv(forecast_file)

        smart = simulate_battery(forecast)
        baseline = simulate_battery_baseline(forecast)

        print("\nSMART SCHEDULING")

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

        available_columns = [
            c for c in display_columns
            if c in smart.columns
        ]

        print(
            smart[available_columns].to_string(
                index=False
            )
        )

        print("\nSMART VALIDATION")
        print(validate_simulation(smart))

        print("\nBASELINE VALIDATION")
        print(validate_simulation(baseline))

        hours = _interval_hours(forecast).to_numpy()

        smart_grid_energy = float(
            (
                smart["grid_backup_required_kw"].to_numpy()
                * hours
            ).sum()
        )

        baseline_grid_energy = float(
            (
                baseline["grid_backup_required_kw"].to_numpy()
                * hours
            ).sum()
        )

        print("\nSUMMARY")

        print(
            f"Smart grid energy (kWh): "
            f"{smart_grid_energy:.2f}"
        )

        print(
            f"Baseline grid energy (kWh): "
            f"{baseline_grid_energy:.2f}"
        )

        print(
            "Smart peak grid demand (kW): "
            f"{smart['grid_backup_required_kw'].max():.2f}"
        )

        print(
            "Baseline peak grid demand (kW): "
            f"{baseline['grid_backup_required_kw'].max():.2f}"
        )

        print(
            "Solar charging intervals: ",
            int(
                (
                    smart["solar_charge_kw"] > EPSILON
                ).sum()
            ),
        )

        print(
            "Grid charging intervals: ",
            int(
                (
                    smart["grid_charge_kw"] > EPSILON
                ).sum()
            ),
        )

        print(
            "Battery discharging intervals: ",
            int(
                (
                    smart["discharge_power_kw"] > EPSILON
                ).sum()
            ),
        )

    else:
        print(
            "forecast_results.csv not found. "
            "Run the Flask dashboard first to generate a forecast."
        )
