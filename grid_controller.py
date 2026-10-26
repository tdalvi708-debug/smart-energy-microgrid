
"""Smart Energy Microgrid - Grid Controller."""

import pandas as pd

# Grid controller configuration
RENEWABLE_STORAGE_DEMAND_THRESHOLD_KW = 160.0
CRITICAL_OVERLOAD_THRESHOLD_KW = 200.0

STORAGE_MODE_START_HOUR = 18
STORAGE_MODE_END_HOUR = 22

HIGH_DEMAND_THRESHOLD_KW = 200.0

NORMAL_RESERVE_SOC_PCT = 20.0
FORECAST_RESERVE_SOC_PCT = 40.0


def get_grid_status(
    timestamp,
    predicted_demand_kw,
    battery_soc_pct=None,
    renewable_generation_kw=0.0,
    max_battery_discharge_kw=100.0,
    min_battery_soc_pct=20.0,
    upcoming_peak_demand_kw=None,
):
    """
    Determine grid mode, battery support, reserve SOC and grid backup.

    Returns:
        dict: Grid status and battery support information.
    """

    timestamp = pd.to_datetime(timestamp, errors="raise")

    demand = float(predicted_demand_kw)
    renewable = float(renewable_generation_kw)
    max_discharge = float(max_battery_discharge_kw)
    min_soc = float(min_battery_soc_pct)

    # Validate input values
    if pd.isna(demand) or demand < 0:
        raise ValueError("Demand must be a valid non-negative number.")

    if pd.isna(renewable) or renewable < 0:
        raise ValueError("Renewable generation cannot be negative.")

    if pd.isna(max_discharge) or max_discharge < 0:
        raise ValueError("Maximum battery discharge cannot be negative.")

    if not 0 <= min_soc <= 100:
        raise ValueError("Minimum battery SOC must be between 0 and 100.")

    if battery_soc_pct is not None:
        battery_soc_pct = float(battery_soc_pct)

        if pd.isna(battery_soc_pct) or not 0 <= battery_soc_pct <= 100:
            raise ValueError("Battery SOC must be between 0 and 100.")

    # 1. Select the grid operating mode
    hour = timestamp.hour

    if (
        demand > RENEWABLE_STORAGE_DEMAND_THRESHOLD_KW
        or STORAGE_MODE_START_HOUR <= hour <= STORAGE_MODE_END_HOUR
    ):
        grid_mode = "Renewable Storage Mode"
    else:
        grid_mode = "Main Grid Mode"

    # 2. Check for critical demand
    critical_overload = demand >= CRITICAL_OVERLOAD_THRESHOLD_KW

    # 3. Calculate direct renewable contribution
    solar_used = min(demand, renewable)
    remaining_demand = max(0.0, demand - solar_used)

    # 4. Estimate upcoming peak demand
    if upcoming_peak_demand_kw is None:
        upcoming_peak = demand
    else:
        upcoming_peak = float(upcoming_peak_demand_kw)

    if pd.isna(upcoming_peak) or upcoming_peak < 0:
        raise ValueError("Upcoming peak demand must be non-negative.")

    # 5. Decide the battery reserve target
    if critical_overload:
        target_reserve_soc = min_soc
    elif upcoming_peak >= HIGH_DEMAND_THRESHOLD_KW:
        target_reserve_soc = max(
            min_soc,
            FORECAST_RESERVE_SOC_PCT,
        )
    else:
        target_reserve_soc = max(
            min_soc,
            NORMAL_RESERVE_SOC_PCT,
        )

    # 6. Estimate how much battery power may be used
    available_battery_power = 0.0

    if battery_soc_pct is not None and battery_soc_pct > min_soc:
        battery_above_reserve = battery_soc_pct > target_reserve_soc

        if battery_above_reserve or critical_overload:
            available_battery_power = min(
                remaining_demand,
                max_discharge,
            )

    # 7. Calculate grid backup requirement
    grid_backup_required = max(
        0.0,
        remaining_demand - available_battery_power,
    )

    # 8. Determine battery status
    if battery_soc_pct is None:
        battery_status = "SOC Not Provided"
    elif battery_soc_pct <= min_soc:
        battery_status = "Minimum SOC Reached"
    elif available_battery_power > 0:
        battery_status = "Battery Support Available"
    else:
        battery_status = "Battery Reserve Protected"

    # 9. Generate an overload alert
    if critical_overload:
        alert = "Critical Overload"
    else:
        alert = "None"

    operational_status = (
        "Grid Backup Required"
        if grid_backup_required > 0.01
        else "Renewables/Battery Meet Demand"
    )

    # 10. Return all controller results
    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_demand_kw": round(demand, 2),
        "renewable_generation_kw": round(renewable, 2),
        "solar_used_kw": round(solar_used, 2),
        "remaining_demand_kw": round(remaining_demand, 2),
        "grid_mode": grid_mode,
        "critical_overload_alert": critical_overload,
        "alert": alert,
        "battery_soc_pct": (
            round(battery_soc_pct, 2)
            if battery_soc_pct is not None
            else None
        ),
        "battery_status": battery_status,
        "upcoming_peak_demand_kw": round(upcoming_peak, 2),
        "target_reserve_soc_pct": round(target_reserve_soc, 2),
        "required_battery_power_kw": round(remaining_demand, 2),
        "available_battery_power_kw": round(
            available_battery_power, 2
        ),
        "grid_backup_required_kw": round(
            grid_backup_required, 2
        ),
        "operational_status": operational_status,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SMART ENERGY MICROGRID - GRID CONTROLLER TEST")
    print("=" * 60)

    test_cases = [
        {
            "timestamp": "2025-01-01 10:00:00",
            "demand": 120,
            "soc": 80,
            "renewable": 100,
            "upcoming_peak": 150,
        },
        {
            "timestamp": "2025-01-01 10:00:00",
            "demand": 175,
            "soc": 80,
            "renewable": 50,
            "upcoming_peak": 220,
        },
        {
            "timestamp": "2025-01-01 18:00:00",
            "demand": 150,
            "soc": 80,
            "renewable": 80,
            "upcoming_peak": 210,
        },
        {
            "timestamp": "2025-01-01 23:00:00",
            "demand": 210,
            "soc": 80,
            "renewable": 50,
            "upcoming_peak": 230,
        },
        {
            "timestamp": "2025-01-01 19:00:00",
            "demand": 250,
            "soc": 20,
            "renewable": 50,
            "upcoming_peak": 250,
        },
    ]

    for number, case in enumerate(test_cases, start=1):
        result = get_grid_status(
            timestamp=case["timestamp"],
            predicted_demand_kw=case["demand"],
            battery_soc_pct=case["soc"],
            renewable_generation_kw=case["renewable"],
            upcoming_peak_demand_kw=case["upcoming_peak"],
        )

        print(f"\nTest Case {number}")
        for key, value in result.items():
            print(f"{key}: {value}")

    print("\nGrid controller tests completed.")
