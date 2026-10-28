import pandas as pd
import pytest

from battery_simulator import (
    simulate_battery,
    simulate_battery_baseline,
    validate_simulation,
)


def make_forecast():
    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01", periods=8, freq="h"
        ),
        "predicted_demand_kw": [
            150.0, 180.0, 220.0, 240.0,
            210.0, 170.0, 140.0, 130.0,
        ],
        "renewable_generation_kw": [
            20.0, 30.0, 25.0, 20.0,
            35.0, 40.0, 45.0, 50.0,
        ],
    })


def test_intelligent_simulation_returns_rows():
    result = simulate_battery(make_forecast())

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8


def test_baseline_simulation_returns_rows():
    result = simulate_battery_baseline(make_forecast())

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8


@pytest.mark.parametrize(
    "simulator",
    [simulate_battery, simulate_battery_baseline],
)
def test_soc_stays_within_limits(simulator):
    result = simulator(make_forecast())

    assert result["battery_soc_pct"].between(
        20.0, 80.0
    ).all()


@pytest.mark.parametrize(
    "simulator",
    [simulate_battery, simulate_battery_baseline],
)
def test_charge_discharge_power_limits(simulator):
    result = simulator(make_forecast())

    assert (result["charge_power_kw"] >= -1e-9).all()
    assert (result["discharge_power_kw"] >= -1e-9).all()
    assert (result["charge_power_kw"] <= 100.0 + 1e-9).all()
    assert (result["discharge_power_kw"] <= 100.0 + 1e-9).all()


def test_intelligent_simulation_validation():
    result = simulate_battery(make_forecast())
    validation = validate_simulation(result)

    assert validation["power_balance_pass"] is True
    assert validation["soc_limits_pass"] is True
    assert validation["charge_power_limit_pass"] is True
    assert validation["discharge_power_limit_pass"] is True
    assert validation["max_power_balance_error_kw"] <= 1e-6