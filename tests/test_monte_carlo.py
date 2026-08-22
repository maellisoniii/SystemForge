import numpy as np
import pandas as pd
import pytest

from monte_carlo import (
    generate_scenarios,
    run_monte_carlo,
)


def test_scenario_generation_is_reproducible():
    load = np.array([10.0, 12.0, 11.0])
    solar = np.array([0.0, 5.0, 8.0])
    price = np.array([0.10, 0.20, 0.50])

    first = generate_scenarios(
        load=load,
        solar=solar,
        hourly_price=price,
        num_scenarios=5,
        random_seed=42,
    )

    second = generate_scenarios(
        load=load,
        solar=solar,
        hourly_price=price,
        num_scenarios=5,
        random_seed=42,
    )

    for first_array, second_array in zip(
        first,
        second,
    ):
        np.testing.assert_allclose(
            first_array,
            second_array,
        )
def test_monte_carlo_is_reproducible():
    kwargs = {
        "load": np.array([
            10.0,
            10.0,
            10.0,
            10.0,
        ]),
        "solar": np.array([
            0.0,
            15.0,
            15.0,
            0.0,
        ]),
        "hourly_price": np.array([
            0.10,
            0.15,
            0.50,
            0.50,
        ]),
        "battery_capacity": 10.0,
        "battery_power": 5.0,
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "degradation_rate": 1e-9,
        "num_scenarios": 3,
        "random_seed": 42,
    }

    first = run_monte_carlo(**kwargs)
    second = run_monte_carlo(**kwargs)

    pd.testing.assert_frame_equal(
        first.scenarios,
        second.scenarios,
    )

def test_monte_carlo_small_batch_all_succeeds():
    result = run_monte_carlo(
        load=np.array([
            10.0, 10.0, 10.0, 10.0
        ]),
        solar=np.array([
            0.0, 15.0, 15.0, 0.0
        ]),
        hourly_price=np.array([
            0.10, 0.15, 0.50, 0.50
        ]),
        battery_capacity=10.0,
        battery_power=5.0,
        num_scenarios=10,
        random_seed=42,
    )

    assert result.successful_scenarios == 10
    assert result.failed_scenarios == 0
    assert result.scenarios["solve_success"].all()

def test_uncertainty_changes_cost_outcomes():
    result = run_monte_carlo(
        load=np.array([
            10.0, 10.0, 10.0, 10.0
        ]),
        solar=np.array([
            0.0, 15.0, 15.0, 0.0
        ]),
        hourly_price=np.array([
            0.10, 0.15, 0.50, 0.50
        ]),
        battery_capacity=10.0,
        battery_power=5.0,
        num_scenarios=20,
        random_seed=42,
    )

    assert (
        result.scenarios["objective_cost"].std()
        > 0.0
    )

def test_negative_variability_is_rejected():
    with pytest.raises(ValueError):
        generate_scenarios(
            load=np.array([10.0]),
            solar=np.array([5.0]),
            hourly_price=np.array([0.20]),
            solar_variability=-0.10,
        )