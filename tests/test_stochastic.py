import numpy as np
import pyomo.environ as pyo
import pytest

from optimization import (
    build_capacity_model,
    solve_dispatch,
)

from stochastic import build_two_stage_model


def test_one_scenario_matches_deterministic_capacity_model():
    load = np.array([
        10.0,
        12.0,
        11.0,
    ])

    solar_cf = np.array([
        0.0,
        0.5,
        0.2,
    ])

    price = np.array([
        0.10,
        0.20,
        0.40,
    ])

    common = {
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "solar_capacity_cost": 1000.0,
        "battery_energy_cost": 400.0,
        "battery_power_cost": 200.0,
        "discount_rate": 0.07,
        "solar_lifetime_years": 25,
        "battery_lifetime_years": 15,
        "degradation_rate": 0.001,
    }

    deterministic = build_capacity_model(
        load=load,
        solar_capacity_factor=solar_cf,
        hourly_price=price,
        **common,
    )

    stochastic = build_two_stage_model(
        scenario_load=load[None, :],
        scenario_solar_cf=solar_cf[None, :],
        scenario_price=price[None, :],
        scenario_probabilities=np.array([
            1.0
        ]),
        **common,
    )

    solve_dispatch(deterministic)
    solve_dispatch(stochastic)

    assert pyo.value(
        stochastic.objective
    ) == pytest.approx(
        pyo.value(deterministic.objective),
        rel=1e-6,
        abs=1e-6,
    )

    assert pyo.value(
        stochastic.solar_capacity
    ) == pytest.approx(
        pyo.value(deterministic.solar_capacity),
        rel=1e-6,
        abs=1e-6,
    )

    assert pyo.value(
        stochastic.battery_capacity
    ) == pytest.approx(
        pyo.value(deterministic.battery_capacity),
        rel=1e-6,
        abs=1e-6,
    )

    assert pyo.value(
        stochastic.battery_power
    ) == pytest.approx(
        pyo.value(deterministic.battery_power),
        rel=1e-6,
        abs=1e-6,
    )