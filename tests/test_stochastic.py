import numpy as np
import pyomo.environ as pyo
import pytest
import pandas as pd

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

def scenario_data():
    load = np.array([
        [10.0, 12.0, 11.0],
        [10.5, 12.5, 11.5], # slightly higher load scenario
        [11.0, 13.0, 12.0], # even higher load scenario
        [12.0, 15.0, 17.0], # extreme load scenario, stress test
    ])

    solar_cf = np.array([
        [0.0, 0.5, 0.2],
        [0.0, 0.45, 0.18],
        [0.0, 0.25, 0.08],
        [0.0, 0.10, 0.03],
    ])

    price = np.array([
        [0.10, 0.20, 0.40],
        [0.15, 0.25, 0.35],
        [0.12, 0.35, 0.70],
        [0.15, 0.55, 1.00],
    ])

    probabilities = np.array([
        0.5,
        0.3,
        0.15,
        0.05,
    ])

    return load, solar_cf, price, probabilities
def stochastic_model(
        risk_aversion=0.0,
        confidence_level=0.80,
):
    load, solar_cf, price, probabilities = (
    scenario_data()
    )
    return build_two_stage_model(
        scenario_load=load,
        scenario_solar_cf=solar_cf,
        scenario_price=price,
        scenario_probabilities=probabilities,
        risk_aversion=risk_aversion,
        confidence_level=confidence_level,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        solar_capacity_cost=1000.0,
        battery_energy_cost=400.0,
        battery_power_cost=200.0,
        discount_rate=0.07,
        solar_lifetime_years=25,
        battery_lifetime_years=15,
        degradation_rate=0.001,
    )

def test_invalid_confidence_level():
    for invalid_value in [
        -0.1,
        0.0,
        1.0,
        1.5,
    ]:
        with pytest.raises(ValueError):
            stochastic_model(
                confidence_level=invalid_value,
            )
def test_invalid_risk_aversion():
    with pytest.raises(ValueError):
        stochastic_model(
            risk_aversion=-0.1,
        )
def test_risk_neutral_model_solves_optimally():
    model = stochastic_model(
        risk_aversion=0.0,
        confidence_level=0.95,
    )
    results = solve_dispatch(model)
    assert (
        results.solver.termination_condition 
        == pyo.TerminationCondition.optimal
    )
    assert np.isfinite(
        pyo.value(model.objective))
            
def test_risk_averse_model_solves_optimally():
    model = stochastic_model(
        risk_aversion=0.25,
        confidence_level=0.95,
    )

    results = solve_dispatch(model)

    assert (
        results.solver.termination_condition
        == pyo.TerminationCondition.optimal
    )

    assert np.isfinite(
        pyo.value(model.objective)
    )

    assert np.isfinite(
        pyo.value(model.cvar)
    )
def test_higher_risk_aversion_reduces_cvar():
    low_risk = stochastic_model(
        risk_aversion=0.1,
        confidence_level=0.80,
    )

    medium_risk = stochastic_model(
        risk_aversion=0.5,
        confidence_level=0.80,
    )

    high_risk = stochastic_model(
        risk_aversion=1.0,
        confidence_level=0.80,
    )

    solve_dispatch(low_risk)
    solve_dispatch(medium_risk)
    solve_dispatch(high_risk)

    low_cvar = pyo.value(low_risk.cvar)
    medium_cvar = pyo.value(medium_risk.cvar)
    high_cvar = pyo.value(high_risk.cvar)

    tolerance = 1e-6

    assert medium_cvar <= low_cvar + tolerance
    assert high_cvar <= medium_cvar + tolerance
    
def test_risk_aversion_can_change_design():
    low_risk = stochastic_model(
        risk_aversion=0.1,
        confidence_level=0.80,
    )

    high_risk = stochastic_model(
        risk_aversion=1.0,
        confidence_level=0.80,
    )

    solve_dispatch(low_risk)
    solve_dispatch(high_risk)

    assert not np.isclose(
        pyo.value(low_risk.solar_capacity),
        pyo.value(high_risk.solar_capacity),
    )
if __name__ == "__main__":
    risk_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        1.0
    ]
    results = []
    for risk_aversion in risk_values:
        model = stochastic_model(
            risk_aversion=risk_aversion,
        )
        solve_dispatch(model)
        results.append({
        "solar_capacity": 
        pyo.value(model.solar_capacity),
        "battery_capacity": 
        pyo.value(model.battery_capacity),
        "battery_power": 
        pyo.value(model.battery_power),
        "objective": pyo.value(model.objective),
        "risk_aversion": risk_aversion,
        "expected_operating_cost": 
        pyo.value(model.expected_operating_cost),
        "cvar": pyo.value(model.cvar),
        "capital_cost": pyo.value(model.capital_cost),
    })
    results_df = pd.DataFrame(results)

    pd.set_option(
    "display.max_columns",
    None,
    )

    pd.set_option(
    "display.width",
    None,   
    )
    print(results_df)

