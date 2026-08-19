import numpy as np

from optimization import (
    build_dispatch_model,
    solve_dispatch,
    extract_dispatch_results,
    validate_dispatch_result,
    optimize_battery_dispatch_scipy,
)
def solve_test_case(
        load,
        solar,
        battery_capacity,
        battery_power,
        hourly_price,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        initial_soc=0.0,
        degradation_rate=1e-9,
    ):

    model = build_dispatch_model(
        load=np.asarray(load, dtype=float),
        solar=np.asarray(solar, dtype=float),
        battery_capacity=battery_capacity,
        battery_power=battery_power,
        battery_charge_efficiency=battery_charge_efficiency,
        battery_discharge_efficiency=battery_discharge_efficiency,
        initial_soc=initial_soc,
        degradation_rate=degradation_rate,
        hourly_price=np.asarray(hourly_price, dtype=float)
    )
    solver_results = solve_dispatch(model)
    dispatch_result = extract_dispatch_results(
        model, solver_results)
    validate_dispatch_result(dispatch_result)
    return dispatch_result

def test_pyomo_matches_scipy_benchmark():
    load = np.array([
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ])

    solar = np.array([
        0.0,
        0.0,
        15.0,
        15.0,
        0.0,
        0.0,
    ])

    price = np.array([
        0.10,
        0.10,
        0.15,
        0.20,
        0.50,
        0.50,
    ])

    pyomo_result = solve_test_case(
        load=load,
        solar=solar,
        hourly_price=price,
        battery_capacity=10.0,
        battery_power=5.0,
        degradation_rate=1e-9,
    )

    scipy_grid, scipy_curtailment, scipy_soc = (
        optimize_battery_dispatch_scipy(
            load=load,
            hourly_price=price,
            battery_capacity=10.0,
            scaled_solar=solar,
            battery_power_ratio=0.5,
            battery_charge_efficiency=0.95,
            battery_discharge_efficiency=0.95,
        )
    )

    np.testing.assert_allclose(
        pyomo_result.grid_import,
        scipy_grid,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        pyomo_result.curtailment,
        scipy_curtailment,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        pyomo_result.soc,
        scipy_soc,
        atol=1e-8,
    )

def test_zero_battery_collapses_to_grid_solar_accounting():
    load = np.array([
        10.0,
        10.0,
        10.0,
        10.0,
    ])

    solar = np.array([
        0.0,
        5.0,
        15.0,
        20.0,
    ])

    price = np.array([
        0.20,
        0.20,
        0.20,
        0.20,
    ])

    result = solve_test_case(
        load=load,
        solar=solar,
        hourly_price=price,
        battery_capacity=0.0,
        battery_power=0.0,
    )

    np.testing.assert_allclose(
        result.charge,
        0.0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result.discharge,
        0.0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result.soc,
        0.0,
        atol=1e-8,
    )
    expected_grid_import = load - solar
    expected_grid_import[expected_grid_import < 0] = 0.0
    np.testing.assert_allclose(
        result.grid_import,
        expected_grid_import,
        atol=1e-8,
    )
    expected_curtailment = solar - load
    expected_curtailment[expected_curtailment < 0] = 0.0
    np.testing.assert_allclose(
        result.curtailment,
        expected_curtailment, 
        atol=1e-8,
    )