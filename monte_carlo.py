import numpy as np
import pandas as pd
import time

#from SystemForge.simulation import battery_capacity_from_duration, solar_scale_from_penetration
from optimization import (
    build_dispatch_model,
    solve_dispatch,
    extract_dispatch_results,
    validate_dispatch_result,
)
from dataclasses import dataclass

@dataclass
class MonteCarloResult:
    scenarios: pd.DataFrame
    random_seed: int
    number_of_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    runtime_seconds: float
    average_runtime_per_scenario: float

start_time = time.perf_counter()

def generate_scenarios(
        load: np.ndarray,
        solar: np.ndarray,
        hourly_price: np.ndarray,
        num_scenarios: int = 100,
        solar_variability: float = 0.15,
        load_variability: float = 0.08,
        price_variability: float = 0.25,
        price_shift_variability: float = 0.01,
        random_seed: int = 42,
    ):
    rng = np.random.default_rng(random_seed)

    load = np.asarray(load, dtype=float)
    solar = np.asarray(solar, dtype=float)
    hourly_price = np.asarray(
        hourly_price,
        dtype=float,
    )

    if not (
        len(load)
        == len(solar)
        == len(hourly_price)
    ):
        raise ValueError(
            "Load, solar, and price profiles "
            "must have equal length."
        )

    if num_scenarios <= 0:
        raise ValueError(
            "num_scenarios must be greater than zero."
        )
    if solar_variability < 0:
        raise ValueError(
        "solar_variability cannot be negative."
    )
    if load_variability < 0:
        raise ValueError(
            "load_variability cannot be negative."
        )

    if price_variability < 0:
        raise ValueError(
            "price_variability cannot be negative."
        )

    if price_shift_variability < 0:

        raise ValueError(
            "price_shift_variability cannot be negative."
        )
    if np.any(load < 0):
        raise ValueError(
        "Load cannot contain negative values."
    )

    if np.any(solar < 0):
        raise ValueError(
        "Solar cannot contain negative values."
    )
    solar_scale = rng.lognormal(
        mean=-0.5 * solar_variability ** 2,
        sigma=solar_variability,
        size=num_scenarios,
        )


    load_scale = np.clip(
            rng.normal(
                1.0,
                load_variability,
                size=num_scenarios,
            ),
            0.1,
            None,
        )

    price_scale = rng.lognormal(
            mean=-0.5 * price_variability ** 2,
            sigma=price_variability,
            size=num_scenarios,
        )

    price_shift = rng.normal(
            0.0,
            price_shift_variability,
            size=num_scenarios,
        )
    scenario_load = (
        load[None, :]
        * load_scale[:, None]
        )

    scenario_solar = (
            solar[None, :]
            * solar_scale[:, None]
        )

    scenario_price = (
            hourly_price[None, :]
            * price_scale[:, None]
            + price_shift[:, None]
        )
    return (
            scenario_load,
            scenario_solar,
            scenario_price,
            load_scale,
            solar_scale,
            price_scale,
            price_shift)

  
def run_monte_carlo(
    load: np.ndarray,
    solar: np.ndarray,
    hourly_price: np.ndarray,
    battery_capacity: float,
    battery_power: float,
    battery_charge_efficiency: float = 0.95,
    battery_discharge_efficiency: float = 0.95,
    degradation_rate: float = 0.0,
    num_scenarios: int = 100,
    solar_variability: float = 0.15,
    load_variability: float = 0.08,
    price_variability: float = 0.25,
    price_shift_variability: float = 0.01,
    random_seed: int = 42,
    ) -> MonteCarloResult:
    """
    Run repeated optimized dispatch under uncertain load, solar, and price profiles.

    Each scenario perturbs the base time series using a seeded random generator.
    The fixed system design is held constant, while operational decisions are
    re-optimized independently for every scenario using optimization.py.
    Monte Carlo does not contain its own battery dispatch policy.
    """
    (
        scenario_load,
        scenario_solar,
        scenario_price,
        load_scale,
        solar_scale,
        price_scale,
        price_shift,
    ) = generate_scenarios(
        load=load,
        solar=solar,
        hourly_price=hourly_price,
        num_scenarios=num_scenarios,
        solar_variability=solar_variability,
        load_variability=load_variability,
        price_variability=price_variability,
        price_shift_variability=price_shift_variability,
        random_seed=random_seed,
    )
    scenario_records = []

    for scenario_id in range(num_scenarios):

        load_s = scenario_load[scenario_id]
        solar_s = scenario_solar[scenario_id]
        price_s = scenario_price[scenario_id]

        try:
            model = build_dispatch_model(
                load=load_s,
                solar=solar_s,
                hourly_price=price_s,
                battery_capacity=battery_capacity,
                battery_power=battery_power,
                battery_charge_efficiency=(
                    battery_charge_efficiency
                ),
                battery_discharge_efficiency=(
                    battery_discharge_efficiency
                ),
                initial_soc=0.0,
                degradation_rate=degradation_rate,
            )

            solver_results = solve_dispatch(
                model
            )

            result = extract_dispatch_results(
                model,
                solver_results,
            )

            validate_dispatch_result(
                result
            )
            battery_throughput = np.sum(
                result.charge
                + result.discharge
            )

            total_grid_import = np.sum(
                result.grid_import
            )

            total_curtailment = np.sum(
                result.curtailment
            )

            peak_grid_import = np.max(
                result.grid_import
            )
            scenario_records.append({
                "scenario_id": scenario_id,
                "solve_success": True,
                "load_scale": 
                load_scale[scenario_id],
                "solar_scale": 
                solar_scale[scenario_id],
                "price_scale": 
                price_scale[scenario_id],
                "price_shift": 
                price_shift[scenario_id],
                "objective_cost":
                    result.objective_cost,
                "grid_cost":
                    result.grid_cost,
                "degradation_cost":
                    result.degradation_cost,
                "grid_import":
                    total_grid_import,
                "curtailment":
                    total_curtailment,
                "battery_throughput":
                    battery_throughput,
                "peak_grid_import":
                    peak_grid_import,
            })
        except (
            RuntimeError,
            ValueError,
            AssertionError) as exc:
            scenario_records.append({
                "scenario_id": scenario_id,
                "solve_success": False,
                "load_scale": 
                load_scale[scenario_id],
                "solar_scale": 
                solar_scale[scenario_id],
                "price_scale": 
                price_scale[scenario_id],
                "price_shift": 
                price_shift[scenario_id],    
                "objective_cost": np.nan,
                "grid_cost": np.nan,
                "degradation_cost": np.nan,
                "grid_import": np.nan,
                "curtailment": np.nan,
                "battery_throughput": np.nan,
                "peak_grid_import": np.nan,
                "error": str(exc),
                "solver_status": 
                result.solver_status,
                "termination_condition":
                result.termination_condition,
                "solver_status": None,
                "termination_condition": None,
            })
    scenario_table = pd.DataFrame(
        scenario_records
    )

    successful_scenarios = int(
        scenario_table["solve_success"].sum()
    )

    failed_scenarios = (
        num_scenarios
        - successful_scenarios
    )
    runtime_seconds = (
        time.perf_counter() - start_time
    )

    average_runtime_per_scenario = (
    runtime_seconds / num_scenarios 
    )
    
    return MonteCarloResult(
        scenarios=scenario_table,
        random_seed=random_seed,
        number_of_scenarios=num_scenarios,
        successful_scenarios=(
            successful_scenarios
        ),
        failed_scenarios=(
            failed_scenarios
        ),
        runtime_seconds=runtime_seconds,
        average_runtime_per_scenario=
        average_runtime_per_scenario,
    )
if __name__ == "__main__": 
    start_time = time.perf_counter()

    result = run_monte_carlo(
        load=np.array([
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]),
        solar=np.array([
            0.0,
            0.0,
            15.0,
            15.0,
            0.0,
            0.0,
        ]),
        hourly_price=np.array([
            0.10,
            0.10,
            0.15,
            0.20,
            0.50,
            0.50,
        ]),
        battery_capacity=10.0,
        battery_power=5.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        degradation_rate=1e-9,
        num_scenarios=500,
        random_seed=42,
    )
    elapsed_time = time.perf_counter() - start_time

    print(result.scenarios)
    print("Successful:", result.successful_scenarios)
    print("Failed:", result.failed_scenarios)
    print(
        f"Runtime: {result.runtime_seconds:.3f} seconds"
    )
    print(
        "Average solve time: "
        f"{result.average_runtime_per_scenario:.4f} "
        "seconds/scenario"
    )


        