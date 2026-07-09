import numpy as np
import pandas as pd

from SystemForge.simulation import battery_capacity_from_duration, solar_scale_from_penetration


def run_monte_carlo(
        profile,
        storage_duration_hours,
        solar_penetration,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
        num_scenarios=1000,
        solar_variability=0.15,
        load_variability=0.08,
        price_variability=0.25,
        price_shift_variability=0.01,
        random_seed=42):
    """Return annual cost outcomes for randomized solar, load, and price scenarios.

    Each scenario keeps the recorded hourly profile, while varying annual solar
    yield, load level, and price level. Price shifts are in EUR/kWh.

    This intentionally preserves the existing faster threshold-style Monte Carlo
    battery dispatch rather than changing it to the linear-program dispatch.
    """
    rng = np.random.default_rng(random_seed)

    load = profile.load_kwh
    solar = profile.solar_kwh
    hourly_price = profile.price_per_kwh

    battery_capacity = battery_capacity_from_duration(profile, storage_duration_hours)
    solar_profile_scale = solar_scale_from_penetration(profile, solar_penetration)

    solar_scale = rng.lognormal(
        mean=-0.5 * solar_variability ** 2,
        sigma=solar_variability,
        size=num_scenarios,
    )
    load_scale = np.clip(
        rng.normal(1, load_variability, size=num_scenarios),
        0.1,
        None,
    )
    price_scale = rng.lognormal(
        mean=-0.5 * price_variability ** 2,
        sigma=price_variability,
        size=num_scenarios,
    )
    price_shift = rng.normal(0, price_shift_variability, size=num_scenarios)

    charging_threshold = np.quantile(hourly_price, 0.25) * price_scale + price_shift
    discharging_threshold = np.quantile(hourly_price, 0.75) * price_scale + price_shift

    battery_soc = np.zeros(num_scenarios)
    grid_cost = np.zeros(num_scenarios)

    for i in range(len(load)):
        scenario_solar = solar[i] * solar_profile_scale * solar_scale
        scenario_load = load[i] * load_scale
        scenario_price = hourly_price[i] * price_scale + price_shift
        surplus = scenario_solar - scenario_load

        charge = np.minimum(
            np.maximum(surplus, 0),
            battery_capacity - battery_soc,
        )
        battery_soc += charge

        deficit = np.maximum(-surplus, 0)
        discharge = np.where(
            scenario_price >= discharging_threshold,
            np.minimum(deficit, battery_soc),
            0,
        )
        battery_soc -= discharge
        grid = deficit - discharge

        grid_charge = np.where(
            scenario_price < charging_threshold,
            np.maximum(battery_capacity - battery_soc, 0),
            0,
        )
        battery_soc += grid_charge
        grid += grid_charge

        grid_cost += grid * scenario_price

    annualized_battery_cost = (
        battery_capacity * battery_cost_per_kwh / battery_lifetime_years
    )
    annualized_solar_cost = (
        solar_penetration * solar_cost_per_penetration / solar_lifetime_years
    )
    total_annual_cost = grid_cost + annualized_battery_cost + annualized_solar_cost

    return total_annual_cost, grid_cost


def run_uncertainty_source_experiments(
        profile,
        storage_duration_hours,
        solar_penetration,
        uncertainty_experiments,
        num_scenarios,
        random_seed,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years):
    """Run one-at-a-time uncertainty experiments and return a comparison table."""
    uncertainty_results = []

    for uncertainty_source, scenario_setting, scenario_parameters in uncertainty_experiments:
        experiment_costs, _ = run_monte_carlo(
            profile,
            storage_duration_hours,
            solar_penetration,
            battery_cost_per_kwh=battery_cost_per_kwh,
            battery_lifetime_years=battery_lifetime_years,
            solar_cost_per_penetration=solar_cost_per_penetration,
            solar_lifetime_years=solar_lifetime_years,
            num_scenarios=num_scenarios,
            random_seed=random_seed,
            **scenario_parameters,
        )

        uncertainty_results.append({
            "Uncertainty Source": uncertainty_source,
            "Scenario Setting": scenario_setting,
            "Cost Standard Deviation (EUR)": np.std(experiment_costs, ddof=1),
        })

    return pd.DataFrame(uncertainty_results)
