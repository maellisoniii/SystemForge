import numpy as np
import pandas as pd

from SystemForge.simulation import battery_capacity_from_duration, solar_scale_from_penetration


def capital_recovery_factor(discount_rate, asset_lifetime_years):
    """Convert an upfront capital cost into an equivalent annual cost."""
    if asset_lifetime_years <= 0:
        raise ValueError("asset_lifetime_years must be greater than zero")
    if discount_rate < 0:
        raise ValueError("discount_rate cannot be negative")
    if discount_rate == 0:
        return 1 / asset_lifetime_years

    return (
        discount_rate * (1 + discount_rate) ** asset_lifetime_years
        / ((1 + discount_rate) ** asset_lifetime_years - 1)
    )


def run_monte_carlo(
        profile,
        storage_duration_hours,
        solar_penetration,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
        discount_rate=0,
        battery_power_ratio=1.0,
        battery_charge_efficiency=1.0,
        battery_discharge_efficiency=1.0,
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

    Dispatch assumptions:
    - This is a heuristic operating rule, not perfect-foresight optimization.
    - The battery charges from same-hour surplus solar first.
    - The battery discharges only when scenario price is above the high-price
      threshold.
    - The battery charges from the grid only when scenario price is below the
      low-price threshold.
    - battery_power_ratio limits charge, discharge, and grid-charge energy per
      hour.
    - battery_charge_efficiency and battery_discharge_efficiency track losses
      inside battery state of charge.
    """
    rng = np.random.default_rng(random_seed)

    if not 0 <= battery_power_ratio:
        raise ValueError("battery_power_ratio cannot be negative")
    if not 0 < battery_charge_efficiency <= 1:
        raise ValueError("battery_charge_efficiency must be greater than 0 and at most 1")
    if not 0 < battery_discharge_efficiency <= 1:
        raise ValueError("battery_discharge_efficiency must be greater than 0 and at most 1")

    load = profile.load_kwh
    solar = profile.solar_kwh
    hourly_price = profile.price_per_kwh

    battery_capacity = battery_capacity_from_duration(profile, storage_duration_hours)
    max_battery_power = battery_capacity * battery_power_ratio
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

        # Charge from surplus solar. Charge is measured as external energy sent
        # to the battery; SOC only increases by charge * charge_efficiency.
        remaining_charge_capacity = np.maximum(
            (battery_capacity - battery_soc) / battery_charge_efficiency,
            0,
        )
        charge = np.minimum(
            np.maximum(surplus, 0),
            np.minimum(max_battery_power, remaining_charge_capacity),
        )
        battery_soc += charge * battery_charge_efficiency

        deficit = np.maximum(-surplus, 0)

        # Discharge is measured as external energy delivered from the battery.
        # SOC falls by discharge / discharge_efficiency.
        available_discharge = np.minimum(
            max_battery_power,
            battery_soc * battery_discharge_efficiency,
        )
        discharge = np.where(
            scenario_price >= discharging_threshold,
            np.minimum(deficit, available_discharge),
            0,
        )
        battery_soc -= discharge / battery_discharge_efficiency
        grid = deficit - discharge

        # Optional grid charging at low prices. It shares the hourly charge power
        # limit with any surplus-solar charging already performed this hour.
        remaining_charge_power = np.maximum(max_battery_power - charge, 0)
        remaining_charge_capacity = np.maximum(
            (battery_capacity - battery_soc) / battery_charge_efficiency,
            0,
        )
        grid_charge = np.where(
            scenario_price < charging_threshold,
            np.minimum(remaining_charge_power, remaining_charge_capacity),
            0,
        )
        battery_soc += grid_charge * battery_charge_efficiency
        grid += grid_charge

        grid_cost += grid * scenario_price

    annualized_battery_cost = (
        battery_capacity
        * battery_cost_per_kwh
        * capital_recovery_factor(discount_rate, battery_lifetime_years)
    )
    annualized_solar_cost = (
        solar_penetration
        * solar_cost_per_penetration
        * capital_recovery_factor(discount_rate, solar_lifetime_years)
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
        solar_lifetime_years,
        discount_rate=0,
        battery_power_ratio=1.0,
        battery_charge_efficiency=1.0,
        battery_discharge_efficiency=1.0):
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
            discount_rate=discount_rate,
            battery_power_ratio=battery_power_ratio,
            battery_charge_efficiency=battery_charge_efficiency,
            battery_discharge_efficiency=battery_discharge_efficiency,
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
