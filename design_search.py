import numpy as np
import pandas as pd

from SystemForge.monte_carlo import run_monte_carlo
from SystemForge.risk import calculate_cost_risk_metrics


def annualized_battery_cost(battery_capacity, battery_cost_per_kwh, battery_lifetime_years):
    return battery_capacity * battery_cost_per_kwh / battery_lifetime_years


def annualized_solar_cost(solar_penetration, solar_cost_per_penetration, solar_lifetime_years):
    return solar_penetration * solar_cost_per_penetration / solar_lifetime_years


def total_annual_cost(
        annual_grid_cost,
        battery_capacity,
        solar_penetration,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years):
    return (
        annual_grid_cost
        + annualized_battery_cost(
            battery_capacity,
            battery_cost_per_kwh,
            battery_lifetime_years,
        )
        + annualized_solar_cost(
            solar_penetration,
            solar_cost_per_penetration,
            solar_lifetime_years,
        )
    )


def find_best_design(
        profile,
        simulator,
        storage_durations_hours,
        solar_penetrations,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years):
    """Search the selected solar/storage grid and return the least-cost design."""
    best_design = None
    best_cost = float("inf")

    for solar_penetration in solar_penetrations:
        for storage_duration in storage_durations_hours:
            (total_grid_import, total_solar_curtailed, annual_grid_cost,
             annual_savings, max_soc) = simulator.simulate_system(
                storage_duration,
                solar_penetration,
            )

            battery_capacity = simulator.battery_capacity(storage_duration)

            annualized_battery = annualized_battery_cost(
                battery_capacity,
                battery_cost_per_kwh,
                battery_lifetime_years,
            )
            annualized_solar = annualized_solar_cost(
                solar_penetration,
                solar_cost_per_penetration,
                solar_lifetime_years,
            )
            design_total_annual_cost = (
                annual_grid_cost
                + annualized_battery
                + annualized_solar
            )

            if design_total_annual_cost < best_cost:
                best_cost = design_total_annual_cost
                best_design = {
                    "solar_penetration": solar_penetration,
                    "storage_duration_hours": storage_duration,
                    "battery_capacity": battery_capacity,
                    "total_annual_cost": design_total_annual_cost,
                    "annual_grid_cost": annual_grid_cost,
                    "annualized_battery_cost": annualized_battery,
                    "annualized_solar_cost": annualized_solar,
                    "max_soc": max_soc,
                    "solar_curtailed": total_solar_curtailed,
                    "annual_savings": annual_savings,
                    "grid_dependence": total_grid_import / profile.annual_load,
                }

    return best_design


def build_cost_map(
        simulator,
        storage_durations_hours,
        solar_penetrations,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years):
    """Build a 2D total-cost map for solar penetration and storage duration."""
    cost_map = np.zeros((len(solar_penetrations), len(storage_durations_hours)))

    for i, design_solar_penetration in enumerate(solar_penetrations):
        for j, design_storage_duration in enumerate(storage_durations_hours):
            (_, _, annual_grid_cost, _, _) = simulator.simulate_system(
                design_storage_duration,
                design_solar_penetration,
            )

            battery_capacity = simulator.battery_capacity(design_storage_duration)
            cost_map[i, j] = total_annual_cost(
                annual_grid_cost,
                battery_capacity,
                design_solar_penetration,
                battery_cost_per_kwh,
                battery_lifetime_years,
                solar_cost_per_penetration,
                solar_lifetime_years,
            )

    return cost_map


def build_design_uncertainty_comparison(
        profile,
        simulator,
        storage_durations_hours,
        solar_penetrations,
        uncertainty_experiments,
        design_comparison_scenarios,
        random_seed,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
        solar_variability,
        load_variability,
        price_variability,
        price_shift_variability):
    """Compare uncertainty and tail-risk metrics across a design grid."""
    design_uncertainty_results = []
    design_uncertainty_maps = {
        uncertainty_source: np.zeros((
            len(solar_penetrations),
            len(storage_durations_hours),
        ))
        for uncertainty_source, _, _ in uncertainty_experiments
    }

    for solar_index, design_solar_penetration in enumerate(solar_penetrations):
        for duration_index, design_storage_duration in enumerate(storage_durations_hours):
            design_battery_capacity = simulator.battery_capacity(
                design_storage_duration
            )
            (design_grid_import, design_solar_curtailed, design_grid_cost,
             _, _) = simulator.simulate_system(
                design_storage_duration,
                design_solar_penetration,
            )
            design_scaled_solar = simulator.scaled_solar(design_solar_penetration)
            design_annual_solar = np.sum(design_scaled_solar)
            design_solar_utilization = (
                (design_annual_solar - design_solar_curtailed) / design_annual_solar
            )
            design_result = {
                "Solar Penetration": design_solar_penetration,
                "Storage Duration (hours)": design_storage_duration,
                "Battery Capacity (kWh)": design_battery_capacity,
                "Annual Cost (EUR)": total_annual_cost(
                    design_grid_cost,
                    design_battery_capacity,
                    design_solar_penetration,
                    battery_cost_per_kwh,
                    battery_lifetime_years,
                    solar_cost_per_penetration,
                    solar_lifetime_years,
                ),
                "Solar Curtailment (kWh)": design_solar_curtailed,
                "Solar Utilization": design_solar_utilization,
                "Grid Dependence": design_grid_import / profile.annual_load,
            }

            for uncertainty_source, _, scenario_parameters in uncertainty_experiments:
                experiment_costs, _ = run_monte_carlo(
                    profile,
                    design_storage_duration,
                    design_solar_penetration,
                    battery_cost_per_kwh=battery_cost_per_kwh,
                    battery_lifetime_years=battery_lifetime_years,
                    solar_cost_per_penetration=solar_cost_per_penetration,
                    solar_lifetime_years=solar_lifetime_years,
                    num_scenarios=design_comparison_scenarios,
                    random_seed=random_seed,
                    **scenario_parameters,
                )

                cost_standard_deviation = np.std(experiment_costs, ddof=1)
                column_name = f"{uncertainty_source} Cost Std Dev (EUR)"
                design_result[column_name] = cost_standard_deviation
                design_uncertainty_maps[uncertainty_source][
                    solar_index,
                    duration_index,
                ] = cost_standard_deviation

            combined_scenario_costs, _ = run_monte_carlo(
                profile,
                design_storage_duration,
                design_solar_penetration,
                battery_cost_per_kwh=battery_cost_per_kwh,
                battery_lifetime_years=battery_lifetime_years,
                solar_cost_per_penetration=solar_cost_per_penetration,
                solar_lifetime_years=solar_lifetime_years,
                num_scenarios=design_comparison_scenarios,
                solar_variability=solar_variability,
                load_variability=load_variability,
                price_variability=price_variability,
                price_shift_variability=price_shift_variability,
                random_seed=random_seed,
            )
            design_value_at_risk, design_conditional_value_at_risk = (
                calculate_cost_risk_metrics(combined_scenario_costs)
            )
            design_expected_cost = np.mean(combined_scenario_costs)

            design_result["Expected Cost (EUR)"] = design_expected_cost
            design_result["All Sources Cost Std Dev (EUR)"] = np.std(
                combined_scenario_costs,
                ddof=1,
            )
            design_result["95% Value at Risk (EUR)"] = design_value_at_risk
            design_result["95% CVaR (EUR)"] = design_conditional_value_at_risk
            design_result["95% CVaR Tail Premium (EUR)"] = (
                design_conditional_value_at_risk - design_expected_cost
            )

            design_uncertainty_results.append(design_result)

    return pd.DataFrame(design_uncertainty_results), design_uncertainty_maps


def run_storage_sensitivity(
        simulator,
        storage_durations_hours,
        analysis_solar_penetration,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years):
    """Evaluate cost, savings, and curtailment against storage duration."""
    sensitivity_results = []

    for storage_duration in storage_durations_hours:
        (total_grid_import, total_solar_curtailed, annual_grid_cost,
         annual_savings, max_soc) = simulator.simulate_system(
            storage_duration,
            analysis_solar_penetration,
        )

        battery_capacity = simulator.battery_capacity(storage_duration)
        annualized_battery = annualized_battery_cost(
            battery_capacity,
            battery_cost_per_kwh,
            battery_lifetime_years,
        )
        annualized_solar = annualized_solar_cost(
            analysis_solar_penetration,
            solar_cost_per_penetration,
            solar_lifetime_years,
        )
        design_total_annual_cost = (
            annual_grid_cost
            + annualized_battery
            + annualized_solar
        )

        sensitivity_results.append({
            "Storage Duration (hours)": storage_duration,
            "Battery Capacity (kWh)": battery_capacity,
            "Total Grid Import (kWh)": total_grid_import,
            "Solar Curtailed (kWh)": total_solar_curtailed,
            "Annual Grid Cost (EUR)": annual_grid_cost,
            "Annualized Battery Cost (EUR)": annualized_battery,
            "Total Annual Cost (EUR)": design_total_annual_cost,
            "Annual Battery Savings (EUR)": annual_savings,
            "Max SOC (kWh)": max_soc,
        })

    return pd.DataFrame(sensitivity_results)
