from pathlib import Path
import sys


project_dir = Path(__file__).parent
workspace_dir = project_dir.parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from SystemForge.config import (
    analysis_solar_penetration,
    battery_cost_per_kwh,
    battery_lifetime_years,
    battery_power_ratio,
    data_profile_config,
    design_comparison_scenarios,
    load_variability,
    monte_carlo_seed,
    num_monte_carlo_scenarios,
    price_shift_variability,
    price_variability,
    solar_cost_reference,
    solar_lifetime_years,
    solar_penetrations,
    solar_variability,
    storage_durations_hours,
    uncertainty_experiments,
)
from SystemForge.data_profiles import EnergyDataProfile
from SystemForge.design_search import (
    build_cost_map,
    build_design_uncertainty_comparison,
    find_best_design,
    run_storage_sensitivity,
)
from SystemForge.monte_carlo import run_monte_carlo, run_uncertainty_source_experiments
from SystemForge.plotting import (
    plot_annual_energy,
    plot_cost_map,
    plot_curtailment_vs_price,
    plot_design_uncertainty_maps,
    plot_expected_cost_vs_risk,
    plot_marginal_savings,
    plot_monte_carlo_distribution,
    plot_storage_cost_sensitivity,
    plot_storage_savings,
    plot_uncertainty_source_table,
)
from SystemForge.reporting import (
    print_best_design,
    print_design_uncertainty_table,
    print_load_solar_price_analysis,
    print_monte_carlo_analysis,
    print_storage_sensitivity,
    print_uncertainty_table,
)
from SystemForge.risk import summarize_cost_distribution
from SystemForge.simulation import SystemSimulator, analyze_load_solar_price


def main():
    """Conductor for the SystemForge analysis workflow."""
    energy_profile = EnergyDataProfile.from_csv(**data_profile_config)
    simulator = SystemSimulator(energy_profile, battery_power_ratio)

    solar_cost_per_penetration = (
        solar_cost_reference / energy_profile.reference_solar_penetration
    )

    best_design = find_best_design(
        energy_profile,
        simulator,
        storage_durations_hours,
        solar_penetrations,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
    )
    print_best_design(energy_profile, best_design)

    monte_carlo_storage_duration = best_design["storage_duration_hours"]
    monte_carlo_solar_penetration = best_design["solar_penetration"]

    scenario_total_costs, _ = run_monte_carlo(
        energy_profile,
        monte_carlo_storage_duration,
        monte_carlo_solar_penetration,
        battery_cost_per_kwh=battery_cost_per_kwh,
        battery_lifetime_years=battery_lifetime_years,
        solar_cost_per_penetration=solar_cost_per_penetration,
        solar_lifetime_years=solar_lifetime_years,
        num_scenarios=num_monte_carlo_scenarios,
        solar_variability=solar_variability,
        load_variability=load_variability,
        price_variability=price_variability,
        price_shift_variability=price_shift_variability,
        random_seed=monte_carlo_seed,
    )
    monte_carlo_summary = summarize_cost_distribution(
        scenario_total_costs,
        num_monte_carlo_scenarios,
    )
    print_monte_carlo_analysis(
        num_monte_carlo_scenarios,
        monte_carlo_storage_duration,
        monte_carlo_solar_penetration,
        monte_carlo_summary,
    )

    uncertainty_table = run_uncertainty_source_experiments(
        energy_profile,
        monte_carlo_storage_duration,
        monte_carlo_solar_penetration,
        uncertainty_experiments,
        num_monte_carlo_scenarios,
        monte_carlo_seed,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
    )
    print_uncertainty_table(uncertainty_table)
    plot_uncertainty_source_table(uncertainty_table)

    design_uncertainty_table, design_uncertainty_maps = (
        build_design_uncertainty_comparison(
            energy_profile,
            simulator,
            storage_durations_hours,
            solar_penetrations,
            uncertainty_experiments,
            design_comparison_scenarios,
            monte_carlo_seed,
            battery_cost_per_kwh,
            battery_lifetime_years,
            solar_cost_per_penetration,
            solar_lifetime_years,
            solar_variability,
            load_variability,
            price_variability,
            price_shift_variability,
        )
    )
    print_design_uncertainty_table(design_uncertainty_table)
    plot_design_uncertainty_maps(
        design_uncertainty_maps,
        storage_durations_hours,
        solar_penetrations,
    )
    plot_expected_cost_vs_risk(design_uncertainty_table)
    plot_monte_carlo_distribution(scenario_total_costs, monte_carlo_summary)

    load_solar_price_analysis = analyze_load_solar_price(
        energy_profile,
        simulator,
        best_design,
    )
    print_load_solar_price_analysis(energy_profile, load_solar_price_analysis)

    cost_map = build_cost_map(
        simulator,
        storage_durations_hours,
        solar_penetrations,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
    )
    plot_cost_map(cost_map, storage_durations_hours, solar_penetrations)

    plot_annual_energy(load_solar_price_analysis)
    plot_curtailment_vs_price(energy_profile, load_solar_price_analysis)

    sensitivity_table = run_storage_sensitivity(
        simulator,
        storage_durations_hours,
        analysis_solar_penetration,
        battery_cost_per_kwh,
        battery_lifetime_years,
        solar_cost_per_penetration,
        solar_lifetime_years,
    )
    print_storage_sensitivity(sensitivity_table)

    plot_storage_cost_sensitivity(sensitivity_table)
    plot_storage_savings(sensitivity_table)
    plot_marginal_savings(sensitivity_table)


if __name__ == "__main__":
    main()
