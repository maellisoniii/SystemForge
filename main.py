from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack


# ===========================
# REAL DATA
# ===========================

data_path = Path(__file__).parent / "data" / "europe_data.csv"
df = pd.read_csv(data_path)

# Use one complete year. The CSV contains multiple years, so multiplying the
# result by 365 would otherwise overstate annual costs and savings.
analysis_year = 2017
system_scale = 1e-5  # Model a small representative share of the German data.
battery_power_ratio = 0.25  # A full battery can charge or discharge in one hour.
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
df = df.loc[df["utc_timestamp"].dt.year == analysis_year].copy()

solar = (
    df["DE_solar_generation_actual"]
    .interpolate(limit_area="inside")
    .values
    * 1000
    * system_scale
)

load = (
    df["DE_load_actual_entsoe_transparency"]
    .interpolate(limit_area="inside")
    .values
    * 1000
    * system_scale
)

prices = (
    df["AT_price_day_ahead"]
    .interpolate(limit_area="inside")
    .values
    / 1000
)

# `prices` is already a NumPy array, so it does not have a "price" column.
hourly_price = prices

# Design metrics based on annual energy and average hourly demand.
annual_load = np.sum(load)
reference_annual_solar = np.sum(solar)
average_load = np.mean(load)
reference_solar_penetration = reference_annual_solar / annual_load


def solar_scale_from_penetration(solar_penetration):
    """Scale the recorded solar profile to the requested annual penetration."""
    return solar_penetration / reference_solar_penetration


def battery_capacity_from_duration(storage_duration_hours):
    """Convert storage duration to capacity using average hourly load."""
    return storage_duration_hours * average_load


# ============================
# INPUT DATA
# ============================

# load = np.array([
#     30, 28, 27, 26, 25, 25,
#     35, 45, 50, 55, 60, 65,
#     70, 72, 70, 68, 65, 70,
#     80, 75, 60, 50, 40, 35
# ])
# load = np.tile(load, 365)

# solar = np.array([
#     0, 0, 0, 0, 0, 5,
#     15, 25, 40, 55, 70, 80,
#     85, 80, 70, 55, 40, 20,
#     5, 0, 0, 0, 0, 0
# ])
# solar = np.tile(solar, 365)


# ============================
# SYSTEM SIMULATION
# ============================

simulation_cache = {}


def optimize_battery_dispatch(battery_capacity, scaled_solar):
    """Solve the hourly battery schedule that minimizes annual grid cost.

    The optimizer has perfect knowledge of the year's price, load, and solar
    profile. It chooses grid import, charging, discharging, curtailment, and
    state of charge subject to energy-balance, capacity, and power constraints.
    """
    number_of_hours = len(load)
    net_load = load - scaled_solar

    if battery_capacity == 0:
        return (
            np.maximum(net_load, 0),
            np.maximum(-net_load, 0),
            np.zeros(number_of_hours + 1),
        )

    identity = eye(number_of_hours, format="csr")
    zero_soc = csr_matrix((number_of_hours, number_of_hours + 1))
    zero_hourly = csr_matrix((number_of_hours, number_of_hours))
    state_of_charge_change = diags(
        [-np.ones(number_of_hours), np.ones(number_of_hours)],
        [0, 1],
        shape=(number_of_hours, number_of_hours + 1),
        format="csr",
    )

    # Variable order: grid import, charge, discharge, curtailment, state of charge.
    energy_balance = hstack(
        [identity, -identity, identity, -identity, zero_soc],
        format="csr",
    )
    battery_balance = hstack(
        [
            zero_hourly,
            -identity,
            identity,
            zero_hourly,
            state_of_charge_change,
        ],
        format="csr",
    )
    equality_constraints = vstack([energy_balance, battery_balance], format="csr")
    equality_values = np.concatenate([net_load, np.zeros(number_of_hours)])

    number_of_variables = 5 * number_of_hours + 1
    grid_start = 0
    charge_start = number_of_hours
    discharge_start = 2 * number_of_hours
    curtailment_start = 3 * number_of_hours
    soc_start = 4 * number_of_hours

    objective = np.zeros(number_of_variables)
    objective[grid_start:charge_start] = hourly_price
    # A tiny throughput penalty removes unnecessary charge/discharge cycles.
    objective[charge_start:discharge_start] = 1e-9
    objective[discharge_start:curtailment_start] = 1e-9

    lower_bounds = np.zeros(number_of_variables)
    upper_bounds = np.full(number_of_variables, np.inf)
    max_power = battery_capacity * battery_power_ratio
    upper_bounds[charge_start:discharge_start] = max_power
    upper_bounds[discharge_start:curtailment_start] = max_power
    upper_bounds[curtailment_start:soc_start] = scaled_solar
    upper_bounds[soc_start:] = battery_capacity

    # Start and end the year empty so the optimizer cannot borrow energy from
    # outside the analysis period or leave a free stored-energy benefit behind.
    upper_bounds[soc_start] = 0
    upper_bounds[-1] = 0

    solution = linprog(
        objective,
        A_eq=equality_constraints,
        b_eq=equality_values,
        bounds=list(zip(lower_bounds, upper_bounds)),
        method="highs",
    )

    if not solution.success:
        raise RuntimeError(f"Battery dispatch optimization failed: {solution.message}")

    return (
        solution.x[grid_start:charge_start],
        solution.x[curtailment_start:soc_start],
        solution.x[soc_start:],
    )


def simulate_system(storage_duration_hours, solar_penetration, return_hourly_data=False):
    battery_capacity = battery_capacity_from_duration(storage_duration_hours)
    scaled_solar = solar * solar_scale_from_penetration(solar_penetration)

    if (len(load) != len(scaled_solar)
            or len(load) != len(hourly_price)):
        raise ValueError("load, solar, and price arrays must have identical lengths")

    cache_key = (storage_duration_hours, solar_penetration)

    if not return_hourly_data and cache_key in simulation_cache:
        return simulation_cache[cache_key]

    grid_import, solar_curtailed, battery_soc = optimize_battery_dispatch(
        battery_capacity,
        scaled_solar,
    )
    grid_cost = np.dot(grid_import, hourly_price)
    cost_without_battery = np.dot(
        np.maximum(load - scaled_solar, 0),
        hourly_price,
    )

    results = (
        np.sum(grid_import),
        np.sum(solar_curtailed),
        grid_cost,
        cost_without_battery - grid_cost,
        np.max(battery_soc),
    )

    if not return_hourly_data:
        simulation_cache[cache_key] = results
        return results

    return results + (grid_import, solar_curtailed)


def run_monte_carlo(
        storage_duration_hours,
        solar_penetration,
        num_scenarios=1000,
        solar_variability=0.15,
        load_variability=0.08,
        price_variability=0.25,
        price_shift_variability=0.01,
        random_seed=42):
    """Return annual cost outcomes for randomized solar, load, and price scenarios.

    Each scenario keeps the recorded hourly profile, while varying annual solar
    yield, load level, and price level. Price shifts are in EUR/kWh.
    """
    rng = np.random.default_rng(random_seed)

    battery_capacity = battery_capacity_from_duration(storage_duration_hours)
    solar_profile_scale = solar_scale_from_penetration(solar_penetration)

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

        surplus_mask = surplus > 0
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


def calculate_cost_risk_metrics(costs, confidence_level=0.95):
    """Return Value at Risk and Conditional Value at Risk for annual costs."""
    value_at_risk = np.percentile(costs, confidence_level * 100)
    conditional_value_at_risk = np.mean(costs[costs >= value_at_risk])
    return value_at_risk, conditional_value_at_risk


# ============================
# DESIGN OPTIMIZATION
# ============================

# Choose designs using energy-system metrics rather than raw asset sizes.
storage_durations_hours = [0, 0.25, 0.5, 1.0, 2.0]
solar_penetrations = [0.05, 0.10, 0.15, 0.20, 0.25]

battery_cost_per_kwh = 300
battery_lifetime_years = 10

solar_cost_per_penetration = 8000 / reference_solar_penetration
solar_lifetime_years = 20

best_design = None
best_cost = float("inf")

for solar_penetration in solar_penetrations:
    for storage_duration in storage_durations_hours:
        (total_grid_import, total_solar_curtailed, annual_grid_cost,
         annual_savings, max_soc) = simulate_system(
            storage_duration,
            solar_penetration,
        )

        battery_capacity = battery_capacity_from_duration(storage_duration)

        battery_cost = battery_capacity * battery_cost_per_kwh
        annualized_battery_cost = battery_cost / battery_lifetime_years

        solar_cost = solar_penetration * solar_cost_per_penetration
        annualized_solar_cost = solar_cost / solar_lifetime_years

        total_annual_cost = (
            annual_grid_cost
            + annualized_battery_cost
            + annualized_solar_cost
        )

        if total_annual_cost < best_cost:
            best_cost = total_annual_cost
            best_design = {
                "solar_penetration": solar_penetration,
                "storage_duration_hours": storage_duration,
                "battery_capacity": battery_capacity,
                "total_annual_cost": total_annual_cost,
                "annual_grid_cost": annual_grid_cost,
                "annualized_battery_cost": annualized_battery_cost,
                "annualized_solar_cost": annualized_solar_cost,
                "max_soc": max_soc,
                "solar_curtailed": total_solar_curtailed,
                "annual_savings": annual_savings,
                "grid_dependence": total_grid_import / annual_load,
            }


print(f"\nBest System Design ({analysis_year})")
print("------------------")
print("Solar Penetration:", round(best_design["solar_penetration"] * 100, 1), "%")
print("Storage Duration:", best_design["storage_duration_hours"], "hours")
print("Battery Capacity:", round(best_design["battery_capacity"], 2), "kWh")
print("Total Annual Cost: EUR", round(best_design["total_annual_cost"], 2))
print("Annual Grid Cost: EUR", round(best_design["annual_grid_cost"], 2))
print("Annualized Battery Cost: EUR", round(best_design["annualized_battery_cost"], 2))
print("Annualized Solar Cost: EUR", round(best_design["annualized_solar_cost"], 2))
print("Annual Battery Savings: EUR", round(best_design["annual_savings"], 2))
print("Max Battery SOC:", round(best_design["max_soc"], 2), "kWh")
print("Solar Curtailed:", round(best_design["solar_curtailed"], 2), "kWh")
print("Grid Dependence:", round(best_design["grid_dependence"] * 100, 2), "%")


# ============================
# MONTE CARLO SCENARIO ANALYSIS
# ============================

num_monte_carlo_scenarios = 1000
monte_carlo_storage_duration = best_design["storage_duration_hours"]
monte_carlo_solar_penetration = best_design["solar_penetration"]
solar_variability = 0.15
load_variability = 0.08
price_variability = 0.25
price_shift_variability = 0.01
monte_carlo_seed = 42

scenario_total_costs, scenario_grid_costs = run_monte_carlo(
    monte_carlo_storage_duration,
    monte_carlo_solar_penetration,
    num_scenarios=num_monte_carlo_scenarios,
    solar_variability=solar_variability,
    load_variability=load_variability,
    price_variability=price_variability,
    price_shift_variability=price_shift_variability,
    random_seed=monte_carlo_seed,
)

expected_cost = np.mean(scenario_total_costs)
worst_case_cost = np.max(scenario_total_costs)
scenario_cost_interval = np.percentile(scenario_total_costs, [2.5, 97.5])
value_at_risk, conditional_value_at_risk = calculate_cost_risk_metrics(
    scenario_total_costs
)
cost_standard_deviation = np.std(scenario_total_costs, ddof=1)
print("Annual Cost Standard Deviation: EUR", round(cost_standard_deviation, 2))
standard_error = np.std(scenario_total_costs, ddof=1) / np.sqrt(num_monte_carlo_scenarios)
expected_cost_confidence_interval = (
    expected_cost - 1.96 * standard_error,
    expected_cost + 1.96 * standard_error,
)

print("\nMonte Carlo Cost Analysis")
print("-------------------------")
print("Scenarios:", num_monte_carlo_scenarios)
print("Storage Duration:", monte_carlo_storage_duration, "hours")
print("Solar Penetration:", round(monte_carlo_solar_penetration * 100, 1), "%")
print("Expected Annual Cost: EUR", round(expected_cost, 2))
print("Worst-Case Annual Cost: EUR", round(worst_case_cost, 2))
print("95% Value at Risk: EUR", round(value_at_risk, 2))
print("95% Conditional Value at Risk: EUR", round(conditional_value_at_risk, 2))
print(
    "95% Scenario Cost Interval: EUR",
    round(scenario_cost_interval[0], 2),
    "to EUR",
    round(scenario_cost_interval[1], 2),
)
print(
    "95% Confidence Interval for Expected Cost: EUR",
    round(expected_cost_confidence_interval[0], 2),
    "to EUR",
    round(expected_cost_confidence_interval[1], 2),
)


# ============================
# UNCERTAINTY SOURCE EXPERIMENTS
# ============================

uncertainty_experiments = [
    (
        "Solar yield",
        f"Solar variation: {solar_variability:.0%}",
        {
            "solar_variability": solar_variability,
            "load_variability": 0,
            "price_variability": 0,
            "price_shift_variability": 0,
        },
    ),
    (
        "Load level",
        f"Load variation: {load_variability:.0%}",
        {
            "solar_variability": 0,
            "load_variability": load_variability,
            "price_variability": 0,
            "price_shift_variability": 0,
        },
    ),
    (
        "Price level and shift",
        (
            f"Price scale: {price_variability:.0%}, "
            f"price shift: {price_shift_variability:.3f} EUR/kWh"
        ),
        {
            "solar_variability": 0,
            "load_variability": 0,
            "price_variability": price_variability,
            "price_shift_variability": price_shift_variability,
        },
    ),
]

uncertainty_results = []

for uncertainty_source, scenario_setting, scenario_parameters in uncertainty_experiments:
    experiment_costs, _ = run_monte_carlo(
        monte_carlo_storage_duration,
        monte_carlo_solar_penetration,
        num_scenarios=num_monte_carlo_scenarios,
        random_seed=monte_carlo_seed,
        **scenario_parameters,
    )

    uncertainty_results.append({
        "Uncertainty Source": uncertainty_source,
        "Scenario Setting": scenario_setting,
        "Cost Standard Deviation (EUR)": np.std(experiment_costs, ddof=1),
    })

uncertainty_table = pd.DataFrame(uncertainty_results)

print("\nUncertainty Source Comparison")
print("-----------------------------")
print(
    uncertainty_table.to_string(
        index=False,
        formatters={
            "Cost Standard Deviation (EUR)": "{:,.2f}".format,
        },
    )
)

table_for_display = uncertainty_table.copy()
table_for_display["Cost Standard Deviation (EUR)"] = (
    table_for_display["Cost Standard Deviation (EUR)"]
    .map(lambda value: f"EUR {value:,.2f}")
)

figure, axis = plt.subplots(figsize=(11, 2.5))
axis.axis("off")
axis.set_title("Monte Carlo Uncertainty Source Comparison", fontsize=14, pad=14)

comparison_table = axis.table(
    cellText=table_for_display.values,
    colLabels=table_for_display.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)
comparison_table.auto_set_font_size(False)
comparison_table.set_fontsize(10)
comparison_table.scale(1, 1.7)

for column_index in range(len(table_for_display.columns)):
    comparison_table[(0, column_index)].set_facecolor("steelblue")
    comparison_table[(0, column_index)].set_text_props(color="white", weight="bold")

plt.show()


# ============================
# DESIGN UNCERTAINTY COMPARISON
# ============================

# This analysis reports every design in the selected duration/penetration grid.
# Reduce either list below when you only need a faster representative sweep.
design_comparison_scenarios = 100
design_comparison_storage_durations = storage_durations_hours
design_comparison_solar_penetrations = solar_penetrations
design_uncertainty_results = []
design_uncertainty_maps = {
    uncertainty_source: np.zeros((
        len(design_comparison_solar_penetrations),
        len(design_comparison_storage_durations),
    ))
    for uncertainty_source, _, _ in uncertainty_experiments
}

for solar_index, design_solar_penetration in enumerate(design_comparison_solar_penetrations):
    for duration_index, design_storage_duration in enumerate(design_comparison_storage_durations):
        design_battery_capacity = battery_capacity_from_duration(design_storage_duration)
        (design_grid_import, design_solar_curtailed, design_grid_cost,
         design_savings, design_max_soc) = simulate_system(
            design_storage_duration,
            design_solar_penetration,
        )
        design_scaled_solar = solar * solar_scale_from_penetration(
            design_solar_penetration
        )
        design_annual_solar = np.sum(design_scaled_solar)
        design_solar_utilization = (
            (design_annual_solar - design_solar_curtailed) / design_annual_solar
        )
        design_result = {
            "Solar Penetration": design_solar_penetration,
            "Storage Duration (hours)": design_storage_duration,
            "Battery Capacity (kWh)": design_battery_capacity,
            "Annual Cost (EUR)": (
                design_grid_cost
                + design_battery_capacity * battery_cost_per_kwh / battery_lifetime_years
                + design_solar_penetration * solar_cost_per_penetration / solar_lifetime_years
            ),
            "Solar Curtailment (kWh)": design_solar_curtailed,
            "Solar Utilization": design_solar_utilization,
            "Grid Dependence": design_grid_import / annual_load,
        }

        for uncertainty_source, _, scenario_parameters in uncertainty_experiments:
            experiment_costs, _ = run_monte_carlo(
                design_storage_duration,
                design_solar_penetration,
                num_scenarios=design_comparison_scenarios,
                random_seed=monte_carlo_seed,
                **scenario_parameters,
            )

            cost_standard_deviation = np.std(experiment_costs, ddof=1)
            column_name = f"{uncertainty_source} Cost Std Dev (EUR)"
            design_result[column_name] = cost_standard_deviation
            design_uncertainty_maps[uncertainty_source][solar_index, duration_index] = (
                cost_standard_deviation
            )

        combined_scenario_costs, _ = run_monte_carlo(
            design_storage_duration,
            design_solar_penetration,
            num_scenarios=design_comparison_scenarios,
            solar_variability=solar_variability,
            load_variability=load_variability,
            price_variability=price_variability,
            price_shift_variability=price_shift_variability,
            random_seed=monte_carlo_seed,
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

design_uncertainty_table = pd.DataFrame(design_uncertainty_results)

print("\nDesign Uncertainty Comparison")
print("-----------------------------")
print(
    design_uncertainty_table.to_string(
        index=False,
        formatters={
            column: "{:,.2f}".format
            for column in design_uncertainty_table.columns
            if "(EUR)" in column
        }
        | {
            "Solar Penetration": "{:.1%}".format,
            "Solar Utilization": "{:.1%}".format,
            "Grid Dependence": "{:.1%}".format,
        },
    )
)

figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for axis, (uncertainty_source, cost_standard_deviation_map) in zip(
        axes,
        design_uncertainty_maps.items()):
    image = axis.imshow(cost_standard_deviation_map, aspect="auto", origin="lower")
    axis.set_title(f"Cost Risk from {uncertainty_source}")
    axis.set_xlabel("Storage Duration (hours)")
    axis.set_ylabel("Solar Penetration")
    axis.set_xticks(
        np.arange(len(design_comparison_storage_durations)),
        design_comparison_storage_durations,
    )
    axis.set_yticks(
        np.arange(len(design_comparison_solar_penetrations)),
        [f"{penetration:.0%}" for penetration in design_comparison_solar_penetrations],
    )
    figure.colorbar(image, ax=axis, label="Cost Std Dev (EUR)")

plt.show()


figure, risk_axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
marker_sizes = 60 + design_uncertainty_table["Storage Duration (hours)"] * 120

standard_deviation_plot = risk_axes[0].scatter(
    design_uncertainty_table["All Sources Cost Std Dev (EUR)"],
    design_uncertainty_table["Expected Cost (EUR)"],
    c=design_uncertainty_table["Solar Penetration"],
    s=marker_sizes,
    cmap="viridis",
    alpha=0.8,
)
risk_axes[0].set_title("Expected Cost vs Standard-Deviation Risk")
risk_axes[0].set_xlabel("Annual Cost Standard Deviation (EUR)")
risk_axes[0].set_ylabel("Expected Annual Cost (EUR)")
risk_axes[0].grid(True)

cvar_plot = risk_axes[1].scatter(
    design_uncertainty_table["95% CVaR Tail Premium (EUR)"],
    design_uncertainty_table["Expected Cost (EUR)"],
    c=design_uncertainty_table["Solar Penetration"],
    s=marker_sizes,
    cmap="viridis",
    alpha=0.8,
)
risk_axes[1].set_title("Expected Cost vs 95% CVaR Tail Risk")
risk_axes[1].set_xlabel("95% CVaR Tail Premium (EUR)")
risk_axes[1].set_ylabel("Expected Annual Cost (EUR)")
risk_axes[1].grid(True)

for _, design in design_uncertainty_table.iterrows():
    label = (
        f"S{design['Solar Penetration']:.0%} / "
        f"D{design['Storage Duration (hours)']:.2g}h"
    )
    risk_axes[0].annotate(
        label,
        (
            design["All Sources Cost Std Dev (EUR)"],
            design["Expected Cost (EUR)"],
        ),
        fontsize=7,
    )
    risk_axes[1].annotate(
        label,
        (
            design["95% CVaR Tail Premium (EUR)"],
            design["Expected Cost (EUR)"],
        ),
        fontsize=7,
    )

figure.colorbar(cvar_plot, ax=risk_axes, label="Solar Penetration")
plt.show()


plt.figure(figsize=(10, 5))
plt.hist(scenario_total_costs, bins=40, color="steelblue", edgecolor="white")
plt.axvline(expected_cost, color="black", linewidth=2, label="Expected cost")
plt.axvline(
    scenario_cost_interval[0],
    color="orange",
    linestyle="--",
    label="95% interval",
)
plt.axvline(scenario_cost_interval[1], color="orange", linestyle="--")
plt.title("Monte Carlo Distribution of Annual System Cost")
plt.xlabel("Annual Cost (EUR)")
plt.ylabel("Number of Scenarios")
plt.legend()
plt.grid(axis="y")
plt.show()


# ============================
# LOAD, SOLAR, AND PRICE ANALYSIS
# ============================

(total_grid_import, total_solar_curtailed, annual_grid_cost,
 annual_savings, max_soc, hourly_grid_import,
 hourly_solar_curtailed) = simulate_system(
    best_design["storage_duration_hours"],
    best_design["solar_penetration"],
    return_hourly_data=True,
)

scaled_solar = solar * solar_scale_from_penetration(best_design["solar_penetration"])
total_load = sum(load)
total_solar = sum(scaled_solar)
total_solar_used = total_solar - total_solar_curtailed
solar_penetration = total_solar / total_load
solar_utilization = total_solar_used / total_solar
grid_dependence = total_grid_import / total_load

min_price_index = np.argmin(hourly_price)
max_price_index = np.argmax(hourly_price)
curtailment_indices = np.where(hourly_solar_curtailed > 0)[0]

if len(curtailment_indices) > 0:
    min_curtailment_price_index = curtailment_indices[
        np.argmin(hourly_price[curtailment_indices])
    ]
    max_curtailment_price_index = curtailment_indices[
        np.argmax(hourly_price[curtailment_indices])
    ]

print("\nLoad, Solar, and Price Analysis")
print("-------------------------------")
print("Total Load:", round(total_load, 2), "kWh")
print("Total Solar Generation:", round(total_solar, 2), "kWh")
print("Total Solar Used:", round(total_solar_used, 2), "kWh")
print("Total Solar Curtailment:", round(total_solar_curtailed, 2), "kWh")
print("Solar Penetration:", round(solar_penetration * 100, 2), "%")
print("Solar Utilization:", round(solar_utilization * 100, 2), "%")
print("Grid Dependence:", round(grid_dependence * 100, 2), "%")
print("Minimum Price:", round(hourly_price[min_price_index], 4), "EUR/kWh")
print("Curtailment at Minimum Price:", round(hourly_solar_curtailed[min_price_index], 2), "kWh")
print("Maximum Price:", round(hourly_price[max_price_index], 4), "EUR/kWh")
print("Curtailment at Maximum Price:", round(hourly_solar_curtailed[max_price_index], 2), "kWh")

if len(curtailment_indices) > 0:
    print("Lowest Price During Curtailment:", round(hourly_price[min_curtailment_price_index], 4), "EUR/kWh")
    print("Curtailment at Lowest Curtailment Price:", round(hourly_solar_curtailed[min_curtailment_price_index], 2), "kWh")
    print("Highest Price During Curtailment:", round(hourly_price[max_curtailment_price_index], 4), "EUR/kWh")
    print("Curtailment at Highest Curtailment Price:", round(hourly_solar_curtailed[max_curtailment_price_index], 2), "kWh")


# ============================
# 2D DESIGN MAP
# ============================

cost_map = np.zeros((len(solar_penetrations), len(storage_durations_hours)))

for i, design_solar_penetration in enumerate(solar_penetrations):
    for j, design_storage_duration in enumerate(storage_durations_hours):
        (total_grid_import, total_solar_curtailed, annual_grid_cost,
         annual_savings, max_soc) = simulate_system(
            design_storage_duration,
            design_solar_penetration,
        )

        battery_capacity = battery_capacity_from_duration(design_storage_duration)

        battery_cost = battery_capacity * battery_cost_per_kwh
        annualized_battery_cost = battery_cost / battery_lifetime_years

        solar_cost = design_solar_penetration * solar_cost_per_penetration
        annualized_solar_cost = solar_cost / solar_lifetime_years

        total_annual_cost = (
            annual_grid_cost
            + annualized_battery_cost
            + annualized_solar_cost
        )
        cost_map[i, j] = total_annual_cost

plt.figure(figsize=(10, 6))
plt.imshow(cost_map, aspect="auto", origin="lower")
plt.colorbar(label="Total Annual Cost (EUR)")
plt.xticks(
    ticks=np.arange(len(storage_durations_hours)),
    labels=storage_durations_hours,
)
plt.yticks(
    ticks=np.arange(len(solar_penetrations)),
    labels=[f"{penetration:.0%}" for penetration in solar_penetrations],
)
plt.title("Solar Penetration + Storage Duration Cost Map")
plt.xlabel("Storage Duration (hours)")
plt.ylabel("Solar Penetration")
plt.show()


# ============================
# LOAD, SOLAR, AND CURTAILMENT VISUALIZATION
# ============================

plt.figure(figsize=(10, 5))
plt.bar(
    ["Total Load", "Solar Generation", "Solar Used", "Solar Curtailed"],
    [total_load, total_solar, total_solar_used, total_solar_curtailed],
    color=["steelblue", "gold", "mediumseagreen", "tomato"],
)
plt.title("Annual Load and Solar Energy")
plt.ylabel("Energy (kWh)")
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(hourly_price, hourly_solar_curtailed, alpha=0.35, s=12)
if len(curtailment_indices) > 0:
    plt.scatter(
        hourly_price[min_curtailment_price_index],
        hourly_solar_curtailed[min_curtailment_price_index],
        color="green",
        s=80,
        label="Lowest price with curtailment",
    )
    plt.scatter(
        hourly_price[max_curtailment_price_index],
        hourly_solar_curtailed[max_curtailment_price_index],
        color="red",
        s=80,
        label="Highest price with curtailment",
    )
plt.title("Solar Curtailment vs Electricity Price")
plt.xlabel("Electricity Price (EUR/kWh)")
plt.ylabel("Solar Curtailment (kWh)")
plt.legend()
plt.grid(True)
plt.show()


# ============================
# SENSITIVITY ANALYSIS
# ============================

analysis_solar_penetration = 0.15

grid_results = []
curtailment_results = []
annual_grid_cost_results = []
annualized_battery_cost_results = []
total_annual_cost_results = []
annual_savings_results = []

for storage_duration in storage_durations_hours:
    (total_grid_import, total_solar_curtailed, annual_grid_cost,
     annual_savings, max_soc) = simulate_system(
        storage_duration,
        analysis_solar_penetration,
    )

    battery_capacity = battery_capacity_from_duration(storage_duration)

    battery_cost = battery_capacity * battery_cost_per_kwh
    annualized_battery_cost = battery_cost / battery_lifetime_years

    solar_cost = analysis_solar_penetration * solar_cost_per_penetration
    annualized_solar_cost = solar_cost / solar_lifetime_years

    total_annual_cost = (
        annual_grid_cost
        + annualized_battery_cost
        + annualized_solar_cost
    )

    grid_results.append(total_grid_import)
    curtailment_results.append(total_solar_curtailed)
    annual_grid_cost_results.append(annual_grid_cost)
    annualized_battery_cost_results.append(annualized_battery_cost)
    total_annual_cost_results.append(total_annual_cost)
    annual_savings_results.append(annual_savings)

    print("\nStorage Duration:", storage_duration, "hours")
    print("Battery Capacity:", round(battery_capacity, 2), "kWh")
    print("Total Grid Import:", round(total_grid_import, 2), "kWh")
    print("Solar Curtailed:", round(total_solar_curtailed, 2), "kWh")
    print("Annual Grid Cost: EUR", round(annual_grid_cost, 2))
    print("Annualized Battery Cost: EUR", round(annualized_battery_cost, 2))
    print("Total Annual Cost: EUR", round(total_annual_cost, 2))
    print("Annual Battery Savings: EUR", round(annual_savings, 2))
    print("Max SOC:", round(max_soc, 2), "kWh")


marginal_savings = np.diff(annual_savings_results)


# ============================
# VISUALIZATION
# ============================

plt.figure(figsize=(10, 5))
plt.plot(storage_durations_hours, annual_grid_cost_results, marker="o", linewidth=2, label="Annual Grid Cost")
plt.plot(storage_durations_hours, annualized_battery_cost_results, marker="o", linewidth=2, label="Annualized Battery Cost")
plt.plot(storage_durations_hours, total_annual_cost_results, marker="o", linewidth=2, label="Total Annual Cost")
plt.title("Storage Duration vs Annual System Cost")
plt.xlabel("Storage Duration (hours)")
plt.ylabel("Annual Cost (EUR)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(storage_durations_hours, annual_savings_results, marker="o", linewidth=2)
plt.title("Storage Duration vs Annual Savings")
plt.xlabel("Storage Duration (hours)")
plt.ylabel("Annual Battery Savings (EUR)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(storage_durations_hours[1:], marginal_savings, marker="o", linewidth=2)
plt.title("Marginal Savings from Additional Storage Duration")
plt.xlabel("Storage Duration (hours)")
plt.ylabel("Additional Annual Savings (EUR)")
plt.grid(True)
plt.show()
