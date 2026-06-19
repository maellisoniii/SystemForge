from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ===========================
# REAL DATA
# ===========================

data_path = Path(__file__).parent / "data" / "europe_data.csv"
df = pd.read_csv(data_path)

# Use one complete year. The CSV contains multiple years, so multiplying the
# result by 365 would otherwise overstate annual costs and savings.
analysis_year = 2017
system_scale = 1e-5  # Model a small representative share of the German data.
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

def simulate_system(battery_capacity, solar_multiplier, return_hourly_data=False):
    scaled_solar = solar * solar_multiplier

    battery_soc = 0
    max_soc = 0

    grid_import = []
    solar_curtailed = []

    grid_cost = 0
    cost_without_battery = 0

    # Charge from the grid during the cheapest 25% of hours and discharge
    # during the most expensive 25% of hours.
    charging_threshold = np.quantile(hourly_price, 0.25)
    high_price_threshold = np.quantile(hourly_price, 0.75)

    if (len(load) != len(scaled_solar)
            or len(load) != len(hourly_price)):
        raise ValueError("load, solar, and price arrays must have identical lengths")

    for i in range(len(load)):
        surplus = scaled_solar[i] - load[i]

        if surplus > 0:
            available_space = battery_capacity - battery_soc
            charge = min(surplus, available_space)

            battery_soc += charge
            curtailed = surplus - charge
            grid = 0

        else:
            deficit = -surplus
            cost_without_battery += deficit * hourly_price[i]

            if hourly_price[i] >= high_price_threshold:
                discharge = min(deficit, battery_soc)
            else:
                discharge = 0

            battery_soc -= discharge
            grid = deficit - discharge
            curtailed = 0

        # When electricity is cheap, fill any remaining battery capacity from
        # the grid. This grid energy is included in both import and cost.
        if hourly_price[i] < charging_threshold:
            available_space = battery_capacity - battery_soc
            grid_charge = max(available_space, 0)

            battery_soc += grid_charge
            grid += grid_charge

        max_soc = max(max_soc, battery_soc)
        grid_cost += grid * hourly_price[i]

        grid_import.append(grid)
        solar_curtailed.append(curtailed)

    total_grid_import = sum(grid_import)
    total_solar_curtailed = sum(solar_curtailed)

    # The simulation covers a full year, so no additional multiplier is needed.
    annual_grid_cost = grid_cost
    annual_savings = cost_without_battery - grid_cost

    results = (
        total_grid_import,
        total_solar_curtailed,
        annual_grid_cost,
        annual_savings,
        max_soc,
    )

    if return_hourly_data:
        return results + (
            np.array(grid_import),
            np.array(solar_curtailed),
        )

    return results


# ============================
# DESIGN OPTIMIZATION
# ============================

battery_sizes = [0, 10, 20, 30, 40, 50, 75, 100]
solar_multipliers = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

battery_cost_per_kwh = 300
battery_lifetime_years = 10

solar_cost_per_multiplier = 8000
solar_lifetime_years = 20

best_design = None
best_cost = float("inf")

for solar_multiplier in solar_multipliers:
    for battery_capacity in battery_sizes:
        (total_grid_import, total_solar_curtailed, annual_grid_cost,
         annual_savings, max_soc) = simulate_system(
            battery_capacity,
            solar_multiplier,
        )

        battery_cost = battery_capacity * battery_cost_per_kwh
        annualized_battery_cost = battery_cost / battery_lifetime_years

        solar_cost = solar_multiplier * solar_cost_per_multiplier
        annualized_solar_cost = solar_cost / solar_lifetime_years

        total_annual_cost = (
            annual_grid_cost
            + annualized_battery_cost
            + annualized_solar_cost
        )

        if total_annual_cost < best_cost:
            best_cost = total_annual_cost
            best_design = {
                "solar_multiplier": solar_multiplier,
                "battery_capacity": battery_capacity,
                "total_annual_cost": total_annual_cost,
                "annual_grid_cost": annual_grid_cost,
                "annualized_battery_cost": annualized_battery_cost,
                "annualized_solar_cost": annualized_solar_cost,
                "max_soc": max_soc,
                "solar_curtailed": total_solar_curtailed,
                "annual_savings": annual_savings,
            }


print(f"\nBest System Design ({analysis_year})")
print("------------------")
print("Solar Multiplier:", best_design["solar_multiplier"])
print("Battery Capacity:", best_design["battery_capacity"], "kWh")
print("Total Annual Cost: EUR", round(best_design["total_annual_cost"], 2))
print("Annual Grid Cost: EUR", round(best_design["annual_grid_cost"], 2))
print("Annualized Battery Cost: EUR", round(best_design["annualized_battery_cost"], 2))
print("Annualized Solar Cost: EUR", round(best_design["annualized_solar_cost"], 2))
print("Annual Battery Savings: EUR", round(best_design["annual_savings"], 2))
print("Max Battery SOC:", round(best_design["max_soc"], 2), "kWh")
print("Solar Curtailed:", round(best_design["solar_curtailed"], 2), "kWh")


# ============================
# LOAD, SOLAR, AND PRICE ANALYSIS
# ============================

(total_grid_import, total_solar_curtailed, annual_grid_cost,
 annual_savings, max_soc, hourly_grid_import,
 hourly_solar_curtailed) = simulate_system(
    best_design["battery_capacity"],
    best_design["solar_multiplier"],
    return_hourly_data=True,
)

scaled_solar = solar * best_design["solar_multiplier"]
total_load = sum(load)
total_solar = sum(scaled_solar)
total_solar_used = total_solar - total_solar_curtailed

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

cost_map = np.zeros((len(solar_multipliers), len(battery_sizes)))

for i, solar_multiplier in enumerate(solar_multipliers):
    for j, battery_capacity in enumerate(battery_sizes):
        (total_grid_import, total_solar_curtailed, annual_grid_cost,
         annual_savings, max_soc) = simulate_system(
            battery_capacity,
            solar_multiplier,
        )

        battery_cost = battery_capacity * battery_cost_per_kwh
        annualized_battery_cost = battery_cost / battery_lifetime_years

        solar_cost = solar_multiplier * solar_cost_per_multiplier
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
plt.xticks(ticks=np.arange(len(battery_sizes)), labels=battery_sizes)
plt.yticks(ticks=np.arange(len(solar_multipliers)), labels=solar_multipliers)
plt.title("Solar + Battery Design Cost Map")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Solar Multiplier")
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

analysis_solar_multiplier = 1.0

grid_results = []
curtailment_results = []
annual_grid_cost_results = []
annualized_battery_cost_results = []
total_annual_cost_results = []
annual_savings_results = []

for battery_capacity in battery_sizes:
    (total_grid_import, total_solar_curtailed, annual_grid_cost,
     annual_savings, max_soc) = simulate_system(
        battery_capacity,
        analysis_solar_multiplier,
    )

    battery_cost = battery_capacity * battery_cost_per_kwh
    annualized_battery_cost = battery_cost / battery_lifetime_years

    solar_cost = analysis_solar_multiplier * solar_cost_per_multiplier
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

    print("\nBattery Capacity:", battery_capacity, "kWh")
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
plt.plot(battery_sizes, annual_grid_cost_results, marker="o", linewidth=2, label="Annual Grid Cost")
plt.plot(battery_sizes, annualized_battery_cost_results, marker="o", linewidth=2, label="Annualized Battery Cost")
plt.plot(battery_sizes, total_annual_cost_results, marker="o", linewidth=2, label="Total Annual Cost")
plt.title("Battery Capacity vs Annual System Cost")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Annual Cost (EUR)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(battery_sizes, annual_savings_results, marker="o", linewidth=2)
plt.title("Battery Capacity vs Annual Savings")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Annual Battery Savings (EUR)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(battery_sizes[1:], marginal_savings, marker="o", linewidth=2)
plt.title("Marginal Savings from Additional Battery Capacity")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Additional Annual Savings (EUR)")
plt.grid(True)
plt.show()
