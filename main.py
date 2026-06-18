import numpy as np
import matplotlib.pyplot as plt


# ============================
# INPUT DATA
# ============================

hours = np.arange(24)

load = np.array([
    30, 28, 27, 26, 25, 25,
    35, 45, 50, 55, 60, 65,
    70, 72, 70, 68, 65, 70,
    80, 75, 60, 50, 40, 35
])

solar = np.array([
    0, 0, 0, 0, 0, 5,
    15, 25, 40, 55, 70, 80,
    85, 80, 70, 55, 40, 20,
    5, 0, 0, 0, 0, 0
])

hourly_price = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.15, 0.15, 0.15, 0.15,
    0.20, 0.20, 0.20, 0.20, 0.20,
    0.30, 0.30, 0.30, 0.30,
    0.40, 0.40, 0.40,
    0.20, 0.15
])


# ============================
# SYSTEM SIMULATION
# ============================

def simulate_system(battery_capacity, solar_multiplier):
    scaled_solar = solar * solar_multiplier

    battery_soc = 0
    max_soc = 0

    grid_import = []
    solar_curtailed = []

    daily_grid_cost = 0
    daily_cost_without_battery = 0

    high_price_threshold = 0.30

    for i in range(24):
        surplus = scaled_solar[i] - load[i]

        if surplus > 0:
            available_space = battery_capacity - battery_soc
            charge = min(surplus, available_space)

            battery_soc += charge
            curtailed = surplus - charge
            grid = 0

        else:
            deficit = -surplus

            daily_cost_without_battery += deficit * hourly_price[i]

            if hourly_price[i] >= high_price_threshold:
                discharge = min(deficit, battery_soc)
            else:
                discharge = 0

            battery_soc -= discharge
            grid = deficit - discharge
            curtailed = 0

        max_soc = max(max_soc, battery_soc)

        daily_grid_cost += grid * hourly_price[i]

        grid_import.append(grid)
        solar_curtailed.append(curtailed)

    total_grid_import = sum(grid_import)
    total_solar_curtailed = sum(solar_curtailed)
    annual_grid_cost = daily_grid_cost * 365
    annual_savings = (daily_cost_without_battery - daily_grid_cost) * 365

    return total_grid_import, total_solar_curtailed, annual_grid_cost, annual_savings, max_soc


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

        total_grid_import, total_solar_curtailed, annual_grid_cost, annual_savings, max_soc = simulate_system(
            battery_capacity,
            solar_multiplier
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
                "annual_savings": annual_savings
            }


print("\nBest System Design")
print("------------------")
print("Solar Multiplier:", best_design["solar_multiplier"])
print("Battery Capacity:", best_design["battery_capacity"], "kWh")
print("Total Annual Cost: $", round(best_design["total_annual_cost"], 2))
print("Annual Grid Cost: $", round(best_design["annual_grid_cost"], 2))
print("Annualized Battery Cost: $", round(best_design["annualized_battery_cost"], 2))
print("Annualized Solar Cost: $", round(best_design["annualized_solar_cost"], 2))
print("Annual Battery Savings: $", round(best_design["annual_savings"], 2))
print("Max Battery SOC:", round(best_design["max_soc"], 2), "kWh")
print("Solar Curtailed:", round(best_design["solar_curtailed"], 2), "kWh/day")


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

    total_grid_import, total_solar_curtailed, annual_grid_cost, annual_savings, max_soc = simulate_system(
        battery_capacity,
        analysis_solar_multiplier
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
    print("Total Grid Import:", round(total_grid_import, 2), "kWh/day")
    print("Solar Curtailed:", round(total_solar_curtailed, 2), "kWh/day")
    print("Annual Grid Cost: $", round(annual_grid_cost, 2))
    print("Annualized Battery Cost: $", round(annualized_battery_cost, 2))
    print("Total Annual Cost: $", round(total_annual_cost, 2))
    print("Annual Battery Savings: $", round(annual_savings, 2))
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
plt.ylabel("Annual Cost ($)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(battery_sizes, annual_savings_results, marker="o", linewidth=2)

plt.title("Battery Capacity vs Annual Savings")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Annual Battery Savings ($)")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(battery_sizes[1:], marginal_savings, marker="o", linewidth=2)

plt.title("Marginal Savings from Additional Battery Capacity")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Additional Annual Savings ($)")
plt.grid(True)
plt.show()