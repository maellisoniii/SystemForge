import numpy as np 
import matplotlib.pyplot as plt 

# Hours of the day
hours = np.arange(24)

# Example building demand (kWh)
load = np.array([
    30, 28, 27, 26, 25, 25,
    35, 45, 50, 55, 60, 65,
    70, 72, 70, 68, 65, 70,
    80, 75, 60, 50, 40, 35
])

# Example solar production (kWh)
solar = np.array([
    0, 0, 0, 0, 0, 5,
    15, 25, 40, 55, 70, 80,
    85, 80, 70, 55, 40, 20,
    5, 0, 0, 0, 0, 0
])
net_load = load - solar
def simulate_battery(battery_capacity):

    battery_soc = 0

    battery_history = []
    grid_import = []
    solar_curtailed = []

    for i in range(24):

        surplus = solar[i] - load[i]

        if surplus > 0:

            available_space = battery_capacity - battery_soc

            charge = min(surplus, available_space)

            battery_soc += charge

            curtailed = surplus - charge

            grid = 0

        else:

            deficit = -surplus

            discharge = min(deficit, battery_soc)

            battery_soc -= discharge

            grid = deficit - discharge

            curtailed = 0

        battery_history.append(battery_soc)
        grid_import.append(grid)
        solar_curtailed.append(curtailed)

    total_grid_import = sum(grid_import)
    total_solar_curtailed = sum(solar_curtailed)

    return total_grid_import, total_solar_curtailed

battery_sizes = [25,50,75,100,125,150,175,200]
electricity_price = 0.15
battery_cost_per_kwh = 300 

grid_results = []
curtailment_results = []
annual_grid_cost_results = []
battery_cost_results = []
total_cost_results = []

for battery_capacity in battery_sizes:
    total_grid_import, total_solar_curtailed = simulate_battery(battery_capacity)

    annual_grid_cost = total_grid_import * electricity_price * 365
    battery_cost = battery_capacity * battery_cost_per_kwh
    total_cost = annual_grid_cost + battery_cost

    annual_grid_cost_results.append(total_grid_import)
    curtailment_results.append(total_solar_curtailed)
    annual_grid_cost_results.append(annual_grid_cost)
    battery_cost_results.append(battery_cost)
    total_cost_results.append(total_cost)

    print("Battery Capacity:", battery_capacity, "kWh")
    print("Total Grid Import:", total_grid_import, "kWh")
    print("Grid Cost: $", annual_grid_cost)
    print("Battery Cost: $", battery_cost)
    print("Total Cost: $", total_cost)
    print()
plt.figure(figsize=(10, 5))

plt.plot(battery_sizes, total_cost_results, marker="o", linewidth=2)

plt.title("Battery Capacity vs Total System Cost")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Total Cost ($)")

plt.grid(True)

plt.show()






