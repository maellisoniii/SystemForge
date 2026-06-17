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
hourly_price = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.15, 0.15, 0.15, 0.15,
    0.20, 0.20, 0.20, 0.20,
    0.20,
    0.30, 0.30, 0.30, 0.30,
    0.40, 0.40, 0.40,
    0.20, 0.15
])

def simulate_battery(battery_capacity):

    battery_soc = 0
    max_soc = 0

    battery_history = []
    grid_import = []
    solar_curtailed = []
    daily_grid_cost = 0
    daily_cost_without_battery = 0

    high_price_threshold = 0.30

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

        battery_history.append(battery_soc)
        grid_import.append(grid)
        solar_curtailed.append(curtailed)

    total_grid_import = sum(grid_import)
    total_solar_curtailed = sum(solar_curtailed)
    annual_grid_cost = daily_grid_cost * 365
    annual_savings = (daily_cost_without_battery - daily_grid_cost) * 365
    
    return total_grid_import, total_solar_curtailed, annual_grid_cost, annual_savings, max_soc

battery_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
battery_cost_per_kwh = 300 
battery_lifetime_years = 10 

grid_results = []
curtailment_results = []
annual_grid_cost_results = []
annualized_battery_cost_results = []
battery_cost_results = []
total_annual_cost_results = []
annual_saving_results = []

for battery_capacity in battery_sizes:
    total_grid_import, total_solar_curtailed, annual_grid_cost, annual_savings, max_soc = simulate_battery(battery_capacity)

    battery_cost = battery_capacity * battery_cost_per_kwh
    annualized_battery_cost = battery_cost / battery_lifetime_years
    total_annual_cost = annual_grid_cost + annualized_battery_cost
   
    grid_results.append(total_grid_import)
    curtailment_results.append(total_solar_curtailed)
    annual_grid_cost_results.append(annual_grid_cost)
    annualized_battery_cost_results.append(annualized_battery_cost)
    total_annual_cost_results.append(total_annual_cost)
    annual_saving_results.append(annual_savings)

    print("Battery Capacity:", battery_capacity, "kWh")
    print("Total Grid Import:", total_grid_import, "kWh")
    print("Solar Curtailed:", total_solar_curtailed, "kWh/day")
    print("Annual Grid Cost: $", round(annual_grid_cost, 2))
    print("Annualized Battery Cost: $", round(annualized_battery_cost,2))
    print("Total Annual Cost: $", round(total_annual_cost, 2))
    print("Annual Battery Savings", round(annual_savings, 2))
    print("Battery:", battery_capacity,"Curtailment:", total_solar_curtailed)
    print("Battery:", battery_capacity, "Max SOC:", round(max_soc, 2))

    print()

marginal_savings = np.diff(annual_saving_results)
print(marginal_savings)

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

plt.plot(battery_sizes, annual_saving_results, marker="o", linewidth=2)

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

