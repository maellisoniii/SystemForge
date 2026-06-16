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
battery_capacity = 100
battery_soc = 50

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
plt.figure(figsize=(10,5))

plt.plot(hours, battery_history, linewidth=2)

plt.title("Battery State of Charge")
plt.xlabel("Hour")
plt.ylabel("Battery Energy Stored (kWh)")

plt.grid(True)

plt.show()
print("Maximum Load:", np.max(load))
print("Maximum Solar:", np.max(solar))
print("Minimum Net Load:", np.min(net_load))
print("Maximum Net Load:", np.max(net_load))

