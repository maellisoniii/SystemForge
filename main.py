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
plt.figure(figsize=(10,5))

plt.plot(hours, net_load)

plt.axhline(0, linestyle="--")

plt.title("Net Load")
plt.xlabel("Hour")
plt.ylabel("kWh")

plt.grid(True)

plt.show()

# Plot
