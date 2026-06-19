import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("data/europe_data.csv")

print(df["DE_load_actual_entsoe_transparency"]
      .isna()
      .sum()
      )
load = df["DE_load_actual_entsoe_transparency"]
plt.plot(load[:168])
plt.title("One week load")
plt.show()

print(load.head())

print(df["DE_load_actual_entsoe_transparency"].describe())
solar = (df["DE_solar_generation_actual"]
         .interpolate()
         .values
)
load = (df["DE_load_actual_entsoe_transparency"]
        .interpolate()
        .values
        )
print(len(load))
print(len(solar))
solar = df["DE_solar_generation_actual"]
net_load = load - solar 
solar_fraction = solar / load
print("Max solar fraction", solar_fraction.max())
idx = np.argmax(solar_fraction)

print("Index:", idx)
print("Solar fraction:", solar_fraction[idx])
print("Load:", load[idx])
print("Solar:", solar[idx])
start = idx - 48
end = idx + 48

plt.figure(figsize=(12,5))
plt.plot(solar_fraction[start:end])
plt.title("High Solar Fraction Event")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(solar_fraction[:168])
plt.title("One Week Solar Fraction")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))

plt.plot(load[:168], label="Load")
plt.plot(solar[:168], label="Solar")
plt.plot(net_load[:168], label="Net Load")
plt.title("Comparison")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.hist(solar_fraction, bins=50)

plt.title("Distribution of Solar Fraction")
plt.xlabel("Solar Fraction")
plt.ylabel("Hours")
plt.grid(True)
plt.show()