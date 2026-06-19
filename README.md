# SystemForge
SystemForge is a systems engineering platform capable of modeling and optimizing complex infrastructure sytems. 

## Current Module 
SystemForge-Energy

### Objective 
Develop a solar + battery + grid optimization platform capable of: 
- Energy flow simulation 
- Cost Optimization
- Emissions analysis
- Resilience analysis
- Infrastructure decision support

## Phase 1 
Energy Flow Simulation 
Goals: 
- Model building load
- Model solar generation
- Calculate net load
- Simulate battery dispatch

A synthetic 24-hour demand and solar profile was developed to visualize model behavior. 
Inputs included: 
- Hourly load (kWh)
- Hourly solar production (kWh)
- Varying electricity prices
- Battery Capacity

The model simualted: 
- Battery charging from energy surplus
- Battery discharging
- Grid imports
- Solar curtailment
- Annual costs and savings

Battery Dispatch Logic
For each hour: 
- Solar Generation was compared against electrical demand
- Surplus solar charged the battery
- Remaining surplus (curtailed energy) was calculated
- During deficits, the battery discharged when economically favorable
- Remaining demand was supplied by the electrical grid

Outputs: 
- Grid imports
- Battery charge level
- Solar curtailment
- Annual costs and savings

Economic Analysis: 
Battery Cost = Capacity x Cost per kWh 

Assumptions: 
- Battery Cost = 300 EUR/kWh
- Battery lifetime = 10 years

Annualized battery cost: 
- Annualized Cost = Battery Cost / Lifetime

Solar Cost: 
Solar capacity was scaled using a solar multiplier. 

Assumptions: 
- Solar cost = 8,000 EUR per unit
- Solar lifetime = 20 years

Annualized Solar Cost = Solar Cost / Lifetime

## Phase 2: Real Data Integration
https://data.open-power-system-data.org/time_series/
This data was aggregated across the EU and some neighboring countries. Data includes load, wind and solar, prices in hourly resolution. 6 years worth of data (2017-2023).

Dataset characteristics: 
- Approximately 50,000 hourly observations
- Germany electrical system was selected
- ENTSO-E transparency data
- Day-ahead electricity prices

Key variables: 

DE_load_actual_entsoe_transparency
DE_solar_generation_actual
AT_price_day_ahead

Data cleaning included: 
- Missing value identification
- Interpolation of missing observations
- Unit verification
- Length consistency checks

# Results 
Load Analysis
- German load data showed realistic demand ranges (35kWh - 70kWh)
- Strong daily cycles

Solar Analysis
- Distinct daytime peaks
- Significant variability

Max solar fraction:
Solar Generation / Load = 0.73
(73% of system demand was met by solar)

Solar Fraction Distribution: 
------

Curtailment Analysis
----- 

Battery Economics 
------

Market-based Battery Operation
----


## Software Tools 
- Python
- Numpy
- Pandas
- Matplotlib
- Visual Studio Code

## Long-Term Vision 
SystemForge will become a portfolio of systems engineering tools focused on energy, sustainability, and optimized infrastructure. 
