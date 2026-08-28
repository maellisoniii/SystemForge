from pathlib import Path


# 
# DATA PROFILE ASSUMPTIONS
# 

project_dir = Path(__file__).parent
system_scale = 1e-5

# Germany/Austria remains the default 
# scenario, but it is now just a scenario.
# To test another building, market, 
# tariff, or weather file, change this mapping
# instead of changing the simulation, 
# optimization, or reporting engines.
data_profile_config = {
    "name": "European sample profile",
    "csv_path": project_dir / "data" / "europe_data.csv",
    "timestamp_column": "utc_timestamp",
    "analysis_year": 2017,
    "load_column": "DE_load_actual_entsoe_transparency",
    "solar_column": "DE_solar_generation_actual",
    "price_column": "AT_price_day_ahead",
    # Source load/solar are MW-like hourly values; convert to kWh and scale down.
    "load_multiplier": 1000 * system_scale,
    "solar_multiplier": 1000 * system_scale,
    # Source price is EUR/MWh; convert to EUR/kWh.
    "price_multiplier": 1 / 1000,
}


# 
# DESIGN ASSUMPTIONS
# 

# Legacy design-search assumptions:
# simulation.py and the original grid-search 
# workflow represent battery
# power as a fixed fraction 
# of battery energy capacity.
#
# The current deterministic capacity 
# and stochastic optimization models
# DO NOT use this assumption. 
# They optimize battery energy capacity and
# battery power capacity independently.
legacy_battery_power_ratio = 0.25


# Legacy design-search assumptions:
# these are retained for comparison with 
# prior results, but they are not used in the 
# current optimization models.
legacy_storage_durations_hours = [
    0, 0.25, 0.5, 1.0, 2.0]
legacy_solar_penetrations = [0.05, 0.10, 0.15, 0.20, 0.25]

### These need to be documented and justified. 
# They are not based on any
### particular market or technology assumptions.
battery_cost_per_kwh = 300
battery_lifetime_years = 10

# Financial assumption used to annualize 
# capital costs with a capital recovery
# factor instead of 
# simple cost / lifetime division.
#
# This is a first-pass project 
# finance assumption, not a 
# market-specific WACC.
# Changing this number can 
# materially change the 
# optimal design because it
# changes how expensive long-lived 
# assets look on an annual basis.
discount_rate = 0.05

# This preserves the prior cost scale. 
# The annualized solar cost per penetration
# is derived after the active profile is 
# loaded because reference penetration is
# profile-specific.
solar_cost_reference = 8000
solar_lifetime_years = 20

# Battery efficiency assumptions.
# These are kept separate so the model can later distinguish energy lost while
# charging from energy lost while discharging.
#
# Current planned interpretation:
# - charge efficiency applies when energy enters the battery
# - discharge efficiency applies when 
# stored energy serves load or offsets grid
# - round-trip efficiency is 
# approx charge_efficiency * discharge_efficiency
#
# Charge and discharge efficiencies are 
# used consistently by the
# deterministic dispatch, 
# Monte Carlo dispatch, capacity co-optimization,
# and stochastic optimization models.
battery_charge_efficiency = 0.95
battery_discharge_efficiency = 0.95
battery_round_trip_efficiency = (
    battery_charge_efficiency * battery_discharge_efficiency
)


# 
# MONTE CARLO ASSUMPTIONS
# 

num_monte_carlo_scenarios = 1000
monte_carlo_seed = 42

solar_variability = 0.15
load_variability = 0.08
price_variability = 0.25
price_shift_variability = 0.01

# 
# OPTIMIZATION ASSUMPTIONS
# 

# Deterministic dispatch:
# - optimization.py uses a perfect-foresight 
# linear program.
# - Load, solar availability, and 
# electricity prices are known over the
#   optimization horizon.
# - This provides a deterministic 
# planning benchmark.

# Monte Carlo dispatch:
# - monte_carlo.py perturbs load, solar, 
# and price using seeded uncertainty.
# - Every scenario is independently 
# re-optimized using the Pyomo dispatch
#   model in optimization.py.
# - Monte Carlo therefore measures 
# the distribution of optimized operating
#   outcomes for a fixed infrastructure design.

# Capacity co-optimization:
# - optimization.py can jointly 
# optimize continuous solar capacity,
#   battery energy capacity, battery 
# power capacity, and dispatch.
# - Capital costs are annualized 
# using a capital recovery factor.
# - Short representative operating 
# horizons are annualized using the
#   model's operating-cost scaling convention.

# Two-stage stochastic optimization:
# - stochastic.py chooses shared 
# first-stage solar and battery capacities.
# - Dispatch decisions are 
# scenario-specific second-stage decisions.
# - Scenario operating costs are 
# probability weighted.
# - Optional CVaR risk aversion 
# penalizes high-cost tail outcomes.

#
# Stochastic optimization 
#
# Default confidence level for CVaR risk aversion.
cvar_confidence_level = 0.95

# Risk-neutral optimization uses zero 
# risk aversion.
# Postive values place increasing weight 
# on high-cost tail outcomes.
default_risk_aversion = 0.0

# Values used when constructing a 
# risk-averse stochastic optimization model for testing.
risk_aversion_values = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]


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


# 
# ANALYSIS AND PLOT ASSUMPTIONS
#
# Legacy reporting/design-search setting.
# Current capacity optimization 
# chooses solar capacity directly.
design_comparison_scenarios = 100
legacy_analysis_solar_penetration = 0.15

# future task is still assumption sourcing 
# and documentation.