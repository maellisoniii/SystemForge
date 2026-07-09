from pathlib import Path


# ============================
# DATA PROFILE ASSUMPTIONS
# ============================

project_dir = Path(__file__).parent
system_scale = 1e-5

# Germany/Austria remains the default scenario, but it is now just a scenario.
# To test another building, market, tariff, or weather file, change this mapping
# instead of changing the simulation, optimization, or reporting engines.
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


# ============================
# DESIGN ASSUMPTIONS
# ============================

# Battery power is modeled as a fraction of battery energy capacity.
# Example: 0.25 means the battery can charge or discharge 25% of its capacity
# per hour, which is equivalent to a four-hour battery.
#
# This assumption should be used consistently in both:
# - perfect-foresight deterministic dispatch in optimization.py
# - threshold-style Monte Carlo dispatch in monte_carlo.py
battery_power_ratio = 0.25

### I should eventually test a wider range of storage durations and solar
### penetrations, but these are the ones that have been used in prior work.
storage_durations_hours = [0, 0.25, 0.5, 1.0, 2.0]
solar_penetrations = [0.05, 0.10, 0.15, 0.20, 0.25]

### These need to be documented and justified. They are not based on any
### particular market or technology assumptions.
battery_cost_per_kwh = 300
battery_lifetime_years = 10

# Financial assumption used to annualize capital costs with a capital recovery
# factor instead of simple cost / lifetime division.
#
# This is a first-pass project finance assumption, not a market-specific WACC.
# Changing this number can materially change the optimal design because it
# changes how expensive long-lived assets look on an annual basis.
discount_rate = 0.05

# This preserves the prior cost scale. The annualized solar cost per penetration
# is derived after the active profile is loaded because reference penetration is
# profile-specific.
solar_cost_reference = 8000
solar_lifetime_years = 20

# Battery efficiency assumptions.
# These are kept separate so the model can later distinguish energy lost while
# charging from energy lost while discharging.
#
# Current planned interpretation:
# - charge efficiency applies when energy enters the battery
# - discharge efficiency applies when stored energy serves load or offsets grid
# - round-trip efficiency is approximately charge_efficiency * discharge_efficiency
#
# These assumptions are wired into the deterministic optimization engine.
# The next refactor step should wire them into monte_carlo.py as well.
battery_charge_efficiency = 0.95
battery_discharge_efficiency = 0.95
battery_round_trip_efficiency = (
    battery_charge_efficiency * battery_discharge_efficiency
)


# ============================
# MONTE CARLO ASSUMPTIONS
# ============================

num_monte_carlo_scenarios = 1000
monte_carlo_seed = 42

solar_variability = 0.15
load_variability = 0.08
price_variability = 0.25
price_shift_variability = 0.01

# Dispatch-method documentation:
#
# Deterministic dispatch:
# - optimization.py uses a perfect-foresight linear program.
# - It assumes the model knows the full price, load, and solar profile for the
#   analysis year before dispatch decisions are made.
# - This is useful as a planning benchmark because it estimates the best
#   possible dispatch under the selected assumptions.
#
# Monte Carlo dispatch:
# - monte_carlo.py currently uses a faster threshold-based dispatch rule.
# - It does not solve a full optimization problem for every scenario.
# - This is useful for scenario screening, but it should be interpreted as a
#   heuristic operating policy rather than the theoretical optimum.
#
# The Monte Carlo dispatch uses battery_power_ratio to enforce max charge,
# max discharge, and max grid-charge limits. It also uses the battery efficiency
# assumptions above to track charge/discharge losses in state of charge.

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


# ============================
# ANALYSIS AND PLOT ASSUMPTIONS
# ============================

design_comparison_scenarios = 100
analysis_solar_penetration = 0.15
