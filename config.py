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

battery_power_ratio = 0.25  # 0.25 means a four-hour battery power rating.

### I should eventually test a wider range of storage durations and solar penetrations, but these are the ones that have been used in prior work.
storage_durations_hours = [0, 0.25, 0.5, 1.0, 2.0]
solar_penetrations = [0.05, 0.10, 0.15, 0.20, 0.25]

### These need to be documented and justified. They are not based on any particular market or technology assumptions.

battery_cost_per_kwh = 300
battery_lifetime_years = 10

# This preserves the prior cost scale. The annualized solar cost per penetration
# is derived after the active profile is loaded because reference penetration is
# profile-specific.
solar_cost_reference = 8000
solar_lifetime_years = 20


# ============================
# MONTE CARLO ASSUMPTIONS
# ============================

num_monte_carlo_scenarios = 1000
monte_carlo_seed = 42

solar_variability = 0.15
load_variability = 0.08
price_variability = 0.25
price_shift_variability = 0.01

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
