import numpy as np

from SystemForge.optimization import optimize_battery_dispatch


def solar_scale_from_penetration(profile, solar_penetration):
    """Scale the recorded solar profile to the requested annual penetration."""
    return solar_penetration / profile.reference_solar_penetration


def battery_capacity_from_duration(profile, storage_duration_hours):
    """Convert storage duration to capacity using average hourly load."""
    return storage_duration_hours * profile.average_load


class SystemSimulator:
    """Runs deterministic simulations for one standardized energy profile."""

    def __init__(
            self,
            profile,
            battery_power_ratio,
            battery_charge_efficiency,
            battery_discharge_efficiency):
        self.profile = profile
        self.battery_power_ratio = battery_power_ratio
        self.battery_charge_efficiency = battery_charge_efficiency
        self.battery_discharge_efficiency = battery_discharge_efficiency
        self.simulation_cache = {}

    def scaled_solar(self, solar_penetration):
        return (
            self.profile.solar_kwh
            * solar_scale_from_penetration(self.profile, solar_penetration)
        )

    def battery_capacity(self, storage_duration_hours):
        return battery_capacity_from_duration(self.profile, storage_duration_hours)

    def simulate_system(
            self,
            storage_duration_hours,
            solar_penetration,
            return_hourly_data=False):
        battery_capacity = self.battery_capacity(storage_duration_hours)
        scaled_solar = self.scaled_solar(solar_penetration)

        load = self.profile.load_kwh
        hourly_price = self.profile.price_per_kwh

        if (len(load) != len(scaled_solar)
                or len(load) != len(hourly_price)):
            raise ValueError("load, solar, and price arrays must have identical lengths")

        cache_key = (storage_duration_hours, solar_penetration)

        if not return_hourly_data and cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]

        grid_import, solar_curtailed, battery_soc = optimize_battery_dispatch(
            load,
            hourly_price,
            battery_capacity,
            scaled_solar,
            self.battery_power_ratio,
            self.battery_charge_efficiency,
            self.battery_discharge_efficiency,
        )
        grid_cost = np.dot(grid_import, hourly_price)

        # This baseline is used to estimate the value of battery dispatch.
        # It is the grid-import cost for the same load and solar design if the
        # battery did not operate. The resulting difference is therefore battery
        # dispatch savings, not total project savings.
        grid_cost_without_battery_dispatch = np.dot(
            np.maximum(load - scaled_solar, 0),
            hourly_price,
        )
        annual_battery_dispatch_savings = (
            grid_cost_without_battery_dispatch - grid_cost
        )

        results = (
            np.sum(grid_import),
            np.sum(solar_curtailed),
            grid_cost,
            annual_battery_dispatch_savings,
            np.max(battery_soc),
        )

        if not return_hourly_data:
            self.simulation_cache[cache_key] = results
            return results

        return results + (grid_import, solar_curtailed)


def analyze_load_solar_price(profile, simulator, best_design):
    """Calculate load, solar, curtailment, and price summary metrics."""
    (total_grid_import, total_solar_curtailed, annual_grid_cost,
     annual_battery_dispatch_savings, max_soc, hourly_grid_import,
     hourly_solar_curtailed) = simulator.simulate_system(
        best_design["storage_duration_hours"],
        best_design["solar_penetration"],
        return_hourly_data=True,
    )

    scaled_solar = simulator.scaled_solar(best_design["solar_penetration"])
    total_load = sum(profile.load_kwh)
    total_solar = sum(scaled_solar)
    total_solar_used = total_solar - total_solar_curtailed

    if total_load <= 0:
        raise ValueError("total load must be greater than zero")
    if total_solar <= 0:
        solar_penetration = 0
        solar_utilization = 0
    else:
        solar_penetration = total_solar / total_load
        solar_utilization = total_solar_used / total_solar

    grid_dependence = total_grid_import / total_load

    min_price_index = np.argmin(profile.price_per_kwh)
    max_price_index = np.argmax(profile.price_per_kwh)
    curtailment_indices = np.where(hourly_solar_curtailed > 0)[0]

    min_curtailment_price_index = None
    max_curtailment_price_index = None

    if len(curtailment_indices) > 0:
        min_curtailment_price_index = curtailment_indices[
            np.argmin(profile.price_per_kwh[curtailment_indices])
        ]
        max_curtailment_price_index = curtailment_indices[
            np.argmax(profile.price_per_kwh[curtailment_indices])
        ]

    return {
        "total_grid_import": total_grid_import,
        "total_solar_curtailed": total_solar_curtailed,
        "annual_grid_cost": annual_grid_cost,
        "annual_battery_dispatch_savings": annual_battery_dispatch_savings,
        "max_soc": max_soc,
        "hourly_grid_import": hourly_grid_import,
        "hourly_solar_curtailed": hourly_solar_curtailed,
        "scaled_solar": scaled_solar,
        "total_load": total_load,
        "total_solar": total_solar,
        "total_solar_used": total_solar_used,
        "solar_penetration": solar_penetration,
        "solar_utilization": solar_utilization,
        "grid_dependence": grid_dependence,
        "min_price_index": min_price_index,
        "max_price_index": max_price_index,
        "curtailment_indices": curtailment_indices,
        "min_curtailment_price_index": min_curtailment_price_index,
        "max_curtailment_price_index": max_curtailment_price_index,
    }
