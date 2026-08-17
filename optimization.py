from pyexpat import model

import numpy as np
import pyomo.environ as pyo
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack

# Legacy SciPy implementation 
# Used as a deterministic validation benchmark during migration

def optimize_battery_dispatch_scipy(
        load,
        hourly_price,
        battery_capacity,
        scaled_solar,
        battery_power_ratio,
        battery_charge_efficiency,
        battery_discharge_efficiency):
    """Solve the hourly battery schedule that minimizes annual grid cost.

    The optimizer has perfect knowledge of the price, load, and solar profile.
    It chooses grid import, charging, discharging, curtailment, and state of
    charge subject to energy-balance, capacity, and power constraints.

    Dispatch model:
    - This is a perfect-foresight planning benchmark.
    - The optimizer sees the full load, solar, and price profile before making
      hourly charge/discharge decisions.
    - Energy balance is modeled on the grid/load side:
      grid import + discharge = net load + charge + curtailment.
    - Battery state of charge is modeled inside the battery:
      SOC_next = SOC + charge_efficiency * charge
                 - discharge / discharge_efficiency.
    - Round-trip efficiency is approximately:
      charge_efficiency * discharge_efficiency.

    Efficiencies are required inputs so the deterministic optimization cannot
    accidentally fall back to an ideal battery assumption.
    """
    number_of_hours = len(load)

    if not 0 < battery_charge_efficiency <= 1:
        raise ValueError("battery_charge_efficiency must be greater than 0 and at most 1")
    if not 0 < battery_discharge_efficiency <= 1:
        raise ValueError("battery_discharge_efficiency must be greater than 0 and at most 1")
    if battery_power_ratio < 0:
        raise ValueError("battery_power_ratio cannot be negative")

    # Net load is load after using same-hour solar directly.
    # Positive net load means the system needs energy from the grid or battery.
    # Negative net load means there is surplus solar that can charge the battery
    # or be curtailed.
    net_load = load - scaled_solar

    if battery_capacity == 0:
        return (
            np.maximum(net_load, 0),
            np.maximum(-net_load, 0),
            np.zeros(number_of_hours + 1),
        )

    identity = eye(number_of_hours, format="csr")
    zero_soc = csr_matrix((number_of_hours, number_of_hours + 1))
    zero_hourly = csr_matrix((number_of_hours, number_of_hours))
    state_of_charge_change = diags(
        [-np.ones(number_of_hours), np.ones(number_of_hours)],
        [0, 1],
        shape=(number_of_hours, number_of_hours + 1),
        format="csr",
    )

    # Variable order: grid import, charge, discharge, curtailment, state of charge.
    #
    # Energy balance uses charge and discharge as grid/load-side energy flows:
    # grid - charge + discharge - curtailment = load - solar
    energy_balance = hstack(
        [identity, -identity, identity, -identity, zero_soc],
        format="csr",
    )
    # Battery balance converts those external energy flows into internal stored
    # energy using charge and discharge efficiency assumptions.
    battery_balance = hstack(
        [
            zero_hourly,
            -battery_charge_efficiency * identity,
            (1 / battery_discharge_efficiency) * identity,
            zero_hourly,
            state_of_charge_change,
        ],
        format="csr",
    )
    equality_constraints = vstack([energy_balance, battery_balance], format="csr")
    equality_values = np.concatenate([net_load, np.zeros(number_of_hours)])

    number_of_variables = 5 * number_of_hours + 1
    grid_start = 0
    charge_start = number_of_hours
    discharge_start = 2 * number_of_hours
    curtailment_start = 3 * number_of_hours
    soc_start = 4 * number_of_hours

    objective = np.zeros(number_of_variables)
    objective[grid_start:charge_start] = hourly_price
    # A tiny throughput penalty removes unnecessary charge/discharge cycles.
    objective[charge_start:discharge_start] = 1e-9
    objective[discharge_start:curtailment_start] = 1e-9

    lower_bounds = np.zeros(number_of_variables)
    upper_bounds = np.full(number_of_variables, np.inf)
    max_power = battery_capacity * battery_power_ratio
    upper_bounds[charge_start:discharge_start] = max_power
    upper_bounds[discharge_start:curtailment_start] = max_power
    upper_bounds[curtailment_start:soc_start] = scaled_solar
    upper_bounds[soc_start:] = battery_capacity

    # Start and end the period empty so the optimizer cannot borrow energy from
    # outside the analysis period or leave a free stored-energy benefit behind.
    upper_bounds[soc_start] = 0
    upper_bounds[-1] = 0

    solution = linprog(
        objective,
        A_eq=equality_constraints,
        b_eq=equality_values,
        bounds=list(zip(lower_bounds, upper_bounds)),
        method="highs",
    )

    if not solution.success:
        raise RuntimeError(f"Battery dispatch optimization failed: {solution.message}")

    return (
        solution.x[grid_start:charge_start],
        solution.x[curtailment_start:soc_start],
        solution.x[soc_start:],
    )

# 
# Pyomo implementation
# 
def build_dispatch_model(
        load: np.ndarray,
        hourly_price: np.ndarray,
        battery_capacity: float,
        solar: np.ndarray,
        battery_power: float,
        battery_charge_efficiency: float,
        battery_discharge_efficiency: float
):
    # Time series inputs are converted to float arrays to ensure compatibility with Pyomo's data handling.
    load = np.asarray(load, dtype=float)
    solar = np.asarray(solar, dtype=float)
    hourly_price = np.asarray(hourly_price, dtype=float)

    model = pyo.ConcreteModel(name="Deterministic_Dispatch")

    # Input validation
    if not (len(load) == len(hourly_price) == len(solar)):
        raise ValueError("Load, solar, and hourly price arrays must have the same length.")

    if len(load) == 0:
        raise ValueError("Load, solar, and hourly price arrays must not be empty.")

    if battery_capacity < 0:
        raise ValueError("Battery capacity must be non-negative.")

    if battery_power < 0:
        raise ValueError("Battery power must be non-negative.")

    if not (0 < battery_charge_efficiency <= 1):
        raise ValueError("Battery charge efficiency must be > 0 and <= 1.")

    if not (0 < battery_discharge_efficiency <= 1):
        raise ValueError("Battery discharge efficiency must be > 0 and <= 1.")

    # Horizon length
    number_of_hours = len(load)

    # Sets
    model.T = pyo.RangeSet(0, number_of_hours - 1)
    # SOC requires one additional timestep for the end-of-horizon state of charge
    model.T_SOC = pyo.RangeSet(0, number_of_hours)

    # Time-varying parameters
    model.load = pyo.Param(
        model.T,
        initialize={t: float(load[t]) for t in range(number_of_hours)},
    )

    model.solar = pyo.Param(
        model.T,
        initialize={t: float(solar[t]) for t in range(number_of_hours)},
    )

    model.hourly_price = pyo.Param(
        model.T,
        initialize={t: float(hourly_price[t]) for t in range(number_of_hours)},
    )

    # Battery parameters
    model.battery_capacity = pyo.Param(initialize=float(battery_capacity))
    model.battery_power = pyo.Param(initialize=float(battery_power))
    model.battery_charge_efficiency = pyo.Param(initialize=float(battery_charge_efficiency))
    model.battery_discharge_efficiency = pyo.Param(initialize=float(battery_discharge_efficiency))

    return model



