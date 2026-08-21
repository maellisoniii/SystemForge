from dataclasses import dataclass
from typing import Any
from pyomo.common.errors import ApplicationError 

import numpy as np
import pyomo.environ as pyo
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack

@dataclass
class DispatchResult:
    solver_status: str
    termination_condition: str

    objective_cost: float
    grid_cost: float 
    degradation_cost: float

    grid_import: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    curtailment: np.ndarray
    solar_used: np.ndarray
    soc: np.ndarray

    balance_residual: np.ndarray
    soc_residual: np.ndarray

    assumptions: dict[str, Any]
    metadata: dict[str, Any]

@dataclass
class CapacityResult:
    solver_status: str
    termination_condition: str
    objective_cost: float

    solar_capacity: float
    battery_capacity: float
    battery_power: float

    capital_cost: float
    grid_cost: float
    degradation_cost: float

    grid_import: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    solar_used: np.ndarray
    curtailment: np.ndarray
    soc: np.ndarray

    assumptions: dict[str, Any]
    metadata: dict[str, Any]

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
        battery_discharge_efficiency: float,
        initial_soc: float = 0.0,
        degradation_rate: float = 0.0,
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

    if not (0 <= initial_soc <= battery_capacity):
        raise ValueError("Initial state of charge must be between 0 and battery capacity.")
    if degradation_rate < 0:
        raise ValueError("Degradation rate must be non-negative.")
    # Horizon length
    number_of_hours = len(load)

    # Sets
    model.T = pyo.RangeSet(0, number_of_hours - 1)
    # SOC requires one additional timestep for the end-of-horizon state of charge
    model.T_SOC = pyo.RangeSet(0, number_of_hours)

    # Time-varying parameters
    model.demand = pyo.Param(
        model.T,
        initialize={t: float(load[t]) for t in range(number_of_hours)},
    )

    model.solar_profile = pyo.Param(
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
    model.degradation_rate = pyo.Param(initialize=float(degradation_rate))
    # Decision variables
    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.curtailment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.solar_used = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T_SOC, domain=pyo.NonNegativeReals)
    model.initial_soc = pyo.Param(initialize=float(initial_soc), mutable=True)
    # Energy balance constraint
    def energy_balance_rule(model, t):
        return (
            model.grid_import[t]
            + model.solar_used[t]
            + model.discharge[t]
            == model.demand[t] + model.charge[t]
        )
    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)
    # Battery state of charge dynamics
    def soc_dynamics_rule(model, t):
        return (
            model.soc[t + 1]
            == model.soc[t]
            + model.battery_charge_efficiency * model.charge[t]
            - (1 / model.battery_discharge_efficiency) * model.discharge[t]
        )
    model.soc_dynamics = pyo.Constraint(
        model.T, rule=soc_dynamics_rule)
    model.initial_soc_constraint = pyo.Constraint(
        expr=model.soc[0] == model.initial_soc)
    # Battery capacity constraint
    model.terminal_soc_constraint = pyo.Constraint(
        expr=model.soc[number_of_hours] == model.soc[0])

    def soc_capacity_rule(model, t):
        return model.soc[t] <= model.battery_capacity

    model.soc_capacity_constraint = pyo.Constraint(
        model.T_SOC,
        rule=soc_capacity_rule,
    )
    # Battery power constraints
    def charge_power_constraint_rule(model, t):
        return model.charge[t] <= model.battery_power


    def discharge_power_constraint_rule(model, t):
        return model.discharge[t] <= model.battery_power


    model.charge_power_constraint = pyo.Constraint(
    model.T,
    rule=charge_power_constraint_rule
    )

    model.discharge_power_constraint = pyo.Constraint(
    model.T,
    rule=discharge_power_constraint_rule
    )
    # Curtailment constraint
    def solar_allocation_rule(model, t):
        return model.solar_used[t] + model.curtailment[t] == model.solar_profile[t]
    model.solar_allocation_constraint = pyo.Constraint(
        model.T, 
        rule=solar_allocation_rule
    )
    # Objective function: Minimize total cost of grid import
    def objective_rule(model):
        grid_cost = sum(model.hourly_price[t] * model.grid_import[t] for t in model.T)
        degradation_cost = sum(
            model.degradation_rate * (model.charge[t] + model.discharge[t]) 
            for t in model.T
        )
        return grid_cost + degradation_cost
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return model
# Investment (Front End)
def capital_recovery_factor(
        discount_rate: float,
        lifetime_years: int, 
) -> float:
    """ 
    Convert an upfront capital cost into an equivalent annual payment factor. 

    Parameters
    --
    discount_rate:
        Annual discount rate as a decimal.

    lifetime_years:
        Economic lifetime of the asset in years.

    """
    ## Come back to this -- Mark
    if discount_rate < 0: 
        raise ValueError(
            "Discount rate cannot be negative."
        )
    if lifetime_years <= 0:
        raise ValueError(
            "Lifetime must be greater than zero."
        )
    if discount_rate == 0:
        return 1.0 / lifetime_years
    growth = (1 + discount_rate) ** lifetime_years
    return (
        discount_rate * growth
        / (growth -1)
    )
def build_capacity_model(
        load:np.ndarray,
        solar_capacity_factor: np.ndarray,
        hourly_price: np.ndarray,
        battery_charge_efficiency: float, 
        battery_discharge_efficiency: float, 
        solar_capacity_cost: float,
        battery_energy_cost: float, 
        battery_power_cost: float, 
        discount_rate: float,
        solar_lifetime_years: int,
        battery_lifetime_years: int,
        degradation_rate: float = 0.0
):
    """ Build a continuous capacity + dispatch co-optimization model
    The model jointly chooses:
    - solar capacity
    - battery energy capacity
    - hourly grid import
    - solar use and curtailment
    - battery charge/discharge
    - battery state of charge 
    All capacity decisions are continuous"""
    # Normalize and validate the capacity model
    load = np.asarray(load, dtype=float)
    solar_capacity_factor = np.asarray(
        solar_capacity_factor,
        dtype=float,
    )
    hourly_price = np.asarray(
        hourly_price,
        dtype=float,
    )

    if not (
        len(load)
        == len(solar_capacity_factor)
        == len(hourly_price)
    ):
        raise ValueError(
            "Load, solar capacity factor, and hourly price "
            "must have the same length."
        )

    if len(load) == 0:
        raise ValueError(
            "Capacity optimization horizon cannot be empty."
        )

    if not 0 < battery_charge_efficiency <= 1:
        raise ValueError(
            "Battery charge efficiency must be > 0 and <= 1."
        )

    if not 0 < battery_discharge_efficiency <= 1:
        raise ValueError(
            "Battery discharge efficiency must be > 0 and <= 1."
        )

    if np.any(solar_capacity_factor < 0):
        raise ValueError(
            "Solar capacity factor cannot be negative."
        )

    if np.any(solar_capacity_factor > 1):
        raise ValueError(
            "Solar capacity factor cannot exceed 1."
        )

    if solar_capacity_cost < 0:
        raise ValueError(
            "Solar capacity cost cannot be negative."
        )

    if battery_energy_cost < 0:
        raise ValueError(
            "Battery energy cost cannot be negative."
        )

    if battery_power_cost < 0:
        raise ValueError(
            "Battery power cost cannot be negative."
        )

    if degradation_rate < 0:
        raise ValueError(
            "Degradation rate cannot be negative."
        )
    if discount_rate < 0:
        raise ValueError(
            "Discount rate cannot be negative."
        )
    if solar_lifetime_years <= 0:
        raise ValueError(
            "Solar lifetime must be greater than zero"
        )
    if battery_lifetime_years <= 0:
        raise ValueError(
            "Battery lifetime must be greater than zero."
        )
    # New model and sets
    number_of_hours = len(load)

    model = pyo.ConcreteModel(
        name="Capacity_Cooptimization"
    )

    model.T = pyo.RangeSet(
        0,
        number_of_hours - 1
    )

    model.T_SOC = pyo.RangeSet(
        0,
        number_of_hours
    )
    # Annualized costs 
    solar_crf = capital_recovery_factor(
        discount_rate, 
        solar_lifetime_years
    )
    battery_crf = capital_recovery_factor(
        discount_rate,
        battery_lifetime_years
    )
    annualized_solar_capacity_cost = (
        solar_capacity_cost * solar_crf
    )
    annualized_battery_energy_cost = (
        battery_energy_cost * battery_crf
    )
    annualized_battery_power_cost = (
        battery_power_cost * battery_crf
    )
    # Assumption
    operating_cost_scale = (
        8760.0 / number_of_hours
    )
    # Parameters 
    model.demand = pyo.Param(
        model.T,
        initialize={
            t: float(load[t])
            for t in range(number_of_hours)
        },
    )
    model.solar_capacity_factor = pyo.Param(
        model.T,
        initialize={
            t: float(solar_capacity_factor[t])
            for t in range(number_of_hours)
        },
    )

    model.annualized_solar_capacity_cost = pyo.Param(
        initialize=float(
            annualized_solar_capacity_cost
        )
    )

    model.hourly_price = pyo.Param(
        model.T,
        initialize={
            t: float(hourly_price[t])
            for t in range(number_of_hours)
        },
    )

    model.battery_charge_efficiency = pyo.Param(
        initialize=float(
            battery_charge_efficiency
        )
    )

    model.battery_discharge_efficiency = pyo.Param(
        initialize=float(
            battery_discharge_efficiency
        )
    )

    model.solar_capacity_cost = pyo.Param(
        initialize=float(solar_capacity_cost)
    )

    model.annualized_battery_energy_cost = pyo.Param(
        initialize=float(
            annualized_battery_energy_cost
        )
    )

    model.annualized_battery_power_cost = pyo.Param(
        initialize=float(
            annualized_battery_power_cost
        )
    )

    model.degradation_rate = pyo.Param(
        initialize=float(degradation_rate)
    )

    model.operating_cost_scale = pyo.Param(
        initialize=float(
            operating_cost_scale
        )
    )
    # Decision (design) variables 
    # First-stage infrastructure design decisions
    model.solar_capacity = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    model.battery_capacity = pyo.Var(
        domain=pyo.NonNegativeReals
    )

    model.battery_power = pyo.Var(
        domain=pyo.NonNegativeReals
    )
    # Operational variables 
    # Operational decisions
    model.grid_import = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals
    )

    model.charge = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals
    )

    model.discharge = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals
    )

    model.solar_used = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals
    )

    model.curtailment = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals
    )

    model.soc = pyo.Var(
        model.T_SOC,
        domain=pyo.NonNegativeReals
    )
    # Physical Limits 
    def energy_balance_rule(model, t):
        return (
            model.grid_import[t]
            + model.solar_used[t]
            + model.discharge[t]
            ==
            model.demand[t]
            + model.charge[t]
        )

    model.energy_balance = pyo.Constraint(
        model.T,
        rule=energy_balance_rule
    )
    def soc_dynamics_rule(model, t):
        return (
            model.soc[t + 1]
            ==
            model.soc[t]
            + model.battery_charge_efficiency
            * model.charge[t]
            - model.discharge[t]
            / model.battery_discharge_efficiency
        )

    model.soc_dynamics = pyo.Constraint(
        model.T,
        rule=soc_dynamics_rule
    )
    def soc_capacity_rule(model, t):
        return (
            model.soc[t]
            <= model.battery_capacity
        )

    model.soc_capacity_constraint = pyo.Constraint(
        model.T_SOC,
        rule=soc_capacity_rule
    )
    def charge_power_rule(model, t):
        return (
            model.charge[t]
            <= model.battery_power
        )

    model.charge_power_constraint = pyo.Constraint(
        model.T,
        rule=charge_power_rule
    )
    def discharge_power_rule(model, t):
        return (
            model.discharge[t]
            <= model.battery_power
        )

    model.discharge_power_constraint = pyo.Constraint(
        model.T,
        rule=discharge_power_rule
    )
    # Capacity dependent solar
    def solar_allocation_rule(model, t):
        return (
            model.solar_used[t]
            + model.curtailment[t]
            ==
            model.solar_capacity
            * model.solar_capacity_factor[t]
        )

    model.solar_allocation = pyo.Constraint(
        model.T,
        rule=solar_allocation_rule
    )
    # Initial and terminal SOC
    model.initial_soc_constraint = pyo.Constraint(
        expr=model.soc[0] == 0.0
    )

    model.terminal_soc_constraint = pyo.Constraint(
        expr=model.soc[number_of_hours]
        == model.soc[0]
    )

    def objective_rule(model):
        capital_cost = (
            model.annualized_solar_capacity_cost
            * model.solar_capacity
            + model.annualized_battery_energy_cost
            * model.battery_capacity
            + model.annualized_battery_power_cost
            * model.battery_power
        )
        grid_cost = model.operating_cost_scale * sum(
            model.hourly_price[t]
            * model.grid_import[t]
            for t in model.T
        )
        degradation_cost = (
            model.operating_cost_scale
            * sum(
                model.degradation_rate
                * (
                    model.charge[t]
                    + model.discharge[t]
                )
                for t in model.T
            )
        )
        return (
            capital_cost
            + grid_cost
            + degradation_cost
        )
        model.objective = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize,
    )
        return model

def solve_dispatch(
        model: pyo.ConcreteModel,
        solver_name: str = "highs",
):
    """
    Solve a previously contructed SystemForge dispatch model.
    Parameters 
    ----------
    model: 
        A Pyomo ConcreteModel returned by build_dispatch_model.
        solver_name:
            Name of the optimization solver to use. Defaults to "highs" (HiGHS solver).
            Returns
            Pyomo solver results object containing the solution and solver information.
            """
    try:
        solver = pyo.SolverFactory(solver_name)

        if not solver.available():
            raise RuntimeError(f"Solver '{solver_name}' is not available. Please ensure it is installed and accessible."
                               )
    except (ApplicationError, ValueError) as exc:
        raise RuntimeError(
            f"Solver '{solver_name}' is not available."
        ) from exc
    results = solver.solve(model)

    solver_status = results.solver.status
    termination_condition = (results.solver.termination_condition
    )
    if solver_status != pyo.SolverStatus.ok:
        raise RuntimeError("Dispatch optimization failed with solver status: "
            f"Solver failed with status: {solver_status}")
    if termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("Dispatch optimization did not terminate optimally. "
            f"Termination condition: {termination_condition}")
    return results


def extract_dispatch_results(
        model: pyo.ConcreteModel,
        results: pyo.SolverResults,
) -> DispatchResult:
    """
    Extracts the dispatch results from a solved Pyomo model and returns a DispatchResult dataclass.
    Parameters
    ----------
    model: 
        A Pyomo ConcreteModel returned by build_dispatch_model.
        """
    time_steps = list(model.T)
    soc_time_steps = list(model.T_SOC)

    grid_import = np.array([pyo.value(model.grid_import[t]) for t in time_steps])
    charge = np.array([pyo.value(model.charge[t]) for t in time_steps])
    discharge = np.array([pyo.value(model.discharge[t]) for t in time_steps])
    curtailment = np.array([pyo.value(model.curtailment[t]) for t in time_steps])
    solar_used = np.array([pyo.value(model.solar_used[t]) for t in time_steps])
    soc = np.array([pyo.value(model.soc[t]) for t in soc_time_steps])

    price = np.array([pyo.value(model.hourly_price[t]) for t in time_steps])
    degradation_rate = pyo.value(model.degradation_rate)
    grid_cost = np.sum(price * grid_import)
    degradation_cost = np.sum(degradation_rate * (charge + discharge))
    objective_cost = pyo.value(model.objective)

    demand = np.array([pyo.value(model.demand[t]) for t in time_steps])
    balance_residual = grid_import + solar_used + discharge - demand - charge

    charge_efficiency = pyo.value(model.battery_charge_efficiency)
    discharge_efficiency = pyo.value(model.battery_discharge_efficiency)

    soc_residual = (
        soc[1:]
        - soc[:-1]
        - charge_efficiency * charge
        + discharge / discharge_efficiency
    )
    assumptions = {
        "battery_capacity": pyo.value(model.battery_capacity),
        "battery_power": pyo.value(model.battery_power),
        "battery_charge_efficiency": charge_efficiency,
        "battery_discharge_efficiency": discharge_efficiency,
        "degradation_rate": degradation_rate,
        "initial_soc": pyo.value(model.initial_soc),
    }
    metadata = {
        "number_of_hours": len(time_steps),
        "model_name": model.name,
    }
    return DispatchResult(
        solver_status=str(results.solver.status),
        termination_condition=str(results.solver.termination_condition),
        objective_cost=float(objective_cost),
        grid_cost=float(grid_cost),
        degradation_cost=float(degradation_cost),
        grid_import=grid_import,
        charge=charge,
        discharge=discharge,
        curtailment=curtailment,
        solar_used=solar_used,
        soc=soc,
        balance_residual=balance_residual,
        soc_residual=soc_residual,
        assumptions=assumptions,
        metadata=metadata,
    )

def extract_capacity_results(
    model: pyo.ConcreteModel,
    results: pyo.SolverResults,
) -> CapacityResult:

    time_steps = list(model.T)
    soc_steps = list(model.T_SOC)

    grid_import = np.array([
        pyo.value(model.grid_import[t])
        for t in time_steps
    ])

    charge = np.array([
        pyo.value(model.charge[t])
        for t in time_steps
    ])

    discharge = np.array([
        pyo.value(model.discharge[t])
        for t in time_steps
    ])

    solar_used = np.array([
        pyo.value(model.solar_used[t])
        for t in time_steps
    ])

    curtailment = np.array([
        pyo.value(model.curtailment[t])
        for t in time_steps
    ])

    soc = np.array([
        pyo.value(model.soc[t])
        for t in soc_steps
    ])

    solar_capacity = pyo.value(
        model.solar_capacity
    )

    battery_capacity = pyo.value(
        model.battery_capacity
    )

    battery_power = pyo.value(
        model.battery_power
    )
    capital_cost = (
        pyo.value(
            model.annualized_solar_capacity_cost
        )
        * solar_capacity
        + pyo.value(
            model.annualized_battery_energy_cost
        )
        * battery_capacity
        + pyo.value(
            model.annualized_battery_power_cost
        )
        * battery_power
    )

    grid_cost = (
        pyo.value(model.operating_cost_scale)
        * sum(
            pyo.value(model.hourly_price[t])
            * grid_import[t]
            for t in time_steps
        )
    )

    degradation_cost = (
        pyo.value(model.operating_cost_scale)
        * pyo.value(model.degradation_rate)
        * np.sum(
            charge + discharge
        )
    )

    objective_cost = pyo.value(
        model.objective
    )
    return CapacityResult(
        solver_status=str(
            results.solver.status
        ),
        termination_condition=str(
            results.solver.termination_condition
        ),
        objective_cost=float(
            objective_cost
        ),
        solar_capacity=float(
            solar_capacity
        ),
        battery_capacity=float(
            battery_capacity
        ),
        battery_power=float(
            battery_power
        ),
        capital_cost=float(
            capital_cost
        ),
        grid_cost=float(
            grid_cost
        ),
        degradation_cost=float(
            degradation_cost
        ),
        grid_import=grid_import,
        charge=charge,
        discharge=discharge,
        solar_used=solar_used,
        curtailment=curtailment,
        soc=soc,
        assumptions={
            "operating_cost_scale":
                pyo.value(
                    model.operating_cost_scale
                ),
            "charge_efficiency":
                pyo.value(
                    model.battery_charge_efficiency
                ),
            "discharge_efficiency":
                pyo.value(
                    model.battery_discharge_efficiency
                ),
        },
        metadata={
            "number_of_hours":
                len(time_steps),
            "model_name":
                model.name,
        },
    )

def validate_dispatch_result(
    result: DispatchResult,
    tolerance: float = 1e-8,
) -> None:
    """
    Validate core physical and numerical invariants of a solved dispatch case.

    Raises
    ------
    AssertionError
        If any required invariant exceeds the specified tolerance.
    """

    max_balance_residual = np.max(
        np.abs(result.balance_residual)
    )

    max_soc_residual = np.max(
        np.abs(result.soc_residual)
    )

    if max_balance_residual > tolerance:
        raise AssertionError(
            "Energy balance validation failed: "
            f"maximum residual = {max_balance_residual:.3e}, "
            f"tolerance = {tolerance:.3e}"
        )

    if max_soc_residual > tolerance:
        raise AssertionError(
            "SOC dynamics validation failed: "
            f"maximum residual = {max_soc_residual:.3e}, "
            f"tolerance = {tolerance:.3e}"
        )

    if np.any(result.grid_import < -tolerance):
        raise AssertionError(
            "Grid import contains negative values."
        )

    if np.any(result.charge < -tolerance):
        raise AssertionError(
            "Battery charge contains negative values."
        )

    if np.any(result.discharge < -tolerance):
        raise AssertionError(
            "Battery discharge contains negative values."
        )

    if np.any(result.curtailment < -tolerance):
        raise AssertionError(
            "Curtailment contains negative values."
        )

    if np.any(result.solar_used < -tolerance):
        raise AssertionError(
            "Solar used contains negative values."
        )

    battery_capacity = result.assumptions[
        "battery_capacity"
    ]

    battery_power = result.assumptions[
        "battery_power"
    ]

    if np.any(result.soc < -tolerance):
        raise AssertionError(
            "State of charge falls below zero."
        )

    if np.any(
        result.soc > battery_capacity + tolerance
    ):
        raise AssertionError(
            "State of charge exceeds battery capacity."
        )

    if np.any(
        result.charge > battery_power + tolerance
    ):
        raise AssertionError(
            "Charge exceeds battery power limit."
        )

    if np.any(
        result.discharge > battery_power + tolerance
    ):
        raise AssertionError(
            "Discharge exceeds battery power limit."
        )

