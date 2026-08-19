from xml.parsers.expat import model

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
    def charge_power_constraint_rule(model, t):
        return model.charge[t] <= model.battery_power
    model.charge_power_constraint = pyo.Constraint(
        model.T, rule=charge_power_constraint_rule)
    # Battery power constraints
    def discharge_power_constraint_rule(model, t):
        return model.discharge[t] <= model.battery_power
    model.battery_power_constraint_charge = pyo.Constraint(
        model.T, rule=lambda model, t: model.charge[t] <= model.battery_power
    )
    model.battery_power_constraint_discharge = pyo.Constraint(
        model.T, rule=lambda model, t: model.discharge[t] <= model.battery_power
    )
    # Curtailment constraint
    def solar_allocation_rule(model, t):
        return model.solar_used[t] + model.curtailment[t] <= model.solar_profile[t]
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
    solver = pyo.SolverFactory(solver_name)

    if not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' is not available. Please ensure it is installed and accessible.")
    results = solver.solve(model, tee=True)

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

if __name__ == "__main__":

    test_load = np.array([
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ])

    test_solar = np.array([
        0.0,
        0.0,
        15.0,
        15.0,
        0.0,
        0.0,
    ])

    test_price = np.array([
        0.10,
        0.10,
        0.15,
        0.20,
        0.50,
        0.50,
    ])

    model = build_dispatch_model(
        load=test_load,
        solar=test_solar,
        hourly_price=test_price,
        battery_capacity=10.0,
        battery_power=5.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        initial_soc=0.0,
        degradation_rate=0.0,
    )

    results = solve_dispatch(model)

    print(
        "Solver status:",
        results.solver.status
    )

    print(
        "Termination:",
        results.solver.termination_condition
    )

    print(
        "Objective value:",
        pyo.value(model.objective)
    )
print("\nHourly dispatch")
print("-" * 70)

for t in model.T:
    print(
        f"Hour {t}: "
        f"grid={pyo.value(model.grid_import[t]):.3f}, "
        f"solar_used={pyo.value(model.solar_used[t]):.3f}, "
        f"charge={pyo.value(model.charge[t]):.3f}, "
        f"discharge={pyo.value(model.discharge[t]):.3f}, "
        f"curtailment={pyo.value(model.curtailment[t]):.3f}, "
        f"soc={pyo.value(model.soc[t]):.3f}"
    )

print(
    f"Final SOC: "
    f"{pyo.value(model.soc[len(test_load)]):.3f}"
)



