import numpy as np
import pyomo.environ as pyo
from optimization import capital_recovery_factor, solve_dispatch


def build_two_stage_model(
    scenario_load: np.ndarray,
    scenario_solar_cf: np.ndarray,
    scenario_price: np.ndarray,
    scenario_probabilities: np.ndarray,
    battery_charge_efficiency: float = 0.95,
    battery_discharge_efficiency: float = 0.95,
    degradation_rate: float = 0.0,
    solar_capacity_cost: float = 0.0,
    battery_energy_cost: float = 0.0,
    battery_power_cost: float = 0.0,
    discount_rate: float = 0.0,
    solar_lifetime_years: int = 25,
    battery_lifetime_years: int = 15,
    confidence_level: float = 0.95,
    risk_aversion: float = 0.0,
):
    scenario_load = np.asarray(
        scenario_load,
        dtype=float,
    )
    scenario_solar_cf = np.asarray(
        scenario_solar_cf,
        dtype=float,
    )
    scenario_price = np.asarray(
        scenario_price,
        dtype=float,
    )
    scenario_probabilities = np.asarray(
        scenario_probabilities,
        dtype=float,
    )

    if scenario_load.ndim != 2:
        raise ValueError(
            "scenario_load must be a 2d array."
        )
    if scenario_solar_cf.shape != scenario_load.shape:
        raise ValueError(
            "scenario_solar_cf must match scenario_load shape."

        )
    if scenario_price.shape != scenario_load.shape:
        raise ValueError(
            "scenario_price must match scenario_load shape."
        )
    number_of_scenarios, number_of_hours = (
        scenario_load.shape
    )
    solar_crf = capital_recovery_factor(
        discount_rate,
        solar_lifetime_years,
    )
    battery_crf = capital_recovery_factor(
        discount_rate,
        battery_lifetime_years,
    )
    annualized_solar_capacity_cost = (
        solar_capacity_cost
        * solar_crf
    )
    annualized_battery_energy_cost = (
        battery_energy_cost
        * battery_crf
    )
    annualized_battery_power_cost = (
        battery_power_cost
        * battery_crf
    )
    operating_cost_scale = (
        8760.0 / number_of_hours
    )
    if len(scenario_probabilities) != number_of_scenarios:
        raise ValueError(
            "scenarios_probabilities must contain one" \
            "probability per scenario."
        )
    if np.any(scenario_probabilities < 0):
        raise ValueError(
            "scenario_probabilities cannot be negative."

        )
    probability_sum = np.sum(
        scenario_probabilities
    )
    if not np.isclose(
        probability_sum,
        1.0,
    ):
        raise ValueError(
            "scenario probabilities must sum to 1."
        )
    if np.any(scenario_load < 0):
        raise ValueError(
            "scenario_load cannot be negative"
        )
    if np.any(scenario_solar_cf > 1):
        raise ValueError(
            "scenario_solar capacity factors cannot exceed 1."

        )
    if np.any(scenario_solar_cf < 0):
        raise ValueError(
            "scenario solar capacity factors cannot be negative."

        )
    if not 0 < battery_charge_efficiency <= 1:
        raise ValueError(
            "battery charge efficiency must be in (0, 1)."
        )
    if not 0 < battery_discharge_efficiency <= 1:
        raise ValueError(
            "battery discharge efficiency must be in (0, 1)."
        )
    if degradation_rate < 0:
        raise ValueError(
            "degradation rate cannot be negative"
        )
    if solar_capacity_cost < 0:
        raise ValueError(
            "solar capacity cost cannot be negative. "
        )
    if battery_energy_cost < 0:
        raise ValueError(
            "battery energy cost cannot be negative."
        )
    if battery_power_cost < 0:
        raise ValueError(
            "battery power cost cannot be negative."
        )
    if discount_rate < 0:
        raise ValueError(
            "discount rate cannot be negative."
        )
    if solar_lifetime_years <= 0:
        raise ValueError(
            "solar_lifetime_years must be positive."
        )
    if battery_lifetime_years <= 0:
        raise ValueError(
            "battery lifetime years must be positive."
        )
    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence_level must be between 0 and 1."
        )
    if risk_aversion < 0:
        raise ValueError(
            "risk_aversion cannot be negative."
        )
    # Sets
    model = pyo.ConcreteModel()

    model.S = pyo.RangeSet(
        0,
        number_of_scenarios - 1,
    )
    model.T = pyo.RangeSet(
        0,
        number_of_hours - 1,
    )
    model.T_SOC = pyo.RangeSet(
        0,
        number_of_hours,
    )
    # Parameters
    model.demand = pyo.Param(
        model.S,
        model.T,
        initialize={
            (s,t): float(scenario_load[s,t]
            )
        for s in range(number_of_scenarios)
        for t in range(number_of_hours)
        },
    )
    model.solar_cf = pyo.Param(
        model.S,
        model.T,
        initialize={
            (s, t): float(scenario_solar_cf[s,t])
        
        for s in range(number_of_scenarios)
        for t in range(number_of_hours)
        },
    )
    model.price = pyo.Param(
        model.S,
        model.T,
        initialize={
            (s,t): float(scenario_price[s,t])
        for s in range(number_of_scenarios)
        for t in range(number_of_hours)
        },
    )
    model.scenario_probability = pyo.Param(
        model.S,
        initialize={
            s: float(scenario_probabilities[s])
        for s in range(number_of_scenarios)
        },
    )
    # scalar param
    model.charge_efficiency = pyo.Param(
        initialize=battery_charge_efficiency
    )
    model.discharge_efficiency = pyo.Param(
        initialize=battery_discharge_efficiency
    )
    model.degradation_rate = pyo.Param(
        initialize=degradation_rate
    )
    model.annualized_solar_capacity_cost = pyo.Param(
        initialize=float(
            annualized_solar_capacity_cost
        )
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
    model.operating_cost_scale = pyo.Param(
        initialize=float(
            operating_cost_scale
        )
    )
    # First stage decisions
    model.solar_capacity = pyo.Var(
        domain=pyo.NonNegativeReals
    )
    model.battery_capacity = pyo.Var(
        domain=pyo.NonNegativeReals
    )
    model.battery_power = pyo.Var(
        domain=pyo.NonNegativeReals
    )
    # Second stage decisions 
    model.grid = pyo.Var(
        model.S,
        model.T,
        domain=pyo.NonNegativeReals
    )
    model.charge = pyo.Var(
        model.S,
        model.T,
        domain=pyo.NonNegativeReals
    )
    model.discharge = pyo.Var(
        model.S,
        model.T,
        domain=pyo.NonNegativeReals
    )
    model.solar_used = pyo.Var(
        model.S,
        model.T,
        domain=pyo.NonNegativeReals
    )
    model.curtailment = pyo.Var(
        model.S,
        model.T,
        domain=pyo.NonNegativeReals
    )
    model.soc = pyo.Var(
        model.S,
        model.T_SOC,
        domain=pyo.NonNegativeReals
    )
    # Constraints - physical
        # Energy balance
    def energy_balance_rule(
            model,
            s,
            t,
    ):
        return (
            model.grid[s, t]
            + model.solar_used[s, t]
            + model.discharge[s, t]
            ==
            model.demand[s, t]
            + model.charge[s, t]
        )
    model.energy_balance = pyo.Constraint(
        model.S,
        model.T,
        rule=energy_balance_rule,
    )
        # Solar availability
    def solar_allocation_rule(
            model, 
            s, 
            t,
    ):
        return (
            model.solar_used[s, t]
            + model.curtailment[s, t]
            == 
            model.solar_capacity
            * model.solar_cf[s, t]
        )
    model.solar_allocation = pyo.Constraint(
        model.S,
        model.T,
        rule=solar_allocation_rule,
    )
        # SOC
    def soc_dynamics_rule(
            model,
            s,
            t,
    ):
        return (
            model.soc[s, t + 1]
            == 
            model.soc[s, t]
            + model.charge_efficiency
            * model.charge[s, t]
            - model.discharge[s, t]
            / model.discharge_efficiency
        )
    model.soc_dynamics = pyo.Constraint(
        model.S,
        model.T,
        rule=soc_dynamics_rule,
    )
        # Battery energy limit
    def soc_capacity_rule(
            model,
            s,
            t,
    ):
        return (
            model.soc[s, t]
            <= model.battery_capacity
        )
    model.soc_capacity = pyo.Constraint(
        model.S,
        model.T_SOC,
        rule=soc_capacity_rule,
    )
        # Charge/discharge limits
    def charge_limit_rule(
            model,
            s,
            t,
    ):
        return (
            model.charge[s, t]
            <= model.battery_power
        )
    def discharge_limit_rule(
            model,
            s,
            t,
    ):
        return (
            model.discharge[s, t]
            <= model.battery_power
        )
    model.charge_limit = pyo.Constraint(
        model.S,
        model.T,
        rule=charge_limit_rule,
    )
    model.discharge_limit = pyo.Constraint(
        model.S,
        model.T,
        rule=discharge_limit_rule,
    )
        # SOC for --  every -- scenario!
    def initial_soc_rule(
            model,
            s,
    ): 
        return (
            model.soc[s, 0]
            == 0.0
        )
    model.initial_soc = pyo.Constraint(
        model.S,
        rule=initial_soc_rule,
    )
    def terminal_soc_rule(
            model, 
            s,
    ):
        return (
            model.soc[s, number_of_hours]
            == model.soc[s, 0]
        )
    model.terminal_soc = pyo.Constraint(
        model.S,
        rule=terminal_soc_rule,
    )
    # Scenario cost expressions 
    def scenario_operating_cost_rule(
            model, 
            s,
    ):
        return (
            model.operating_cost_scale
            * sum(
                model.price[s, t]
                * model.grid[s, t]
                + model.degradation_rate
                * (model.charge[s, t]
                   + model.discharge[s, t])
            for t in model.T
            )
        )
    model.scenario_operating_cost = pyo.Expression(
        model.S,
        rule=scenario_operating_cost_rule,
    )
    # cVar variables 
    model.var_threshold = pyo.Var(
        domain=pyo.Reals
    )
    model.excess_cost = pyo.Var(
        model.S,
        domain=pyo.NonNegativeReals,
    )
    # excess cost constraint
    def excess_cost_rule(
            model, 
            s,
    ):
        return (
            model.excess_cost[s]
            >=
            model.scenario_operating_cost[s]
            - model.var_threshold
        )
    model.excess_cost_constraint = pyo.Constraint(
        model.S, 
        rule=excess_cost_rule,
    )
    model.cvar = pyo.Expression(
    expr=(
        model.var_threshold
        + (
            1.0
            / (1.0 - confidence_level)
        )
        * sum(
            model.scenario_probability[s]
            * model.excess_cost[s]
            for s in model.S
        )
    )
)
    # Capital and expected operating cost
    model.capital_cost = pyo.Expression(
        expr=(
            model.annualized_solar_capacity_cost 
            * model.solar_capacity
            + model.annualized_battery_energy_cost
            * model.battery_capacity
            + model.annualized_battery_power_cost
            * model.battery_power
        )
    )
    model.expected_operating_cost = pyo.Expression(
        expr=(
            sum(
        model.scenario_probability[s]
        * model.scenario_operating_cost[s]
        for s in model.S
        )
    )
    )
    # Objective: first stage capacity cost
    # and probability-weighted second stage operating cost
    def objective_rule(model):
    
        return (
                model.capital_cost
                + model.expected_operating_cost
                + risk_aversion
                * model.cvar
            )
    model.objective = pyo.Objective(
            rule=objective_rule,
            sense=pyo.minimize,
        )
    return model
# test run
if __name__ == "__main__":
    model = build_two_stage_model(
        scenario_load=np.array([
            [10.0, 12.0, 11.0],
            [11.0, 13.0, 12.0],
        ]),
        scenario_solar_cf=np.array([
            [0.0, 0.5, 0.2],
            [0.0, 0.3, 0.1],
        ]),
        scenario_price=np.array([
            [0.10, 0.20, 0.40],
            [0.12, 0.25, 0.50],
        ]),
        scenario_probabilities=np.array([
            0.5,
            0.5,
        ]),
        solar_capacity_cost=0.08,
        battery_energy_cost=0.05,
        battery_power_cost=0.03,
        degradation_rate=0.001,
    )

    model.pprint()
    solver = pyo.SolverFactory("highs")
    results = solver.solve(
        model,
    )
    ###
    print(
        "Solver status:",
        results.solver.status,
    )
    print(
        "Termination:",
        results.solver.termination_condition,
    )
    print("\nOptimal Shared Design")
    print("-" * 40)
    print(
        "Solar capacity:",
        pyo.value(model.solar_capacity),
    )
    print(
        "Battery capacity:",
        pyo.value(model.battery_capacity),
    )
    print(
        "Battery power:",
        pyo.value(model.battery_power),
    )
    print(
        "Expected objective:",
        pyo.value(model.objective),
    )
    for s in model.S:
        print(
            f"\nScenario {s}"
        )

        for t in model.T:
            print(
                f"t={t} | "
                f"grid={pyo.value(model.grid[s, t]):.3f} | "
                f"charge={pyo.value(model.charge[s, t]):.3f} | "
                f"discharge={pyo.value(model.discharge[s, t]):.3f} | "
                f"soc={pyo.value(model.soc[s, t]):.3f}"
            )


