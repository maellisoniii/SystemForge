import numpy as np
import pandas as pd

def extract_successful_metric(
        scenarios: pd.DataFrame,
        metric: str,
) -> np.ndarray:
    """ Return finite values for a metric
    from successful scenarios only."""
    if "solve_success" not in scenarios.columns:
        raise ValueError(
            "Scenario table must contain 'solve_success'."
        )
    if metric not in scenarios.columns:
        raise ValueError(
            f"Scenario table does not contain '{metric}'."
        )
    successful_values = scenarios.loc[
        scenarios["solve_success"],
        metric,
    ].to_numpy(dtype=float)
    successful_values = successful_values[
        np.isfinite(successful_values)
    ]
    if len(successful_values) == 0:
        raise ValueError(
            f"No successful finite values found for '{metric}'."
        )
    return successful_values

def calculate_cost_risk_metrics(costs, 
                                confidence_level=0.95):
    """Return Value at Risk and Conditional Value at Risk for annual costs."""
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence level must be between 0 and 1"
        )
    value_at_risk = np.percentile(
        costs, 
        confidence_level * 100)
    conditional_value_at_risk = np.mean(
        costs[costs >= value_at_risk])
    return value_at_risk, conditional_value_at_risk


def summarize_cost_distribution(
        scenarios: pd.DataFrame,
        confidence_level: float=0.95,
        ):
    """Summarize expected cost, worst-case cost, intervals, VaR, and CVaR.""" 
    costs = extract_successful_metric(
        scenarios,
        "objective_cost",
    )
    num_scenarios = len(costs)
    if num_scenarios < 2:
        raise ValueError(
    "At least two succesful scenarios are required"
    "to summarize the cost distribution."
)  
    expected_cost = np.mean(costs)
    worst_case_cost = np.max(costs)
    scenario_cost_interval = np.percentile(
        costs, [2.5, 97.5])

    value_at_risk, conditional_value_at_risk = calculate_cost_risk_metrics(
        costs,
        confidence_level=confidence_level,
    )
    cost_standard_deviation = np.std(costs, ddof=1)

    standard_error = cost_standard_deviation / np.sqrt(num_scenarios)

    expected_cost_confidence_interval = (
        expected_cost - 1.96 * standard_error,
        expected_cost + 1.96 * standard_error,
    )
    p5 = np.percentile(costs, 5)
    p50 = np.percentile(costs, 50)
    p95 = np.percentile(costs, 95)

    return {
        "expected_cost": 
        expected_cost,
        "worst_case_cost": 
        worst_case_cost,
        "p5":
        p5,
        "p50":
        p50,
        "p95":
        p95,
        "scenario_cost_interval": 
        scenario_cost_interval,
        "value_at_risk": 
        value_at_risk,
        "conditional_value_at_risk": 
        conditional_value_at_risk,
        "cost_standard_deviation": 
        cost_standard_deviation,
        "expected_cost_confidence_interval": 
        expected_cost_confidence_interval,
        "number_of_successful_scenarios":
        num_scenarios,
    }
def calculate_exceedance_probability(
        scenarios: pd.DataFrame,
        metric: str,
        threshold: float,   
    ) -> float:
    values = extract_successful_metric(
        scenarios,
        metric,
    )
    return float(
        np.mean(values > threshold)
    )
