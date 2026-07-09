import numpy as np


def calculate_cost_risk_metrics(costs, confidence_level=0.95):
    """Return Value at Risk and Conditional Value at Risk for annual costs."""
    value_at_risk = np.percentile(costs, confidence_level * 100)
    conditional_value_at_risk = np.mean(costs[costs >= value_at_risk])
    return value_at_risk, conditional_value_at_risk


def summarize_cost_distribution(costs, num_scenarios, confidence_level=0.95):
    """Summarize expected cost, worst-case cost, intervals, VaR, and CVaR."""
    expected_cost = np.mean(costs)
    worst_case_cost = np.max(costs)
    scenario_cost_interval = np.percentile(costs, [2.5, 97.5])
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

    return {
        "expected_cost": expected_cost,
        "worst_case_cost": worst_case_cost,
        "scenario_cost_interval": scenario_cost_interval,
        "value_at_risk": value_at_risk,
        "conditional_value_at_risk": conditional_value_at_risk,
        "cost_standard_deviation": cost_standard_deviation,
        "expected_cost_confidence_interval": expected_cost_confidence_interval,
    }
