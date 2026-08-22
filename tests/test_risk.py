import numpy as np
import pandas as pd
import pytest

from risk import (
    extract_successful_metric,
    calculate_cost_risk_metrics,
    summarize_cost_distribution,
    calculate_exceedance_probability,
)
def make_test_scenarios():
    return pd.DataFrame({
        "solve_success": [
            True,
            True,
            True,
            True,
            True,
        ],
        "objective_cost": [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
        ],
    })
def test_extract_successful_metric_ignores_failed_scenarios():
    scenarios = pd.DataFrame({
        "solve_success": [
            True,
            False,
            True,
        ],
        "objective_cost": [
            10.0,
            np.nan,
            30.0,
        ],
    })

    values = extract_successful_metric(
        scenarios,
        "objective_cost",
    )

    np.testing.assert_allclose(
        values,
        np.array([10.0, 30.0]),
    )


def test_cost_summary_known_distribution():
    scenarios = make_test_scenarios()

    summary = summarize_cost_distribution(
        scenarios,
    )

    assert summary["expected_cost"] == pytest.approx(
        30.0
    )

    assert summary["p50"] == pytest.approx(
        30.0
    )

    assert (
        summary["number_of_successful_scenarios"]
        == 5
    )
def test_exceedance_probability():
    scenarios = make_test_scenarios()

    probability = calculate_exceedance_probability(
        scenarios,
        metric="objective_cost",
        threshold=30.0,
    )

    assert probability == pytest.approx(
        0.40
    )
def test_cvar_is_at_least_var():
    costs = np.array([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ])

    var, cvar = calculate_cost_risk_metrics(
        costs,
        confidence_level=0.80,
    )

    assert cvar >= var