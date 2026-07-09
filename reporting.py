def print_best_design(profile, best_design):
    print(f"\nBest System Design ({profile.name}, {profile.analysis_period})")
    print("------------------")
    print("Data Source:", profile.source)
    print("Solar Penetration:", round(best_design["solar_penetration"] * 100, 1), "%")
    print("Storage Duration:", best_design["storage_duration_hours"], "hours")
    print("Battery Capacity:", round(best_design["battery_capacity"], 2), "kWh")
    print("Total Annual Cost: EUR", round(best_design["total_annual_cost"], 2))
    print("Annual Grid Cost: EUR", round(best_design["annual_grid_cost"], 2))
    print("Annualized Battery Cost: EUR", round(best_design["annualized_battery_cost"], 2))
    print("Annualized Solar Cost: EUR", round(best_design["annualized_solar_cost"], 2))
    print(
        "Annual Battery Dispatch Savings: EUR",
        round(best_design["annual_battery_dispatch_savings"], 2),
    )
    print("Max Battery SOC:", round(best_design["max_soc"], 2), "kWh")
    print("Solar Curtailed:", round(best_design["solar_curtailed"], 2), "kWh")
    print("Grid Dependence:", round(best_design["grid_dependence"] * 100, 2), "%")


def print_monte_carlo_analysis(
        num_scenarios,
        storage_duration,
        solar_penetration,
        cost_summary):
    print(
        "Annual Cost Standard Deviation: EUR",
        round(cost_summary["cost_standard_deviation"], 2),
    )

    print("\nMonte Carlo Cost Analysis")
    print("-------------------------")
    print("Scenarios:", num_scenarios)
    print("Storage Duration:", storage_duration, "hours")
    print("Solar Penetration:", round(solar_penetration * 100, 1), "%")
    print("Expected Annual Cost: EUR", round(cost_summary["expected_cost"], 2))
    print("Worst-Case Annual Cost: EUR", round(cost_summary["worst_case_cost"], 2))
    print("95% Value at Risk: EUR", round(cost_summary["value_at_risk"], 2))
    print(
        "95% Conditional Value at Risk: EUR",
        round(cost_summary["conditional_value_at_risk"], 2),
    )
    print(
        "95% Scenario Cost Interval: EUR",
        round(cost_summary["scenario_cost_interval"][0], 2),
        "to EUR",
        round(cost_summary["scenario_cost_interval"][1], 2),
    )
    print(
        "95% Confidence Interval for Expected Cost: EUR",
        round(cost_summary["expected_cost_confidence_interval"][0], 2),
        "to EUR",
        round(cost_summary["expected_cost_confidence_interval"][1], 2),
    )


def print_uncertainty_table(uncertainty_table):
    print("\nUncertainty Source Comparison")
    print("-----------------------------")
    print(
        uncertainty_table.to_string(
            index=False,
            formatters={
                "Cost Standard Deviation (EUR)": "{:,.2f}".format,
            },
        )
    )


def print_design_uncertainty_table(design_uncertainty_table):
    print("\nDesign Uncertainty Comparison")
    print("-----------------------------")
    formatters = {
        column: "{:,.2f}".format
        for column in design_uncertainty_table.columns
        if "(EUR)" in column
    }
    formatters.update({
        "Solar Penetration": "{:.1%}".format,
        "Solar Utilization": "{:.1%}".format,
        "Grid Dependence": "{:.1%}".format,
    })
    print(
        design_uncertainty_table.to_string(
            index=False,
            formatters=formatters,
        )
    )


def print_load_solar_price_analysis(profile, analysis):
    hourly_price = profile.price_per_kwh
    hourly_solar_curtailed = analysis["hourly_solar_curtailed"]
    min_price_index = analysis["min_price_index"]
    max_price_index = analysis["max_price_index"]
    min_curtailment_price_index = analysis["min_curtailment_price_index"]
    max_curtailment_price_index = analysis["max_curtailment_price_index"]

    print("\nLoad, Solar, and Price Analysis")
    print("-------------------------------")
    print("Total Load:", round(analysis["total_load"], 2), "kWh")
    print("Total Solar Generation:", round(analysis["total_solar"], 2), "kWh")
    print("Total Solar Used:", round(analysis["total_solar_used"], 2), "kWh")
    print("Total Solar Curtailment:", round(analysis["total_solar_curtailed"], 2), "kWh")
    print("Solar Penetration:", round(analysis["solar_penetration"] * 100, 2), "%")
    print("Solar Utilization:", round(analysis["solar_utilization"] * 100, 2), "%")
    print("Grid Dependence:", round(analysis["grid_dependence"] * 100, 2), "%")
    print("Minimum Price:", round(hourly_price[min_price_index], 4), "EUR/kWh")
    print(
        "Curtailment at Minimum Price:",
        round(hourly_solar_curtailed[min_price_index], 2),
        "kWh",
    )
    print("Maximum Price:", round(hourly_price[max_price_index], 4), "EUR/kWh")
    print(
        "Curtailment at Maximum Price:",
        round(hourly_solar_curtailed[max_price_index], 2),
        "kWh",
    )

    if min_curtailment_price_index is not None:
        print(
            "Lowest Price During Curtailment:",
            round(hourly_price[min_curtailment_price_index], 4),
            "EUR/kWh",
        )
        print(
            "Curtailment at Lowest Curtailment Price:",
            round(hourly_solar_curtailed[min_curtailment_price_index], 2),
            "kWh",
        )
        print(
            "Highest Price During Curtailment:",
            round(hourly_price[max_curtailment_price_index], 4),
            "EUR/kWh",
        )
        print(
            "Curtailment at Highest Curtailment Price:",
            round(hourly_solar_curtailed[max_curtailment_price_index], 2),
            "kWh",
        )


def print_storage_sensitivity(sensitivity_table):
    for _, result in sensitivity_table.iterrows():
        print("\nStorage Duration:", f"{result['Storage Duration (hours)']:g}", "hours")
        print("Battery Capacity:", round(result["Battery Capacity (kWh)"], 2), "kWh")
        print("Total Grid Import:", round(result["Total Grid Import (kWh)"], 2), "kWh")
        print("Solar Curtailed:", round(result["Solar Curtailed (kWh)"], 2), "kWh")
        print("Annual Grid Cost: EUR", round(result["Annual Grid Cost (EUR)"], 2))
        print(
            "Annualized Battery Cost: EUR",
            round(result["Annualized Battery Cost (EUR)"], 2),
        )
        print("Total Annual Cost: EUR", round(result["Total Annual Cost (EUR)"], 2))
        print(
            "Annual Battery Dispatch Savings: EUR",
            round(result["Annual Battery Dispatch Savings (EUR)"], 2),
        )
        print("Max SOC:", round(result["Max SOC (kWh)"], 2), "kWh")
