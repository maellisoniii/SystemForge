import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_source_table(uncertainty_table):
    table_for_display = uncertainty_table.copy()
    table_for_display["Cost Standard Deviation (EUR)"] = (
        table_for_display["Cost Standard Deviation (EUR)"]
        .map(lambda value: f"EUR {value:,.2f}")
    )

    figure, axis = plt.subplots(figsize=(11, 2.5))
    axis.axis("off")
    axis.set_title("Monte Carlo Uncertainty Source Comparison", fontsize=14, pad=14)

    comparison_table = axis.table(
        cellText=table_for_display.values,
        colLabels=table_for_display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    comparison_table.auto_set_font_size(False)
    comparison_table.set_fontsize(10)
    comparison_table.scale(1, 1.7)

    for column_index in range(len(table_for_display.columns)):
        comparison_table[(0, column_index)].set_facecolor("steelblue")
        comparison_table[(0, column_index)].set_text_props(
            color="white",
            weight="bold",
        )

    plt.show()


def plot_design_uncertainty_maps(
        design_uncertainty_maps,
        storage_durations_hours,
        solar_penetrations):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for axis, (uncertainty_source, cost_standard_deviation_map) in zip(
            axes,
            design_uncertainty_maps.items()):
        image = axis.imshow(cost_standard_deviation_map, aspect="auto", origin="lower")
        axis.set_title(f"Cost Risk from {uncertainty_source}")
        axis.set_xlabel("Storage Duration (hours)")
        axis.set_ylabel("Solar Penetration")
        axis.set_xticks(
            np.arange(len(storage_durations_hours)),
            storage_durations_hours,
        )
        axis.set_yticks(
            np.arange(len(solar_penetrations)),
            [f"{penetration:.0%}" for penetration in solar_penetrations],
        )
        figure.colorbar(image, ax=axis, label="Cost Std Dev (EUR)")

    plt.show()


def plot_expected_cost_vs_risk(design_uncertainty_table):
    figure, risk_axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    marker_sizes = 60 + design_uncertainty_table["Storage Duration (hours)"] * 120

    risk_axes[0].scatter(
        design_uncertainty_table["All Sources Cost Std Dev (EUR)"],
        design_uncertainty_table["Expected Cost (EUR)"],
        c=design_uncertainty_table["Solar Penetration"],
        s=marker_sizes,
        cmap="viridis",
        alpha=0.8,
    )
    risk_axes[0].set_title("Expected Cost vs Standard-Deviation Risk")
    risk_axes[0].set_xlabel("Annual Cost Standard Deviation (EUR)")
    risk_axes[0].set_ylabel("Expected Annual Cost (EUR)")
    risk_axes[0].grid(True)

    cvar_plot = risk_axes[1].scatter(
        design_uncertainty_table["95% CVaR Tail Premium (EUR)"],
        design_uncertainty_table["Expected Cost (EUR)"],
        c=design_uncertainty_table["Solar Penetration"],
        s=marker_sizes,
        cmap="viridis",
        alpha=0.8,
    )
    risk_axes[1].set_title("Expected Cost vs 95% CVaR Tail Risk")
    risk_axes[1].set_xlabel("95% CVaR Tail Premium (EUR)")
    risk_axes[1].set_ylabel("Expected Annual Cost (EUR)")
    risk_axes[1].grid(True)

    for _, design in design_uncertainty_table.iterrows():
        label = (
            f"S{design['Solar Penetration']:.0%} / "
            f"D{design['Storage Duration (hours)']:.2g}h"
        )
        risk_axes[0].annotate(
            label,
            (
                design["All Sources Cost Std Dev (EUR)"],
                design["Expected Cost (EUR)"],
            ),
            fontsize=7,
        )
        risk_axes[1].annotate(
            label,
            (
                design["95% CVaR Tail Premium (EUR)"],
                design["Expected Cost (EUR)"],
            ),
            fontsize=7,
        )

    figure.colorbar(cvar_plot, ax=risk_axes, label="Solar Penetration")
    plt.show()


def plot_monte_carlo_distribution(scenario_total_costs, cost_summary):
    plt.figure(figsize=(10, 5))
    plt.hist(scenario_total_costs, bins=40, color="steelblue", edgecolor="white")
    plt.axvline(
        cost_summary["expected_cost"],
        color="black",
        linewidth=2,
        label="Expected cost",
    )
    plt.axvline(
        cost_summary["scenario_cost_interval"][0],
        color="orange",
        linestyle="--",
        label="95% interval",
    )
    plt.axvline(
        cost_summary["scenario_cost_interval"][1],
        color="orange",
        linestyle="--",
    )
    plt.title("Monte Carlo Distribution of Annual System Cost")
    plt.xlabel("Annual Cost (EUR)")
    plt.ylabel("Number of Scenarios")
    plt.legend()
    plt.grid(axis="y")
    plt.show()


def plot_cost_map(cost_map, storage_durations_hours, solar_penetrations):
    plt.figure(figsize=(10, 6))
    plt.imshow(cost_map, aspect="auto", origin="lower")
    plt.colorbar(label="Total Annual Cost (EUR)")
    plt.xticks(
        ticks=np.arange(len(storage_durations_hours)),
        labels=storage_durations_hours,
    )
    plt.yticks(
        ticks=np.arange(len(solar_penetrations)),
        labels=[f"{penetration:.0%}" for penetration in solar_penetrations],
    )
    plt.title("Solar Penetration + Storage Duration Cost Map")
    plt.xlabel("Storage Duration (hours)")
    plt.ylabel("Solar Penetration")
    plt.show()


def plot_annual_energy(analysis):
    plt.figure(figsize=(10, 5))
    plt.bar(
        ["Total Load", "Solar Generation", "Solar Used", "Solar Curtailed"],
        [
            analysis["total_load"],
            analysis["total_solar"],
            analysis["total_solar_used"],
            analysis["total_solar_curtailed"],
        ],
        color=["steelblue", "gold", "mediumseagreen", "tomato"],
    )
    plt.title("Annual Load and Solar Energy")
    plt.ylabel("Energy (kWh)")
    plt.grid(axis="y")
    plt.show()


def plot_curtailment_vs_price(profile, analysis):
    hourly_price = profile.price_per_kwh
    hourly_solar_curtailed = analysis["hourly_solar_curtailed"]
    curtailment_indices = analysis["curtailment_indices"]

    plt.figure(figsize=(10, 5))
    plt.scatter(hourly_price, hourly_solar_curtailed, alpha=0.35, s=12)
    if len(curtailment_indices) > 0:
        min_index = analysis["min_curtailment_price_index"]
        max_index = analysis["max_curtailment_price_index"]
        plt.scatter(
            hourly_price[min_index],
            hourly_solar_curtailed[min_index],
            color="green",
            s=80,
            label="Lowest price with curtailment",
        )
        plt.scatter(
            hourly_price[max_index],
            hourly_solar_curtailed[max_index],
            color="red",
            s=80,
            label="Highest price with curtailment",
        )
    plt.title("Solar Curtailment vs Electricity Price")
    plt.xlabel("Electricity Price (EUR/kWh)")
    plt.ylabel("Solar Curtailment (kWh)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_storage_cost_sensitivity(sensitivity_table):
    storage_durations = sensitivity_table["Storage Duration (hours)"]

    plt.figure(figsize=(10, 5))
    plt.plot(
        storage_durations,
        sensitivity_table["Annual Grid Cost (EUR)"],
        marker="o",
        linewidth=2,
        label="Annual Grid Cost",
    )
    plt.plot(
        storage_durations,
        sensitivity_table["Annualized Battery Cost (EUR)"],
        marker="o",
        linewidth=2,
        label="Annualized Battery Cost",
    )
    plt.plot(
        storage_durations,
        sensitivity_table["Total Annual Cost (EUR)"],
        marker="o",
        linewidth=2,
        label="Total Annual Cost",
    )
    plt.title("Storage Duration vs Annual System Cost")
    plt.xlabel("Storage Duration (hours)")
    plt.ylabel("Annual Cost (EUR)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_storage_savings(sensitivity_table):
    plt.figure(figsize=(10, 5))
    plt.plot(
        sensitivity_table["Storage Duration (hours)"],
        sensitivity_table["Annual Battery Savings (EUR)"],
        marker="o",
        linewidth=2,
    )
    plt.title("Storage Duration vs Annual Savings")
    plt.xlabel("Storage Duration (hours)")
    plt.ylabel("Annual Battery Savings (EUR)")
    plt.grid(True)
    plt.show()


def plot_marginal_savings(sensitivity_table):
    storage_durations = sensitivity_table["Storage Duration (hours)"].to_numpy()
    annual_savings = sensitivity_table["Annual Battery Savings (EUR)"].to_numpy()
    marginal_savings = np.diff(annual_savings)

    plt.figure(figsize=(10, 5))
    plt.plot(storage_durations[1:], marginal_savings, marker="o", linewidth=2)
    plt.title("Marginal Savings from Additional Storage Duration")
    plt.xlabel("Storage Duration (hours)")
    plt.ylabel("Additional Annual Savings (EUR)")
    plt.grid(True)
    plt.show()
