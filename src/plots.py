from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_rolling_forecast_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    observed = predictions.groupby("year")["observed"].first()
    pivot = predictions.pivot(index="year", columns="model", values="predicted")

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(observed.index, observed.values, marker="o", linewidth=2, label="Observed")

    for model in [
        "Persistence",
        "Residual Ridge",
        "Residual SVR",
        "PG-LiteTransformer",
    ]:
        if model in pivot.columns:
            plt.plot(
                pivot.index,
                pivot[model],
                marker=".",
                linewidth=1.3,
                label=model,
            )

    plt.xlabel("Forecast year")
    plt.ylabel("Production (10,000 t)")
    plt.title("Expanding-window rolling forecasts")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_ablation_plot(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = metrics.sort_values("rmse", ascending=True)

    plt.figure(figsize=(7.7, 4.5))
    plt.barh(ordered["model"], ordered["rmse"])
    plt.xlabel("RMSE (10,000 t)")
    plt.title("Trend-weather ablation")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_permutation_plot(
    permutation_result: Dict[str, object],
    output_path: Path,
) -> None:
    null_values = np.asarray(
        permutation_result["permutation_improvements"],
        dtype=float,
    )
    observed = float(
        permutation_result["observed_rmse_improvement"]
    )

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(null_values, bins=28, edgecolor="black", alpha=0.8)
    plt.axvline(
        observed,
        linestyle="--",
        linewidth=2,
        label=f"Observed improvement = {observed:.2f}",
    )
    plt.xlabel("RMSE improvement over trend-only model (10,000 t)")
    plt.ylabel("Permutation frequency")
    plt.title("Year-weather permutation test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_bootstrap_plot(
    intervals: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.2, 4.8))
    plt.fill_between(
        intervals["year"],
        intervals["lower_95"],
        intervals["upper_95"],
        alpha=0.25,
        label="95% bootstrap interval",
    )
    plt.plot(
        intervals["year"],
        intervals["observed"],
        marker="o",
        linewidth=2,
        label="Observed",
    )
    plt.plot(
        intervals["year"],
        intervals["predicted"],
        marker="s",
        linewidth=1.5,
        label="Predicted",
    )
    plt.xlabel("Forecast year")
    plt.ylabel("Production (10,000 t)")
    plt.title("Trend + complete-weather forecast uncertainty")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def save_area_yield_standardized_plot(
    area_yield: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = {
        "sown_area_1000_ha": "Sown area",
        "production_10k_t": "Production",
        "apparent_yield_t_ha": "Apparent yield",
    }
    standardized = area_yield[["year", *columns]].copy()
    for column in columns:
        values = standardized[column].to_numpy(dtype=float)
        standardized[column] = (values - values.mean()) / values.std(ddof=0)

    plt.figure(figsize=(8.2, 4.8))
    for column, label in columns.items():
        plt.plot(
            standardized["year"],
            standardized[column],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Year")
    plt.ylabel("Standardized value (z score)")
    plt.title("Sown area, production, and apparent yield (2017-2024)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_area_yield_decomposition_plot(
    area_yield: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = area_yield.dropna(subset=["area_effect_10k_t", "yield_effect_10k_t"])
    x = np.arange(len(frame))
    width = 0.36
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x - width / 2, frame["area_effect_10k_t"], width, label="Area effect")
    plt.bar(x + width / 2, frame["yield_effect_10k_t"], width, label="Yield effect")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, frame["year"].astype(int))
    plt.xlabel("Year")
    plt.ylabel("Contribution to annual production change (10,000 t)")
    plt.title("Symmetric decomposition of annual production change")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_yield_sensitivity_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    observed = predictions.groupby("year")["observed"].first()
    pivot = predictions.pivot(index="year", columns="model", values="predicted")
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(observed.index, observed.values, marker="o", linewidth=2.2, label="Observed yield")
    for model in ["Yield persistence", "Yield trend only", "Yield trend + weather"]:
        if model in pivot.columns:
            plt.plot(pivot.index, pivot[model], marker="s", linewidth=1.4, label=model)
    plt.xlabel("Forecast year")
    plt.ylabel("Apparent yield (t/ha)")
    plt.title("Exploratory yield sensitivity analysis")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
