from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.area_sensitivity import (
    area_yield_summary,
    build_area_yield_table,
    exploratory_yield_rolling_validation,
    load_sown_area,
)
from src.data import (
    StageDefinition,
    build_residual_samples,
    build_stage_features,
    load_production,
    load_weather,
)
from src.evaluation import (
    TransformerConfig,
    rolling_model_evaluation,
    trend_weather_ablation,
)
from src.plots import (
    save_ablation_plot,
    save_area_yield_decomposition_plot,
    save_area_yield_standardized_plot,
    save_bootstrap_plot,
    save_permutation_plot,
    save_rolling_forecast_plot,
    save_yield_sensitivity_plot,
)
from src.statistics import (
    bootstrap_complete_weather_intervals,
    permutation_test,
)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else project_root / path


def relative_display(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Chongzuo production models, weather-information tests, "
            "and the limited-period sown-area/yield sensitivity analysis."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.json"), help="Configuration file"
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent
    config = load_config(config_path)

    weather_path = resolve(project_root, config["data"]["weather_csv"])
    production_path = resolve(project_root, config["data"]["production_csv"])
    sown_area_path = resolve(project_root, config["data"]["sown_area_csv"])
    output_dir = resolve(project_root, config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    weather = load_weather(weather_path)
    production = load_production(production_path)
    sown_area = load_sown_area(sown_area_path)

    stages = [
        StageDefinition(
            name=item["name"],
            start_month=int(item["start_month"]),
            end_month=int(item["end_month"]),
        )
        for item in config["analysis"]["phenology_stages"]
    ]
    stage_features = build_stage_features(weather, stages)
    samples = build_residual_samples(production, stage_features)

    forecast_years = list(
        range(
            int(config["analysis"]["forecast_start_year"]),
            int(config["analysis"]["forecast_end_year"]) + 1,
        )
    )
    transformer_settings = config["models"]["transformer"]
    transformer_config = TransformerConfig(
        latent_dim=int(transformer_settings["latent_dim"]),
        trend_hidden_dim=int(transformer_settings["trend_hidden_dim"]),
        head_hidden_dim=int(transformer_settings["head_hidden_dim"]),
        dropout=float(transformer_settings["dropout"]),
        learning_rate=float(transformer_settings["learning_rate"]),
        weight_decay=float(transformer_settings["weight_decay"]),
        epochs=int(transformer_settings["epochs"]),
        seeds=[int(seed) for seed in transformer_settings["seeds"]],
        device=str(transformer_settings["device"]),
    )

    rolling_predictions, rolling_metrics = rolling_model_evaluation(
        samples=samples,
        forecast_years=forecast_years,
        residual_ridge_alpha=float(config["models"]["residual_ridge_alpha"]),
        svr_c=float(config["models"]["svr_c"]),
        svr_epsilon=float(config["models"]["svr_epsilon"]),
        transformer_config=transformer_config,
    )
    ablation_predictions, ablation_metrics = trend_weather_ablation(
        production=production,
        stage_features=stage_features,
        forecast_years=forecast_years,
        trend_alpha=float(config["models"]["trend_ridge_alpha"]),
        weather_alpha=float(config["models"]["weather_ridge_alpha"]),
    )
    permutation_result = permutation_test(
        production=production,
        stage_features=stage_features,
        forecast_years=forecast_years,
        trend_alpha=float(config["models"]["trend_ridge_alpha"]),
        weather_alpha=float(config["models"]["weather_ridge_alpha"]),
        permutation_count=int(config["analysis"]["permutation_count"]),
        seed=int(config["analysis"]["permutation_seed"]),
    )
    bootstrap_intervals = bootstrap_complete_weather_intervals(
        production=production,
        stage_features=stage_features,
        forecast_years=forecast_years,
        trend_alpha=float(config["models"]["trend_ridge_alpha"]),
        weather_alpha=float(config["models"]["weather_ridge_alpha"]),
        resamples=int(config["analysis"]["bootstrap_resamples"]),
        seed=int(config["analysis"]["bootstrap_seed"]),
    )

    area_yield = build_area_yield_table(sown_area, production)
    area_forecast_years = list(
        range(
            int(config["analysis"]["area_sensitivity_forecast_start_year"]),
            int(config["analysis"]["area_sensitivity_forecast_end_year"]) + 1,
        )
    )
    yield_predictions, yield_metrics = exploratory_yield_rolling_validation(
        area_yield=area_yield,
        stage_features=stage_features,
        forecast_years=area_forecast_years,
        trend_alpha=float(config["models"]["yield_trend_ridge_alpha"]),
        weather_alpha=float(config["models"]["yield_weather_ridge_alpha"]),
    )
    area_summary = area_yield_summary(area_yield)

    rolling_predictions.to_csv(output_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
    rolling_metrics.to_csv(output_dir / "rolling_metrics.csv", index=False, encoding="utf-8-sig")
    ablation_predictions.to_csv(output_dir / "ablation_predictions.csv", index=False, encoding="utf-8-sig")
    ablation_metrics.to_csv(output_dir / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap_intervals.to_csv(output_dir / "bootstrap_intervals.csv", index=False, encoding="utf-8-sig")
    area_yield.to_csv(output_dir / "area_yield_decomposition.csv", index=False, encoding="utf-8-sig")
    yield_predictions.to_csv(output_dir / "yield_sensitivity_predictions.csv", index=False, encoding="utf-8-sig")
    yield_metrics.to_csv(output_dir / "yield_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "area_yield_summary.json").write_text(json.dumps(area_summary, indent=2), encoding="utf-8")

    permutation_values = np.asarray(permutation_result.pop("permutation_improvements"), dtype=float)
    np.savetxt(
        output_dir / "permutation_improvements.csv",
        permutation_values,
        delimiter=",",
        header="rmse_improvement",
        comments="",
    )
    (output_dir / "permutation_summary.json").write_text(
        json.dumps(permutation_result, indent=2), encoding="utf-8"
    )

    summary = {
        "forecast_years": forecast_years,
        "rolling_metrics": rolling_metrics.to_dict(orient="records"),
        "ablation_metrics": ablation_metrics.to_dict(orient="records"),
        "permutation": permutation_result,
        "bootstrap_coverage_rate": float(bootstrap_intervals["covered"].mean()),
        "area_yield_summary": area_summary,
        "yield_sensitivity_metrics": yield_metrics.to_dict(orient="records"),
        "note": (
            "The 2017-2024 area/yield analysis is exploratory because only eight "
            "area observations and four rolling forecast years are available."
        ),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    save_rolling_forecast_plot(rolling_predictions, output_dir / "rolling_forecasts.png")
    save_ablation_plot(ablation_metrics, output_dir / "ablation_rmse.png")
    save_permutation_plot(
        {**permutation_result, "permutation_improvements": permutation_values},
        output_dir / "permutation_test.png",
    )
    save_bootstrap_plot(bootstrap_intervals, output_dir / "bootstrap_intervals.png")
    save_area_yield_standardized_plot(area_yield, output_dir / "area_yield_standardized_trends.png")
    save_area_yield_decomposition_plot(area_yield, output_dir / "area_yield_decomposition.png")
    save_yield_sensitivity_plot(yield_predictions, output_dir / "yield_weather_sensitivity.png")

    print("\nUnified rolling-validation metrics")
    print(rolling_metrics.to_string(index=False))
    print("\nTrend-weather ablation metrics")
    print(ablation_metrics.to_string(index=False))
    print("\nPermutation test")
    print(json.dumps(permutation_result, indent=2))
    print("\nLimited-period area/yield summary")
    print(json.dumps(area_summary, indent=2))
    print("\nExploratory yield sensitivity metrics")
    print(yield_metrics.to_string(index=False))
    print(f"\nOutputs written to: {relative_display(output_dir, project_root)}")


if __name__ == "__main__":
    main()
