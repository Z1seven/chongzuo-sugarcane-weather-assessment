from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (x_train - mean) / scale, (x_test - mean) / scale


def _ridge_fit_predict_closed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, float]:
    """Equivalent to StandardScaler + Ridge(fit_intercept=True) for dense data."""
    z_train, z_test = _standardize_train_test(x_train, x_test)
    y_mean = float(y_train.mean())
    centered_y = y_train - y_mean
    gram = z_train.T @ z_train + alpha * np.eye(z_train.shape[1])
    coefficient = np.linalg.solve(gram, z_train.T @ centered_y)
    fitted = y_mean + z_train @ coefficient
    predicted = y_mean + z_test @ coefficient
    return fitted, float(np.asarray(predicted).reshape(-1)[0])


def permutation_test(
    production: Dict[int, float | None],
    stage_features: Dict[int, np.ndarray],
    forecast_years: Sequence[int],
    trend_alpha: float,
    weather_alpha: float,
    permutation_count: int,
    seed: int,
) -> Dict[str, float | int | np.ndarray]:
    """Year-weather permutation test with leakage-free rolling folds.

    The same global permutation map is applied to the year-to-weather correspondence
    in every fold, matching the manuscript analysis. Closed-form ridge calculations
    are used so that 5000 permutations finish in a practical amount of time.
    """
    valid_years = np.asarray(
        [year for year, value in production.items() if value is not None],
        dtype=int,
    )

    flattened_features = {
        int(year): np.asarray(values, dtype=float).ravel()
        for year, values in stage_features.items()
    }

    fold_cache = []
    observed = []

    for test_year in forecast_years:
        train_years = valid_years[valid_years < test_year]
        train_production = np.asarray(
            [production[int(year)] for year in train_years],
            dtype=float,
        )

        fitted_trend, test_trend = _ridge_fit_predict_closed_form(
            train_years.reshape(-1, 1).astype(float),
            train_production,
            np.asarray([[test_year]], dtype=float),
            trend_alpha,
        )
        residual_target = train_production - fitted_trend

        fold_cache.append(
            {
                "test_year": int(test_year),
                "train_years": train_years.copy(),
                "residual_target": residual_target,
                "test_trend": test_trend,
            }
        )
        observed.append(float(production[test_year]))

    observed_array = np.asarray(observed, dtype=float)
    trend_predictions = np.asarray(
        [fold["test_trend"] for fold in fold_cache], dtype=float
    )
    trend_rmse = float(
        math.sqrt(mean_squared_error(observed_array, trend_predictions))
    )

    def weather_rmse(mapping: Dict[int, int] | None) -> float:
        predictions: List[float] = []

        for fold in fold_cache:
            train_years = fold["train_years"]
            mapped_train_years = [
                mapping.get(int(year), int(year)) if mapping else int(year)
                for year in train_years
            ]
            mapped_test_year = (
                mapping.get(fold["test_year"], fold["test_year"])
                if mapping
                else fold["test_year"]
            )

            x_train = np.stack(
                [flattened_features[year] for year in mapped_train_years]
            )
            x_test = flattened_features[mapped_test_year].reshape(1, -1)

            _, weather_residual = _ridge_fit_predict_closed_form(
                x_train,
                fold["residual_target"],
                x_test,
                weather_alpha,
            )
            predictions.append(fold["test_trend"] + weather_residual)

        return float(
            math.sqrt(
                mean_squared_error(
                    observed_array,
                    np.asarray(predictions, dtype=float),
                )
            )
        )

    complete_weather_rmse = weather_rmse(mapping=None)
    observed_improvement = trend_rmse - complete_weather_rmse

    rng = np.random.default_rng(seed)
    permutation_improvements = np.empty(permutation_count, dtype=float)

    for index in range(permutation_count):
        shuffled = valid_years.copy()
        rng.shuffle(shuffled)
        mapping = {
            int(original): int(permuted)
            for original, permuted in zip(valid_years, shuffled)
        }
        permutation_improvements[index] = trend_rmse - weather_rmse(mapping)

    p_value = float(
        (1 + np.sum(permutation_improvements >= observed_improvement))
        / (permutation_count + 1)
    )

    return {
        "trend_rmse": trend_rmse,
        "complete_weather_rmse": complete_weather_rmse,
        "observed_rmse_improvement": float(observed_improvement),
        "permutation_count": int(permutation_count),
        "seed": int(seed),
        "one_sided_p_value": p_value,
        "null_mean": float(permutation_improvements.mean()),
        "null_sd": float(permutation_improvements.std()),
        "null_q025": float(np.quantile(permutation_improvements, 0.025)),
        "null_q975": float(np.quantile(permutation_improvements, 0.975)),
        "permutation_improvements": permutation_improvements,
    }


def bootstrap_complete_weather_intervals(
    production: Dict[int, float | None],
    stage_features: Dict[int, np.ndarray],
    forecast_years: Sequence[int],
    trend_alpha: float,
    weather_alpha: float,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    valid_years = np.asarray(
        [year for year, value in production.items() if value is not None],
        dtype=int,
    )
    flattened_features = {
        int(year): np.asarray(values, dtype=float).ravel()
        for year, values in stage_features.items()
    }
    rows = []

    for test_year in forecast_years:
        train_years = valid_years[valid_years < test_year]
        train_production = np.asarray(
            [production[int(year)] for year in train_years],
            dtype=float,
        )
        x_weather = np.stack(
            [flattened_features[int(year)] for year in train_years]
        )
        x_test_weather = flattened_features[test_year].reshape(1, -1)

        fitted_trend, test_trend = _ridge_fit_predict_closed_form(
            train_years.reshape(-1, 1).astype(float),
            train_production,
            np.asarray([[test_year]], dtype=float),
            trend_alpha,
        )
        residual_target = train_production - fitted_trend
        _, weather_residual = _ridge_fit_predict_closed_form(
            x_weather,
            residual_target,
            x_test_weather,
            weather_alpha,
        )
        point_prediction = test_trend + weather_residual

        rng = np.random.default_rng(seed + int(test_year))
        bootstrap_predictions: List[float] = []
        sample_count = len(train_years)

        attempts = 0
        while len(bootstrap_predictions) < resamples and attempts < resamples * 20:
            attempts += 1
            indices = rng.integers(0, sample_count, size=sample_count)
            sampled_years = train_years[indices]
            sampled_production = train_production[indices]

            if len(np.unique(sampled_years)) < 3:
                continue
            if float(sampled_years.std()) == 0.0:
                continue

            sampled_weather = x_weather[indices]

            sampled_fitted_trend, sampled_test_trend = _ridge_fit_predict_closed_form(
                sampled_years.reshape(-1, 1).astype(float),
                sampled_production,
                np.asarray([[test_year]], dtype=float),
                trend_alpha,
            )
            sampled_residual = sampled_production - sampled_fitted_trend

            _, sampled_weather_residual = _ridge_fit_predict_closed_form(
                sampled_weather,
                sampled_residual,
                x_test_weather,
                weather_alpha,
            )
            bootstrap_predictions.append(
                sampled_test_trend + sampled_weather_residual
            )

        if not bootstrap_predictions:
            raise RuntimeError(f"Bootstrap failed for test year {test_year}")

        lower, upper = np.quantile(
            np.asarray(bootstrap_predictions, dtype=float),
            [0.025, 0.975],
        )
        observed = float(production[test_year])

        rows.append(
            {
                "year": int(test_year),
                "observed": observed,
                "predicted": float(point_prediction),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "covered": int(lower <= observed <= upper),
                "successful_resamples": int(len(bootstrap_predictions)),
            }
        )

    return pd.DataFrame(rows)
