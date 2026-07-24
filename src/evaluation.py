from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn

from .data import AnnualSample
from .models import PGLiteTransformer


@dataclass
class TransformerConfig:
    latent_dim: int
    trend_hidden_dim: int
    head_hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seeds: Sequence[int]
    device: str


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "mape_percent": float(np.mean(np.abs((observed - predicted) / observed)) * 100.0),
        "r2": float(r2_score(observed, predicted)),
    }


def train_transformer_and_predict(
    train_samples: Sequence[AnnualSample],
    test_sample: AnnualSample,
    config: TransformerConfig,
) -> Tuple[float, float]:
    if len(train_samples) < 3:
        raise ValueError("At least three training samples are required")

    sequence_train = np.stack([sample.stage_sequence for sample in train_samples])
    trend_train = np.stack([sample.trend_features for sample in train_samples])
    residual_train = np.asarray(
        [sample.residual for sample in train_samples], dtype=np.float32
    )

    sequence_scaler = StandardScaler().fit(
        sequence_train.reshape(-1, sequence_train.shape[-1])
    )
    trend_scaler = StandardScaler().fit(trend_train)

    sequence_scaled = sequence_scaler.transform(
        sequence_train.reshape(-1, sequence_train.shape[-1])
    ).reshape(sequence_train.shape).astype(np.float32)
    trend_scaled = trend_scaler.transform(trend_train).astype(np.float32)

    residual_mean = float(residual_train.mean())
    residual_std = float(residual_train.std() + 1e-6)
    residual_scaled = ((residual_train - residual_mean) / residual_std).astype(np.float32)

    test_sequence = sequence_scaler.transform(
        test_sample.stage_sequence.reshape(-1, test_sample.stage_sequence.shape[-1])
    ).reshape(1, *test_sample.stage_sequence.shape).astype(np.float32)
    test_trend = trend_scaler.transform(
        test_sample.trend_features.reshape(1, -1)
    ).astype(np.float32)

    device = torch.device(config.device)
    x_sequence = torch.tensor(sequence_scaled, device=device)
    x_trend = torch.tensor(trend_scaled, device=device)
    y_residual = torch.tensor(residual_scaled, device=device)
    test_sequence_tensor = torch.tensor(test_sequence, device=device)
    test_trend_tensor = torch.tensor(test_trend, device=device)

    predictions: List[float] = []

    for seed in config.seeds:
        set_global_seed(int(seed))

        model = PGLiteTransformer(
            input_dim=sequence_train.shape[-1],
            latent_dim=config.latent_dim,
            trend_input_dim=trend_train.shape[-1],
            trend_hidden_dim=config.trend_hidden_dim,
            head_hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
            stage_count=sequence_train.shape[1],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_function = nn.SmoothL1Loss()

        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad()
            predicted = model(x_sequence, x_trend)
            loss = loss_function(predicted, y_residual)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scaled_prediction = model(
                test_sequence_tensor, test_trend_tensor
            ).item()

        residual_prediction = scaled_prediction * residual_std + residual_mean
        predictions.append(float(residual_prediction))

    return float(np.median(predictions)), float(np.std(predictions))


def rolling_model_evaluation(
    samples: Sequence[AnnualSample],
    forecast_years: Iterable[int],
    residual_ridge_alpha: float,
    svr_c: float,
    svr_epsilon: float,
    transformer_config: TransformerConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_by_year = {sample.year: sample for sample in samples}
    result_rows: List[Dict[str, float | str | int]] = []

    for test_year in forecast_years:
        if test_year not in sample_by_year:
            raise ValueError(
                f"Test year {test_year} cannot be evaluated because its "
                "current or previous-year production is missing"
            )

        test_sample = sample_by_year[test_year]
        train_samples = [
            sample for sample in samples if sample.year < test_year
        ]

        feature_train = np.asarray(
            [
                np.concatenate(
                    [sample.stage_sequence.ravel(), sample.trend_features]
                )
                for sample in train_samples
            ]
        )
        residual_train = np.asarray(
            [sample.residual for sample in train_samples]
        )
        feature_test = np.concatenate(
            [test_sample.stage_sequence.ravel(), test_sample.trend_features]
        ).reshape(1, -1)

        feature_scaler = StandardScaler().fit(feature_train)
        feature_train_scaled = feature_scaler.transform(feature_train)
        feature_test_scaled = feature_scaler.transform(feature_test)

        ridge = Ridge(alpha=residual_ridge_alpha)
        ridge.fit(feature_train_scaled, residual_train)
        ridge_residual = float(ridge.predict(feature_test_scaled)[0])

        svr = SVR(C=svr_c, epsilon=svr_epsilon, gamma="scale")
        svr.fit(feature_train_scaled, residual_train)
        svr_residual = float(svr.predict(feature_test_scaled)[0])

        transformer_residual, seed_sd = train_transformer_and_predict(
            train_samples,
            test_sample,
            transformer_config,
        )

        forecasts = [
            ("Persistence", test_sample.previous_production, 0.0),
            (
                "Residual Ridge",
                test_sample.previous_production + ridge_residual,
                0.0,
            ),
            (
                "Residual SVR",
                test_sample.previous_production + svr_residual,
                0.0,
            ),
            (
                "PG-LiteTransformer",
                test_sample.previous_production + transformer_residual,
                seed_sd,
            ),
        ]

        for model_name, predicted, model_seed_sd in forecasts:
            result_rows.append(
                {
                    "year": test_year,
                    "model": model_name,
                    "observed": test_sample.production,
                    "predicted": predicted,
                    "error": predicted - test_sample.production,
                    "absolute_error": abs(predicted - test_sample.production),
                    "seed_prediction_sd": model_seed_sd,
                }
            )

    predictions = pd.DataFrame(result_rows)

    metric_rows: List[Dict[str, float | str | int]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        metrics = regression_metrics(
            group["observed"].to_numpy(dtype=float),
            group["predicted"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "model": model_name,
                "forecast_years": int(len(group)),
                **metrics,
            }
        )

    return predictions, pd.DataFrame(metric_rows)


def _fit_standardized_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> float:
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha)
    model.fit(scaler.transform(x_train), y_train)
    return float(model.predict(scaler.transform(x_test))[0])


def trend_weather_ablation(
    production: Dict[int, float | None],
    stage_features: Dict[int, np.ndarray],
    forecast_years: Sequence[int],
    trend_alpha: float,
    weather_alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid_years = np.asarray(
        [year for year, value in production.items() if value is not None],
        dtype=int,
    )

    groups = {
        "Trend only": None,
        "Trend + temperature": [0, 1, 2, 5],
        "Trend + water": [3, 4],
        "Trend + extreme stress": [6, 7],
        "Trend + complete weather": list(range(8)),
    }

    prediction_rows: List[Dict[str, float | str | int]] = []

    for test_year in forecast_years:
        train_years = valid_years[valid_years < test_year]
        train_production = np.asarray(
            [production[int(year)] for year in train_years],
            dtype=float,
        )

        trend_prediction = _fit_standardized_ridge(
            train_years.reshape(-1, 1).astype(float),
            train_production,
            np.asarray([[test_year]], dtype=float),
            trend_alpha,
        )

        trend_scaler = StandardScaler().fit(
            train_years.reshape(-1, 1).astype(float)
        )
        trend_model = Ridge(alpha=trend_alpha)
        trend_model.fit(
            trend_scaler.transform(
                train_years.reshape(-1, 1).astype(float)
            ),
            train_production,
        )
        fitted_trend = trend_model.predict(
            trend_scaler.transform(
                train_years.reshape(-1, 1).astype(float)
            )
        )
        residual_target = train_production - fitted_trend

        for group_name, feature_indices in groups.items():
            predicted = trend_prediction
            if feature_indices is not None:
                x_train = np.stack(
                    [
                        stage_features[int(year)][:, feature_indices].ravel()
                        for year in train_years
                    ]
                )
                x_test = stage_features[test_year][:, feature_indices].ravel().reshape(1, -1)
                predicted += _fit_standardized_ridge(
                    x_train,
                    residual_target,
                    x_test,
                    weather_alpha,
                )

            observed = float(production[test_year])
            prediction_rows.append(
                {
                    "year": int(test_year),
                    "model": group_name,
                    "observed": observed,
                    "predicted": predicted,
                    "error": predicted - observed,
                    "absolute_error": abs(predicted - observed),
                }
            )

    predictions = pd.DataFrame(prediction_rows)

    metric_rows: List[Dict[str, float | str | int]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        metrics = regression_metrics(
            group["observed"].to_numpy(dtype=float),
            group["predicted"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "model": model_name,
                "forecast_years": int(len(group)),
                **metrics,
            }
        )

    return predictions, pd.DataFrame(metric_rows)
