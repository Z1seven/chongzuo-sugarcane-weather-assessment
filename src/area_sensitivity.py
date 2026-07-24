from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_sown_area(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sown-area file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "sown_area_1000_ha"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sown-area file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["sown_area_1000_ha"] = pd.to_numeric(
        frame["sown_area_1000_ha"], errors="raise"
    )
    if frame["year"].duplicated().any():
        raise ValueError("Duplicate years detected in the sown-area file")
    if (frame["sown_area_1000_ha"] <= 0).any():
        raise ValueError("Sown area must be positive")
    return frame.sort_values("year").reset_index(drop=True)


def build_area_yield_table(
    area: pd.DataFrame,
    production: Dict[int, float | None],
) -> pd.DataFrame:
    rows = []
    for row in area.itertuples(index=False):
        year = int(row.year)
        value = production.get(year)
        if value is None:
            raise ValueError(f"Production is unavailable for area year {year}")
        production_10k_t = float(value)
        area_1000_ha = float(row.sown_area_1000_ha)
        # (10,000 t)/(1,000 ha) = 10 t/ha.
        yield_t_ha = production_10k_t * 10.0 / area_1000_ha
        rows.append(
            {
                "year": year,
                "sown_area_1000_ha": area_1000_ha,
                "production_10k_t": production_10k_t,
                "apparent_yield_t_ha": yield_t_ha,
            }
        )
    frame = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    frame["area_change_1000_ha"] = frame["sown_area_1000_ha"].diff()
    frame["yield_change_t_ha"] = frame["apparent_yield_t_ha"].diff()
    frame["production_change_10k_t"] = frame["production_10k_t"].diff()

    previous_area = frame["sown_area_1000_ha"].shift(1)
    previous_yield = frame["apparent_yield_t_ha"].shift(1)
    # Exact symmetric two-factor decomposition of P=A*Y/10.
    frame["area_effect_10k_t"] = (
        0.5
        * (frame["apparent_yield_t_ha"] + previous_yield)
        * (frame["sown_area_1000_ha"] - previous_area)
        / 10.0
    )
    frame["yield_effect_10k_t"] = (
        0.5
        * (frame["sown_area_1000_ha"] + previous_area)
        * (frame["apparent_yield_t_ha"] - previous_yield)
        / 10.0
    )
    frame["decomposition_error_10k_t"] = (
        frame["production_change_10k_t"]
        - frame["area_effect_10k_t"]
        - frame["yield_effect_10k_t"]
    )
    return frame


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    result = {
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "mape_percent": float(
            np.mean(np.abs((observed - predicted) / observed)) * 100.0
        ),
    }
    result["r2"] = (
        float(r2_score(observed, predicted)) if len(observed) >= 2 else float("nan")
    )
    return result


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_train)), model.predict(scaler.transform(x_test))


def exploratory_yield_rolling_validation(
    area_yield: pd.DataFrame,
    stage_features: Dict[int, np.ndarray],
    forecast_years: Sequence[int],
    trend_alpha: float,
    weather_alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available = area_yield.set_index("year")
    rows = []
    for test_year in forecast_years:
        train = area_yield[area_yield["year"] < test_year].copy()
        if len(train) < 4 or test_year not in available.index:
            continue
        observed = float(available.loc[test_year, "apparent_yield_t_ha"])
        previous = float(available.loc[test_year - 1, "apparent_yield_t_ha"])
        rows.append(
            {"year": test_year, "model": "Yield persistence", "observed": observed, "predicted": previous}
        )

        years = train["year"].to_numpy(dtype=float).reshape(-1, 1)
        targets = train["apparent_yield_t_ha"].to_numpy(dtype=float)
        fitted_trend, test_trend_arr = _ridge_predict(
            years,
            targets,
            np.asarray([[test_year]], dtype=float),
            trend_alpha,
        )
        test_trend = float(test_trend_arr[0])
        rows.append(
            {"year": test_year, "model": "Yield trend only", "observed": observed, "predicted": test_trend}
        )

        residual_target = targets - fitted_trend
        x_train = np.stack(
            [np.asarray(stage_features[int(y)], dtype=float).ravel() for y in train["year"]]
        )
        x_test = np.asarray(stage_features[int(test_year)], dtype=float).ravel().reshape(1, -1)
        _, residual_pred = _ridge_predict(
            x_train, residual_target, x_test, weather_alpha
        )
        rows.append(
            {
                "year": test_year,
                "model": "Yield trend + weather",
                "observed": observed,
                "predicted": test_trend + float(residual_pred[0]),
            }
        )

    predictions = pd.DataFrame(rows)
    predictions["error"] = predictions["predicted"] - predictions["observed"]
    predictions["absolute_error"] = predictions["error"].abs()

    metric_rows = []
    for model, group in predictions.groupby("model", sort=False):
        metric_rows.append(
            {
                "model": model,
                "forecast_years": int(len(group)),
                **_metrics(
                    group["observed"].to_numpy(dtype=float),
                    group["predicted"].to_numpy(dtype=float),
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    return predictions, metrics


def area_yield_summary(area_yield: pd.DataFrame) -> Dict[str, float | int]:
    first = area_yield.iloc[0]
    last = area_yield.iloc[-1]
    area_corr = float(
        area_yield["sown_area_1000_ha"].corr(area_yield["production_10k_t"])
    )
    yield_corr = float(
        area_yield["apparent_yield_t_ha"].corr(area_yield["production_10k_t"])
    )
    return {
        "years": int(len(area_yield)),
        "start_year": int(first["year"]),
        "end_year": int(last["year"]),
        "area_change_percent": float(
            (last["sown_area_1000_ha"] / first["sown_area_1000_ha"] - 1.0) * 100.0
        ),
        "production_change_percent": float(
            (last["production_10k_t"] / first["production_10k_t"] - 1.0) * 100.0
        ),
        "yield_change_percent": float(
            (last["apparent_yield_t_ha"] / first["apparent_yield_t_ha"] - 1.0) * 100.0
        ),
        "area_cv_percent": float(
            area_yield["sown_area_1000_ha"].std(ddof=1)
            / area_yield["sown_area_1000_ha"].mean()
            * 100.0
        ),
        "yield_cv_percent": float(
            area_yield["apparent_yield_t_ha"].std(ddof=1)
            / area_yield["apparent_yield_t_ha"].mean()
            * 100.0
        ),
        "area_production_pearson_r": area_corr,
        "yield_production_pearson_r": yield_corr,
        "max_abs_decomposition_error_10k_t": float(
            area_yield["decomposition_error_10k_t"].abs().fillna(0.0).max()
        ),
    }
