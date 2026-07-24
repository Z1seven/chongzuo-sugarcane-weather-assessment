from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REQUIRED_WEATHER_COLUMNS = {
    "date",
    "tmean_c",
    "tmax_c",
    "tmin_c",
    "relative_humidity_pct",
    "soil_moisture_0_7cm_m3m3",
    "gdd_base10",
    "heat_day_tmax_ge35",
    "cold_day_tmin_lt13",
}


@dataclass(frozen=True)
class StageDefinition:
    name: str
    start_month: int
    end_month: int


@dataclass
class AnnualSample:
    year: int
    stage_sequence: np.ndarray
    trend_features: np.ndarray
    previous_production: float
    production: float
    residual: float


def load_weather(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_WEATHER_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Weather file is missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month

    numeric_columns = sorted(REQUIRED_WEATHER_COLUMNS.difference({"date"}))
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].astype(str).tolist()
        raise ValueError(f"Duplicate dates detected: {duplicates[:5]}")

    return frame.sort_values("date").reset_index(drop=True)


def load_production(path: Path) -> Dict[int, float | None]:
    if not path.exists():
        raise FileNotFoundError(f"Production file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "production_10k_t"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Production file is missing columns: {sorted(missing)}")

    result: Dict[int, float | None] = {}
    for row in frame.itertuples(index=False):
        year = int(row.year)
        raw = row.production_10k_t
        value = None if pd.isna(raw) or str(raw).strip() == "" else float(raw)
        result[year] = value
    return result


def build_stage_features(
    weather: pd.DataFrame,
    stages: Iterable[StageDefinition],
) -> Dict[int, np.ndarray]:
    stage_list = list(stages)
    features: Dict[int, np.ndarray] = {}

    for year, year_frame in weather.groupby("year"):
        tokens: List[List[float]] = []
        for stage in stage_list:
            part = year_frame[
                (year_frame["month"] >= stage.start_month)
                & (year_frame["month"] <= stage.end_month)
            ]
            if part.empty:
                raise ValueError(f"No weather rows for year={year}, stage={stage.name}")

            token = [
                float(part["tmean_c"].mean()),
                float(part["tmax_c"].mean()),
                float(part["tmin_c"].mean()),
                float(part["relative_humidity_pct"].mean()),
                float(part["soil_moisture_0_7cm_m3m3"].mean()),
                float(part["gdd_base10"].sum()),
                float(part["heat_day_tmax_ge35"].sum()),
                float(part["cold_day_tmin_lt13"].sum()),
            ]
            tokens.append(token)

        features[int(year)] = np.asarray(tokens, dtype=np.float32)

    return features


def build_trend_features(
    target_year: int,
    production: Dict[int, float | None],
) -> np.ndarray:
    previous = production.get(target_year - 1)
    if previous is None:
        raise ValueError(f"Previous-year production is unavailable for {target_year}")

    history = [
        production[year]
        for year in range(target_year - 3, target_year)
        if production.get(year) is not None
    ]
    moving_mean = float(np.mean(history)) if history else float(previous)
    return np.asarray(
        [float(previous), moving_mean, float(target_year - 2005)],
        dtype=np.float32,
    )


def build_residual_samples(
    production: Dict[int, float | None],
    stage_features: Dict[int, np.ndarray],
) -> List[AnnualSample]:
    samples: List[AnnualSample] = []

    for year in sorted(production):
        current = production.get(year)
        previous = production.get(year - 1)
        if current is None or previous is None:
            continue
        if year not in stage_features:
            raise ValueError(f"Weather features are unavailable for year {year}")

        samples.append(
            AnnualSample(
                year=year,
                stage_sequence=stage_features[year],
                trend_features=build_trend_features(year, production),
                previous_production=float(previous),
                production=float(current),
                residual=float(current - previous),
            )
        )

    return samples
