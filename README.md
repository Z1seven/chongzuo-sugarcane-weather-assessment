# Chongzuo Sugarcane Weather-Information Assessment

This repository contains the complete analysis code and processed data for the manuscript:

**A Leakage-Free Framework for Assessing Phenology-Aware Weather Information in Small-Sample Regional Sugarcane Production Modeling**

The repository evaluates whether phenology-aware weather features add information beyond strong temporal baselines. It does not assume that the Transformer must outperform persistence. A limited-period (2017-2024) sown-area and apparent-yield analysis is included as an exploratory sensitivity assessment.

## Contents

1. Phenology-stage weather feature engineering.
2. Previous-year persistence baseline.
3. Residual Ridge and residual SVR baselines.
4. PG-LiteTransformer residual model.
5. Expanding-window validation for 2013-2024.
6. Trend-weather ablation.
7. A 5000-permutation year-weather test.
8. Bootstrap prediction intervals.
9. Sown-area, production, and apparent-yield decomposition for 2017-2024.
10. Exploratory rolling yield sensitivity analysis for 2021-2024.

## Project structure

```text
.
├── config.json
├── config_test.json
├── requirements.txt
├── run_all.py
├── data/
│   ├── daily_weather_2005_2024.csv
│   ├── production_2005_2024.csv
│   ├── sown_area_2017_2024.csv
│   └── raw/
│       └── guangxi_statistics_A0303_chongzuo_sown_area_2017_2024.xlsx
├── src/
│   ├── area_sensitivity.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── plots.py
│   └── statistics.py
└── outputs/
```

All file paths are project-relative, and the code contains no machine-specific absolute paths.

## Installation

Python 3.10 or later is recommended. The included outputs were regenerated with Python 3.13.5; see `environment_tested.txt`.

```bash
python -m pip install -r requirements.txt
```

A Conda `base` environment can be used directly, although an isolated environment is safer for dependency management.

## Fast verification

```bash
python run_all.py --config config_test.json
```

The test configuration reduces the number of epochs, permutations, and bootstrap resamples. It validates the pipeline but is not intended to reproduce the manuscript's final numbers.

## Full reproduction

```bash
python run_all.py --config config.json
```

Results are written to `outputs/`.

## Expected primary results

The exact Transformer result can vary slightly with PyTorch and numerical-library versions. In the validated run included with this package:

- Persistence RMSE: approximately 70.011 x 10^4 t.
- Residual Ridge RMSE: approximately 89.673 x 10^4 t.
- Residual SVR RMSE: approximately 108.152 x 10^4 t.
- PG-LiteTransformer RMSE: approximately 85.004 x 10^4 t.
- Trend-only RMSE: approximately 165.292 x 10^4 t.
- Trend plus complete weather RMSE: approximately 143.205 x 10^4 t.
- Relative RMSE reduction over trend only: approximately 13.36%.
- 5000-permutation one-sided p-value: approximately 0.0392.

The persistence model remains the strongest overall production forecast. The main contribution is the leakage-free assessment framework and the evidence that complete phenology-aware weather information adds limited but statistically detectable information beyond a long-term trend.

## Area and apparent-yield sensitivity analysis

Sown-area records for 2017-2024 were obtained from the Guangxi Statistical Data Query System, table A0303, accessed on 24 July 2026:

https://gxsj.tjj.gxzf.gov.cn:18090/pub/advquery/advquery.htm?m=advquery&cn=A0303&selkey=cd75175af8854168918afd89cd203197

Apparent yield is calculated as:

```text
apparent_yield_t_ha = production_10k_t * 10 / sown_area_1000_ha
```

Because only eight area observations and four exploratory forecast years are available, these results must not be interpreted as an independent validation of a high-capacity model.

## Reproducibility notes

- The missing 2009 production value is excluded from the primary analysis.
- Every scaler is fitted within each historical training window.
- All core models use the same 2013-2024 evaluation years.
- Transformer seeds: 7, 19, and 31.
- Permutation seed: 20260721.
- Bootstrap seed: 20260721.
- No archived machine-specific execution logs are required for reproduction.

## License and data attribution

The MIT License applies to the source code. Statistical and reanalysis data remain subject to the terms and attribution requirements of their original providers.

## Maintainer

Shiqi Zhao, Guangxi Polytechnic University  
Correspondence: wxyaizsq@gmail.com
