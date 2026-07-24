# New GitHub Repository Checklist

1. Create a new public repository, preferably named `chongzuo-sugarcane-weather-assessment`.
2. Upload the contents of this folder, not the outer ZIP folder.
3. Confirm that `README.md` is visible on the repository homepage.
4. Run `python run_all.py --config config_test.json` after downloading a fresh copy.
5. Run the full command and compare `outputs/rolling_metrics.csv`, `outputs/permutation_summary.json`, and `outputs/yield_sensitivity_metrics.csv` with the included files.
6. Create a release tagged `v2.0.0` after verification.
7. Add the final repository URL to `CITATION.cff` and to the manuscript Data Availability Statement.
8. Do not upload local logs, `.DS_Store`, virtual environments, or `__pycache__` directories.
