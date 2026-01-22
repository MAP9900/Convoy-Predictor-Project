# Project Map
Project root: /Users/matthewplambeck/Desktop/Convoy Predictor

- `Plots/` — Generated figures and model output visuals.
- `artifacts/` — Saved model artifacts (joblib/json).
- `data/` — Raw, external, and processed datasets.
- `docs/` — Project report site (HTML/CSS) and images.
- `misc/` — Miscellaneous assets.
- `notebooks/` — Exploration, modeling, and visualization notebooks.
- `src/` — Python source scripts (scraping, data prep, models, tests).
- `Python.gitignore` — Git ignore template.

# Scripts

## Data cleaning and preparation
- `src/data_cleaning/DataFrames.py` — Cleans raw convoy Excel files and computes per-convoy aggregates (ships, sunk, tons, escorts, stragglers), writing transformed Excel outputs.
- `src/data_cleaning/Convert_To_CSV.py` — Merges transformed convoy data with date sheets, standardizes date formats, and writes CSVs; includes a U-boat monthly data table.
- `src/data_cleaning/Inconsistency_Test.py` — Compares convoy sink counts against UBoat.net data and surfaces mismatches for manual correction.
- `src/data_cleaning/Compile_Data_Frames.py` — Combines route-level datasets into a single CSV and engineers derived features (sink percentages, escort ratio, time at sea, historical rates, etc.).

## Web scraping
- `src/scraping/SC_Convoy_Web-Scrape.py` — Scrapes SC convoy tables and dates from convoyweb.org.uk.
- `src/scraping/HX_Convoy_Web-Scrape.py` — Scrapes HX convoy tables and dates from convoyweb.org.uk.
- `src/scraping/OB_Convoy_Web-Scrape.py` — Scrapes OB convoy tables and dates from convoyweb.org.uk.
- `src/scraping/ON_Convoy_Web-Scrape.py` — Scrapes ON convoy tables and dates from convoyweb.org.uk.
- `src/scraping/ONS_Convoy_Web-Scrape.py` — Scrapes ONS convoy tables and dates from convoyweb.org.uk.
- `src/scraping/UBoat.net_Data.py` — Scrapes convoy sink totals from uboat.net into a dataframe.

## Modeling and evaluation
- `src/models/ML_Class_1.py` — Baseline model testing class with K-fold, grid search, and evaluation plots.
- `src/models/ML_Class_2.py` — Expanded model testing class with additional optimization hooks and configurable scoring.
- `src/models/Gradient_Boosting_Optimization.py` — Gradient boosting optimizer with threshold calibration and optional plot saving.
- `src/models/CNB_Tester.py` — Complement Naive Bayes workflow with calibration, grid search, and evaluation utilities.
- `src/models/Model_Refiner_1.py` — Experimental gradient-descent model refiner (noted as untested/optional).
- `src/models/model_specs.py` — Model registry with estimator definitions and tuned parameter grids.
- `src/models/model_artifacts.py` — Save/load helpers for trained models and metadata.
- `src/models/perf_utils.py` — Performance-tracking decorator for timing, CPU, and memory.
- `src/models/cache_preprocessor.py` — Utility to cache train/test/val splits for faster model runs.

## Tests and exploratory scripts
- `src/tests/Test_1.py` — Mixed exploratory analysis and KNN cross-validation (includes plotting and stats).
- `src/tests/Test_2.py` — SC convoy time-series plots and moving averages.
- `src/tests/Test_3.py` — SC convoy trend plots plus correlation and binning analysis.
- `src/tests/Test_4.py` — Learning-curve plotting helper.
- `src/tests/Test_5.py` — U-boat data table export and PCA/classification experiments.
- `src/tests/Test_6.py` — PCA 3D visualization demo on synthetic data.
