# Project Map
Project root: /Users/matthewplambeck/Desktop/Convoy Predictor

- `Plots/` — Generated figures and model output visuals.
- `artifacts/` — Saved model artifacts (joblib/json).
- `data/` — Raw, external, and processed datasets.
- `docs/` — Project report site (HTML/CSS) and images.
- `misc/` — Miscellaneous assets.
- `notebooks/` — Exploration, modeling, and visualization notebooks.
- `src/` — Python source scripts (scraping, data prep, models, tests).

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

## Results workflows
- `src/results/model_loading_core.py` — Shared seed/split utilities, model-tester preparation, and voting-ensemble evaluation metrics.
- `src/results/ensemble_models.py` — Loads saved base models and runs the calibrated five-model soft-voting ensemble at threshold 0.25.
- `src/results/visualization_functions.py` — ROC/confusion plotting plus permutation, aggregated, and SHAP feature-importance plot functions.
- `src/results/confusion_groups.py` — Builds FN/FP/TP/TN slices from scored test rows and compares grouped `describe()` statistics.
- `src/results/statistical_testing.py` — Nonparametric feature-testing pipeline (global Kruskal, pairwise Mann-Whitney, Cliff's delta, FDR/Holm correction, and ranked summary table).
- `src/results/calibration_threshold_eval.py` — Calibration quality and operating-threshold evaluation utilities (Brier/calibration curve, FN probability diagnostics, threshold sweep/selection, and CV threshold stability).
- `src/results/feature_importance_triangulation.py` — Multi-view feature-importance triangulation (permutation, aggregated base-model importance, SHAP, rank agreement, stability flags, and FN-specific SHAP analysis).
- `src/results/segment_temporal_robustness.py` — Segment/time robustness evaluation (time bins, segment metrics, FN concentration, shift testing, and per-segment threshold stability).
- `src/results/leakage_data_quality_checks.py` — Leakage/data-quality audit utilities (column leakage flags, split/alignment checks, missingness/outlier diagnostics by confusion group, preprocessing audit, and ranked risk summary).
- `src/results/run_results.md` — Notebook-ready execution cells (imports + function calls) for the full results workflow.
- `src/results/results.md` — Detailed narrative summary of all results-module additions, goals, execution flow, assumptions, and caveats.

## Tests and exploratory scripts
- `src/tests/Test_1.py` — Mixed exploratory analysis and KNN cross-validation (includes plotting and stats).
- `src/tests/Test_2.py` — SC convoy time-series plots and moving averages.
- `src/tests/Test_3.py` — SC convoy trend plots plus correlation and binning analysis.
- `src/tests/Test_4.py` — Learning-curve plotting helper.
- `src/tests/Test_5.py` — U-boat data table export and PCA/classification experiments.
- `src/tests/Test_6.py` — PCA 3D visualization demo on synthetic data.

# Notebooks

## Exploration notebooks
- `notebooks/exploration/Algorithm_Test_1.ipynb` — Early algorithm comparison experiments.
- `notebooks/exploration/Algorithm_Test_2.ipynb` — Follow-up algorithm testing and metric checks.
- `notebooks/exploration/Algorithm_Test_3.ipynb` — Later algorithm benchmarking and tuning trials.
- `notebooks/exploration/CNB_and_QDA_Test.ipynb` — Complement Naive Bayes vs QDA evaluation.
- `notebooks/exploration/Classification_Test_1.ipynb` — Initial classification workflow experiments.
- `notebooks/exploration/Classification_Test_2.ipynb` — Extended classification testing.
- `notebooks/exploration/Ensemble_Learning_Test.ipynb` — Ensemble model experiments.
- `notebooks/exploration/Regression_Test_1.ipynb` — Initial regression analysis.
- `notebooks/exploration/Regression_Test_2.ipynb` — Follow-up regression experiments.
- `notebooks/exploration/XGBRFClassifier_Test_1.ipynb` — XGBRF classifier-specific testing.

## Modeling notebooks
- `notebooks/models/Convoy_Feature_Importance.ipynb` — Feature-importance analysis for trained models.
- `notebooks/models/GB_Model_Vis.ipynb` — Gradient boosting model visualization.
- `notebooks/models/Results.ipynb` — Consolidated model result review.
- `notebooks/models/Results_OLD.ipynb` — Archived/previous results notebook.

## Visualization notebooks
- `notebooks/visualization/Convoy_Data_Vis.ipynb` — General convoy dataset visualization.
- `notebooks/visualization/Convoy_Routes_Viz.ipynb` — Convoy route-focused plots/maps.
- `notebooks/visualization/HX_Plots.ipynb` — HX route visualization notebook.
- `notebooks/visualization/OB_Plots.ipynb` — OB route visualization notebook.
- `notebooks/visualization/SC_Plots.ipynb` — SC route visualization notebook.
