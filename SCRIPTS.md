# SCRIPTS.md
Project root: `/Users/matthewplambeck/Desktop/Convoy Predictor`

This document is the practical map/runbook for the repository.

## 1) Canonical Workflow (Current)
If your goal is to reproduce and extend final model analysis, use this path:

1. Data source: `data/processed/Complete_Convoy_Data.csv`
2. Model artifacts: `artifacts/algorithm_test_3/*.joblib`
3. Results notebooks:
   - `notebooks/models/Results.ipynb` (core analysis + exports)
   - `notebooks/models/Results_Viz.ipynb` (final report visualizations)
4. Modular backend for results: `src/results/*.py`
5. Final outputs: `results/*.xlsx`, `results/*.png`, plus report content files in project root/docs

For Section 6 report writing:
- analysis narrative source: `results-analysis.md`
- webpage target: `docs/section6.html`

## 2) Directory Map
- `data/`
  - `raw/`: original convoy spreadsheets
  - `external/`: external references (U-boat data)
  - `processed/`: model-ready combined data
- `artifacts/`: persisted trained estimators + metadata
- `notebooks/`
  - `models/Results.ipynb`: final integrated analysis notebook
- `src/`
  - `results/`: modular analytics library used by results workflow
  - `models/`: model classes, tuning logic, artifact helpers
  - `data_cleaning/`: legacy preprocessing scripts
  - `scraping/`: legacy web-scraping scripts
  - `tests/`: exploratory/legacy scripts (not formal test suite)
- `results/`: generated result tables/figures
- `docs/`: static report website
- `Plots/`: earlier or duplicated plot exports

## 3) `src/results` Modules (Primary)

### Core loading and ensemble execution
- `src/results/model_loading_core.py`
  - Shared seed/split helpers and voting-ensemble evaluation.
- `src/results/ensemble_models.py`
  - Loads final base models and builds calibrated five-model soft-voting ensemble.

### Visualization and interpretability
- `src/results/visualization_functions.py`
  - ROC, confusion matrix, permutation importance, aggregated base importance, SHAP importance plots.
- `src/results/performance_panel_viz.py`
  - Operating-point panel (ROC, PR, confusion matrix, KPI block).
- `src/results/threshold_tradeoff_viz.py`
  - Threshold trade-off curve for recall/precision/accuracy/F1.
- `src/results/calibration_distribution_viz.py`
  - Calibration curve + class-wise probability distribution plot.
- `src/results/robustness_viz.py`
  - Temporal robustness heatmap by segment.
- `src/results/interpretability_viz.py`
  - Feature-rank bump chart and FN-focused SHAP delta insight view.
- `src/results/case_study_cards_viz.py`
  - TP/FN/FP case-study cards for report storytelling.

### Error cohort analysis
- `src/results/confusion_groups.py`
  - Builds FN/FP/TP/TN subsets and exports descriptive comparisons.

### Statistical inference
- `src/results/statistical_testing.py`
  - Kruskal global screens, Mann-Whitney pairwise tests, Cliff's delta, p-value correction, summary ranking.

### Calibration and thresholding
- `src/results/calibration_threshold_eval.py`
  - Calibration diagnostics, FN-threshold diagnostics, classification report table at fixed threshold, threshold sweeps, objective-based threshold selection, CV stability.

### Feature importance triangulation
- `src/results/feature_importance_triangulation.py`
  - Merges permutation/native/SHAP importance views, computes rank agreement and stability, performs FN-specific SHAP analysis.

### Segment and temporal robustness
- `src/results/segment_temporal_robustness.py`
  - Segment metrics, FN concentration, distribution-shift tests, segment-specific threshold stability.

### Leakage and data quality checks
- `src/results/leakage_data_quality_checks.py`
  - Leakage flags, split integrity checks, alignment audits, missingness/outlier diagnostics, risk summary.

### Notebook cell runbook
- `src/results/run_results.md`
  - Ordered notebook cells to run the full analysis stack.

## 4) `src/models` (Modeling Infrastructure)

### Active/core utilities
- `src/models/ML_Class_2.py`
  - Main model runner class with search strategies and threshold calibration.
- `src/models/model_specs.py`
  - Estimator registry + small/large parameter grids.
- `src/models/model_artifacts.py`
  - Save/load helpers for trained models.
- `src/models/perf_utils.py`
  - Runtime and resource instrumentation helpers.
- `src/models/cache_preprocessor.py`
  - Preprocessed split caching utility.

### Legacy/experimental scripts
- `src/models/ML_Class_1.py`
- `src/models/CNB_Tester.py`
- `src/models/QDA_Tester.py`
- `src/models/Gradient_Boosting_Optimization.py`
- `src/models/Model_Refiner_1.py`

Use these for historical context unless you explicitly need their behavior.

## 5) Data-Cleaning and Scraping Scripts (Legacy Pipeline)
These scripts were part of the original acquisition/prep workflow and contain hardcoded relative assumptions.

### Data preparation
- `src/data_cleaning/DataFrames.py`
- `src/data_cleaning/Convert_To_CSV.py`
- `src/data_cleaning/Inconsistency_Test.py`
- `src/data_cleaning/Compile_Data_Frames.py`

### Scraping
- `src/scraping/SC_Convoy_Web-Scrape.py`
- `src/scraping/HX_Convoy_Web-Scrape.py`
- `src/scraping/OB_Convoy_Web-Scrape.py`
- `src/scraping/ON_Convoy_Web-Scrape.py`
- `src/scraping/ONS_Convoy_Web-Scrape.py`
- `src/scraping/UBoat.net_Data.py`

## 6) Tests Folder Status
`src/tests/` currently holds exploratory scripts and visual demos rather than assertive, automated tests.

- `Test_1.py` to `Test_6.py`: ad hoc experiments/plots/PCA demos.

Recommendation:
- treat as notebooks-in-script-form, not CI-safe validation.

## 7) Inputs and Outputs

### Key input files
- `data/processed/Complete_Convoy_Data.csv`
- `artifacts/algorithm_test_3/*.joblib`

### Key output files
- `results/Threshold_Sweep.xlsx`
- `results/Threshold_Selections.xlsx`
- `results/Classification_Report_t0.25.xlsx`
- `results/Statistical_Testing_Summary.xlsx`
- `results/Feature_Triangulation_*.xlsx`
- `results/Segment_*.xlsx`
- `results/Leakage_Data_Quality_All_In_One.xlsx`
- `results/FiveModel_CalSoft_t0.25_*.png`

## 8) Documentation Pointers
- high-level project summary: `README.md`
- results module details: `src/results/README.md`
- section 6 narrative draft: `results-analysis.md`
- engineering/code review: `PROJECT_CODE_REVIEW.md`

## 9) Known Operational Caveats
1. Several modules still include absolute local paths and are not fully portable.
2. Environment dependencies are not pinned in a lock file.
3. End-to-end execution is notebook-driven (manual orchestration).
4. Formal automated tests are not yet implemented for core analytics modules.
