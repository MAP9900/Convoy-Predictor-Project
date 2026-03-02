# `src/results` README

Purpose: this folder is the modular backend for the final analysis workflow used in `notebooks/models/Results.ipynb`.

It turns the notebook analysis into reusable functions for:
- model loading and ensemble scoring,
- error cohort analysis (FN/FP/TP/TN),
- statistical testing,
- calibration and threshold tuning,
- feature-importance triangulation,
- temporal/segment robustness,
- leakage and data-quality audits.

A dedicated visualization notebook is also available at `notebooks/models/Results_Viz.ipynb` for Section 6 report figures.

## Canonical Inputs
- Processed data: `data/processed/Complete_Convoy_Data.csv`
- Saved models: `artifacts/algorithm_test_3/*.joblib`

## Canonical Outputs
Generated into `results/` (repo root), including:
- performance plots (`*_CM.png`, `*_Permutation_Importance.png`, etc.)
- threshold diagnostics (`Threshold_Sweep.xlsx`, `Threshold_Selections.xlsx`)
- statistical outputs (`Kruskal_Global_Screen.xlsx`, `MWU_*.xlsx`)
- triangulation outputs (`Feature_Triangulation_*.xlsx`)
- segment robustness outputs (`Segment_*.xlsx`)
- leakage/data-quality outputs (`Leakage_Data_Quality_All_In_One.xlsx`)

## Module Guide

### `model_loading_core.py`
Shared utilities for reproducibility and evaluation:
- seed setup
- train/test split helper
- tester preparation from `MODEL_SPECS`
- voting ensemble evaluation function

### `ensemble_models.py`
Loads the final base models and runs the calibrated five-model soft-voting ensemble (`t=0.25`).

### `visualization_functions.py`
Plotting functions for:
- ROC curve
- confusion matrix
- permutation importance
- aggregated base-model importance
- SHAP importance

### `performance_panel_viz.py`
Final operating-point panel:
- ROC with threshold marker
- PR with threshold marker
- confusion matrix
- KPI summary block

### `threshold_tradeoff_viz.py`
Threshold trade-off line chart for recall, precision, accuracy, and F1.

### `calibration_distribution_viz.py`
Calibration and probability diagnostics:
- reliability curve
- class-wise probability histogram with threshold line

### `robustness_viz.py`
Temporal robustness heatmap across segment metrics.

### `interpretability_viz.py`
Section-6 interpretability visuals:
- feature-rank bump chart across methods
- FN-focused SHAP delta insight view

### `case_study_cards_viz.py`
Storytelling cards for representative TP/FN/FP convoy cases.

### `confusion_groups.py`
Builds scored test-row subsets for:
- false negatives
- false positives
- true positives
- true negatives

Also includes grouped descriptive comparison helpers.

### `statistical_testing.py`
Nonparametric testing pipeline:
- global Kruskal screens
- pairwise Mann-Whitney tests
- Cliff's delta and rank-biserial effect sizes
- multiple-testing adjustment (`fdr_bh`, `holm`)
- ranked summary tables

### `calibration_threshold_eval.py`
Calibration and operating-threshold diagnostics:
- Brier score report
- calibration curve support
- FN probability diagnostics near threshold
- classification report table at chosen threshold
- threshold sweep and objective-based selection
- optional CV threshold stability

### `feature_importance_triangulation.py`
Integrates three importance perspectives:
- permutation importance
- native base-model importance
- SHAP importance

Adds:
- rank agreement,
- stable/unstable feature flags,
- FN-specific SHAP delta analysis,
- one-call consolidated triangulation report.

### `segment_temporal_robustness.py`
Temporal and segment reliability checks:
- segment creation (`Year`, `YearMonth`, `EarlyLate`)
- per-segment metrics and FN distribution
- distribution-shift testing
- per-segment threshold stability

### `leakage_data_quality_checks.py`
Audit layer for leakage and data quality:
- leakage flagging by feature names/metadata
- split integrity checks
- alignment checks for X/y/raw frames
- missingness and outlier concentration by confusion group
- preprocessing audit heuristics
- ranked risk summary

## Recommended Run Order
Use `run_results.md` (or notebook equivalents) in this order:
1. Load data, split, load models, run ensemble.
2. Baseline plots + confusion-group extraction.
3. Statistical testing.
4. Calibration + threshold diagnostics.
5. Feature importance triangulation.
6. Segment/temporal robustness.
7. Leakage/data quality audit.

For final report visuals, run `notebooks/models/Results_Viz.ipynb` after exports are available.

## Related Docs
- Top-level runbook: `SCRIPTS.md`
- Project overview: `README.md`
- Results narrative for Section 6: `results-analysis.md`
- Repo engineering review: `PROJECT_CODE_REVIEW.md`

## Current Caveats
1. Some modules still use machine-specific absolute paths for defaults.
2. The execution pattern is notebook-driven, not yet one-click script orchestration.
3. `src/tests/` is exploratory and does not yet provide automated regression coverage for these modules.
