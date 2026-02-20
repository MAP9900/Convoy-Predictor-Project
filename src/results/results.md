# Results Workflow Additions

## Purpose
This document summarizes all additions implemented under `src/results/` from `Model_Loading.md` TODOs and follow-on analysis requests. The aim is to turn notebook-heavy analysis into reusable, modular scripts while preserving your existing output paths and notebook-driven execution style.

The additions support:
- Reproducible model loading and ensemble evaluation
- Plotting and interpretation utilities
- Confusion-group extraction and comparison
- Statistical testing across confusion groups
- Calibration and threshold quality analysis
- Feature-importance triangulation across multiple methods
- Segment/time robustness analysis
- Leakage and data-quality audits

## What Was Added

### 1) Core model loading and evaluation
- `src/results/model_loading_core.py`
- `src/results/ensemble_models.py`

Goal:
- Centralize shared split/seed setup and model tester preparation
- Load artifacted base models and run the calibrated five-model soft-voting ensemble

Key outputs:
- Ensemble metrics table
- Trained voting model object
- Confusion matrix/report objects

### 2) Visualization functions
- `src/results/visualization_functions.py`

Goal:
- Keep visualization logic separate and editable per plot (no monolithic runner)

Includes:
- ROC curve plot
- Confusion matrix plot
- Permutation importance plot
- Aggregated base-model importance plot
- SHAP importance plot

Note:
- Save paths are preserved to `/Users/matthewplambeck/Desktop/Convoy Predictor/results/...`

### 3) Confusion-group extraction and summary
- `src/results/confusion_groups.py`

Goal:
- Build FN/FP/TP/TN subsets from consistent split definitions
- Compare descriptive statistics across groups

Includes:
- FN/FP/TP/TN extraction functions
- `compare_confusion_group_describes(...)`

### 4) Statistical testing pipeline (nonparametric)
- `src/results/statistical_testing.py`

Goal:
- Quantify feature differences across confusion groups with corrected significance

Includes:
- Global Kruskal–Wallis feature screen
- Pairwise Mann–Whitney tests
- Manual Cliff's delta + magnitude labels
- FDR (Benjamini–Hochberg) and Holm corrections
- Ranked summary table builder
- Conditional targeted post-hoc testing on globally significant features

### 5) Calibration and threshold-quality evaluation
- `src/results/calibration_threshold_eval.py`

Goal:
- Evaluate probability calibration quality and operating-threshold behavior

Includes:
- Brier score and baseline comparison
- Calibration curve plotting (pre/post optional)
- FN probability diagnostics near threshold
- Threshold sweep metrics
- Objective-based threshold selection:
  - `min_fn_bounded_fp`
  - `max_mcc_with_recall_constraint`
  - `max_bal_acc_with_recall_constraint`
- CV threshold stability analysis

### 6) Feature importance triangulation
- `src/results/feature_importance_triangulation.py`

Goal:
- Compare feature-importance rankings from multiple viewpoints and detect disagreement

Includes:
- Ensemble permutation importance
- Aggregated native base-model importance (mean/var/CV)
- Tree-model SHAP aggregation (lazy SHAP import)
- Normalized multi-method comparison table
- Rank agreement (Spearman)
- Stable vs unstable feature identification
- FN-specific SHAP delta analysis
- One-call full report builder

### 7) Segment and temporal robustness
- `src/results/segment_temporal_robustness.py`

Goal:
- Validate whether model behavior shifts by time period or route segment

Includes:
- Time segment creation (`Year`, `YearMonth`, `EarlyLate`)
- Segment-wise performance metrics and confusion counts
- FN distribution by segment
- Covariate shift tests:
  - Early vs Late (Mann–Whitney + Cliff's delta)
  - Multi-group (Kruskal)
- Segment-wise threshold stability and material deviation flags
- Optional matplotlib plots (no seaborn)

### 8) Leakage and data-quality checks
- `src/results/leakage_data_quality_checks.py`

Goal:
- Surface leakage risks and dataset integrity issues before relying on results

Includes:
- Leakage flagging by column-name patterns + optional metadata
- Split integrity checks (duplicates, ID overlap, convoy/group overlap)
- Alignment audits for `X`, `y`, and optional raw frames
- Missingness-by-confusion-group tests with p-value correction
- Train-derived outlier bounds + outlier concentration by group
- Preprocessing audit heuristics (train-only fit consistency checks)
- Ranked risk summary table for reporting/prioritization

## Notebook Execution Integration
- `src/results/run_results.md` now contains executable notebook cells for all modules.
- Cells are organized into phases:
  1. Core load/evaluate/plot
  2. Confusion-group tables
  3. Statistical testing
  4. Calibration + threshold tuning
  5. Importance triangulation
  6. Segment/time robustness
  7. Leakage + data quality checks

## Documentation Updates
- `SCRIPTS.md` was updated with a **Results workflows** section that lists and describes each added results script.

## Design Principles Used
- Keep functions simple and composable
- Prefer explicit inputs/outputs over hidden state
- Preserve your existing absolute save paths
- Add comments/docstrings for maintainability
- Make plots optional where analysis can run headless
- Handle NaNs and small-sample edge cases defensively

## Known Limitations and Heuristics
- Leakage and preprocessing checks are best-effort heuristics, not formal proofs.
- Preprocessing audit cannot always prove train-only fitting without training logs/artifacts.
- SHAP support depends on installed package and tree-compatible estimators.
- Some segment/time analyses require enough segment sample size (`min_segment_n`, `min_pos_n`).
- Statistical significance depends on group counts and missingness patterns.

## Suggested Usage Pattern
1. Run base workflow cells in `run_results.md` through model scoring.
2. Run diagnostic modules in order (stats -> threshold -> triangulation -> segment robustness -> leakage quality).
3. Export generated tables to `/results` and use them for reporting.
4. Treat `risk_summary` as a gating checklist before final conclusions.

## File Inventory (Results)
- `src/results/model_loading_core.py`
- `src/results/ensemble_models.py`
- `src/results/visualization_functions.py`
- `src/results/confusion_groups.py`
- `src/results/statistical_testing.py`
- `src/results/calibration_threshold_eval.py`
- `src/results/feature_importance_triangulation.py`
- `src/results/segment_temporal_robustness.py`
- `src/results/leakage_data_quality_checks.py`
- `src/results/run_results.md`
- `src/results/results.md` (this file)
