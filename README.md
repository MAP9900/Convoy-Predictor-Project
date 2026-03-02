# Convoy Predictor

## Overview
Convoy Predictor is an end-to-end historical analytics and machine learning project focused on WWII Atlantic convoy risk.

The repository contains:
- data ingestion and cleaning workflows,
- model training and artifact persistence,
- a modular results-analysis stack,
- report-ready outputs and a static documentation site.

## Current Project State
The most complete and current workflow is the final analysis pipeline in:
- `notebooks/models/Results.ipynb`
- `notebooks/models/Results_Viz.ipynb`
- `src/results/*.py`

This workflow loads saved models, evaluates the calibrated ensemble, runs deeper diagnostics (thresholding, statistical testing, feature triangulation, temporal robustness, and leakage/data-quality checks), and generates final report visualizations.

## Quick Navigation
- Script map/runbook: `SCRIPTS.md`
- Results module detail: `src/results/README.md`
- Findings narrative for report writing: `results-analysis.md`
- Repository-level engineering review: `PROJECT_CODE_REVIEW.md`

## Repository Layout
- `data/` - raw/external/processed data
- `artifacts/` - persisted model objects and metadata
- `notebooks/` - exploration, modeling, and visualization notebooks
- `src/` - code for data prep, modeling, results analysis, scraping, and legacy tests
- `results/` - exported analysis tables and figures
- `docs/` - static report website (`section6.html` is results section target)
- `Plots/` - historical and supplemental plots

## Core Inputs and Outputs

### Inputs
- Processed dataset: `data/processed/Complete_Convoy_Data.csv`
- Model artifacts: `artifacts/algorithm_test_3/*.joblib`

### Outputs
- Metrics/diagnostics tables in `results/*.xlsx`
- Figures in `results/*.png`
- Report prose assets in markdown/html under repo root and `docs/`
- Includes threshold operating-point exports such as `results/Classification_Report_t0.25.xlsx`

## Recommended Workflow
1. Use the existing processed data and artifacts.
2. Run/inspect `notebooks/models/Results.ipynb` (guided by `src/results/run_results.md`).
3. Run `notebooks/models/Results_Viz.ipynb` to generate final Section 6 figures.
4. Use `results-analysis.md` to populate `docs/section6.html`.
5. Use `PROJECT_CODE_REVIEW.md` and `SCRIPTS.md` for maintenance and refactoring priorities.

## Important Notes
- Some scripts still rely on machine-specific absolute paths.
- Dependency versions are not yet pinned in a lock file.
- `src/tests/` is exploratory and not a formal automated test suite.
