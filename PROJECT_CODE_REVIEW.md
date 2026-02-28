# Project Code Review

Scope: repository-wide engineering review with emphasis on reproducibility, maintainability, and correctness for the current model/results workflow.

Date: 2026-02-28

## Findings (Prioritized)

### P1: Reproducibility depends on machine-specific absolute paths
Files under `src/results/` and parts of `src/models/` hardcode absolute paths like:
- `/Users/matthewplambeck/Desktop/Convoy Predictor/results`
- `/Users/matthewplambeck/Desktop/Convoy Predictor/data/...`

Impact:
- Breaks portability on any other machine or CI runner.
- Makes it hard to package or automate end-to-end runs.

Recommendation:
- Centralize path resolution in one config module using `Path(__file__).resolve().parents[...]`.
- Accept optional environment overrides (`CONVOY_PROJECT_ROOT`, `CONVOY_RESULTS_DIR`).
- Treat all output directories as parameters, not constants.

### P1: Runtime/dependency contract is undocumented and likely incomplete
No active dependency lock file (`requirements.txt`, `pyproject.toml`, or `environment.yml`) is present.

Impact:
- Fresh environment setup is uncertain.
- Re-running historical notebooks may fail due to version drift.

Recommendation:
- Add a pinned environment file.
- Include optional groups for heavy/optional libs (`shap`, `xgboost`, `lightgbm`, `catboost`, `selenium`).

### P1: Pipeline has no automated regression tests for core results modules
`src/tests/` is mostly exploratory plotting scripts, not assertions over behavior.

Impact:
- Refactors can silently break metric calculations, leakage checks, or threshold selection.
- Hard to trust future changes.

Recommendation:
- Add unit tests for:
  - threshold sweep/selection invariants,
  - confusion-group partition counts,
  - statistical correction functions,
  - leakage/split-integrity checks,
  - segment-threshold stability outputs.
- Add a small synthetic fixture dataset for deterministic testing.

### P2: Two parallel code tracks increase maintenance cost
There are two model classes (`ML_Class_1.py`, `ML_Class_2.py`) and mixed older scripts with newer modular results code.

Impact:
- Duplicated logic and naming drift.
- New contributors cannot quickly identify canonical entrypoints.

Recommendation:
- Declare canonical path: `src/results/*` + `Model_Tester_V2`.
- Mark legacy modules as archived in docs and optionally move to `src/legacy/`.

### P2: Data pipeline scripts are script-style and rely on relative legacy paths
Many `src/data_cleaning/` and `src/scraping/` scripts still assume `Excel_Files/...` style paths and manual post-processing steps.

Impact:
- Hard to execute end-to-end from clean checkout.
- Requires manual working-directory assumptions.

Recommendation:
- Introduce argparse-based CLIs with explicit `--input`/`--output`.
- Add one orchestrator script or makefile for deterministic data build.

### P2: Results notebook/reporting flow is strong, but orchestration is manual
`notebooks/models/Results.ipynb` and `src/results/run_results.md` are comprehensive, but execution remains notebook-cell driven.

Impact:
- Repeatability depends on user discipline and notebook state.

Recommendation:
- Add one script entrypoint (for example `src/results/run_full_results.py`) that runs all stages and writes standard artifacts.

### P3: Plot naming and titles have minor inconsistencies
Example: ROC plot function saves `*_PR_Curve.png` in one location.

Impact:
- Confusing artifact naming and downstream doc links.

Recommendation:
- Standardize artifact naming conventions (`ROC`, `PR`, `CM`, etc.).

### P3: Warnings and style debt remain in visualization layer
Seaborn deprecation warnings appear in plotting output (`palette` usage pattern).

Impact:
- Noise and future break risk on library upgrade.

Recommendation:
- Update seaborn calls (`hue`/`legend=False` pattern) and run a style/lint pass.

## Strengths
- Clear modularization of the final analysis pipeline under `src/results/`.
- Good analytical coverage: calibration, thresholding, stats testing, feature triangulation, segment drift, leakage checks.
- Artifact outputs are comprehensive and report-ready.
- `model_artifacts.py` cleanly handles persisted estimators and metadata.

## Architecture Snapshot
- Data sources: `data/raw/`, `data/external/`
- Processed dataset: `data/processed/Complete_Convoy_Data.csv`
- Training/eval abstractions: `src/models/`
- Final analysis modules: `src/results/`
- Notebook orchestration: `notebooks/models/Results.ipynb`
- Report site: `docs/` (notably `section6.html`)

## Recommended Refactor Roadmap

### Phase 1 (High ROI, low risk)
1. Add pinned environment file and setup instructions.
2. Centralize path config and remove absolute path constants from results modules.
3. Add smoke tests for critical functions.

### Phase 2 (Medium effort)
1. Introduce script-based results runner.
2. Convert key data-cleaning scripts to parameterized CLI tools.
3. Standardize artifact naming and metadata.

### Phase 3 (Longer-term hardening)
1. Consolidate legacy vs canonical modules.
2. Add CI checks (lint + tests + minimal notebook validation).
3. Add model monitoring protocol (drift thresholds, retraining cadence, threshold governance).

## Suggested Technical Debt Backlog
1. Path abstraction/config module.
2. Dependency lock file.
3. Core unit test suite.
4. Results orchestration script.
5. Legacy module segregation.
6. Data pipeline CLI conversion.
7. Plot API modernization.

## Bottom Line
The project has strong analytical depth and meaningful modular progress. The largest remaining risks are operational: environment drift, path portability, and insufficient automated verification. Addressing those will make the current results pipeline reproducible and maintainable for future iterations.
