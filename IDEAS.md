# Project Next Steps

## Core ML Goals (predict risk, optimize, explain)
- Define the prediction target clearly (e.g., any ships sunk vs. high-loss threshold) and document it in `README.md`.
- Establish a single training dataset pipeline (raw → cleaned → features) with a versioned output in `data/processed/`.
- Lock in an evaluation protocol: stratified splits + time-based split to test historical generalization.
- Expand model benchmarking to include calibrated probability models and threshold tuning for recall vs. precision trade-offs.
- Add model interpretability outputs (SHAP/perm importance) and tie them to domain narrative.

## Feature Discovery and Risk Drivers
- Run permutation importance across top models to identify robust risk predictors.
- Test interaction effects (e.g., escort ratio x time at sea, convoy size x U-Boat presence).
- Build time-lagged features (previous month sink rates, seasonal effects) and compare impact.
- Add domain-driven features: convoy route class, war-year phase, or escort types if available.

## Data and Pipeline Expansion
- Automate scraping + ingestion into a repeatable pipeline (with caching of raw sources).
- Create a data dictionary and feature catalog for the processed dataset.
- Add quality checks: missingness reports, outlier detection, and schema validation.

## Visualization and Storytelling
- Build a “risk over time” dashboard with key drivers and historical events annotated.
- Add route-level comparisons: SC vs HX vs OB vs ON vs ONS risk trends and size distributions.
- Map convoy routes and overlay risk or sink counts for spatial storytelling.
- Create a one-page model report with key metrics, plots, and top features.

## Other Additions
- Package a reproducible “runbook” (inputs, outputs, and expected artifacts).
- Add a short demo notebook that walks from data → model → insights in <10 minutes.
