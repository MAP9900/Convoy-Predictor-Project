# Convoy Predictor

## Overview
Convoy Predictor is an end-to-end data science project centered on WWII Atlantic convoy operations. It covers
data acquisition, cleaning, feature engineering, modeling, and reporting, with a focus on predicting convoy risk
and analyzing historical drivers of sinkings.

## Highlights
- Full pipeline ownership: scraping raw data, validating inconsistencies, engineering features, and training models.
- Multi-model experimentation with reusable evaluation utilities and saved artifacts for reproducibility.
- Practical focus on interpretability and decision thresholds, with plots for ROC/PR and feature importance.
- Strong data hygiene: cross-source consistency checks and curated processed datasets.
- Clear project organization: scripts, notebooks, reports, and artifacts separated by purpose.

## Project Map
See `SCRIPTS.md` for a concise map of the top-level folders and the purpose of each script.

## Data
- `data/raw/` — Raw convoy datasets (Excel/CSV).
- `data/external/` — External reference datasets (e.g., U-Boat data).
- `data/processed/` — Cleaned and compiled CSV outputs.

## Notebooks
- `notebooks/exploration/` — Early exploration and algorithm tests.
- `notebooks/models/` — Model-focused analysis and visuals.
- `notebooks/visualization/` — Plotting and storytelling notebooks.

## Scripts
- `src/scraping/` — Selenium-based scrapers for convoy sources.
- `src/data_cleaning/` — Data preparation and feature engineering utilities.
- `src/models/` — Model training, evaluation, and artifact helpers.
- `src/tests/` — Exploratory analysis scripts and ad hoc tests.

## Outputs
- `Plots/` — Generated plots and figures.
- `artifacts/` — Saved model artifacts and metadata.
- `docs/` — Static report site assets.

## Findings (Still need to add!)
- 

## Notes
- Some scripts assume local paths and ChromeDriver availability.
- Several scripts are exploratory or one-off; `SCRIPTS.md` clarifies their intent.
