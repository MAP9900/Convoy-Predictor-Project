# Convoy Predictor

## Overview
Convoy Predictor is a data and modeling project focused on WWII Atlantic convoy operations. The repository includes
data ingestion and cleaning utilities, feature engineering pipelines, modeling experiments, and reporting assets.

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

## Notes
- Some scripts assume local paths and ChromeDriver availability.
- Several scripts are exploratory or one-off; `SCRIPTS.md` clarifies their intent.
