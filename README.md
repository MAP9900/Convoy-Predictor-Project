# Convoy Predictor

## Overview
Convoy Predictor is an end-to-end data science project centered on WWII Atlantic convoy operations. It covers
data acquisition, cleaning, feature engineering, modeling, and reporting, with a focus on predicting convoy risk
and analyzing historical drivers of sinkings.

## Highlights
- End-to-end WWII convoy dataset built from multiple routes (SC, HX, OB, ON, ONS) plus external U-Boat sources.
- Data integrity checks cross-validate convoy sink counts against UBoat.net records, with mismatch reports for manual fixes.
- Feature engineering adds operational context such as escort ratio, time at sea, monthly historical sink rates, and U-Boat presence.
- Modeling bench spans classic ML baselines and ensembles (e.g., QDA, Complement Naive Bayes, Gradient Boosting) with reusable evaluation utilities.
- Interpretability-first outputs: ROC/PR curves, confusion matrices, and feature importance plots for decision support.
- Reproducible artifacts saved to `artifacts/` with metadata, plus static report assets in `docs/`.

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
