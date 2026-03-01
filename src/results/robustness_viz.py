"""Temporal robustness visualization helpers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def plot_temporal_robustness_heatmap(
    segment_metrics_df,
    segment_type="Year",
    metrics=("recall1", "fpr", "mcc", "bal_acc"),
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot a heatmap of key metrics across temporal segments."""
    df = segment_metrics_df.copy()
    df = df[df["segment_type"].astype(str) == str(segment_type)].copy()

    if df.empty:
        raise ValueError(f"No rows found for segment_type='{segment_type}'.")

    keep_cols = ["segment_value"] + list(metrics)
    heat_df = df[keep_cols].copy()

    # Robust sort for year-like labels.
    heat_df["segment_sort"] = pd.to_numeric(heat_df["segment_value"], errors="coerce")
    heat_df = heat_df.sort_values(["segment_sort", "segment_value"], na_position="last").drop(columns=["segment_sort"])

    heat_df = heat_df.set_index("segment_value")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="lightgrey")
    ax.set_facecolor("lightgrey")

    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=ax,
    )

    ax.set_title(f"Temporal Robustness Heatmap ({segment_type})")
    ax.set_xlabel("Metric")
    ax.set_ylabel(segment_type)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_Temporal_Robustness_Heatmap_{segment_type}.png")
    plt.show()

    return heat_df
