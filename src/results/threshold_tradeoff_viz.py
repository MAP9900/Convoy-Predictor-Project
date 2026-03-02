"""Threshold trade-off visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def plot_threshold_tradeoff_curve(
    sweep_df,
    threshold_col="threshold",
    recall_col="recall1",
    precision_col="precision1",
    accuracy_col="accuracy",
    f1_col="f1",
    marker_thresholds=(0.25, 0.21, 0.19),
    title="Final Ensemble Metrics vs Decision Threshold",
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,):
    """Plot recall, precision, accuracy, and F1 across decision thresholds."""
    df = sweep_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="lightgrey")
    ax.set_facecolor("lightgrey")

    ax.plot(df[threshold_col], df[recall_col], label="Recall", marker="o", color="#0398fc")
    ax.plot(df[threshold_col], df[precision_col], label="Precision", marker="s", color="#fc6f03")
    ax.plot(df[threshold_col], df[accuracy_col], label="Accuracy", marker="^", color="#3f8a06")
    ax.plot(df[threshold_col], df[f1_col], label="F1-score", marker="d", color="#ab0003")

    for t in marker_thresholds:
        ax.axvline(x=t, linestyle="--", linewidth=1, color="grey", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_xticks(np.round(np.arange(df[threshold_col].min(), df[threshold_col].max() + 0.001, 0.01), 2))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}Threshold_Tradeoff_Plot.png", bbox_inches='tight', dpi=300)
    # plt.savefig(f"{results_dir}/{model_name}_Threshold_Tradeoff_Curve.png")
    plt.show()

    return ax
