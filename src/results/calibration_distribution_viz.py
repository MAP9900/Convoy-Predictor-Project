"""Calibration and probability-distribution visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def plot_calibration_probability_distribution(
    y_true,
    y_proba,
    threshold=0.25,
    n_bins=10,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot reliability curve and class-wise probability distributions."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    frac_pos, mean_pred = calibration_curve(y_true_arr, y_proba_arr, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true_arr, y_proba_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="lightgrey")
    ax_cal, ax_hist = axes

    # Calibration / reliability curve
    ax_cal.set_facecolor("lightgrey")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax_cal.plot(mean_pred, frac_pos, marker="o", color="#06768d", linewidth=2, label=f"Model (Brier={brier:.3f})")
    ax_cal.set_title("Calibration Curve")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Observed Positive Rate")
    ax_cal.set_xticks(np.arange(0, 1.01, 0.1))
    ax_cal.set_yticks(np.arange(0, 1.01, 0.1))
    ax_cal.grid(True, linestyle="--", alpha=0.5)
    ax_cal.legend(loc="upper left")

    # Probability distribution by class
    ax_hist.set_facecolor("lightgrey")
    proba_neg = y_proba_arr[y_true_arr == 0]
    proba_pos = y_proba_arr[y_true_arr == 1]

    bins = np.linspace(0, 1, 21)
    ax_hist.hist(proba_neg, bins=bins, alpha=0.65, color="#3f8a06", label="True Class 0")
    ax_hist.hist(proba_pos, bins=bins, alpha=0.65, color="#ab0003", label="True Class 1")
    ax_hist.axvline(x=threshold, linestyle="--", linewidth=2, color="#06768d", label=f"Threshold={threshold:.2f}")

    ax_hist.set_title("Predicted Probability Distribution by True Class")
    ax_hist.set_xlabel("Predicted Probability (Class 1)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_hist.legend(loc="upper right")

    for ax in [ax_cal, ax_hist]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_Calibration_Probability_Distribution.png")
    plt.show()

    return {
        "brier": brier,
        "threshold": threshold,
        "n_class0": int((y_true_arr == 0).sum()),
        "n_class1": int((y_true_arr == 1).sum()),
    }
