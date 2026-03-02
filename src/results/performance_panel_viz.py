"""Performance visualization helpers for final ensemble reporting."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def _safe_div(num, den):
    if den == 0:
        return np.nan
    return num / den


def plot_operating_point_panel(
    y_true,
    y_proba,
    threshold=0.25,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot an operating-point panel with ROC, PR, confusion matrix, and KPI text."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_proba_arr)
    roc_auc = roc_auc_score(y_true_arr, y_proba_arr)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_arr, y_proba_arr)
    pr_auc = auc(recall_curve, precision_curve)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    recall_1 = recall_score(y_true_arr, y_pred, pos_label=1)
    precision_1 = precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true_arr, y_pred)
    mcc = matthews_corrcoef(y_true_arr, y_pred)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred)
    fpr_at_threshold = _safe_div(fp, fp + tn)

    roc_idx = int(np.argmin(np.abs(roc_thresholds - threshold)))

    pr_idx = 0
    if pr_thresholds.size > 0:
        pr_idx = int(np.argmin(np.abs(pr_thresholds - threshold)))

    cm_counts = np.array([[tn, fp], [fn, tp]], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="lightgrey")
    ax_roc, ax_pr = axes[0, 0], axes[0, 1]
    ax_cm, ax_text = axes[1, 0], axes[1, 1]

    # ROC
    ax_roc.set_facecolor("lightgrey")
    ax_roc.plot(fpr, tpr, color="#06768d", linewidth=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax_roc.scatter(
        fpr[roc_idx],
        tpr[roc_idx],
        color="#ab0003",
        s=70,
        zorder=5,
        label=f"t={threshold:.2f}",
    )
    ax_roc.set_title("ROC Curve with Operating Threshold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, linestyle="--", alpha=0.5)
    ax_roc.legend(loc="lower right")

    # PR
    ax_pr.set_facecolor("lightgrey")
    ax_pr.plot(recall_curve, precision_curve, color="#06768d", linewidth=2, label=f"PR (AUC={pr_auc:.3f})")
    ax_pr.scatter(
        recall_curve[pr_idx],
        precision_curve[pr_idx],
        color="#ab0003",
        s=70,
        zorder=5,
        label=f"t={threshold:.2f}",
    )
    ax_pr.set_title("Precision-Recall Curve with Operating Threshold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.5)
    ax_pr.legend(loc="lower left")

    # Confusion matrix
    ax_cm.set_facecolor("lightgrey")
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax_cm,
    )
    ax_cm.set_title(f"Confusion Matrix \\n Threshold={threshold:.2f}")

    # KPI text block
    ax_text.set_facecolor("lightgrey")
    ax_text.axis("off")
    kpi_text = (
        f"Model: {model_name}\n"
        f"Threshold: {threshold:.2f}\n\n"
        f"ROC AUC: {roc_auc:.3f}\n"
        f"PR AUC: {pr_auc:.3f}\n"
        f"Accuracy: {acc:.3f}\n"
        f"Recall (Class 1): {recall_1:.3f}\n"
        f"Precision (Class 1): {precision_1:.3f}\n"
        f"F1 (Class 1): {f1_1:.3f}\n"
        f"MCC: {mcc:.3f}\n"
        f"Balanced Accuracy: {bal_acc:.3f}\n"
        f"FPR: {fpr_at_threshold:.3f}\n\n"
        f"TN={tn} | FP={fp} | FN={fn} | TP={tp}"
    )
    ax_text.text(
        0.5,
        0.5,
        kpi_text,
        va="center",
        ha="center",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "boxstyle": "round,pad=0.5"},
    )

    for ax in [ax_roc, ax_pr, ax_text]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"{model_name} Operating Point Panel", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_Operating_Point_Panel.png")
    plt.show()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "recall_1": recall_1,
        "precision_1": precision_1,
        "f1_1": f1_1,
        "mcc": mcc,
        "bal_acc": bal_acc,
        "fpr": fpr_at_threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
