"""Calibration and threshold-quality utilities for binary classifiers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix


def _prepare_binary_inputs(y_true, y_proba, pos_label=1):
    """Convert inputs to aligned numeric arrays and drop NaN pairs."""
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_true_bin = (y_true_arr == pos_label).astype(int)

    # Keep rows where probability is finite and label is present.
    mask = np.isfinite(y_proba_arr) & pd.notna(y_true_arr)
    y_true_bin = y_true_bin[mask]
    y_proba_arr = y_proba_arr[mask]
    return y_true_bin, y_proba_arr


def _safe_div(numerator, denominator):
    """Safe division that returns NaN if denominator is zero."""
    if denominator == 0:
        return np.nan
    return numerator / denominator


def _metrics_at_threshold(y_true_bin, y_proba, threshold):
    """Compute confusion counts and classification metrics at one threshold."""
    y_hat = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_hat, labels=[0, 1]).ravel()

    recall1 = _safe_div(tp, tp + fn)
    precision1 = _safe_div(tp, tp + fp)
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)
    bal_acc = np.nanmean([recall1, tnr])

    # Manual MCC with zero-denominator guard.
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div((tp * tn) - (fp * fn), mcc_den)
    if np.isnan(mcc):
        mcc = 0.0

    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "recall1": recall1,
        "precision1": precision1,
        "fpr": fpr,
        "mcc": mcc,
        "bal_acc": bal_acc,
    }


def _safe_calibration_curve(y_true_bin, y_proba, n_bins=10, strategy="uniform"):
    """Return calibration-curve coordinates without failing on degenerate labels."""
    # calibration_curve needs both classes for meaningful bin positives.
    if np.unique(y_true_bin).size < 2:
        return np.array([]), np.array([])
    frac_pos, mean_pred = calibration_curve(
        y_true_bin,
        y_proba,
        n_bins=n_bins,
        strategy=strategy,
    )
    return frac_pos, mean_pred


def calibration_report(
    y_true,
    y_proba,
    threshold=0.25,
    y_proba_pre=None,
    pos_label=1,
    n_bins=10,
    strategy="uniform",
    plot=True,
    ax=None,
):
    """Return Brier/calibration details and optionally plot reliability curves."""
    y_true_bin, y_proba_post = _prepare_binary_inputs(y_true, y_proba, pos_label=pos_label)
    if y_true_bin.size == 0:
        raise ValueError("No valid rows after input cleaning.")

    base_rate = float(np.mean(y_true_bin))
    brier_post = float(brier_score_loss(y_true_bin, y_proba_post))
    baseline_pred = np.full_like(y_proba_post, fill_value=base_rate, dtype=float)
    brier_baseline = float(brier_score_loss(y_true_bin, baseline_pred))

    # Calibration-curve inputs.
    frac_post, mean_post = _safe_calibration_curve(
        y_true_bin,
        y_proba_post,
        n_bins=n_bins,
        strategy=strategy,
    )

    brier_pre = np.nan
    frac_pre = np.array([])
    mean_pre = np.array([])
    if y_proba_pre is not None:
        y_true_pre, y_proba_pre_arr = _prepare_binary_inputs(y_true, y_proba_pre, pos_label=pos_label)
        min_len = min(len(y_true_pre), len(y_true_bin))
        y_true_pre = y_true_pre[:min_len]
        y_proba_pre_arr = y_proba_pre_arr[:min_len]
        brier_pre = float(brier_score_loss(y_true_pre, y_proba_pre_arr))
        frac_pre, mean_pre = _safe_calibration_curve(
            y_true_pre,
            y_proba_pre_arr,
            n_bins=n_bins,
            strategy=strategy,
        )

    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        ax.plot(mean_post, frac_post, marker="o", label="Post Calibration")
        if y_proba_pre is not None:
            ax.plot(mean_pre, frac_pre, marker="o", label="Pre Calibration")
        ax.set_xlabel("Mean Predicted Value")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Curve (threshold={threshold:.2f})")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return {
        "threshold": float(threshold),
        "base_rate": base_rate,
        "brier_post": brier_post,
        "brier_pre": brier_pre,
        "brier_baseline": brier_baseline,
        "brier_improvement_vs_baseline": brier_baseline - brier_post,
        "curve_post": pd.DataFrame(
            {"mean_predicted_value": mean_post, "fraction_of_positives": frac_post}
        ),
        "curve_pre": pd.DataFrame(
            {"mean_predicted_value": mean_pre, "fraction_of_positives": frac_pre}
        ),
    }


def fn_probability_diagnostics(
    y_true,
    y_proba,
    threshold=0.25,
    pos_label=1,
    band_below=(0.20, 0.25),
    band_above=(0.25, 0.35),
    near_window=0.05,
    threshold_artifact_cutoff=0.30,
):
    """Inspect FN probability concentration around the operating threshold."""
    y_true_bin, y_proba_arr = _prepare_binary_inputs(y_true, y_proba, pos_label=pos_label)
    y_hat = (y_proba_arr >= threshold).astype(int)
    fn_mask = (y_true_bin == 1) & (y_hat == 0)
    fn_probs = y_proba_arr[fn_mask]
    fn_count = int(fn_probs.size)

    if fn_count == 0:
        return {
            "threshold": float(threshold),
            "fn_count": 0,
            "median_fn_proba": np.nan,
            "pct_fn_in_band_below": 0.0,
            "pct_fn_in_band_above": 0.0,
            "pct_fn_near_threshold": 0.0,
            "heuristic": "no_false_negatives",
        }

    below_mask = (fn_probs >= band_below[0]) & (fn_probs < band_below[1])
    above_mask = (fn_probs >= band_above[0]) & (fn_probs <= band_above[1])

    # Near-threshold band (symmetric window around threshold).
    near_low = threshold - near_window
    near_high = threshold + near_window
    near_mask = (fn_probs >= near_low) & (fn_probs <= near_high)
    near_share = float(np.mean(near_mask))

    heuristic = "threshold artifact" if near_share >= threshold_artifact_cutoff else "structural miss"

    return {
        "threshold": float(threshold),
        "fn_count": fn_count,
        "median_fn_proba": float(np.median(fn_probs)),
        "pct_fn_in_band_below": float(np.mean(below_mask) * 100.0),
        "pct_fn_in_band_above": float(np.mean(above_mask) * 100.0),
        "pct_fn_near_threshold": float(near_share * 100.0),
        "heuristic": heuristic,
        "near_window": float(near_window),
        "threshold_artifact_cutoff": float(threshold_artifact_cutoff),
    }


def threshold_sweep(
    y_true,
    y_proba,
    threshold_min=0.15,
    threshold_max=0.35,
    threshold_step=0.01,
    pos_label=1,
):
    """Evaluate confusion + metrics over a threshold grid."""
    y_true_bin, y_proba_arr = _prepare_binary_inputs(y_true, y_proba, pos_label=pos_label)
    thresholds = np.round(np.arange(threshold_min, threshold_max + 1e-9, threshold_step), 10)

    rows = []
    for threshold in thresholds:
        rows.append(_metrics_at_threshold(y_true_bin, y_proba_arr, threshold))
    return pd.DataFrame(rows)


def select_threshold(
    sweep_df,
    current_threshold=0.25,
    objective="min_fn_bounded_fp",
    fp_increase_bound=0.20,
    recall_min=0.70,
):
    """Select threshold using one of three business-objective rules."""
    if sweep_df.empty:
        return {
            "chosen_threshold": np.nan,
            "selected_row": {},
            "objective": objective,
            "feasible": False,
            "note": "No sweep results provided.",
        }

    # Use nearest row if exact threshold is absent.
    idx_current = (sweep_df["threshold"] - current_threshold).abs().idxmin()
    current_row = sweep_df.loc[idx_current]

    feasible = True
    note = "Constraints satisfied."

    if objective == "min_fn_bounded_fp":
        fp_limit = float(current_row["FP"] * (1.0 + fp_increase_bound))
        candidates = sweep_df[sweep_df["FP"] <= fp_limit]
        if candidates.empty:
            feasible = False
            note = "FP bound infeasible; selected closest FP bound with minimum FN."
            candidates = sweep_df.copy()
            candidates["fp_gap"] = (candidates["FP"] - fp_limit).abs()
            candidates = candidates.sort_values(["fp_gap", "FN", "threshold"], ascending=[True, True, True])
            chosen = candidates.iloc[0]
        else:
            chosen = candidates.sort_values(["FN", "FP", "threshold"], ascending=[True, True, True]).iloc[0]

    elif objective == "max_mcc_with_recall_constraint":
        candidates = sweep_df[sweep_df["recall1"] >= recall_min]
        if candidates.empty:
            feasible = False
            note = "Recall constraint infeasible; selected highest recall then highest MCC."
            chosen = sweep_df.sort_values(["recall1", "mcc", "threshold"], ascending=[False, False, True]).iloc[0]
        else:
            chosen = candidates.sort_values(["mcc", "recall1", "threshold"], ascending=[False, False, True]).iloc[0]

    elif objective == "max_bal_acc_with_recall_constraint":
        candidates = sweep_df[sweep_df["recall1"] >= recall_min]
        if candidates.empty:
            feasible = False
            note = "Recall constraint infeasible; selected highest recall then highest balanced accuracy."
            chosen = sweep_df.sort_values(["recall1", "bal_acc", "threshold"], ascending=[False, False, True]).iloc[0]
        else:
            chosen = candidates.sort_values(["bal_acc", "recall1", "threshold"], ascending=[False, False, True]).iloc[0]

    else:
        raise ValueError(
            "objective must be one of: 'min_fn_bounded_fp', "
            "'max_mcc_with_recall_constraint', 'max_bal_acc_with_recall_constraint'"
        )

    chosen_dict = chosen.drop(labels=["fp_gap"], errors="ignore").to_dict()
    return {
        "chosen_threshold": float(chosen_dict["threshold"]),
        "selected_row": chosen_dict,
        "objective": objective,
        "feasible": feasible,
        "note": note,
    }


def threshold_stability_cv(
    cv_folds,
    objective="min_fn_bounded_fp",
    current_threshold=0.25,
    fp_increase_bound=0.20,
    recall_min=0.70,
    threshold_min=0.15,
    threshold_max=0.35,
    threshold_step=0.01,
    pos_label=1,
    plot=False,
    plot_kind="hist",
):
    """Assess threshold stability across CV folds and summarize variability."""
    if cv_folds is None or len(cv_folds) == 0:
        return {
            "per_fold_optimal_thresholds": [],
            "threshold_stats": {},
            "mean_metrics_fold_optimal": {},
            "mean_metrics_global_threshold": {},
            "global_threshold": np.nan,
            "per_fold_results": pd.DataFrame(),
        }

    fold_rows = []
    fold_optimal_thresholds = []

    for fold_idx, fold in enumerate(cv_folds):
        if isinstance(fold, dict):
            y_true_fold = fold["y_true"]
            y_proba_fold = fold["y_proba"]
        elif len(fold) >= 2:
            y_true_fold, y_proba_fold = fold[0], fold[1]
        else:
            raise ValueError("Each cv fold must contain at least y_true and y_proba.")

        sweep_df = threshold_sweep(
            y_true=y_true_fold,
            y_proba=y_proba_fold,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            pos_label=pos_label,
        )
        selected = select_threshold(
            sweep_df=sweep_df,
            current_threshold=current_threshold,
            objective=objective,
            fp_increase_bound=fp_increase_bound,
            recall_min=recall_min,
        )
        row = selected["selected_row"].copy()
        row["fold"] = fold_idx
        row["selection_feasible"] = selected["feasible"]
        row["selection_note"] = selected["note"]
        fold_rows.append(row)
        fold_optimal_thresholds.append(selected["chosen_threshold"])

    fold_results_df = pd.DataFrame(fold_rows)
    threshold_array = np.asarray(fold_optimal_thresholds, dtype=float)
    global_threshold = float(np.nanmedian(threshold_array))

    q25 = float(np.nanpercentile(threshold_array, 25))
    q75 = float(np.nanpercentile(threshold_array, 75))

    # Evaluate each fold at the global threshold for comparison.
    global_rows = []
    for fold in cv_folds:
        if isinstance(fold, dict):
            y_true_fold = fold["y_true"]
            y_proba_fold = fold["y_proba"]
        else:
            y_true_fold, y_proba_fold = fold[0], fold[1]
        sweep_df = threshold_sweep(
            y_true=y_true_fold,
            y_proba=y_proba_fold,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            pos_label=pos_label,
        )
        idx = (sweep_df["threshold"] - global_threshold).abs().idxmin()
        global_rows.append(sweep_df.loc[idx].to_dict())

    global_df = pd.DataFrame(global_rows)
    metric_cols = ["TP", "FP", "TN", "FN", "recall1", "precision1", "fpr", "mcc", "bal_acc"]

    if plot:
        if plot_kind == "box":
            plt.figure(figsize=(6, 4))
            plt.boxplot(threshold_array, vert=True)
            plt.ylabel("Optimal Threshold")
            plt.title("CV Threshold Stability (Boxplot)")
        else:
            plt.figure(figsize=(6, 4))
            plt.hist(threshold_array, bins="auto")
            plt.xlabel("Optimal Threshold")
            plt.ylabel("Count")
            plt.title("CV Threshold Stability (Histogram)")
        plt.tight_layout()
        plt.show()

    return {
        "per_fold_optimal_thresholds": [float(x) for x in threshold_array],
        "threshold_stats": {
            "mean": float(np.nanmean(threshold_array)),
            "std": float(np.nanstd(threshold_array, ddof=1)) if threshold_array.size > 1 else 0.0,
            "iqr": float(q75 - q25),
            "q25": q25,
            "q75": q75,
        },
        "mean_metrics_fold_optimal": fold_results_df[metric_cols].mean(numeric_only=True).to_dict(),
        "mean_metrics_global_threshold": global_df[metric_cols].mean(numeric_only=True).to_dict(),
        "global_threshold": global_threshold,
        "per_fold_results": fold_results_df,
    }
