"""Segment and temporal robustness evaluation for binary classifiers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from src.results.calibration_threshold_eval import select_threshold
from src.results.statistical_testing import adjust_pvalues, cliffs_delta


def _safe_div(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def _effect_magnitude(delta):
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def add_time_segments(df, time_col, early_late_split=None, unknown_label="Unknown", drop_invalid_time=False):
    """Add Year, YearMonth, and EarlyLate segment columns."""
    out = df.copy()
    out["_time_parsed"] = pd.to_datetime(out[time_col], errors="coerce")

    if drop_invalid_time:
        out = out[out["_time_parsed"].notna()].copy()

    out["Year"] = out["_time_parsed"].dt.year.astype("Int64").astype(str)
    out["Year"] = out["Year"].replace("<NA>", unknown_label)

    out["YearMonth"] = out["_time_parsed"].dt.to_period("M").astype(str)
    out["YearMonth"] = out["YearMonth"].replace("NaT", unknown_label)

    config = early_late_split or {"type": "quantile", "q": 0.5}
    split_type = str(config.get("type", "quantile")).lower()

    out["EarlyLate"] = unknown_label
    valid_time = out["_time_parsed"].notna()

    if split_type == "year":
        cutoff_year = int(config.get("cutoff_year", out.loc[valid_time, "_time_parsed"].dt.year.median()))
        is_early = out["_time_parsed"].dt.year <= cutoff_year
        out.loc[valid_time & is_early, "EarlyLate"] = "Early"
        out.loc[valid_time & (~is_early), "EarlyLate"] = "Late"
    else:
        q = float(config.get("q", 0.5))
        cutoff = out.loc[valid_time, "_time_parsed"].quantile(q) if valid_time.any() else pd.NaT
        if pd.isna(cutoff):
            out.loc[valid_time, "EarlyLate"] = unknown_label
        else:
            is_early = out["_time_parsed"] <= cutoff
            out.loc[valid_time & is_early, "EarlyLate"] = "Early"
            out.loc[valid_time & (~is_early), "EarlyLate"] = "Late"

    return out


def compute_segment_metrics(
    df,
    segment_col,
    y_true_col,
    y_proba_col,
    threshold,
    segment_type=None,
    min_segment_n=30,
    min_pos_n=5,
    pos_label=1,
):
    """Compute confusion + metrics by segment at a fixed threshold."""
    clean = df[[segment_col, y_true_col, y_proba_col]].copy()
    clean = clean.dropna(subset=[segment_col, y_true_col, y_proba_col])
    clean["y_true_bin"] = (clean[y_true_col] == pos_label).astype(int)
    clean["y_hat"] = (clean[y_proba_col] >= threshold).astype(int)

    rows = []
    for seg_value, grp in clean.groupby(segment_col, dropna=False):
        n = int(len(grp))
        if n < min_segment_n:
            continue

        pos_n = int(grp["y_true_bin"].sum())
        y_true = grp["y_true_bin"].to_numpy()
        y_hat = grp["y_hat"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        recall1 = _safe_div(tp, tp + fn)
        if pos_n < min_pos_n:
            recall1 = np.nan

        fpr = _safe_div(fp, fp + tn)
        tnr = _safe_div(tn, tn + fp)
        bal_acc = np.nanmean([recall1, tnr])

        try:
            mcc = float(matthews_corrcoef(y_true, y_hat))
        except Exception:
            mcc = np.nan

        rows.append(
            {
                "segment_type": segment_type or segment_col,
                "segment_value": seg_value,
                "n": n,
                "pos_n": pos_n,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "recall1": recall1,
                "fpr": fpr,
                "mcc": mcc,
                "bal_acc": bal_acc,
            }
        )

    return pd.DataFrame(rows)


def compute_fn_distribution(
    df,
    segment_col,
    y_true_col,
    y_proba_col,
    threshold,
    segment_type=None,
    min_segment_n=30,
    pos_label=1,
):
    """Compute FN count/share/rates by segment."""
    clean = df[[segment_col, y_true_col, y_proba_col]].copy()
    clean = clean.dropna(subset=[segment_col, y_true_col, y_proba_col])
    clean["y_true_bin"] = (clean[y_true_col] == pos_label).astype(int)
    clean["y_hat"] = (clean[y_proba_col] >= threshold).astype(int)
    clean["is_fn"] = (clean["y_true_bin"] == 1) & (clean["y_hat"] == 0)

    total_fns = int(clean["is_fn"].sum())
    rows = []
    for seg_value, grp in clean.groupby(segment_col, dropna=False):
        n = int(len(grp))
        if n < min_segment_n:
            continue

        pos_n = int(grp["y_true_bin"].sum())
        fn_count = int(grp["is_fn"].sum())
        rows.append(
            {
                "segment_type": segment_type or segment_col,
                "segment_value": seg_value,
                "n": n,
                "pos_n": pos_n,
                "fn_count": fn_count,
                "fn_share": _safe_div(fn_count, total_fns),
                "fn_rate_in_segment": _safe_div(fn_count, n),
                "pos_rate_in_segment": _safe_div(pos_n, n),
            }
        )

    return pd.DataFrame(rows)


def _infer_feature_cols(df, exclude_cols):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in set(exclude_cols)]


def distribution_shift_tests(
    df,
    feature_cols=None,
    segment_definitions=None,
    multiple_test_method="fdr_bh",
    min_segment_n=30,
):
    """Run covariate shift tests by segment definition."""
    segment_definitions = segment_definitions or {}
    if feature_cols is None:
        feature_cols = _infer_feature_cols(df, exclude_cols=list(segment_definitions.values()))

    all_rows = []

    for seg_type, seg_col in segment_definitions.items():
        if seg_col not in df.columns:
            continue

        seg_series = df[seg_col].copy()
        valid_seg = seg_series.notna()
        if valid_seg.sum() == 0:
            continue

        # Two-group early/late test.
        if seg_type.lower() == "earlylate":
            early_mask = valid_seg & (df[seg_col].astype(str) == "Early")
            late_mask = valid_seg & (df[seg_col].astype(str) == "Late")

            for feature in feature_cols:
                s_early = pd.to_numeric(df.loc[valid_seg & early_mask, feature], errors="coerce").dropna()
                s_late = pd.to_numeric(df.loc[valid_seg & late_mask, feature], errors="coerce").dropna()
                if len(s_early) < min_segment_n or len(s_late) < min_segment_n:
                    p_raw = np.nan
                    effect = np.nan
                    effect_mag = "insufficient_data"
                    direction = "insufficient_data"
                    medians = {"Early": np.nan, "Late": np.nan}
                    iqrs = {"Early": np.nan, "Late": np.nan}
                else:
                    _, p_raw = stats.mannwhitneyu(s_early, s_late, alternative="two-sided")
                    effect, effect_mag = cliffs_delta(s_early, s_late)
                    medians = {"Early": float(s_early.median()), "Late": float(s_late.median())}
                    iqrs = {
                        "Early": float(s_early.quantile(0.75) - s_early.quantile(0.25)),
                        "Late": float(s_late.quantile(0.75) - s_late.quantile(0.25)),
                    }
                    if medians["Early"] > medians["Late"]:
                        direction = "Early > Late"
                    elif medians["Early"] < medians["Late"]:
                        direction = "Late > Early"
                    else:
                        direction = "equal_median"

                all_rows.append(
                    {
                        "segment_type": seg_type,
                        "feature": feature,
                        "test_type": "mannwhitney_early_vs_late",
                        "p_raw": p_raw,
                        "effect_size": effect,
                        "effect_mag": effect_mag,
                        "direction": direction,
                        "group_medians": medians,
                        "group_iqrs": iqrs,
                    }
                )
        else:
            # Multi-group Kruskal test (Year / YearMonth / Route if passed in).
            group_sizes = df.groupby(seg_col).size()
            eligible_segments = group_sizes[group_sizes >= min_segment_n].index.tolist()
            if len(eligible_segments) < 2:
                continue

            for feature in feature_cols:
                sample_groups = []
                medians = {}
                iqrs = {}
                for seg in eligible_segments:
                    series = pd.to_numeric(df.loc[df[seg_col] == seg, feature], errors="coerce").dropna()
                    if len(series) < min_segment_n:
                        continue
                    sample_groups.append(series.to_numpy())
                    medians[str(seg)] = float(series.median())
                    iqrs[str(seg)] = float(series.quantile(0.75) - series.quantile(0.25))

                if len(sample_groups) < 2:
                    p_raw = np.nan
                    stat = np.nan
                else:
                    stat, p_raw = stats.kruskal(*sample_groups)

                all_rows.append(
                    {
                        "segment_type": seg_type,
                        "feature": feature,
                        "test_type": "kruskal_multigroup",
                        "statistic": stat,
                        "p_raw": p_raw,
                        "effect_size": np.nan,
                        "effect_mag": np.nan,
                        "direction": "multigroup",
                        "group_medians": medians,
                        "group_iqrs": iqrs,
                    }
                )

    out = pd.DataFrame(all_rows)
    if out.empty:
        return out

    # Adjust p-values inside each segment_type + test_type family.
    out["p_adj"] = np.nan
    for (seg_type, test_type), idx in out.groupby(["segment_type", "test_type"]).groups.items():
        pvals = out.loc[idx, "p_raw"].to_numpy(dtype=float)
        out.loc[idx, "p_adj"] = adjust_pvalues(pvals, method=multiple_test_method)

    return out.sort_values(["segment_type", "test_type", "p_adj", "p_raw"], na_position="last").reset_index(drop=True)


def threshold_sweep_metrics(y_true, y_proba, threshold_grid, pos_label=1):
    """Compute threshold sweep metrics for an arbitrary threshold grid."""
    y_true_arr = (np.asarray(y_true) == pos_label).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    valid = np.isfinite(y_proba_arr) & pd.notna(y_true_arr)
    y_true_arr = y_true_arr[valid]
    y_proba_arr = y_proba_arr[valid]

    rows = []
    for threshold in threshold_grid:
        y_hat = (y_proba_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_hat, labels=[0, 1]).ravel()
        recall1 = _safe_div(tp, tp + fn)
        precision1 = _safe_div(tp, tp + fp)
        fpr = _safe_div(fp, fp + tn)
        tnr = _safe_div(tn, tn + fp)
        bal_acc = np.nanmean([recall1, tnr])

        try:
            mcc = float(matthews_corrcoef(y_true_arr, y_hat))
        except Exception:
            mcc = np.nan

        rows.append(
            {
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
        )

    return pd.DataFrame(rows)


def select_threshold_from_sweep(
    sweep_df,
    objective="max_mcc_with_recall_constraint",
    constraints=None,
):
    """Select an operating threshold from a sweep dataframe."""
    constraints = constraints or {}
    current_threshold = float(constraints.get("current_threshold", 0.25))
    fp_increase_bound = float(constraints.get("fp_increase_bound", 0.20))
    recall_min = float(constraints.get("recall_min", 0.70))

    selection = select_threshold(
        sweep_df=sweep_df,
        current_threshold=current_threshold,
        objective=objective,
        fp_increase_bound=fp_increase_bound,
        recall_min=recall_min,
    )
    return selection


def threshold_stability_by_segment(
    df,
    segment_col,
    y_true_col,
    y_proba_col,
    threshold_grid,
    segment_type=None,
    objective="max_mcc_with_recall_constraint",
    constraints=None,
    min_segment_n=30,
    min_pos_n=5,
    delta_threshold=0.03,
    pos_label=1,
):
    """Estimate optimal threshold per segment and summarize stability."""
    constraints = constraints or {}

    full_sweep = threshold_sweep_metrics(
        y_true=df[y_true_col],
        y_proba=df[y_proba_col],
        threshold_grid=threshold_grid,
        pos_label=pos_label,
    )
    global_selection = select_threshold_from_sweep(
        full_sweep,
        objective=objective,
        constraints=constraints,
    )
    global_opt_threshold = global_selection["chosen_threshold"]

    rows = []
    for seg_value, grp in df.groupby(segment_col, dropna=False):
        n = int(len(grp))
        if n < min_segment_n:
            continue
        pos_n = int((grp[y_true_col] == pos_label).sum())
        if pos_n < min_pos_n:
            continue

        sweep_df = threshold_sweep_metrics(
            y_true=grp[y_true_col],
            y_proba=grp[y_proba_col],
            threshold_grid=threshold_grid,
            pos_label=pos_label,
        )
        selection = select_threshold_from_sweep(
            sweep_df=sweep_df,
            objective=objective,
            constraints=constraints,
        )
        selected_row = selection.get("selected_row", {})
        if not selected_row:
            continue

        opt_threshold = float(selection["chosen_threshold"])
        rows.append(
            {
                "segment_type": segment_type or segment_col,
                "segment_value": seg_value,
                "n": n,
                "pos_n": pos_n,
                "opt_threshold": opt_threshold,
                "opt_recall1": selected_row.get("recall1", np.nan),
                "opt_fpr": selected_row.get("fpr", np.nan),
                "opt_mcc": selected_row.get("mcc", np.nan),
                "opt_bal_acc": selected_row.get("bal_acc", np.nan),
                "material_deviation": abs(opt_threshold - global_opt_threshold) >= delta_threshold,
                "selection_feasible": selection.get("feasible", True),
                "selection_note": selection.get("note", ""),
            }
        )

    per_segment_df = pd.DataFrame(rows)
    if per_segment_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "segment_type": segment_type or segment_col,
                    "global_opt_threshold": global_opt_threshold,
                    "mean_opt_threshold": np.nan,
                    "std_opt_threshold": np.nan,
                    "iqr_opt_threshold": np.nan,
                    "range_opt_threshold": np.nan,
                    "n_segments": 0,
                    "n_material_deviation": 0,
                }
            ]
        )
        return {
            "per_segment_opt_threshold": per_segment_df,
            "stability_summary": summary_df,
            "global_opt_threshold": global_opt_threshold,
        }

    q25 = per_segment_df["opt_threshold"].quantile(0.25)
    q75 = per_segment_df["opt_threshold"].quantile(0.75)
    summary_df = pd.DataFrame(
        [
            {
                "segment_type": segment_type or segment_col,
                "global_opt_threshold": global_opt_threshold,
                "mean_opt_threshold": float(per_segment_df["opt_threshold"].mean()),
                "std_opt_threshold": float(per_segment_df["opt_threshold"].std(ddof=1))
                if len(per_segment_df) > 1
                else 0.0,
                "iqr_opt_threshold": float(q75 - q25),
                "range_opt_threshold": float(per_segment_df["opt_threshold"].max() - per_segment_df["opt_threshold"].min()),
                "n_segments": int(len(per_segment_df)),
                "n_material_deviation": int(per_segment_df["material_deviation"].sum()),
            }
        ]
    )

    return {
        "per_segment_opt_threshold": per_segment_df,
        "stability_summary": summary_df,
        "global_opt_threshold": global_opt_threshold,
    }


def _plot_yearmonth_metrics(segment_metrics_df):
    ym = segment_metrics_df[segment_metrics_df["segment_type"] == "YearMonth"].copy()
    if ym.empty:
        return
    ym = ym.sort_values("segment_value")
    plt.figure(figsize=(9, 4))
    plt.plot(ym["segment_value"], ym["recall1"], marker="o", label="Recall1")
    plt.plot(ym["segment_value"], ym["fpr"], marker="o", label="FPR")
    plt.plot(ym["segment_value"], ym["mcc"], marker="o", label="MCC")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("YearMonth")
    plt.ylabel("Metric Value")
    plt.title("Segment Metrics Over Time (YearMonth)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_fn_share(fn_distribution_df, segment_type):
    subset = fn_distribution_df[fn_distribution_df["segment_type"] == segment_type].copy()
    if subset.empty:
        return
    subset = subset.sort_values("segment_value")
    plt.figure(figsize=(9, 4))
    plt.bar(subset["segment_value"].astype(str), subset["fn_share"].fillna(0.0))
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Segment")
    plt.ylabel("FN Share")
    plt.title(f"FN Share by {segment_type}")
    plt.tight_layout()
    plt.show()


def _plot_opt_threshold_over_time(per_segment_opt_df):
    ym = per_segment_opt_df[per_segment_opt_df["segment_type"] == "YearMonth"].copy()
    if ym.empty:
        return
    ym = ym.sort_values("segment_value")
    plt.figure(figsize=(9, 4))
    plt.scatter(ym["segment_value"].astype(str), ym["opt_threshold"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("YearMonth")
    plt.ylabel("Optimal Threshold")
    plt.title("Optimal Threshold by YearMonth")
    plt.tight_layout()
    plt.show()


def build_segment_temporal_report(
    df,
    y_true_col,
    y_proba_col,
    time_col,
    route_col=None,
    feature_cols=None,
    threshold=0.25,
    threshold_grid=None,
    early_late_split=None,
    min_segment_n=30,
    min_pos_n=5,
    multiple_test_method="fdr_bh",
    objective="max_mcc_with_recall_constraint",
    constraints=None,
    delta_threshold=0.03,
    plot=False,
):
    """Run full segment/temporal robustness pipeline and return report objects."""
    threshold_grid = threshold_grid if threshold_grid is not None else np.arange(0.15, 0.351, 0.01)
    constraints = constraints or {"current_threshold": threshold, "recall_min": 0.70, "fp_increase_bound": 0.20}

    work = add_time_segments(df=df, time_col=time_col, early_late_split=early_late_split)

    # Build segment definitions in one place for metrics/shift testing.
    segment_defs = {"Year": "Year", "YearMonth": "YearMonth", "EarlyLate": "EarlyLate"}
    if route_col is not None and route_col in work.columns:
        segment_defs["Route"] = route_col

    # Segment-level performance table.
    metrics_tables = []
    for seg_type, seg_col in segment_defs.items():
        metrics_df = compute_segment_metrics(
            work,
            segment_col=seg_col,
            y_true_col=y_true_col,
            y_proba_col=y_proba_col,
            threshold=threshold,
            segment_type=seg_type,
            min_segment_n=min_segment_n,
            min_pos_n=min_pos_n,
        )
        metrics_tables.append(metrics_df)
    segment_metrics_df = pd.concat(metrics_tables, ignore_index=True) if metrics_tables else pd.DataFrame()

    # FN distribution table by segment.
    fn_tables = []
    for seg_type, seg_col in segment_defs.items():
        fn_df = compute_fn_distribution(
            work,
            segment_col=seg_col,
            y_true_col=y_true_col,
            y_proba_col=y_proba_col,
            threshold=threshold,
            segment_type=seg_type,
            min_segment_n=min_segment_n,
        )
        fn_tables.append(fn_df)
    fn_distribution_df = pd.concat(fn_tables, ignore_index=True) if fn_tables else pd.DataFrame()

    # Shift tests focus on time segments.
    exclude_cols = [y_true_col, y_proba_col, time_col]
    if route_col:
        exclude_cols.append(route_col)
    if feature_cols is None:
        feature_cols = _infer_feature_cols(work, exclude_cols=exclude_cols)
    shift_segment_defs = {"EarlyLate": "EarlyLate", "Year": "Year", "YearMonth": "YearMonth"}
    shift_results_df = distribution_shift_tests(
        work,
        feature_cols=feature_cols,
        segment_definitions=shift_segment_defs,
        multiple_test_method=multiple_test_method,
        min_segment_n=min_segment_n,
    )

    # Threshold stability for Year / EarlyLate / Route (if present) plus YearMonth for time tracking.
    stability_segment_defs = {"Year": "Year", "YearMonth": "YearMonth", "EarlyLate": "EarlyLate"}
    if "Route" in segment_defs:
        stability_segment_defs["Route"] = segment_defs["Route"]

    per_segment_opt_tables = []
    summary_tables = []
    global_thresholds = {}
    for seg_type, seg_col in stability_segment_defs.items():
        stab = threshold_stability_by_segment(
            work,
            segment_col=seg_col,
            y_true_col=y_true_col,
            y_proba_col=y_proba_col,
            threshold_grid=threshold_grid,
            segment_type=seg_type,
            objective=objective,
            constraints=constraints,
            min_segment_n=min_segment_n,
            min_pos_n=min_pos_n,
            delta_threshold=delta_threshold,
        )
        per_segment_opt_tables.append(stab["per_segment_opt_threshold"])
        summary_tables.append(stab["stability_summary"])
        global_thresholds[seg_type] = stab["global_opt_threshold"]

    per_segment_opt_threshold_df = (
        pd.concat(per_segment_opt_tables, ignore_index=True) if per_segment_opt_tables else pd.DataFrame()
    )
    stability_summary_df = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()

    if plot:
        _plot_yearmonth_metrics(segment_metrics_df)
        _plot_fn_share(fn_distribution_df, segment_type="YearMonth")
        _plot_opt_threshold_over_time(per_segment_opt_threshold_df)

    return {
        "df_with_segments": work,
        "segment_metrics": segment_metrics_df,
        "fn_distribution": fn_distribution_df,
        "shift_results": shift_results_df,
        "per_segment_opt_threshold": per_segment_opt_threshold_df,
        "stability_summary": stability_summary_df,
        "global_opt_thresholds": global_thresholds,
    }
