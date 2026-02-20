"""Leakage and data quality checks for binary classification pipelines."""

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.results.statistical_testing import adjust_pvalues


DEFAULT_LEAKAGE_PATTERNS = [
    "outcome",
    "loss",
    "sunk",
    "after",
    "post",
    "future",
    "label",
    "target",
]


def _safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def flag_leakage_columns(X, feature_metadata=None, patterns=None):
    """Flag suspicious leakage columns by name patterns and optional metadata."""
    patterns = patterns or DEFAULT_LEAKAGE_PATTERNS
    regex = re.compile("|".join(patterns), flags=re.IGNORECASE)

    meta_map = {}
    if feature_metadata is not None:
        if isinstance(feature_metadata, pd.DataFrame):
            if "feature" in feature_metadata.columns:
                for _, row in feature_metadata.iterrows():
                    meta_map[str(row["feature"])] = row.to_dict()
        elif isinstance(feature_metadata, dict):
            meta_map = feature_metadata

    rows = []
    for col in X.columns:
        pattern_hit = bool(regex.search(str(col)))
        meta = meta_map.get(str(col), {})

        meta_flag = False
        meta_reason = None
        if isinstance(meta, dict):
            for key in ["post_outcome", "future_derived", "label_proxy"]:
                if bool(meta.get(key, False)):
                    meta_flag = True
                    meta_reason = key
                    break

        if not pattern_hit and not meta_flag:
            continue

        if meta_flag or pattern_hit:
            severity = "high" if (meta_flag or pattern_hit) else "low"
        else:
            severity = "medium"

        reasons = []
        if pattern_hit:
            reasons.append("name_pattern")
        if meta_flag:
            reasons.append(f"metadata:{meta_reason}")

        rows.append(
            {
                "feature": col,
                "reason": ";".join(reasons),
                "severity": severity,
                "pattern_matched": pattern_hit,
                "metadata_flag": meta_flag,
            }
        )

    return pd.DataFrame(rows).sort_values(["severity", "feature"], ascending=[True, True]).reset_index(drop=True)


def check_split_integrity(X_train, X_test, id_col=None, convoy_id_col=None, sample_n=10):
    """Check row duplication and identifier overlap across train/test splits."""
    train = X_train.copy()
    test = X_test.copy()

    compare_cols = [c for c in train.columns.intersection(test.columns) if c != id_col]
    train_hash = pd.util.hash_pandas_object(train[compare_cols], index=False)
    test_hash = pd.util.hash_pandas_object(test[compare_cols], index=False)

    dup_hashes = np.intersect1d(train_hash.values, test_hash.values)
    dup_count = int(len(dup_hashes))

    overlapping_ids = []
    if id_col is not None and id_col in train.columns and id_col in test.columns:
        overlapping_ids = sorted(set(train[id_col].dropna()).intersection(set(test[id_col].dropna())))

    overlapping_convoys = []
    if convoy_id_col is not None and convoy_id_col in train.columns and convoy_id_col in test.columns:
        overlapping_convoys = sorted(
            set(train[convoy_id_col].dropna()).intersection(set(test[convoy_id_col].dropna()))
        )

    return {
        "duplicates_across_splits_count": dup_count,
        "overlapping_ids_count": len(overlapping_ids),
        "overlapping_convoy_ids_count": len(overlapping_convoys),
        "example_ids": overlapping_ids[:sample_n],
        "example_convoy_ids": overlapping_convoys[:sample_n],
        "pass_duplicates_check": dup_count == 0,
        "pass_id_overlap_check": len(overlapping_ids) == 0,
        "pass_convoy_overlap_check": len(overlapping_convoys) == 0,
    }


def audit_alignment(X, y, df_raw=None, id_col=None):
    """Audit X/y/raw alignment, uniqueness, and potential join mismatches."""
    y_series = pd.Series(y)
    report = {
        "x_len": int(len(X)),
        "y_len": int(len(y_series)),
        "xy_length_match": len(X) == len(y_series),
        "xy_index_match": bool(pd.Index(X.index).equals(y_series.index)),
        "x_index_unique": bool(X.index.is_unique),
        "suggested_fixes": [],
    }

    if not report["xy_length_match"]:
        report["suggested_fixes"].append("Ensure y is sliced with the same row mask/index as X.")
    if not report["xy_index_match"]:
        report["suggested_fixes"].append("Realign with y = y.reindex(X.index) before modeling.")

    if df_raw is not None:
        report["raw_len"] = int(len(df_raw))
        report["x_raw_length_match"] = len(X) == len(df_raw)
        report["x_raw_index_match"] = bool(pd.Index(X.index).equals(pd.Index(df_raw.index)))
        if not report["x_raw_length_match"]:
            report["suggested_fixes"].append("Rebuild split artifacts from one merged source before preprocessing.")
        if not report["x_raw_index_match"]:
            report["suggested_fixes"].append("Use id-based joins and explicit sort by id before split.")

        if id_col is not None and id_col in df_raw.columns:
            duplicated_ids = int(df_raw[id_col].duplicated().sum())
            report["raw_duplicated_ids_count"] = duplicated_ids
            report["raw_id_unique"] = duplicated_ids == 0
            if duplicated_ids > 0:
                report["suggested_fixes"].append(f"Deduplicate {id_col} in raw data before joins.")

    if id_col is not None and id_col in X.columns:
        x_dup_ids = int(X[id_col].duplicated().sum())
        report["x_duplicated_ids_count"] = x_dup_ids
        report["x_id_unique"] = x_dup_ids == 0
        if x_dup_ids > 0:
            report["suggested_fixes"].append(f"{id_col} should be unique in model matrix.")

    if not report["suggested_fixes"]:
        report["suggested_fixes"] = ["No alignment issues detected."]

    return report


def confusion_groups(y_true, y_proba, threshold):
    """Map each row to FN/FP/TP/TN based on thresholded probabilities."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_hat = (y_proba_arr >= threshold).astype(int)

    groups = np.full(len(y_true_arr), "Unknown", dtype=object)
    groups[(y_true_arr == 1) & (y_hat == 0)] = "FN"
    groups[(y_true_arr == 0) & (y_hat == 1)] = "FP"
    groups[(y_true_arr == 1) & (y_hat == 1)] = "TP"
    groups[(y_true_arr == 0) & (y_hat == 0)] = "TN"
    return pd.Series(groups, index=range(len(groups)), name="confusion_group")


def missingness_by_confusion_group(
    X_test,
    groups,
    multiple_test_method="fdr_bh",
    missingness_test=None,
):
    """Test whether feature missingness differs across confusion groups."""
    test_method = (missingness_test or {}).get("method", "chi2").lower() if missingness_test else "chi2"
    grp = pd.Series(groups).reset_index(drop=True)
    out_rows = []

    for col in X_test.columns:
        miss = X_test[col].isna().astype(int).reset_index(drop=True)
        tmp = pd.DataFrame({"g": grp, "m": miss}).dropna(subset=["g"])

        rates = {f"missing_rate_{g}": float(tmp.loc[tmp["g"] == g, "m"].mean()) if (tmp["g"] == g).any() else np.nan for g in ["FN", "FP", "TP", "TN"]}

        p_raw = np.nan
        try:
            if test_method == "kruskal":
                samples = [tmp.loc[tmp["g"] == g, "m"].values for g in ["FN", "FP", "TP", "TN"] if (tmp["g"] == g).any()]
                if len(samples) >= 2:
                    _, p_raw = stats.kruskal(*samples)
            else:
                contingency = pd.crosstab(tmp["g"], tmp["m"])
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    # Fisher exact is only for 2x2; fallback to chi2 otherwise.
                    if test_method == "fisher" and contingency.shape == (2, 2):
                        _, p_raw = stats.fisher_exact(contingency.values)
                    else:
                        _, p_raw, _, _ = stats.chi2_contingency(contingency.values)
        except Exception:
            p_raw = np.nan

        rate_map = {g: rates.get(f"missing_rate_{g}", np.nan) for g in ["FN", "FP", "TP", "TN"]}
        top_group = max(rate_map, key=lambda x: -np.inf if pd.isna(rate_map[x]) else rate_map[x])
        direction = f"highest_missing={top_group}" if not pd.isna(rate_map[top_group]) else "insufficient_data"

        out_rows.append(
            {
                "feature": col,
                **rates,
                "p_raw": p_raw,
                "direction": direction,
                "delta_FN_vs_TP": rates["missing_rate_FN"] - rates["missing_rate_TP"]
                if not (pd.isna(rates["missing_rate_FN"]) or pd.isna(rates["missing_rate_TP"]))
                else np.nan,
                "delta_FP_vs_TN": rates["missing_rate_FP"] - rates["missing_rate_TN"]
                if not (pd.isna(rates["missing_rate_FP"]) or pd.isna(rates["missing_rate_TN"]))
                else np.nan,
            }
        )

    out = pd.DataFrame(out_rows)
    out["p_adj"] = adjust_pvalues(out["p_raw"].to_numpy(dtype=float), method=multiple_test_method)
    return out.sort_values(["p_adj", "p_raw"], na_position="last").reset_index(drop=True)


def outlier_bounds_from_train(X_train, numeric_cols, method="iqr", params=None):
    """Compute train-derived outlier bounds for numeric columns."""
    params = params or {}
    bounds = {}

    for col in numeric_cols:
        series = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if series.empty:
            bounds[col] = {"low": np.nan, "high": np.nan, "method": method}
            continue

        if method == "zscore":
            z = float(params.get("z", 3.0))
            mu = float(series.mean())
            sd = float(series.std(ddof=0))
            low, high = mu - z * sd, mu + z * sd
        elif method == "robust_z":
            z = float(params.get("z", 3.5))
            med = float(series.median())
            mad = float(np.median(np.abs(series - med)))
            scale = 1.4826 * mad if mad > 0 else np.nan
            low, high = med - z * scale, med + z * scale
        else:
            k = float(params.get("k", 1.5))
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            low, high = q1 - k * iqr, q3 + k * iqr

        bounds[col] = {"low": float(low), "high": float(high), "method": method}

    return bounds


def outliers_by_confusion_group(X_test, groups, bounds):
    """Estimate outlier concentration in confusion groups using train-based bounds."""
    grp = pd.Series(groups).reset_index(drop=True)
    rows = []

    for col, b in bounds.items():
        s = pd.to_numeric(X_test[col], errors="coerce").reset_index(drop=True)
        low, high = b.get("low"), b.get("high")
        outlier = (s < low) | (s > high)
        tmp = pd.DataFrame({"g": grp, "o": outlier.astype(float)})

        rates = {
            f"outlier_rate_{g}": float(tmp.loc[tmp["g"] == g, "o"].mean()) if (tmp["g"] == g).any() else np.nan
            for g in ["FN", "FP", "TP", "TN"]
        }

        notes = []
        if not pd.isna(rates["outlier_rate_FN"]) and not pd.isna(rates["outlier_rate_TP"]):
            if rates["outlier_rate_FN"] - rates["outlier_rate_TP"] > 0.10:
                notes.append("FN_outlier_concentration")
        if not pd.isna(rates["outlier_rate_FP"]) and not pd.isna(rates["outlier_rate_TN"]):
            if rates["outlier_rate_FP"] - rates["outlier_rate_TN"] > 0.10:
                notes.append("FP_outlier_concentration")

        rows.append(
            {
                "feature": col,
                **rates,
                "train_outlier_bounds_low": low,
                "train_outlier_bounds_high": high,
                "notes": ";".join(notes) if notes else "",
            }
        )

    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def _collect_estimators_for_audit(obj):
    estimators = []
    if obj is None:
        return estimators

    # Pipeline-like.
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            estimators.extend(_collect_estimators_for_audit(step))
        return estimators

    # ColumnTransformer-like.
    if hasattr(obj, "transformers_"):
        for _, tr, _ in obj.transformers_:
            estimators.extend(_collect_estimators_for_audit(tr))
        return estimators
    if hasattr(obj, "transformers"):
        for _, tr, _ in obj.transformers:
            estimators.extend(_collect_estimators_for_audit(tr))
        return estimators

    estimators.append(obj)
    return estimators


def audit_preprocessing(preprocess_pipeline, X_train, X_test, feature_groups=None, tolerance=1e-6):
    """Sanity-check preprocessing fit status and train-only consistency heuristics."""
    if preprocess_pipeline is None:
        return {
            "fit_status": "not_provided",
            "category_leakage_flags": [],
            "scaler_leakage_flags": [],
            "tolerance_used": tolerance,
            "notes": ["No preprocessing object provided."],
        }

    if isinstance(preprocess_pipeline, tuple) and len(preprocess_pipeline) == 2:
        _, fitted_obj = preprocess_pipeline
        fit_status = "tuple_provided_unfitted_and_fitted"
    else:
        fitted_obj = preprocess_pipeline
        fit_status = "single_fitted_object_assumed"

    notes = []
    category_flags = []
    scaler_flags = []

    # Basic fitted-attribute sanity.
    fitted_like = any(
        hasattr(fitted_obj, attr)
        for attr in ["transformers_", "named_steps", "feature_names_in_", "n_features_in_"]
    )
    if not fitted_like:
        notes.append("Could not confirm fitted attributes on preprocessing object.")

    numeric_cols = (
        feature_groups.get("numeric", [])
        if isinstance(feature_groups, dict) and "numeric" in feature_groups
        else X_train.select_dtypes(include=[np.number]).columns.tolist()
    )
    categorical_cols = (
        feature_groups.get("categorical", [])
        if isinstance(feature_groups, dict) and "categorical" in feature_groups
        else X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    )

    train_only_cats = {}
    for col in categorical_cols:
        if col in X_train.columns and col in X_test.columns:
            train_set = set(X_train[col].dropna().astype(str))
            test_set = set(X_test[col].dropna().astype(str))
            train_only_cats[col] = {"train": train_set, "test_only": sorted(test_set - train_set)}

    estimators = _collect_estimators_for_audit(fitted_obj)
    for est in estimators:
        # Category leakage heuristic for encoders.
        if hasattr(est, "categories_"):
            cats = est.categories_
            for i, col in enumerate(categorical_cols[: len(cats)]):
                learned = set(pd.Series(cats[i]).astype(str))
                test_only = set(train_only_cats.get(col, {}).get("test_only", []))
                leak_hits = sorted(learned.intersection(test_only))
                if leak_hits:
                    category_flags.append(
                        {"feature": col, "categories_in_encoder_only_seen_in_test": leak_hits[:10]}
                    )

        # Scaler leakage heuristic for mean_/var_.
        if hasattr(est, "mean_"):
            means = np.asarray(est.mean_, dtype=float).ravel()
            cols = numeric_cols[: len(means)]
            for i, col in enumerate(cols):
                train_mean = pd.to_numeric(X_train[col], errors="coerce").mean()
                if pd.notna(train_mean) and abs(float(means[i]) - float(train_mean)) > tolerance:
                    scaler_flags.append(
                        {
                            "feature": col,
                            "learned_mean": float(means[i]),
                            "train_mean": float(train_mean),
                            "abs_diff": abs(float(means[i]) - float(train_mean)),
                        }
                    )

    if not category_flags:
        notes.append("No encoder category leakage flags found.")
    if not scaler_flags:
        notes.append("No scaler mean mismatch flags above tolerance.")

    return {
        "fit_status": fit_status,
        "category_leakage_flags": category_flags,
        "scaler_leakage_flags": scaler_flags,
        "tolerance_used": tolerance,
        "notes": notes,
    }


def build_risk_summary(
    leakage_flags=None,
    split_integrity=None,
    alignment_train=None,
    alignment_test=None,
    missingness_df=None,
    outliers_df=None,
    preprocessing_audit=None,
    alpha=0.05,
    report_top_n=30,
):
    """Build a ranked risk summary table across all audits."""
    rows = []

    if leakage_flags is not None and not leakage_flags.empty:
        for _, row in leakage_flags.iterrows():
            rows.append(
                {
                    "issue_type": "leakage_flag",
                    "feature_or_id": row["feature"],
                    "severity": row.get("severity", "high"),
                    "key_stat": row.get("reason", ""),
                    "recommendation": "Remove or justify feature before training.",
                }
            )

    if split_integrity is not None:
        if split_integrity.get("duplicates_across_splits_count", 0) > 0:
            rows.append(
                {
                    "issue_type": "split_integrity",
                    "feature_or_id": "duplicate_rows",
                    "severity": "high",
                    "key_stat": split_integrity["duplicates_across_splits_count"],
                    "recommendation": "Rebuild split with strict disjointness constraints.",
                }
            )
        if split_integrity.get("overlapping_convoy_ids_count", 0) > 0:
            rows.append(
                {
                    "issue_type": "group_leakage",
                    "feature_or_id": "convoy_id_overlap",
                    "severity": "high",
                    "key_stat": split_integrity["overlapping_convoy_ids_count"],
                    "recommendation": "Use group-aware split by convoy identifier.",
                }
            )
        if split_integrity.get("overlapping_ids_count", 0) > 0:
            rows.append(
                {
                    "issue_type": "id_overlap",
                    "feature_or_id": "id_overlap",
                    "severity": "high",
                    "key_stat": split_integrity["overlapping_ids_count"],
                    "recommendation": "Ensure unique ids never cross train/test.",
                }
            )

    for tag, rep in [("alignment_train", alignment_train), ("alignment_test", alignment_test)]:
        if rep is None:
            continue
        if not rep.get("xy_length_match", True) or not rep.get("xy_index_match", True):
            rows.append(
                {
                    "issue_type": tag,
                    "feature_or_id": "X/y_alignment",
                    "severity": "high",
                    "key_stat": f"len_match={rep.get('xy_length_match')} idx_match={rep.get('xy_index_match')}",
                    "recommendation": "; ".join(rep.get("suggested_fixes", [])),
                }
            )

    if missingness_df is not None and not missingness_df.empty:
        sig = missingness_df[missingness_df["p_adj"] < alpha].copy()
        sig = sig.sort_values("p_adj")
        for _, row in sig.head(report_top_n).iterrows():
            rows.append(
                {
                    "issue_type": "missingness_disparity",
                    "feature_or_id": row["feature"],
                    "severity": "medium",
                    "key_stat": f"p_adj={row['p_adj']:.4g}; {row.get('direction','')}",
                    "recommendation": "Investigate mechanism of missingness and add robust imputation indicators.",
                }
            )

    if outliers_df is not None and not outliers_df.empty:
        for _, row in outliers_df.iterrows():
            note = str(row.get("notes", ""))
            if note:
                rows.append(
                    {
                        "issue_type": "outlier_group_concentration",
                        "feature_or_id": row["feature"],
                        "severity": "medium",
                        "key_stat": note,
                        "recommendation": "Inspect tails, apply clipping/winsorization, or robust transforms.",
                    }
                )

    if preprocessing_audit is not None:
        if preprocessing_audit.get("category_leakage_flags"):
            rows.append(
                {
                    "issue_type": "preprocess_category_leakage",
                    "feature_or_id": "encoder_categories",
                    "severity": "high",
                    "key_stat": len(preprocessing_audit["category_leakage_flags"]),
                    "recommendation": "Fit encoder on train only and re-run split transform.",
                }
            )
        if preprocessing_audit.get("scaler_leakage_flags"):
            rows.append(
                {
                    "issue_type": "preprocess_scaler_mismatch",
                    "feature_or_id": "scaler_mean_var",
                    "severity": "medium",
                    "key_stat": len(preprocessing_audit["scaler_leakage_flags"]),
                    "recommendation": "Refit scaler using train data only.",
                }
            )

    summary = pd.DataFrame(rows, columns=["issue_type", "feature_or_id", "severity", "key_stat", "recommendation"])
    if summary.empty:
        return summary

    severity_order = {"high": 0, "medium": 1, "low": 2}
    summary["_sev"] = summary["severity"].map(severity_order).fillna(3)
    summary = summary.sort_values(["_sev", "issue_type", "feature_or_id"]).drop(columns="_sev").head(report_top_n)
    return summary.reset_index(drop=True)
