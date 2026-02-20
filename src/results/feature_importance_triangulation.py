"""Feature importance triangulation across permutation, native model, and SHAP views."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


def _prepare_feature_matrix(X_val, X_val_transformed=None, preprocess=None):
    # Choose the matrix that downstream models will consume.
    if X_val_transformed is not None:
        return X_val_transformed

    if preprocess is not None:
        if hasattr(preprocess, "transform"):
            return preprocess.transform(X_val)
        if callable(preprocess):
            return preprocess(X_val)

    return X_val


def _resolve_feature_names(X_val, feature_names=None, expected_len=None):
    if feature_names is None:
        if isinstance(X_val, pd.DataFrame):
            names = list(X_val.columns)
        elif expected_len is not None:
            names = [f"feature_{i}" for i in range(expected_len)]
        else:
            raise ValueError("feature_names must be provided when X_val is not a DataFrame.")
    else:
        names = list(feature_names)

    if expected_len is not None and len(names) != expected_len:
        names = [f"feature_{i}" for i in range(expected_len)]
    return names


def _align_importance_vector(values, feature_names):
    # Align vector length to feature length to keep downstream merges stable.
    vec = np.asarray(values, dtype=float).ravel()
    target = len(feature_names)
    out = np.full(target, np.nan, dtype=float)
    n = min(target, vec.size)
    out[:n] = vec[:n]
    return out


def _rank_desc(series):
    return series.rank(method="min", ascending=False)


def _minmax_norm(series):
    if series is None:
        return pd.Series(dtype=float)
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    vmin, vmax = float(valid.min()), float(valid.max())
    if np.isclose(vmin, vmax):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (vals - vmin) / (vmax - vmin)


def _extract_native_importance(model):
    # Native model importance: feature_importances_ preferred, else |coef_|.
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float), "feature_importances_"

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            imp = np.mean(np.abs(coef), axis=0)
        return np.asarray(imp, dtype=float), "coef_"

    return None, "none"


def _unwrap_model(model):
    # Handle calibrated wrappers or pipelines where possible.
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        est = getattr(model.calibrated_classifiers_[0], "estimator", model)
    else:
        est = model

    if hasattr(est, "named_steps") and "model" in est.named_steps:
        return est.named_steps["model"]
    return est


def _is_tree_compatible(model):
    est = _unwrap_model(model)
    cls_name = est.__class__.__name__.lower()
    module_name = est.__class__.__module__.lower()

    tree_markers = [
        "randomforest",
        "extratrees",
        "gradientboosting",
        "histgradientboosting",
        "decisiontree",
        "xgb",
        "lgbm",
        "catboost",
    ]
    if any(marker in cls_name for marker in tree_markers):
        return True
    if "xgboost" in module_name or "lightgbm" in module_name or "catboost" in module_name:
        return True
    return False


def _reduce_shap_values(shap_values, n_features):
    # Convert different SHAP output shapes to sample x feature matrix.
    values = shap_values[-1] if isinstance(shap_values, list) else shap_values
    arr = np.asarray(values)

    if arr.ndim == 3:
        # Common binary-class layout: samples x features x classes.
        if arr.shape[1] == n_features:
            arr = np.mean(np.abs(arr), axis=2)
        elif arr.shape[2] == n_features:
            arr = np.mean(np.abs(arr), axis=1)
        else:
            arr = np.abs(arr.reshape(arr.shape[0], -1)[:, :n_features])
    elif arr.ndim == 2:
        arr = np.abs(arr[:, :n_features])
    elif arr.ndim == 1:
        arr = np.abs(arr[:n_features]).reshape(1, -1)
    else:
        arr = np.abs(arr.reshape(arr.shape[0], -1)[:, :n_features])

    return np.asarray(arr, dtype=float)


def compute_permutation_importance(
    ensemble_model,
    X_val,
    y_val,
    feature_names=None,
    X_val_transformed=None,
    preprocess=None,
    n_repeats_perm=10,
    scoring="balanced_accuracy",
    random_state=1945,
):
    """Compute ensemble-level permutation importance on validation data."""
    X_eval = _prepare_feature_matrix(X_val, X_val_transformed=X_val_transformed, preprocess=preprocess)
    y_eval = np.asarray(y_val)
    if y_eval.shape[0] != len(X_eval):
        raise ValueError("y_val length must match X_val rows.")

    resolved_names = _resolve_feature_names(X_val, feature_names=feature_names, expected_len=X_eval.shape[1])

    perm = permutation_importance(
        estimator=ensemble_model,
        X=X_eval,
        y=y_eval,
        n_repeats=n_repeats_perm,
        scoring=scoring,
        random_state=random_state,
    )
    out = pd.DataFrame(
        {
            "feature": resolved_names,
            "perm_mean": _align_importance_vector(perm.importances_mean, resolved_names),
            "perm_std": _align_importance_vector(perm.importances_std, resolved_names),
        }
    )
    out["perm_rank"] = _rank_desc(out["perm_mean"])
    out = out.sort_values("perm_rank").reset_index(drop=True)
    return out


def compute_base_model_importance(base_models, feature_names):
    """Aggregate native importance across base models (mean/variance/CV)."""
    model_vectors = []
    model_details = []

    for model_name, model in base_models.items():
        est = _unwrap_model(model)
        raw_imp, source = _extract_native_importance(est)

        if raw_imp is None:
            model_details.append({"model": model_name, "used": False, "source": source, "note": "No native importance"})
            continue

        aligned = _align_importance_vector(raw_imp, feature_names)
        model_vectors.append(aligned)
        model_details.append({"model": model_name, "used": True, "source": source, "note": ""})

    if not model_vectors:
        return {
            "importance_df": pd.DataFrame(
                {
                    "feature": feature_names,
                    "base_mean": np.nan,
                    "base_var": np.nan,
                    "base_cv": np.nan,
                    "base_rank": np.nan,
                }
            ),
            "model_details": pd.DataFrame(model_details),
            "note": "No base models exposed native importance.",
        }

    mat = np.vstack(model_vectors)
    base_mean = np.nanmean(mat, axis=0)
    base_var = np.nanvar(mat, axis=0)
    base_std = np.sqrt(base_var)

    with np.errstate(divide="ignore", invalid="ignore"):
        base_cv = np.where(np.abs(base_mean) > 1e-12, base_std / np.abs(base_mean), np.nan)

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "base_mean": base_mean,
            "base_var": base_var,
            "base_cv": base_cv,
        }
    )
    out["base_rank"] = _rank_desc(out["base_mean"])
    out = out.sort_values("base_rank").reset_index(drop=True)

    return {
        "importance_df": out,
        "model_details": pd.DataFrame(model_details),
        "note": "",
    }


def compute_shap_importance(
    base_models,
    X_val,
    feature_names=None,
    X_val_transformed=None,
    preprocess=None,
):
    """Compute aggregated SHAP importance for tree-compatible base models only."""
    X_eval = _prepare_feature_matrix(X_val, X_val_transformed=X_val_transformed, preprocess=preprocess)
    resolved_names = _resolve_feature_names(X_val, feature_names=feature_names, expected_len=X_eval.shape[1])

    try:
        import shap
    except ImportError:
        empty = pd.DataFrame(
            {
                "feature": resolved_names,
                "shap_mean": np.nan,
                "shap_var": np.nan,
                "shap_rank": np.nan,
            }
        )
        return {
            "importance_df": empty,
            "tree_models_used": [],
            "skipped_models": list(base_models.keys()),
            "sample_abs_shap_mean": None,
            "note": "SHAP library is not installed.",
        }

    per_model_feature_means = []
    per_model_sample_abs = []
    used_models = []
    skipped_models = []

    for model_name, model in base_models.items():
        if not _is_tree_compatible(model):
            skipped_models.append(model_name)
            continue

        est = _unwrap_model(model)
        try:
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_eval)
            abs_matrix = _reduce_shap_values(shap_values, n_features=len(resolved_names))
            # Ensure sample x feature shape.
            if abs_matrix.ndim != 2:
                skipped_models.append(model_name)
                continue
            # Align feature width if needed.
            if abs_matrix.shape[1] != len(resolved_names):
                aligned = np.full((abs_matrix.shape[0], len(resolved_names)), np.nan, dtype=float)
                n = min(abs_matrix.shape[1], len(resolved_names))
                aligned[:, :n] = abs_matrix[:, :n]
                abs_matrix = aligned

            per_model_sample_abs.append(abs_matrix)
            per_model_feature_means.append(np.nanmean(abs_matrix, axis=0))
            used_models.append(model_name)
        except Exception:
            skipped_models.append(model_name)
            continue

    if not per_model_feature_means:
        empty = pd.DataFrame(
            {
                "feature": resolved_names,
                "shap_mean": np.nan,
                "shap_var": np.nan,
                "shap_rank": np.nan,
            }
        )
        return {
            "importance_df": empty,
            "tree_models_used": used_models,
            "skipped_models": skipped_models,
            "sample_abs_shap_mean": None,
            "note": "No tree-compatible base models produced SHAP values.",
        }

    mat = np.vstack(per_model_feature_means)
    shap_mean = np.nanmean(mat, axis=0)
    shap_var = np.nanvar(mat, axis=0)

    out = pd.DataFrame(
        {
            "feature": resolved_names,
            "shap_mean": shap_mean,
            "shap_var": shap_var,
        }
    )
    out["shap_rank"] = _rank_desc(out["shap_mean"])
    out = out.sort_values("shap_rank").reset_index(drop=True)

    sample_abs_shap_mean = np.nanmean(np.stack(per_model_sample_abs, axis=0), axis=0)
    return {
        "importance_df": out,
        "tree_models_used": used_models,
        "skipped_models": skipped_models,
        "sample_abs_shap_mean": sample_abs_shap_mean,  # sample x feature
        "note": "",
    }


def normalize_importances(comparison_df):
    """Add normalized method scores and combined rank summary columns."""
    out = comparison_df.copy()
    out["perm_norm"] = _minmax_norm(out.get("perm_mean"))
    out["base_norm"] = _minmax_norm(out.get("base_mean"))
    out["shap_norm"] = _minmax_norm(out.get("shap_mean"))

    rank_cols = [col for col in ["perm_rank", "base_rank", "shap_rank"] if col in out.columns]
    out["rank_sum"] = out[rank_cols].sum(axis=1, min_count=1)
    out["rank_avg"] = out[rank_cols].mean(axis=1)
    return out


def compute_rank_agreement(comparison_df):
    """Compute pairwise Spearman rank agreement between methods."""
    pairs = [
        ("perm_rank", "base_rank"),
        ("perm_rank", "shap_rank"),
        ("base_rank", "shap_rank"),
    ]
    rows = []
    for col1, col2 in pairs:
        if col1 not in comparison_df.columns or col2 not in comparison_df.columns:
            rows.append({"pair": f"{col1} vs {col2}", "spearman_rho": np.nan, "p_value": np.nan, "n_features": 0})
            continue

        aligned = comparison_df[[col1, col2]].dropna()
        if aligned.empty:
            rows.append({"pair": f"{col1} vs {col2}", "spearman_rho": np.nan, "p_value": np.nan, "n_features": 0})
            continue

        rho, pval = spearmanr(aligned[col1], aligned[col2])
        rows.append(
            {
                "pair": f"{col1} vs {col2}",
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_features": int(aligned.shape[0]),
            }
        )

    agreement_df = pd.DataFrame(rows)
    matrix = pd.DataFrame(
        np.nan,
        index=["perm_rank", "base_rank", "shap_rank"],
        columns=["perm_rank", "base_rank", "shap_rank"],
    )
    for _, row in agreement_df.iterrows():
        left, _, right = row["pair"].partition(" vs ")
        matrix.loc[left, right] = row["spearman_rho"]
        matrix.loc[right, left] = row["spearman_rho"]
    np.fill_diagonal(matrix.values, 1.0)
    return {"agreement_table": agreement_df, "agreement_matrix": matrix}


def identify_stable_unstable_features(
    comparison_df,
    top_k=20,
    rank_spread_threshold=20,
    base_var_threshold=None,
    shap_var_threshold=None,
    variance_quantile=0.90,
):
    """Classify features as stable/unstable using rank overlap and disagreement rules."""
    out = comparison_df.copy()

    # Count in how many methods the feature lands in top_k.
    top_hits = np.zeros(len(out), dtype=int)
    for rank_col in ["perm_rank", "base_rank", "shap_rank"]:
        if rank_col in out.columns:
            top_hits += ((pd.to_numeric(out[rank_col], errors="coerce") <= top_k).fillna(False)).astype(int).to_numpy()
    out["stability_count"] = top_hits
    out["is_stable"] = out["stability_count"] >= 2

    # Rank spread among available methods.
    rank_cols = [c for c in ["perm_rank", "base_rank", "shap_rank"] if c in out.columns]
    out["rank_min"] = out[rank_cols].min(axis=1)
    out["rank_max"] = out[rank_cols].max(axis=1)
    out["rank_spread"] = out["rank_max"] - out["rank_min"]

    # Auto variance thresholds if user did not provide explicit values.
    if base_var_threshold is None and "base_var" in out.columns:
        base_var_threshold = float(pd.to_numeric(out["base_var"], errors="coerce").dropna().quantile(variance_quantile))
    if shap_var_threshold is None and "shap_var" in out.columns:
        shap_var_threshold = float(pd.to_numeric(out["shap_var"], errors="coerce").dropna().quantile(variance_quantile))

    base_var_flag = (
        (pd.to_numeric(out.get("base_var"), errors="coerce") >= base_var_threshold)
        if base_var_threshold is not None and "base_var" in out.columns
        else pd.Series(False, index=out.index)
    )
    shap_var_flag = (
        (pd.to_numeric(out.get("shap_var"), errors="coerce") >= shap_var_threshold)
        if shap_var_threshold is not None and "shap_var" in out.columns
        else pd.Series(False, index=out.index)
    )
    spread_flag = out["rank_spread"] >= rank_spread_threshold

    out["is_unstable"] = spread_flag | base_var_flag | shap_var_flag

    stable_features = out.loc[out["is_stable"], "feature"].tolist()
    unstable_features = out.loc[out["is_unstable"], "feature"].tolist()

    return {
        "feature_stability_df": out,
        "stable_features": stable_features,
        "unstable_features": unstable_features,
        "thresholds": {
            "top_k": top_k,
            "rank_spread_threshold": rank_spread_threshold,
            "base_var_threshold": base_var_threshold,
            "shap_var_threshold": shap_var_threshold,
        },
    }


def fn_specific_shap_analysis(shap_report, fn_mask=None, top_k=20):
    """Compute FN-only SHAP importance and compare against global SHAP importance."""
    if shap_report is None or shap_report.get("sample_abs_shap_mean") is None:
        return {
            "fn_shap_df": pd.DataFrame(),
            "fn_top_features": pd.DataFrame(),
            "positive_delta_features": pd.DataFrame(),
            "note": "FN-specific SHAP unavailable (no SHAP sample matrix).",
        }

    if fn_mask is None:
        return {
            "fn_shap_df": pd.DataFrame(),
            "fn_top_features": pd.DataFrame(),
            "positive_delta_features": pd.DataFrame(),
            "note": "fn_mask not provided.",
        }

    sample_abs = np.asarray(shap_report["sample_abs_shap_mean"], dtype=float)  # sample x feature
    n_samples = sample_abs.shape[0]

    # Support bool mask, integer indices, or list-like.
    fn_mask_arr = np.asarray(fn_mask)
    if fn_mask_arr.dtype == bool and fn_mask_arr.size == n_samples:
        mask = fn_mask_arr
    else:
        mask = np.zeros(n_samples, dtype=bool)
        idx = fn_mask_arr.astype(int, copy=False)
        idx = idx[(idx >= 0) & (idx < n_samples)]
        mask[idx] = True

    if not np.any(mask):
        return {
            "fn_shap_df": pd.DataFrame(),
            "fn_top_features": pd.DataFrame(),
            "positive_delta_features": pd.DataFrame(),
            "note": "No FN samples selected by fn_mask.",
        }

    importance_df = shap_report["importance_df"].copy()
    if importance_df.empty:
        return {
            "fn_shap_df": pd.DataFrame(),
            "fn_top_features": pd.DataFrame(),
            "positive_delta_features": pd.DataFrame(),
            "note": "Global SHAP importance dataframe is empty.",
        }

    fn_shap_mean = np.nanmean(sample_abs[mask], axis=0)
    fn_shap_df = importance_df[["feature", "shap_mean"]].copy()
    fn_shap_df["fn_shap_mean"] = fn_shap_mean
    fn_shap_df["delta_fn_vs_global"] = fn_shap_df["fn_shap_mean"] - fn_shap_df["shap_mean"]
    fn_shap_df["fn_rank"] = _rank_desc(fn_shap_df["fn_shap_mean"])
    fn_shap_df = fn_shap_df.sort_values("fn_rank").reset_index(drop=True)

    fn_top = fn_shap_df.head(top_k).copy()
    positive_delta = fn_shap_df.sort_values("delta_fn_vs_global", ascending=False).head(top_k).copy()

    return {
        "fn_shap_df": fn_shap_df,
        "fn_top_features": fn_top,
        "positive_delta_features": positive_delta,
        "note": "",
    }


def build_triangulation_report(
    X_val,
    y_val,
    ensemble_model,
    base_models,
    feature_names=None,
    X_val_transformed=None,
    preprocess=None,
    fn_mask=None,
    top_k=20,
    n_repeats_perm=10,
    scoring="balanced_accuracy",
    random_state=1945,
    rank_spread_threshold=20,
    base_var_threshold=None,
    shap_var_threshold=None,
):
    """Run full triangulation workflow and return structured report objects."""
    X_eval = _prepare_feature_matrix(X_val, X_val_transformed=X_val_transformed, preprocess=preprocess)
    resolved_names = _resolve_feature_names(X_val, feature_names=feature_names, expected_len=X_eval.shape[1])

    perm_df = compute_permutation_importance(
        ensemble_model=ensemble_model,
        X_val=X_val,
        y_val=y_val,
        feature_names=resolved_names,
        X_val_transformed=X_val_transformed,
        preprocess=preprocess,
        n_repeats_perm=n_repeats_perm,
        scoring=scoring,
        random_state=random_state,
    )
    base_report = compute_base_model_importance(base_models=base_models, feature_names=resolved_names)
    shap_report = compute_shap_importance(
        base_models=base_models,
        X_val=X_val,
        feature_names=resolved_names,
        X_val_transformed=X_val_transformed,
        preprocess=preprocess,
    )

    # Outer-merge keeps features even when one method is missing.
    comparison_df = perm_df.merge(base_report["importance_df"], on="feature", how="outer")
    comparison_df = comparison_df.merge(shap_report["importance_df"], on="feature", how="outer")
    comparison_df = normalize_importances(comparison_df)
    comparison_df = comparison_df.sort_values(["rank_avg", "rank_sum"], na_position="last").reset_index(drop=True)

    agreement = compute_rank_agreement(comparison_df)
    stability = identify_stable_unstable_features(
        comparison_df,
        top_k=top_k,
        rank_spread_threshold=rank_spread_threshold,
        base_var_threshold=base_var_threshold,
        shap_var_threshold=shap_var_threshold,
    )
    fn_shap = fn_specific_shap_analysis(shap_report=shap_report, fn_mask=fn_mask, top_k=top_k)

    return {
        "comparison_df": comparison_df,
        "permutation_df": perm_df,
        "base_importance_df": base_report["importance_df"],
        "shap_importance_df": shap_report["importance_df"],
        "base_model_details": base_report["model_details"],
        "agreement_table": agreement["agreement_table"],
        "agreement_matrix": agreement["agreement_matrix"],
        "stability_df": stability["feature_stability_df"],
        "stable_features": stability["stable_features"],
        "unstable_features": stability["unstable_features"],
        "fn_shap_df": fn_shap["fn_shap_df"],
        "fn_shap_top_features": fn_shap["fn_top_features"],
        "fn_shap_positive_delta": fn_shap["positive_delta_features"],
        "notes": {
            "base": base_report["note"],
            "shap": shap_report["note"],
            "fn_shap": fn_shap["note"],
        },
    }
