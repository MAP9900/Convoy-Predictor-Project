import itertools

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_GROUP_ORDER = ("FN", "FP", "TP", "TN")
DEFAULT_COMPARISONS = (("FN", "TP"), ("FP", "TN"), ("FN", "FP"))
EXCLUDED_NUMERIC_COLUMNS = {
    "At Risk (0/1)",
    "Predicted At Risk (0/1)",
    "Pred_Prob",
    "Risk",
    "Overall Sink Percentage",
    "Is_Test",
    "Is_False_Negative",
    "Is_False_Positive",
    "Is_True_Positive",
    "Is_True_Negative",
}


def adjust_pvalues(pvalues, method="fdr_bh"):
    # Multiple-testing correction for a 1D sequence of p-values.
    pvals = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvals, np.nan, dtype=float)
    valid_mask = np.isfinite(pvals)
    valid_p = pvals[valid_mask]

    if valid_p.size == 0:
        return adjusted

    method_key = method.lower()
    m = valid_p.size
    order = np.argsort(valid_p)
    ranked = valid_p[order]

    if method_key in {"fdr_bh", "bh", "fdr"}:
        ranks = np.arange(1, m + 1)
        bh = ranked * m / ranks
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        corrected_sorted = np.clip(bh, 0.0, 1.0)
    elif method_key == "holm":
        factors = m - np.arange(m)
        holm = ranked * factors
        holm = np.maximum.accumulate(holm)
        corrected_sorted = np.clip(holm, 0.0, 1.0)
    else:
        raise ValueError("method must be one of: 'fdr_bh' or 'holm'")

    corrected = np.empty_like(corrected_sorted)
    corrected[order] = corrected_sorted
    adjusted[valid_mask] = corrected
    return adjusted


def cliffs_delta(group1, group2):
    # Manual Cliff's delta: P(group1 > group2) - P(group1 < group2).
    g1 = np.asarray(pd.Series(group1).dropna(), dtype=float)
    g2 = np.asarray(pd.Series(group2).dropna(), dtype=float)
    n1 = g1.size
    n2 = g2.size

    if n1 == 0 or n2 == 0:
        return np.nan, "insufficient_data"

    diffs = np.subtract.outer(g1, g2)
    greater = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    delta = (greater - less) / (n1 * n2)
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return float(delta), magnitude


def rank_biserial_correlation(u_stat, n1, n2):
    # Rank-biserial correlation derived from Mann-Whitney U.
    if n1 == 0 or n2 == 0:
        return np.nan
    return (2.0 * u_stat / (n1 * n2)) - 1.0


def _numeric_feature_candidates(groups):
    # Keep only numeric columns shared by every group.
    common = None
    for frame in groups.values():
        numeric_cols = set(frame.select_dtypes(include=[np.number]).columns)
        common = numeric_cols if common is None else common.intersection(numeric_cols)
    if not common:
        return []
    filtered = [col for col in sorted(common) if col not in EXCLUDED_NUMERIC_COLUMNS]
    return filtered


def run_global_kruskal(
    groups,
    features=None,
    group_order=DEFAULT_GROUP_ORDER,
    p_adjust_method="fdr_bh",
):
    # Global 4-group screen feature-by-feature (Kruskal-Wallis).
    if features is None:
        features = _numeric_feature_candidates(groups)

    results = []
    for feature in features:
        samples = []
        medians = {}
        valid = True

        for group_name in group_order:
            frame = groups[group_name]
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            samples.append(values.values)
            medians[f"median_{group_name}"] = float(values.median()) if not values.empty else np.nan
            if values.empty:
                valid = False

        if valid:
            h_stat, p_raw = stats.kruskal(*samples)
        else:
            h_stat, p_raw = np.nan, np.nan

        row = {
            "feature": feature,
            "kruskal_h": h_stat,
            "kruskal_p_raw": p_raw,
        }
        row.update(medians)
        results.append(row)

    global_df = pd.DataFrame(results)
    if global_df.empty:
        return global_df

    global_df["kruskal_p_adj"] = adjust_pvalues(global_df["kruskal_p_raw"].to_numpy(), method=p_adjust_method)
    global_df = global_df.sort_values(["kruskal_p_adj", "kruskal_p_raw"], na_position="last").reset_index(drop=True)
    return global_df


def run_pairwise_tests(
    groups,
    features=None,
    comparisons=DEFAULT_COMPARISONS,
    p_adjust_method="fdr_bh",
    kruskal_results=None,
):
    # Pairwise Mann-Whitney U tests with effect sizes for each feature.
    if features is None:
        features = _numeric_feature_candidates(groups)

    kruskal_map = {}
    if kruskal_results is not None and not kruskal_results.empty:
        kruskal_map = kruskal_results.set_index("feature")["kruskal_p_adj"].to_dict()

    rows = []
    for feature in features:
        for group1, group2 in comparisons:
            s1 = pd.to_numeric(groups[group1][feature], errors="coerce").dropna()
            s2 = pd.to_numeric(groups[group2][feature], errors="coerce").dropna()

            median1 = float(s1.median()) if not s1.empty else np.nan
            median2 = float(s2.median()) if not s2.empty else np.nan

            if s1.empty or s2.empty:
                u_stat, p_raw = np.nan, np.nan
                delta, magnitude = np.nan, "insufficient_data"
                rbc = np.nan
            else:
                u_stat, p_raw = stats.mannwhitneyu(s1, s2, alternative="two-sided")
                delta, magnitude = cliffs_delta(s1, s2)
                rbc = rank_biserial_correlation(u_stat, len(s1), len(s2))

            if np.isnan(median1) or np.isnan(median2):
                direction = "insufficient_data"
            elif median1 > median2:
                direction = f"{group1} > {group2}"
            elif median1 < median2:
                direction = f"{group2} > {group1}"
            else:
                direction = "equal_median"

            rows.append(
                {
                    "feature": feature,
                    "comparison": f"{group1} vs {group2}",
                    "group1": group1,
                    "group2": group2,
                    "median_group1": median1,
                    "median_group2": median2,
                    "u_stat": u_stat,
                    "p_raw": p_raw,
                    "effect_size": delta,
                    "effect_magnitude": magnitude,
                    "rank_biserial": rbc,
                    "direction": direction,
                    "kruskal_p_adj": kruskal_map.get(feature, np.nan),
                }
            )

    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        return pairwise_df

    pairwise_df["p_adj"] = adjust_pvalues(pairwise_df["p_raw"].to_numpy(), method=p_adjust_method)
    return pairwise_df


def run_conditional_targeted_tests(
    groups,
    kruskal_results,
    alpha=0.05,
    p_adjust_method="holm",
    comparisons=None,
):
    # Post-hoc testing restricted to features passing global screen.
    if kruskal_results is None or kruskal_results.empty:
        return pd.DataFrame()

    significant = kruskal_results.loc[kruskal_results["kruskal_p_adj"] < alpha, "feature"].tolist()
    if not significant:
        return pd.DataFrame()

    if comparisons is None:
        # Full targeted post-hoc across all 4 groups using MW pairwise tests.
        comparisons = tuple(itertools.combinations(DEFAULT_GROUP_ORDER, 2))

    return run_pairwise_tests(
        groups=groups,
        features=significant,
        comparisons=comparisons,
        p_adjust_method=p_adjust_method,
        kruskal_results=kruskal_results,
    )


def build_summary_table(global_kruskal_results, pairwise_results):
    # Final tidy table ranked by global significance then effect size.
    if pairwise_results is None or pairwise_results.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "comparison",
                "median_group1",
                "median_group2",
                "p_raw",
                "p_adj",
                "effect_size",
                "effect_magnitude",
                "direction",
                "kruskal_p_adj",
                "rank_biserial",
            ]
        )

    summary = pairwise_results.copy()

    if (
        "kruskal_p_adj" not in summary.columns
        and global_kruskal_results is not None
        and not global_kruskal_results.empty
    ):
        k_map = global_kruskal_results.set_index("feature")["kruskal_p_adj"].to_dict()
        summary["kruskal_p_adj"] = summary["feature"].map(k_map)

    summary = summary[
        [
            "feature",
            "comparison",
            "median_group1",
            "median_group2",
            "p_raw",
            "p_adj",
            "effect_size",
            "effect_magnitude",
            "direction",
            "kruskal_p_adj",
            "rank_biserial",
        ]
    ].copy()

    summary["abs_effect_size"] = summary["effect_size"].abs()
    summary = summary.sort_values(
        ["kruskal_p_adj", "abs_effect_size"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["abs_effect_size"])
    summary = summary.reset_index(drop=True)
    return summary
