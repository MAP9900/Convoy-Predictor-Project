"""Interpretability-focused visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def plot_feature_rank_bump_chart(
    triangulation_df,
    feature_col="feature",
    rank_cols=("perm_rank", "base_rank", "shap_rank"),
    top_k=10,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot a bump/slope chart for feature ranks across importance methods."""
    df = triangulation_df.copy()

    if "rank_avg" in df.columns:
        df = df.sort_values("rank_avg", ascending=True)
    elif "rank_sum" in df.columns:
        df = df.sort_values("rank_sum", ascending=True)
    else:
        df["rank_proxy"] = df[list(rank_cols)].mean(axis=1)
        df = df.sort_values("rank_proxy", ascending=True)

    df = df.head(top_k).copy()
    df = df.dropna(subset=list(rank_cols))

    x = np.arange(len(rank_cols))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="lightgrey")
    ax.set_facecolor("lightgrey")

    for i, (_, row) in enumerate(df.iterrows()):
        y = [row[col] for col in rank_cols]
        color = "#06768d" if i % 2 == 0 else "#fc6f03"
        ax.plot(x, y, marker="o", linewidth=2, color=color, alpha=0.9)

        ax.text(x[0] - 0.06, y[0], str(row[feature_col]), ha="right", va="center", fontsize=9)
        ax.text(x[-1] + 0.06, y[-1], str(row[feature_col]), ha="left", va="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_rank", "").upper() for c in rank_cols])
    ax.set_ylabel("Rank (Lower is More Important)")
    ax.set_title(f"Feature Rank Bump Chart (Top {top_k})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_Feature_Rank_Bump_Chart.png")
    plt.show()

    return df[[feature_col] + list(rank_cols)]


def plot_fn_focused_insight_view(
    fn_shap_df,
    feature_col="feature",
    delta_col="delta_fn_vs_global",
    top_k=10,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot FN-focused SHAP deltas to highlight features amplified in misses."""
    df = fn_shap_df.copy()
    df = df.sort_values(delta_col, ascending=False).head(top_k)

    colors = ["#06768d" if v >= 0 else "#ab0003" for v in df[delta_col]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="lightgrey")
    ax.set_facecolor("lightgrey")

    ax.barh(df[feature_col], df[delta_col], color=colors, alpha=0.9)
    ax.axvline(x=0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"FN-Focused Insight View (Top {top_k} by Delta)")
    ax.set_xlabel("FN SHAP Delta vs Global SHAP")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_FN_Focused_Insight_View.png")
    plt.show()

    return df[[feature_col, delta_col]]
