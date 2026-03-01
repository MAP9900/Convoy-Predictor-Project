"""Case-study card visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def _select_case_row(df, strategy="max"):
    if df is None or df.empty:
        return None
    if "Pred_Prob" in df.columns:
        if strategy == "min":
            return df.sort_values("Pred_Prob", ascending=True).iloc[0]
        return df.sort_values("Pred_Prob", ascending=False).iloc[0]
    return df.iloc[0]


def _build_card_text(row, group_name):
    if row is None:
        return f"{group_name}\n\nNo rows available"

    fields = [
        ("Convoy", "Convoy Number"),
        ("Pred Prob", "Pred_Prob"),
        ("True Label", "At Risk (0/1)"),
        ("Pred Label", "Predicted At Risk (0/1)"),
        ("Ships", "Number of Ships"),
        ("Escorts", "Number of Escort Ships"),
        ("Stragglers", "Number of Stragglers"),
        ("Tons", "Total Tons of Convoy"),
        ("U-Boats", "Avg Number of U-Boats in Atlantic"),
        ("Month", "Month"),
        ("Year", "Year"),
    ]

    lines = [group_name, ""]
    for label, col in fields:
        if col in row.index:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                if col == "Pred_Prob":
                    lines.append(f"{label}: {val:.3f}")
                elif col in {"Total Tons of Convoy"}:
                    lines.append(f"{label}: {val:,.0f}")
                else:
                    lines.append(f"{label}: {val:.2f}")
            else:
                lines.append(f"{label}: {val}")

    return "\n".join(lines)


def plot_case_study_cards(
    true_positive_df,
    false_negative_df,
    false_positive_df,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    """Plot three case-study cards (TP, FN, FP) for report storytelling."""
    tp_row = _select_case_row(true_positive_df, strategy="max")
    fn_row = _select_case_row(false_negative_df, strategy="max")
    fp_row = _select_case_row(false_positive_df, strategy="max")

    cards = [
        ("True Positive Case", tp_row),
        ("False Negative Case", fn_row),
        ("False Positive Case", fp_row),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="lightgrey")

    for ax, (title, row) in zip(axes, cards):
        ax.set_facecolor("lightgrey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

        text = _build_card_text(row, title)
        ax.text(
            0.03,
            0.97,
            text,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "boxstyle": "round,pad=0.5"},
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_Case_Study_Cards.png")
    plt.show()

    return {
        "tp": None if tp_row is None else tp_row.to_dict(),
        "fn": None if fn_row is None else fn_row.to_dict(),
        "fp": None if fp_row is None else fp_row.to_dict(),
    }
