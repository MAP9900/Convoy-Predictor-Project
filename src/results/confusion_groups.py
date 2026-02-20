import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATA_PATH = "/Users/matthewplambeck/Desktop/Convoy Predictor/data/processed/Complete_Convoy_Data.csv"
DROP_COLUMNS_FOR_MODEL = [
    "Convoy Number",
    "Number of Ships Sunk",
    "Depart_Date",
    "Arrival/Dispersal Date",
    "Number of Escorts Sunk",
    "Number of Stragglers Sunk",
    "Total Tons of Ships Sunk",
    "Escort Sink Percentage",
    "Straggler Sink Percentage",
]
DROP_COLUMNS_FOR_OUTPUT = [
    "Depart_Date",
    "Arrival/Dispersal Date",
    "Number of Escorts Sunk",
    "Number of Stragglers Sunk",
    "Total Tons of Ships Sunk",
    "Escort Sink Percentage",
    "Straggler Sink Percentage",
]


def _build_scored_rows(voter_calsoft_025, threshold=0.25, data_path=DEFAULT_DATA_PATH):
    raw_df = pd.read_csv(data_path)
    raw_df = raw_df.drop(columns=["Unnamed: 0"]).reset_index(drop=True)

    model_df = raw_df.drop(columns=DROP_COLUMNS_FOR_MODEL)
    model_df["Risk"] = (model_df["Overall Sink Percentage"] > 0).astype(int)
    X_df = model_df.drop(columns=["Overall Sink Percentage", "Risk"])
    y_arr = model_df["Risk"].values

    row_idx = np.arange(len(X_df))
    _, X_test_meta, _, y_test_meta, _, idx_test = train_test_split(
        X_df,
        y_arr,
        row_idx,
        train_size=0.8,
        random_state=1945,
        stratify=y_arr,
    )

    y_proba = voter_calsoft_025.predict_proba(X_test_meta)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    all_rows_scored = raw_df.copy()
    all_rows_scored["At Risk (0/1)"] = np.nan
    all_rows_scored["Predicted At Risk (0/1)"] = np.nan
    all_rows_scored["Pred_Prob"] = np.nan
    all_rows_scored["Is_Test"] = False

    all_rows_scored.loc[idx_test, "At Risk (0/1)"] = y_test_meta
    all_rows_scored.loc[idx_test, "Predicted At Risk (0/1)"] = y_pred
    all_rows_scored.loc[idx_test, "Pred_Prob"] = y_proba
    all_rows_scored.loc[idx_test, "Is_Test"] = True

    return all_rows_scored, y_test_meta, y_pred


def get_false_negatives(voter_calsoft_025, threshold=0.25, data_path=DEFAULT_DATA_PATH):
    all_rows_scored, y_test_meta, y_pred = _build_scored_rows(
        voter_calsoft_025,
        threshold=threshold,
        data_path=data_path,
    )
    mask = (y_test_meta == 1) & (y_pred == 0)
    idx_test = all_rows_scored.index[all_rows_scored["Is_Test"]]
    all_rows_scored.loc[idx_test, "Is_False_Negative"] = mask

    false_negatives = all_rows_scored[all_rows_scored["Is_False_Negative"]].copy()
    false_negatives = false_negatives.sort_values("Pred_Prob")
    false_negatives = false_negatives.drop(columns=DROP_COLUMNS_FOR_OUTPUT)

    return all_rows_scored, false_negatives


def get_false_positives(voter_calsoft_025, threshold=0.25, data_path=DEFAULT_DATA_PATH):
    all_rows_scored, y_test_meta, y_pred = _build_scored_rows(
        voter_calsoft_025,
        threshold=threshold,
        data_path=data_path,
    )
    mask = (y_test_meta == 0) & (y_pred == 1)
    idx_test = all_rows_scored.index[all_rows_scored["Is_Test"]]
    all_rows_scored.loc[idx_test, "Is_False_Positive"] = mask

    false_positives = all_rows_scored[all_rows_scored["Is_False_Positive"]].copy()
    false_positives = false_positives.sort_values("Pred_Prob", ascending=False)
    false_positives = false_positives.drop(columns=DROP_COLUMNS_FOR_OUTPUT)

    return all_rows_scored, false_positives


def get_true_positives(voter_calsoft_025, threshold=0.25, data_path=DEFAULT_DATA_PATH):
    all_rows_scored, y_test_meta, y_pred = _build_scored_rows(
        voter_calsoft_025,
        threshold=threshold,
        data_path=data_path,
    )
    mask = (y_test_meta == 1) & (y_pred == 1)
    idx_test = all_rows_scored.index[all_rows_scored["Is_Test"]]
    all_rows_scored.loc[idx_test, "Is_True_Positive"] = mask

    true_positives = all_rows_scored[all_rows_scored["Is_True_Positive"]].copy()
    true_positives = true_positives.sort_values("Pred_Prob", ascending=False)
    true_positives = true_positives.drop(columns=DROP_COLUMNS_FOR_OUTPUT)

    return all_rows_scored, true_positives


def get_true_negatives(voter_calsoft_025, threshold=0.25, data_path=DEFAULT_DATA_PATH):
    all_rows_scored, y_test_meta, y_pred = _build_scored_rows(
        voter_calsoft_025,
        threshold=threshold,
        data_path=data_path,
    )
    mask = (y_test_meta == 0) & (y_pred == 0)
    idx_test = all_rows_scored.index[all_rows_scored["Is_Test"]]
    all_rows_scored.loc[idx_test, "Is_True_Negative"] = mask

    true_negatives = all_rows_scored[all_rows_scored["Is_True_Negative"]].copy()
    true_negatives = true_negatives.sort_values("Pred_Prob", ascending=True)
    true_negatives = true_negatives.drop(columns=DROP_COLUMNS_FOR_OUTPUT)

    return all_rows_scored, true_negatives


def compare_confusion_group_describes(
    false_negatives,
    false_positives,
    true_positives,
    true_negatives,
    include="all",
):
    groups = {
        "False_Negatives": false_negatives,
        "False_Positives": false_positives,
        "True_Positives": true_positives,
        "True_Negatives": true_negatives,
    }

    summary = {}
    for name, frame in groups.items():
        if frame is None or frame.empty:
            summary[name] = pd.DataFrame()
            continue

        if include == "numeric":
            desc = frame.describe(include=[np.number]).T
        elif include == "categorical":
            desc = frame.describe(include=["object", "category", "bool"]).T
        else:
            desc = frame.describe(include="all").T

        summary[name] = desc

    non_empty = {k: v for k, v in summary.items() if not v.empty}
    if not non_empty:
        return summary, pd.DataFrame()

    combined = pd.concat(non_empty, axis=1)
    return summary, combined
