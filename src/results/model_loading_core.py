import random
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.ML_Class_2 import Model_Tester_V2
from src.models.model_specs import MODEL_SPECS


warnings.filterwarnings("ignore", module="skopt")
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 1945


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_shared_split(X, y, *, train_size: float = 0.8, random_state: int = SEED):
    return train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_tester(
    model_key,
    *,
    feature_names,
    X_train,
    X_test,
    y_train,
    y_test,
    scaler=None,
    cv_folds=None,
):
    spec = MODEL_SPECS[model_key]
    tester = Model_Tester_V2(
        model=spec["estimator"],
        scaler=scaler,
        parameter_grid=spec["grid_large"],
        cv_folds=cv_folds or spec.get("cv_folds", 5),
        feature_names=feature_names,
        model_config=spec["config"],
    )
    tester.X_train, tester.X_test = X_train, X_test
    tester.y_train, tester.y_test = y_train, y_test
    if callable(tester.parameter_grid):
        tester.parameter_grid = tester.parameter_grid(tester.y_train)
    return tester


def evaluate_voting_ensemble(
    estimators_dict,
    X_train,
    X_test,
    y_train,
    y_test,
    threshold=0.5,
    voting="soft",
    ensemble_name=None,
    pos_label=1,
    verbose=True,
    weights=None,
):
    estimators_list = [(name, est) for name, est in estimators_dict.items()]
    if ensemble_name is None:
        ensemble_name = "VotingEnsemble_" + "_".join(sorted(estimators_dict.keys()))

    voter = VotingClassifier(
        estimators=estimators_list,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )
    voter.fit(X_train, y_train)

    if voting == "soft":
        proba_pos = voter.predict_proba(X_test)[:, 1]
        y_pred = (proba_pos >= threshold).astype(int)
    else:
        y_pred = voter.predict(X_test)
        proba_pos = None

    roc_auc = roc_auc_score(y_test, proba_pos) if proba_pos is not None else np.nan
    acc = np.mean(y_pred == y_test)
    recall_1 = recall_score(y_test, y_pred, pos_label=pos_label)
    precision_1 = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    cls_report = classification_report(y_test, y_pred, digits=3, zero_division=0)

    if verbose:
        print(f"Ensemble: {ensemble_name}")
        print(f"Voting: {voting}")
        if voting == "soft":
            print(f"Threshold: {threshold:.2f}")
        if weights is not None:
            print(f"Weights: {weights}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Recall (class {pos_label}): {recall_1:.3f}")
        print(f"Precision (class {pos_label}): {precision_1:.3f}")
        print(f"F1 (class {pos_label}): {f1_1:.3f}")
        print(f"MCC: {mcc:.3f}")
        print(f"Balanced Accuracy: {bal_acc:.3f}")
        print("Confusion Matrix:")
        print(cm_df)
        print("Classification Report (digits=3):")
        print(cls_report)

    results_df = pd.DataFrame(
        [
            {
                "Model": ensemble_name,
                "Threshold": threshold if voting == "soft" else np.nan,
                "Acc": acc,
                "ROC_AUC": roc_auc,
                "MCC": mcc,
                "Bal_Acc": bal_acc,
                "Recall_1": recall_1,
                "Precision_1": precision_1,
                "F1_1": f1_1,
            }
        ]
    )

    return results_df, voter, cm_df, cls_report
