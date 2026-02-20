import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance


DEFAULT_RESULTS_DIR = "/Users/matthewplambeck/Desktop/Convoy Predictor/results"


def plot_roc_curve(fpr, tpr, ROC_AUC, model_name, results_dir=DEFAULT_RESULTS_DIR):
    plt.figure(figsize=(6, 4), facecolor="lightgrey")
    plt.plot(fpr, tpr, color="#06768d", label=f"ROC Curve (AUC = {ROC_AUC:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FiveModel_CalSoft_t0.25 \n ROC Curve")
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.set_facecolor("lightgrey")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f"{results_dir}/{model_name}_PR_Curve.png")
    plt.show()


def plot_confusion_matrix(cm, model_name, class_labels=None, results_dir=DEFAULT_RESULTS_DIR):
    if class_labels is None:
        labels_to_use = [str(i) for i in range(cm.shape[0])]
    else:
        labels_to_use = [str(label) for label in class_labels]

    plt.figure(figsize=(6, 4), facecolor="lightgrey")
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_to_use,
        yticklabels=labels_to_use,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("FiveModel_CalSoft_t0.25 \n Confusion Matrix")
    plt.savefig(f"{results_dir}/{model_name}_CM.png")
    plt.show()


def plot_permutation_importance(
    voter_calsoft_025,
    X_test,
    y_test,
    feature_names,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    perm = permutation_importance(
        voter_calsoft_025,
        X_test,
        y_test,
        n_repeats=30,
        random_state=1945,
        scoring="recall",
    )

    perm_df = pd.DataFrame(
        {
            "Feature": feature_names[: len(perm.importances_mean)],
            "Importance": perm.importances_mean,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 4), facecolor="lightgrey")
    ax = sns.barplot(x="Importance", y="Feature", data=perm_df, palette="crest_r")
    plt.title("FiveModel_CalSoft_t0.25 \n Permutation Importance")
    ax.set_facecolor("lightgrey")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f"{results_dir}/{model_name}_Permutation_Importance.png")
    plt.show()

    return perm_df


def _extract_importance_from_estimator(est):
    if hasattr(est, "named_steps") and "model" in est.named_steps:
        est = est.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_)
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)
    return None


def _unwrap_calibrated(est):
    if hasattr(est, "calibrated_classifiers_") and len(est.calibrated_classifiers_) > 0:
        imps = []
        for cc in est.calibrated_classifiers_:
            base = getattr(cc, "estimator", None)
            if base is None:
                continue
            imp = _extract_importance_from_estimator(base)
            if imp is not None:
                imps.append(np.asarray(imp))
        if imps:
            min_len = min(len(i) for i in imps)
            return np.mean([i[:min_len] for i in imps], axis=0)
    return _extract_importance_from_estimator(est)


def plot_aggregated_base_model_importance(
    voter_calsoft_025,
    feature_names,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    base_imps = []
    for est in voter_calsoft_025.estimators_:
        imp = _unwrap_calibrated(est)
        if imp is not None:
            base_imps.append(np.asarray(imp))

    if not base_imps:
        print("No base estimators exposed feature importances/coefficients.")
        return pd.DataFrame(columns=["Feature", "Importance"])

    min_len = min(len(i) for i in base_imps)
    agg_imp = np.mean([i[:min_len] for i in base_imps], axis=0)
    agg_df = pd.DataFrame(
        {
            "Feature": feature_names[: len(agg_imp)],
            "Importance": agg_imp,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 4), facecolor="lightgrey")
    ax = sns.barplot(x="Importance", y="Feature", data=agg_df, palette="crest_r")
    plt.title("FiveModel_CalSoft_t0.25 \n Aggregated Base-Model Importance")
    ax.set_facecolor("lightgrey")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f"{results_dir}/{model_name}_Aggregated_Importance.png")
    plt.show()

    return agg_df


def plot_shap_importance(
    voter_calsoft_025,
    X_test,
    feature_names,
    model_name="FiveModel_CalSoft_t0.25",
    results_dir=DEFAULT_RESULTS_DIR,
):
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Install with: pip install shap")
        return pd.DataFrame(columns=["Feature", "Importance"])

    shap_imps = []

    def _unwrap_for_shap(est):
        if hasattr(est, "calibrated_classifiers_") and len(est.calibrated_classifiers_) > 0:
            return getattr(est.calibrated_classifiers_[0], "estimator", est)
        return est

    n_features = X_test.shape[1]
    active_feature_names = feature_names[:n_features]

    for est in voter_calsoft_025.estimators_:
        base = _unwrap_for_shap(est)
        if hasattr(base, "named_steps") and "model" in base.named_steps:
            base = base.named_steps["model"]

        if not hasattr(base, "predict"):
            continue

        try:
            explainer = shap.TreeExplainer(base)
            shap_vals = explainer.shap_values(X_test)

            if isinstance(shap_vals, list):
                vals = np.asarray(shap_vals[-1])
            else:
                vals = np.asarray(shap_vals)

            mean_abs = np.mean(np.abs(vals), axis=0)
            mean_abs = np.asarray(mean_abs).squeeze()

            if mean_abs.ndim > 1:
                if mean_abs.shape[0] == n_features:
                    mean_abs = np.mean(mean_abs, axis=tuple(range(1, mean_abs.ndim)))
                elif mean_abs.shape[-1] == n_features:
                    mean_abs = np.mean(mean_abs, axis=tuple(range(0, mean_abs.ndim - 1)))
                else:
                    mean_abs = np.ravel(mean_abs)[:n_features]

            mean_abs = np.ravel(mean_abs)
            if mean_abs.size >= n_features:
                shap_imps.append(mean_abs[:n_features])
        except Exception:
            continue

    if not shap_imps:
        print("No compatible tree estimators available for SHAP in this ensemble.")
        return pd.DataFrame(columns=["Feature", "Importance"])

    min_len = min(len(i) for i in shap_imps)
    shap_mean = np.mean([i[:min_len] for i in shap_imps], axis=0)
    shap_mean = np.ravel(shap_mean)

    shap_df = pd.DataFrame(
        {
            "Feature": active_feature_names[: len(shap_mean)],
            "Importance": shap_mean,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 4), facecolor="lightgrey")
    ax = sns.barplot(x="Importance", y="Feature", data=shap_df, palette="crest_r")
    plt.title("FiveModel_CalSoft_t0.25 \n SHAP Feature Importance (Tree Base Models)")
    ax.set_facecolor("lightgrey")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f"{results_dir}/{model_name}_SHAP_Importance.png")
    plt.show()

    return shap_df
