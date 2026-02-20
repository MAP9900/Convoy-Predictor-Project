from sklearn.calibration import CalibratedClassifierCV

from src.models.model_artifacts import load_model
from src.results.model_loading_core import evaluate_voting_ensemble, prepare_tester


def load_final_ensemble_models(*, artifact_dir, feature_names, X_train, X_test, y_train, y_test):
    dt = prepare_tester(
        "dt",
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    load_model("dt", directory=artifact_dir, assign_to=dt)

    rf = prepare_tester(
        "rf",
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    load_model("rf", directory=artifact_dir, assign_to=rf)

    et = prepare_tester(
        "et",
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    load_model("et", directory=artifact_dir, assign_to=et)

    ada = prepare_tester(
        "ada",
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    load_model("ada", directory=artifact_dir, assign_to=ada)

    qda = prepare_tester(
        "qda",
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    load_model("qda", directory=artifact_dir, assign_to=qda)

    return {
        "dt": dt,
        "rf": rf,
        "et": et,
        "ada": ada,
        "qda": qda,
    }


def run_five_model_calsoft_t025(
    X_train,
    X_test,
    y_train,
    y_test,
    qda_ensemble,
    ada_ensemble,
    dt_ensemble,
    rf_ensemble,
    et_ensemble,
):
    five_model_ensemble = {
        "qda": qda_ensemble,
        "ada": ada_ensemble,
        "dt": dt_ensemble,
        "rf": rf_ensemble,
        "et": et_ensemble,
    }

    calibrated_five = {}
    for name, est in five_model_ensemble.items():
        cal = CalibratedClassifierCV(est, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        calibrated_five[name] = cal

    return evaluate_voting_ensemble(
        calibrated_five,
        X_train,
        X_test,
        y_train,
        y_test,
        threshold=0.25,
        voting="soft",
        ensemble_name="FiveModel_CalSoft_t0.25",
    )
