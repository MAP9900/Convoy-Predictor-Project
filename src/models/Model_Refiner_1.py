"""Utilities for optimizing gradient-descent classifiers with recall-focused evaluation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:  # Successive halving search (sklearn>=0.24)
    from sklearn.experimental import enable_halving_search_cv  # type: ignore  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

    _HALVING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    HalvingGridSearchCV = None  # type: ignore
    HalvingRandomSearchCV = None  # type: ignore
    _HALVING_AVAILABLE = False

try:  
    from xgboost import XGBClassifier 

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGBOOST_AVAILABLE = False


class GradientDescentModelRefiner:
    """Optimize, evaluate, and compare gradient-descent models with recall priority."""

    SUPPORTED_STRATEGIES = {"none", "grid", "random", "halving_grid", "halving_random"}

    def __init__(
        self,
        models_config: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        cv_folds: int = 5,
        primary_metric: str = "recall",
        positive_label: Any = 1,
        n_jobs: int = -1,
        random_state: int = 1945,
        default_scaler: Optional[StandardScaler] = None,
        default_n_iter: int = 40,
        verbose: bool = True,
    ) -> None:
        self.cv_folds = cv_folds
        self.primary_metric = primary_metric
        self.positive_label = positive_label
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.default_scaler = default_scaler
        self.default_n_iter = default_n_iter
        self.verbose = verbose

        self.models_config: Dict[str, Dict[str, Any]] = {}
        self.best_models: Dict[str, Pipeline] = {}
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.k_fold_results: Dict[str, Dict[str, Any]] = {}

        self.X_train: Optional[Any] = None
        self.X_test: Optional[Any] = None
        self.y_train: Optional[Any] = None
        self.y_test: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None

        if models_config:
            for name, cfg in models_config.items():
                self.register_model(name, **cfg)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def register_model(
        self,
        name: str,
        estimator: Any,
        *,
        param_grid: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = None,
        search_strategy: str = "grid",
        fit_params: Optional[Dict[str, Any]] = None,
        n_iter: Optional[int] = None,
        resource_param: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a gradient-descent model and its optimization settings."""
        if not hasattr(estimator, "fit"):
            raise ValueError(f"Estimator for '{name}' must implement fit().")

        strategy = search_strategy.lower()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Search strategy '{search_strategy}' not supported. Choose from {sorted(self.SUPPORTED_STRATEGIES)}."
            )
        if strategy in {"grid", "halving_grid"} and param_grid is None:
            raise ValueError(f"Parameter grid is required for strategy '{search_strategy}'.")
        if strategy in {"random", "halving_random"} and param_grid is None:
            raise ValueError(f"Parameter distributions are required for strategy '{search_strategy}'.")

        if resource_param is None and strategy.startswith("halving"):
            resource_param = "max_iter"

        self.models_config[name] = {
            "estimator": estimator,
            "param_grid": param_grid,
            "scaler": scaler,
            "search_strategy": strategy,
            "fit_params": fit_params or {},
            "n_iter": n_iter or self.default_n_iter,
            "resource_param": resource_param,
            "search_kwargs": dict(search_kwargs or {}),
        }

    def remove_model(self, name: str) -> None:
        """Remove a registered model and its cached results."""
        self.models_config.pop(name, None)
        self.best_models.pop(name, None)
        self.optimization_results.pop(name, None)
        self.evaluation_results.pop(name, None)
        self.k_fold_results.pop(name, None)

    def train_test_split(
        self,
        X: Any,
        y: Any,
        *,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify: bool = True,
    ) -> None:
        """Hold out data for optimization and evaluation."""
        stratify_target = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=random_state or self.random_state,
            stratify=stratify_target,
        )

        if self.feature_names is None:
            if hasattr(X, "columns"):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]

        self.k_fold_results = {}
        self._log(
            f"Train/Test split -> train: {len(self.y_train)}, test: {len(self.y_test)}"
        )

    def _build_pipeline(self, estimator: Any, scaler: Optional[Any]) -> Pipeline:
        steps = []
        scaler = scaler if scaler is not None else self.default_scaler
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", clone(estimator)))
        return Pipeline(steps)

    def _coerce_param_grid(
        self, estimator: Pipeline, param_grid: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if param_grid is None:
            return None
        if not isinstance(estimator, Pipeline):
            return param_grid

        coerced: Dict[str, Any] = {}
        for key, value in param_grid.items():
            if "__" in key:
                coerced[key] = value
            else:
                coerced[f"model__{key}"] = value
        return coerced

    def _coerce_resource_param(self, estimator: Pipeline, resource_param: Optional[str]) -> Optional[str]:
        if resource_param is None:
            return None
        if not isinstance(estimator, Pipeline):
            return resource_param
        if "__" in resource_param:
            return resource_param
        return f"model__{resource_param}"

    def _coerce_fit_params(
        self, estimator: Pipeline, fit_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not fit_params:
            return {}
        if not isinstance(estimator, Pipeline):
            return dict(fit_params)

        coerced: Dict[str, Any] = {}
        for key, value in fit_params.items():
            if "__" in key:
                coerced[key] = value
            else:
                coerced[f"model__{key}"] = value
        return coerced

    def _inject_halving_resource_bounds(
        self,
        param_grid: Optional[Dict[str, Any]],
        resource: Optional[str],
        search_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if resource is None or param_grid is None:
            return search_kwargs
        if resource not in param_grid:
            raise ValueError(
                f"Resource parameter '{resource}' not present in param grid. "
                "Include it or supply explicit search_kwargs for halving."
            )
        values = param_grid[resource]
        if isinstance(values, (list, tuple, np.ndarray)) and values:
            numeric_values = [val for val in values if isinstance(val, (int, float))]
            if numeric_values and len(numeric_values) == len(values):
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                search_kwargs.setdefault("min_resources", min_val)
                search_kwargs.setdefault("max_resources", max_val)
        return search_kwargs

    def _create_search(
        self,
        estimator: Pipeline,
        param_grid: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Optional[Any]:
        strategy = config["search_strategy"]
        if strategy == "none":
            return None

        search_kwargs = dict(config.get("search_kwargs", {}))
        if strategy == "grid":
            return GridSearchCV(
                estimator,
                param_grid,
                cv=self.cv_folds,
                scoring=self.primary_metric,
                refit=self.primary_metric,
                n_jobs=self.n_jobs,
                return_train_score=True,
                **search_kwargs,
            )
        if strategy == "random":
            return RandomizedSearchCV(
                estimator,
                param_grid,
                n_iter=config["n_iter"],
                cv=self.cv_folds,
                scoring=self.primary_metric,
                refit=self.primary_metric,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                return_train_score=True,
                **search_kwargs,
            )
        if strategy == "halving_grid":
            if not _HALVING_AVAILABLE or HalvingGridSearchCV is None:
                raise RuntimeError(
                    "Halving searches require scikit-learn with experimental halving support."
                )
            resource = self._coerce_resource_param(
                estimator, config.get("resource_param", "max_iter")
            )
            config["resource_param"] = resource
            if resource and resource not in estimator.get_params(deep=True):
                raise ValueError(
                    f"Resource parameter '{resource}' not found on estimator. "
                    "Ensure your param_grid includes this parameter."
                )
            search_kwargs = self._inject_halving_resource_bounds(param_grid, resource, search_kwargs)
            return HalvingGridSearchCV(
                estimator,
                param_grid,
                cv=self.cv_folds,
                scoring=self.primary_metric,
                refit=self.primary_metric,
                factor=search_kwargs.pop("factor", 3),
                resource=resource,
                n_jobs=self.n_jobs,
                **search_kwargs,
            )
        if strategy == "halving_random":
            if not _HALVING_AVAILABLE or HalvingRandomSearchCV is None:
                raise RuntimeError(
                    "Halving searches require scikit-learn with experimental halving support."
                )
            resource = self._coerce_resource_param(
                estimator, config.get("resource_param", "max_iter")
            )
            config["resource_param"] = resource
            if resource and resource not in estimator.get_params(deep=True):
                raise ValueError(
                    f"Resource parameter '{resource}' not found on estimator. "
                    "Ensure your param distributions include this parameter."
                )
            search_kwargs = self._inject_halving_resource_bounds(param_grid, resource, search_kwargs)
            return HalvingRandomSearchCV(
                estimator,
                param_grid,
                n_candidates=config["n_iter"],
                cv=self.cv_folds,
                scoring=self.primary_metric,
                refit=self.primary_metric,
                factor=search_kwargs.pop("factor", 3),
                resource=resource,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **search_kwargs,
            )
        raise ValueError(f"Unsupported search strategy '{strategy}'.")

    def optimize_model(self, name: str) -> None:
        """Run the configured search/fit routine for a single model."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize_model().")
        if name not in self.models_config:
            raise KeyError(f"Model '{name}' is not registered.")

        config = self.models_config[name]
        pipeline = self._build_pipeline(config["estimator"], config.get("scaler"))
        param_grid = self._coerce_param_grid(pipeline, config.get("param_grid"))
        search_obj = self._create_search(pipeline, param_grid, config)
        fit_params = self._coerce_fit_params(pipeline, config.get("fit_params", {}))

        start = time.time()
        if search_obj is None:
            pipeline.fit(self.X_train, self.y_train, **fit_params)
            best_estimator = pipeline
            best_params = pipeline.get_params()
            best_score = recall_score(
                self.y_train, pipeline.predict(self.X_train), pos_label=self.positive_label
            )
            cv_results = None
        else:
            search_obj.fit(self.X_train, self.y_train, **fit_params)
            best_estimator = search_obj.best_estimator_
            best_params = getattr(search_obj, "best_params_", {})
            best_score = getattr(search_obj, "best_score_", None)
            cv_results = getattr(search_obj, "cv_results_", None)

        duration = time.time() - start
        self.best_models[name] = best_estimator
        self.optimization_results[name] = {
            "best_params": best_params,
            "best_score": best_score,
            "duration_sec": duration,
            "search_strategy": config["search_strategy"],
            "cv_results": cv_results,
            "search_object": search_obj,
        }

        score_msg = (
            f"best_{self.primary_metric}={best_score:.4f}" if best_score is not None else "fitted"
        )
        self._log(
            f"[{name}] optimization via {config['search_strategy']} completed in {duration:.2f}s ({score_msg})."
        )

    def optimize_all(self) -> None:
        """Optimize every registered model."""
        for name in self.models_config:
            self.optimize_model(name)

    def evaluate_model(self, name: str, *, plot: bool = True) -> Dict[str, Any]:
        """Evaluate a fitted model on the held-out test set with recall emphasis."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Call train_test_split() before evaluate_model().")
        if name not in self.best_models:
            raise KeyError(f"Model '{name}' has not been optimized yet.")

        estimator = self.best_models[name]
        try:
            check_is_fitted(estimator)
        except NotFittedError as exc:
            raise RuntimeError(f"Model '{name}' is not fitted. Run optimize_model().") from exc

        y_pred = estimator.predict(self.X_test)
        classes = getattr(estimator, "classes_", None)
        if classes is None and isinstance(estimator, Pipeline):
            model_step = estimator.named_steps.get("model")
            if model_step is not None:
                classes = getattr(model_step, "classes_", None)
        metrics = {
            "recall": recall_score(self.y_test, y_pred, pos_label=self.positive_label),
            "precision": precision_score(
                self.y_test, y_pred, pos_label=self.positive_label, zero_division=0
            ),
            "f1": f1_score(self.y_test, y_pred, pos_label=self.positive_label),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, y_pred),
            "mcc": matthews_corrcoef(self.y_test, y_pred),
        }

        proba_available = hasattr(estimator, "predict_proba")
        decision_available = hasattr(estimator, "decision_function")
        y_scores = None
        if proba_available:
            probas = estimator.predict_proba(self.X_test)
            class_list = list(classes) if classes is not None else list(estimator.classes_)
            try:
                positive_index = class_list.index(self.positive_label)
            except ValueError as exc:
                raise ValueError(
                    f"Positive label {self.positive_label} not found in model classes {class_list}."
                ) from exc
            y_scores = probas[:, positive_index]
        elif decision_available:
            decision_values = estimator.decision_function(self.X_test)
            if decision_values.ndim == 1:
                y_scores = decision_values

        fpr, tpr = None, None
        roc_auc = None
        avg_precision = None
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_scores, pos_label=self.positive_label)
            roc_auc = auc(fpr, tpr)
            avg_precision = average_precision_score(self.y_test, y_scores)

        metrics["roc_auc"] = roc_auc
        metrics["average_precision"] = avg_precision

        cm = confusion_matrix(self.y_test, y_pred)
        specificity = None
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) else 0.0

        report = classification_report(
            self.y_test, y_pred, output_dict=True, zero_division=0
        )

        evaluation_payload = {
            "metrics": metrics,
            "specificity": specificity,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }
        self.evaluation_results[name] = evaluation_payload

        self._log(
            f"[{name}] recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )

        if plot:
            if fpr is not None and tpr is not None and roc_auc is not None:
                self.plot_roc_curve(fpr, tpr, roc_auc, title=f"{name} ROC Curve")
            if y_scores is not None:
                self.plot_precision_recall_curve(
                    self.y_test, y_scores, title=f"{name} Precision-Recall"
                )
            self.plot_confusion_matrix(cm, model_name=name)
            self.plot_feature_importance(estimator, model_name=name)

        return evaluation_payload

    def evaluate_all(self, *, plot: bool = True) -> Dict[str, Dict[str, Any]]:
        """Evaluate every optimized model."""
        results = {}
        for name in self.best_models:
            results[name] = self.evaluate_model(name, plot=plot)
        return results

    def cross_validate_model(
        self,
        name: str,
        *,
        K: Optional[int] = None,
        stratified: bool = True,
        scoring: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Cross-validate a fitted model on the training split."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before cross_validate_model().")
        if name not in self.best_models:
            raise KeyError(f"Model '{name}' has not been optimized yet.")

        estimator = self.best_models[name]
        splits = K or self.cv_folds
        cv = (
            StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.random_state)
            if stratified
            else KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        )

        scoring = scoring or {
            "recall": "recall",
            "precision": "precision",
            "balanced_accuracy": "balanced_accuracy",
        }

        cv_results = cross_validate(
            estimator,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            return_train_score=True,
        )

        summary: Dict[str, Dict[str, float]] = {}
        for key, values in cv_results.items():
            if isinstance(values, np.ndarray):
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        self.k_fold_results[name] = summary

        recall_stats = summary.get("test_recall", {})
        self._log(
            f"[{name}] CV recall mean={recall_stats.get('mean', float('nan')):.4f} "
            f"± {recall_stats.get('std', float('nan')):.4f}"
        )
        return cv_results

    def compare_models(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Return a comparison table sorted by the metric of interest (defaults to recall)."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results cached. Run evaluate_model() first.")

        metric = metric or self.primary_metric
        rows: List[Dict[str, Any]] = []
        for name, payload in self.evaluation_results.items():
            row = {
                "model_name": name,
                **payload["metrics"],
                "specificity": payload.get("specificity"),
                "best_params": self.optimization_results.get(name, {}).get("best_params"),
                "best_cv_score": self.optimization_results.get(name, {}).get("best_score"),
            }
            rows.append(row)

        comparison = pd.DataFrame(rows).set_index("model_name")
        if metric in comparison.columns:
            comparison = comparison.sort_values(by=metric, ascending=False)

        if metric in comparison.columns:
            self._log(f"Model ranking by '{metric}':\n{comparison[[metric]].round(4)}")
        else:
            self._log("Requested metric not found in comparison table.")
        return comparison

    def plot_roc_curve(
        self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, *, title: str = "ROC Curve"
    ) -> None:
        plt.figure(figsize=(6, 4), facecolor="lightgrey")
        plt.plot(fpr, tpr, color="#06768d", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        ax = plt.gca()
        ax.set_facecolor("lightgrey")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_scores: np.ndarray, *, title: str = "Precision-Recall Curve"
    ) -> None:
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=self.positive_label)
        ap = average_precision_score(y_true, y_scores)
        plt.figure(figsize=(6, 4), facecolor="lightgrey")
        plt.plot(recall, precision, color="#e1723a", label=f"AP = {ap:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        ax = plt.gca()
        ax.set_facecolor("lightgrey")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, *, model_name: Optional[str] = None) -> None:
        labels = [str(i) for i in range(cm.shape[0])]
        plt.figure(figsize=(6, 4), facecolor="lightgrey")
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{model_name} Confusion Matrix" if model_name else "Confusion Matrix")
        ax.set_facecolor("lightgrey")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_feature_importance(self, estimator: Any, *, model_name: Optional[str] = None) -> None:
        model_step = estimator
        if isinstance(estimator, Pipeline):
            model_step = estimator.named_steps.get("model", estimator)

        feature_labels: Optional[List[str]] = None
        if hasattr(model_step, "feature_importances_"):
            importances = np.asarray(model_step.feature_importances_)
        elif hasattr(model_step, "coef_"):
            coef = np.asarray(model_step.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        elif _XGBOOST_AVAILABLE and isinstance(model_step, XGBClassifier):
            booster = model_step.get_booster()
            scores = booster.get_score(importance_type="gain")
            if not scores:
                self._log(f"[{model_name}] No feature importance found for XGBoost model.")
                return
            importances = []
            feature_labels = []
            if self.feature_names:
                for idx, feature_name in enumerate(self.feature_names):
                    feature_labels.append(feature_name)
                    importances.append(scores.get(f"f{idx}", 0.0))
            else:
                for key, value in scores.items():
                    feature_labels.append(key)
                    importances.append(value)
            importances = np.asarray(importances)
        else:
            self._log(f"[{model_name}] Feature importance unavailable for this estimator.")
            return

        if feature_labels is None:
            if self.feature_names and len(self.feature_names) == len(importances):
                feature_labels = self.feature_names
            else:
                feature_labels = [f"Feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({"Feature": feature_labels, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 4), facecolor="lightgrey")
        ax = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="crest_r")
        plt.title(f"{model_name} Feature Importance" if model_name else "Feature Importance")
        ax.set_facecolor("lightgrey")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.show()


# Backwards compatibility for existing imports
Model_Tester = GradientDescentModelRefiner