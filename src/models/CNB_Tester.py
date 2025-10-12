#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import ComplementNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect


class CNB_Tester:

    def __init__(self, model=None, scaler=None, parameter_grid=None, cv_folds: int = 5, feature_names: list = None, random_state: int = 1945):
        if model is not None and not hasattr(model, "fit"):
            raise ValueError(f"Error: model must be a scikit-learn classifier, but got {type(model)}")
        self.model = model if model is not None else ComplementNB()

        self.scaler = scaler if scaler is not None else MaxAbsScaler()
        self.parameter_grid = parameter_grid if parameter_grid is not None else {
            "model__alpha": [0.1, 0.25, 0.5, 1.0],
            "model__norm": [True, False]
        }
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.feature_names = feature_names
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.positive_label = None
        self.decision_threshold = 0.5
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return

    def _build_estimator(self):
        steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        if self.scaler is not None:
            steps.append(("scaler", clone(self.scaler)))
        steps.append(("model", clone(self.model)))
        return Pipeline(steps)

    def _coerce_param_grid(self, estimator, param_grid):
        if not isinstance(estimator, Pipeline):
            return param_grid
        if param_grid is None:
            return None
        coerced = {}
        for k, v in param_grid.items():
            if "__" in k:
                coerced[k] = v
            else:
                coerced[f"model__{k}"] = v
        return coerced

    def _final_estimator(self, estimator):
        if isinstance(estimator, Pipeline):
            return estimator.named_steps.get("model", estimator)
        return estimator

    def _supports_sample_weight(self, estimator):
        final_est = self._final_estimator(estimator)
        if final_est is None:
            return False
        signature = inspect.signature(final_est.fit)
        return "sample_weight" in signature.parameters

    def train_test_split(self, X, y, train_size=0.8, random_state=None):
        random_state = random_state if random_state is not None else self.random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y
        )
        if self.feature_names is None:
            if hasattr(X, "columns"):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        classes, counts = np.unique(self.y_train, return_counts=True)
        if len(classes) == 2:
            self.positive_label = classes[np.argmin(counts)]
        else:
            self.positive_label = None
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return

    def k_folds(self, K=None, random_state=None, stratified: bool = True):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before k_folds().")

        K = K if K else self.cv_folds
        random_state = random_state if random_state is not None else self.random_state

        X_train_array = np.array(self.X_train)
        y_train_array = np.array(self.y_train)
        if stratified:
            kf = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
        else:
            kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
        train_scores, test_scores = [], []

        for idxTrain, idxTest in kf.split(X_train_array, y_train_array):
            X_train_fold, X_test_fold = X_train_array[idxTrain], X_train_array[idxTest]
            y_train_fold, y_test_fold = y_train_array[idxTrain], y_train_array[idxTest]

            estimator = self._build_estimator()
            fit_kwargs = {}
            if self._supports_sample_weight(estimator):
                fold_weights = compute_sample_weight(class_weight="balanced", y=y_train_fold)
                if isinstance(estimator, Pipeline):
                    fit_kwargs["model__sample_weight"] = fold_weights
                else:
                    fit_kwargs["sample_weight"] = fold_weights
            estimator.fit(X_train_fold, y_train_fold, **fit_kwargs)
            train_scores.append(estimator.score(X_train_fold, y_train_fold))
            test_scores.append(estimator.score(X_test_fold, y_test_fold))

        self.k_fold_results = {"train_scores": train_scores, "test_scores": test_scores}
        print(f"Average Train Score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"Average Test Score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

        return train_scores, test_scores

    def optimize(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize().")

        estimator = self._build_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        if self.parameter_grid:
            param_grid = self._coerce_param_grid(estimator, self.parameter_grid)
            scoring = {
                "recall_macro": "recall_macro",
                "f1_macro": "f1_macro",
                "recall_weighted": "recall_weighted",
                "f1_weighted": "f1_weighted"
            }
            grid_search = GridSearchCV(
                estimator,
                param_grid,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                refit="f1_weighted",
                n_jobs=-1
            )
            fit_kwargs = {}
            if self._supports_sample_weight(estimator):
                weights = compute_sample_weight(class_weight="balanced", y=self.y_train)
                fit_kwargs["model__sample_weight"] = weights
            grid_search.fit(self.X_train, self.y_train, **fit_kwargs)
            self.best_model = grid_search.best_estimator_
            print(f"\nBest Hyperparameters Found:\n{grid_search.best_params_}")
            print(f"Best Cross-Validation F1 (weighted): {grid_search.best_score_:.4f}")
        else:
            self.best_model = estimator
            fit_kwargs = {}
            if self._supports_sample_weight(estimator):
                weights = compute_sample_weight(class_weight="balanced", y=self.y_train)
                fit_kwargs["model__sample_weight"] = weights
            self.best_model.fit(self.X_train, self.y_train, **fit_kwargs)

        self._calibrate_threshold()
        return

    def _calibrate_threshold(self):
        self.decision_threshold = 0.5
        if self.best_model is None:
            return
        if self.positive_label is None:
            return
        if not hasattr(self.best_model, "predict_proba"):
            return
        y_train_binary = (self.y_train == self.positive_label).astype(int)
        probabilities = self.best_model.predict_proba(self.X_train)
        positive_index = list(self.best_model.classes_).index(self.positive_label)
        y_scores = probabilities[:, positive_index]
        precision, recall, thresholds = precision_recall_curve(y_train_binary, y_scores)
        f1_scores = (2 * precision * recall) / np.maximum((precision + recall), np.finfo(float).eps)
        best_index = np.argmax(f1_scores)
        if best_index < len(thresholds):
            self.decision_threshold = thresholds[best_index]
        else:
            self.decision_threshold = 0.5
        print(f"Tuned decision threshold for positive class ({self.positive_label}): {self.decision_threshold:.3f}")
        return

    def evaluate(self, show_plots: bool = False):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Call train_test_split() before evaluate().")
        if self.best_model is None:
            raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.")

        try:
            check_is_fitted(self.best_model)
        except NotFittedError as exc:
            raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.") from exc

        _name_estimator = self._final_estimator(self.best_model)
        model_name = type(_name_estimator).__name__

        class_labels = getattr(self.best_model, "classes_", None)
        y_predict_probability = None
        if class_labels is not None and len(class_labels) == 2 and hasattr(self.best_model, "predict_proba"):
            positive_index = list(class_labels).index(self.positive_label if self.positive_label is not None else class_labels[-1])
            probabilities = self.best_model.predict_proba(self.X_test)
            y_predict_probability = probabilities[:, positive_index]
            threshold = self.decision_threshold if self.decision_threshold is not None else 0.5
            y_predict = (y_predict_probability >= threshold).astype(class_labels.dtype)
            y_predict = np.where(y_predict == 1, class_labels[positive_index], class_labels[0])
        else:
            y_predict = self.best_model.predict(self.X_test)

        print(f"\n{model_name} Evaluation:")

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_predict))

        if y_predict_probability is not None:
            pos_label = self.positive_label if self.positive_label is not None else class_labels[-1]
            fpr, tpr, _ = roc_curve(self.y_test, y_predict_probability, pos_label=pos_label)
            ROC_AUC = auc(fpr, tpr)
            print(f"\nROC AUC Score: {ROC_AUC:.4f}")
        else:
            fpr, tpr, ROC_AUC = None, None, None
            print("\nROC AUC Score: N/A (requires binary classification with probability or decision scores)")

        mcc = matthews_corrcoef(self.y_test, y_predict)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        balanced_acc = balanced_accuracy_score(self.y_test, y_predict)
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        cm = confusion_matrix(self.y_test, y_predict, labels=class_labels) if class_labels is not None else confusion_matrix(self.y_test, y_predict)
        if show_plots:
            print(f"\n{model_name} Plots:\n")

            if fpr is not None and tpr is not None:
                self.plot_roc_curve(fpr, tpr, ROC_AUC)
            if y_predict_probability is not None:
                self.plot_precision_recall_curve(y_predict_probability)

            print(f"{model_name} Confusion Matrix:")
            self.plot_confusion_matrix(cm, class_labels=class_labels)

            final_estimator = self._final_estimator(self.best_model)
            if hasattr(final_estimator, "feature_importances_") or hasattr(final_estimator, "coef_"):
                print(f"{model_name} Feature Importance Plot:")
                self.plot_feature_importance()
            else:
                print("Model has no attribute: Feature Importances")
        else:
            print(f"{model_name} Confusion Matrix (values only):\n{cm}")

        results = {
            "model_name": model_name,
            "classification_report": classification_report(self.y_test, y_predict, output_dict=True),
            "confusion_matrix": cm,
            "roc_auc": ROC_AUC,
            "mcc": mcc,
            "balanced_accuracy": balanced_acc,
            "threshold": self.decision_threshold
        }
        return results

    def plot_roc_curve(self, fpr, tpr, ROC_AUC):
        plt.figure(figsize=(6,4), facecolor="lightgrey")
        plt.plot(fpr, tpr, color="#06768d", label=f"ROC Curve (AUC = {ROC_AUC:.2f})")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        ax = plt.gca()
        ax.set_facecolor("lightgrey")
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_precision_recall_curve(self, y_scores):
        if self.positive_label is None:
            return
        y_true = (self.y_test == self.positive_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(6,4), facecolor="lightgrey")
        plt.plot(recall, precision, color="#ff8c32")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        ax = plt.gca()
        ax.set_facecolor("lightgrey")
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_confusion_matrix(self, cm, class_labels=None):
        if class_labels is None:
            labels_to_use = [str(i) for i in range(cm.shape[0])]
        else:
            labels_to_use = [str(label) for label in class_labels]

        plt.figure(figsize=(6,4), facecolor="lightgrey")
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_to_use,
                    yticklabels=labels_to_use)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_feature_importance(self):
        estimator = self._final_estimator(self.best_model)

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.mean(np.abs(coef), axis=0)
        else:
            print("Model has no attribute: Feature Importances")
            return

        importances = np.asarray(importances)
        if importances.ndim > 1:
            importances = importances.ravel()

        if self.feature_names and len(self.feature_names) == len(importances):
            feature_labels = self.feature_names
        else:
            feature_labels = [f"Feature_{i}" for i in range(len(importances))]

        feature_importance_df = pd.DataFrame({
                                "Feature": feature_labels,
                                "Importance": importances
                                }).sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(8,4), facecolor="lightgrey")
        ax = sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="crest_r")
        plt.title("Feature Importance")
        ax.set_facecolor("lightgrey")
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()
        return
