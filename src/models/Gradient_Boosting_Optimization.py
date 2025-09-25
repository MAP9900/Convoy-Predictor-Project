
#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, roc_curve, auc, matthews_corrcoef,
                             balanced_accuracy_score, confusion_matrix, recall_score,
                             fbeta_score, precision_recall_curve, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning Class Number 1:

class Model_Tester:

    def __init__(self, model = None, scaler = None, parameter_grid = None, cv_folds:int = 5,
                 feature_names:list = No+ne, positive_label = 1, optimize_scoring = 'recall',
                 auto_calibrate_threshold: bool = True, threshold_beta: float = 2.0,
                 random_state: int = 1945):
        """
        Class Initializer

            model: Model used for machine learning (defaults to random classifier)
            scaler: Scaler used for K-Fold cross-validation
            parameter_grid: Dictionary of hyperparamters for use in Grid Search Cross-Validation
            cv_folds: The number (integer) of folds used in cross-validation (defaults to 5)
            feature_names: List of feature names for use in feature importance plot. Class stores feature names automatically if they are 
            there. Otherwise a list must be given to return a feature plot. 
            
            Note: When a scaler is provided, the scaler and model are combined into a scikit-learn Pipeline
            for consistent preprocessing across K-Folds, optimization, and evaluation.
        """
        if model is not None and not hasattr(model, "fit"):
            raise ValueError(f"Error: model must be a scikit-learn classifier, but got {type(model)}")
        self.random_state = random_state
        self.model = model if model is not None else GradientBoostingClassifier(random_state=self.random_state) 
        #Default model at deafault random state (1945)
        self.scaler = scaler
        self.parameter_grid = parameter_grid
        self.cv_folds = cv_folds
        self.feature_names = feature_names
        self.positive_label = positive_label
        self.optimize_scoring = optimize_scoring
        self.auto_calibrate_threshold = auto_calibrate_threshold
        self.threshold_beta = threshold_beta
        self.decision_threshold = None
        self.threshold_metric = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return

    def _build_estimator(self):
        """
        Internal helper to construct the estimator used for training/evaluation.
        If a scaler is provided, returns a Pipeline(scaler -> model); otherwise returns a cloned model.
        """
        if self.scaler is not None and self.model is not None:
            return Pipeline([
                ("scaler", self.scaler),
                ("model", clone(self.model))
            ])
        #Fall back to model as-is (cloned to avoid reusing fitted state across folds)
        return clone(self.model) if self.model is not None else None

    def _coerce_param_grid(self, estimator, param_grid):
        """
        If using a Pipeline, ensure parameters are namespaced (e.g., 'model__C').
        If keys are already namespaced, leave as-is.
        """
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

    def _resolve_scoring(self, scoring):
        if scoring is not None:
            return scoring
        if self.optimize_scoring == 'recall':
            return make_scorer(recall_score, pos_label=self.positive_label)
        return self.optimize_scoring

    def _get_classes(self):
        estimator = self.best_model
        classes = getattr(estimator, "classes_", None)
        if classes is None and isinstance(estimator, Pipeline):
            model_step = estimator.named_steps.get("model")
            if model_step is not None:
                classes = getattr(model_step, "classes_", None)
        return classes

    def _calibrate_threshold(self):
        if self.best_model is None or self.X_train is None or self.y_train is None:
            self.decision_threshold = None
            self.threshold_metric = None
            return
        if not hasattr(self.best_model, "predict_proba"):
            self.decision_threshold = None
            self.threshold_metric = None
            return
        class_labels = self._get_classes()
        if class_labels is None or len(class_labels) != 2:
            self.decision_threshold = None
            self.threshold_metric = None
            return
        if self.positive_label not in class_labels:
            self.decision_threshold = None
            self.threshold_metric = None
            return
        probas = self.best_model.predict_proba(self.X_train)
        pos_index = list(class_labels).index(self.positive_label)
        pos_proba = probas[:, pos_index]
        y_true = np.asarray(self.y_train)
        precision, recall, thresholds = precision_recall_curve(y_true, pos_proba, pos_label=self.positive_label)
        if thresholds.size == 0:
            self.decision_threshold = 0.5
            self.threshold_metric = None
            return
        beta = self.threshold_beta if self.threshold_beta and self.threshold_beta > 0 else 1.0
        precision = precision[:-1]
        recall = recall[:-1]
        if precision.size == 0 or recall.size == 0 or thresholds.size == 0:
            self.decision_threshold = 0.5
            self.threshold_metric = None
            return
        numerator = (1 + beta ** 2) * precision * recall
        denominator = (beta ** 2 * precision) + recall + 1e-12
        fbeta_scores = numerator / denominator
        best_idx = int(np.nanargmax(fbeta_scores))
        self.decision_threshold = float(thresholds[best_idx])
        self.threshold_metric = float(fbeta_scores[best_idx])

    def set_decision_threshold(self, threshold = None):
        """Manually set the decision threshold for probability-to-class conversion."""
        if threshold is None:
            self.decision_threshold = None
        else:
            self.decision_threshold = float(threshold)

    def train_test_split(self, X, y, train_size = 0.8, random_state = 1945):
        """
        Perform Train Test Split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y)
        if self.feature_names is None:
            if hasattr(X, "columns"):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return
        
    def k_folds(self, K = None, random_state= 1945, stratified: bool = True):
        """
        Perform K-Fold Cross-Validation

        If stratified=True (default), uses StratifiedKFold to preserve class ratios
        in each fold — recommended for imbalanced classification tasks.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before k_folds().")

        K = K if K else self.cv_folds

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

            #Build estimator (Pipeline if scaler is provided) for each fold
            estimator = self._build_estimator()
            estimator.fit(X_train_fold, y_train_fold)
            train_scores.append(estimator.score(X_train_fold, y_train_fold))
            test_scores.append(estimator.score(X_test_fold, y_test_fold))

        self.k_fold_results = {'train_scores': train_scores, 'test_scores': test_scores}
        print(f"Average Train Score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"Average Test Score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

        return train_scores, test_scores
    
    def optimize(self, scoring = None, fit_params = None):
        """
        Optimization of model/classifier through Grid Search Cross-Validation
        """

        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize().")

        estimator = self._build_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")
        scoring_to_use = self._resolve_scoring(scoring)
        fit_params = fit_params or {}

        if self.parameter_grid:
            #Ensure parameter grid works with a Pipeline
            param_grid = self._coerce_param_grid(estimator, self.parameter_grid)
            grid_search = GridSearchCV(estimator, param_grid, cv=self.cv_folds, scoring=scoring_to_use, n_jobs=-1)
            #n_jobs = -1 uses all available CPUs (Parallel Execution). Switch to -2 to avoid freezing (ex: running server side)
            grid_search.fit(self.X_train, self.y_train, **fit_params)
            self.best_model = grid_search.best_estimator_
            print(f"\nBest Hyperparameters Found:\n{grid_search.best_params_}")
            print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        else:
            #Fit the estimator directly when no grid is provided
            self.best_model = estimator
            self.best_model.fit(self.X_train, self.y_train, **fit_params)

        if self.auto_calibrate_threshold:
            self._calibrate_threshold()
        else:
            self.threshold_metric = None
        return

    def evaluate(self, show_plots: bool = False):
        """
        Evaluate the fitted model using the held-out test data and generate reports/plots.

        Returns a dictionary containing key metrics for downstream comparison.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Call train_test_split() before evaluate().")
        if self.best_model is None:
            raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.")

        try:
            check_is_fitted(self.best_model)
        except NotFittedError as exc:
            raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.") from exc

        #Use underlying model's name when using a Pipeline
        _name_estimator = self.best_model
        if isinstance(_name_estimator, Pipeline):
            _name_estimator = _name_estimator.named_steps.get("model", _name_estimator)
        model_name = type(_name_estimator).__name__

        y_predict = self.best_model.predict(self.X_test)

        #Determine class labels for downstream reporting
        class_labels = getattr(self.best_model, "classes_", None)
        if class_labels is None and isinstance(self.best_model, Pipeline):
            model_step = self.best_model.named_steps.get("model")
            if model_step is not None:
                class_labels = getattr(model_step, "classes_", None)

        #Probabilities/scores for ROC (only for binary problems)
        y_predict_probability = None
        if class_labels is not None and len(class_labels) == 2:
            if hasattr(self.best_model, "predict_proba"):
                probas = self.best_model.predict_proba(self.X_test)
                if self.positive_label in class_labels:
                    positive_index = list(class_labels).index(self.positive_label)
                else:
                    positive_index = 1
                y_predict_probability = probas[:, positive_index]
            elif hasattr(self.best_model, "decision_function"):
                scores = self.best_model.decision_function(self.X_test)
                if scores.ndim == 1:
                    y_predict_probability = scores

        if y_predict_probability is not None and self.decision_threshold is not None and class_labels is not None and len(class_labels) == 2:
            if self.positive_label in class_labels:
                positive_label = self.positive_label
            else:
                positive_label = class_labels[1]
            negative_label = class_labels[0] if class_labels[0] != positive_label else class_labels[1]
            y_predict = np.where(y_predict_probability >= self.decision_threshold, positive_label, negative_label)
            print(f"Applied custom decision threshold: {self.decision_threshold:.3f}")

        print(f"\n{model_name} Evaluation:")

        #Classification Report
        print('\nClassification Report:')
        print(classification_report(self.y_test, y_predict))

        #ROC Curve and AUC Score (binary classification only)
        if y_predict_probability is not None:
            if class_labels is not None and self.positive_label in class_labels:
                pos_label = self.positive_label
            else:
                pos_label = class_labels[1] if class_labels is not None else None
            fpr, tpr, _ = roc_curve(self.y_test, y_predict_probability, pos_label=pos_label)
            ROC_AUC = auc(fpr, tpr)
            print(f"\nROC AUC Score: {ROC_AUC:.4f}")
        else:
            fpr, tpr, ROC_AUC = None, None, None
            print("\nROC AUC Score: N/A (requires binary classification with probability or decision scores)")

        #Matthews Correlation Coefficient
        mcc = matthews_corrcoef(self.y_test, y_predict)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        #Balanced Accuracy Score
        balanced_acc = balanced_accuracy_score(self.y_test, y_predict)
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        recall_pos = None
        f2_score = None
        if class_labels is not None and self.positive_label in class_labels:
            recall_pos = recall_score(self.y_test, y_predict, pos_label=self.positive_label)
            f2_score = fbeta_score(self.y_test, y_predict, beta=2.0, pos_label=self.positive_label)
            print(f"Recall (positive={self.positive_label}): {recall_pos:.4f}")
            print(f"F2 Score: {f2_score:.4f}")

        cm = confusion_matrix(self.y_test, y_predict, labels=class_labels) if class_labels is not None else confusion_matrix(self.y_test, y_predict)

        if class_labels is not None and self.positive_label in class_labels and len(class_labels) == 2:
            pos_idx = list(class_labels).index(self.positive_label)
            neg_idx = 0 if pos_idx == 1 else 1
            false_negatives = cm[pos_idx, neg_idx]
            print(f"False Negatives: {false_negatives}")
        else:
            false_negatives = None

        if show_plots:
            print(f"\n{model_name} Plots:\n")

            if fpr is not None and tpr is not None:
                self.plot_roc_curve(fpr, tpr, ROC_AUC)

            print(f"{model_name} Confusion Matrix:")
            self.plot_confusion_matrix(cm, class_labels=class_labels)

            final_estimator = self.best_model
            if isinstance(final_estimator, Pipeline):
                final_estimator = final_estimator.named_steps.get("model", final_estimator)

            if hasattr(final_estimator, "feature_importances_") or hasattr(final_estimator, "coef_"):
                print(f"{model_name} Feature Importance Plot:")
                self.plot_feature_importance()
            else:
                print('Model has no attribute: Feature Importances')
        else:
            print(f"{model_name} Confusion Matrix (values only):\n{cm}")

        results = {
            "model_name": model_name,
            "classification_report": classification_report(self.y_test, y_predict, output_dict=True),
            "confusion_matrix": cm,
            "roc_auc": ROC_AUC,
            "mcc": mcc,
            "balanced_accuracy": balanced_acc,
            "recall": recall_pos,
            "f2_score": f2_score,
            "decision_threshold": self.decision_threshold,
            "threshold_metric": self.threshold_metric,
            "false_negatives": false_negatives,
        }
        return results

    def plot_roc_curve(self, fpr, tpr, ROC_AUC):
        """
        ROC Curve Plot
        """
        plt.figure(figsize=(6,4), facecolor='lightgrey')
        plt.plot(fpr, tpr, color='#06768d', label=f'ROC Curve (AUC = {ROC_AUC:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        ax = plt.gca()
        ax.set_facecolor('lightgrey')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()

    def plot_confusion_matrix(self, cm, class_labels=None):
        """
        Plot Confusion Matrix
        """
        if class_labels is None:
            labels_to_use = [str(i) for i in range(cm.shape[0])]
        else:
            labels_to_use = [str(label) for label in class_labels]

        plt.figure(figsize=(6,4), facecolor='lightgrey')
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_to_use, \
                    yticklabels=labels_to_use)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_feature_importance(self):
        """
        Feature Importance Plot
        """

        #Support feature importances for tree models and coefficients for linear models
        estimator = self.best_model
        if isinstance(estimator, Pipeline):
            estimator = estimator.named_steps.get("model", estimator)

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.mean(np.abs(coef), axis=0)
        else:
            print('Model has no attribute: Feature Importances')
            return

        importances = np.asarray(importances)
        if importances.ndim > 1:
            importances = importances.ravel()

        if self.feature_names and len(self.feature_names) == len(importances):
            feature_labels = self.feature_names
        else:
            feature_labels = [f"Feature_{i}" for i in range(len(importances))]

        feature_importance_df = pd.DataFrame({
                                'Feature': feature_labels,
                                'Importance': importances
                                }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(8,4), facecolor='lightgrey')
        ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='crest_r')  
        plt.title('Feature Importance')
        ax.set_facecolor('lightgrey')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()
        return

# Set up Model_Tester so gradient boosting is the default and recall remains the primary optimisation target, 
# with optional threshold tuning to squeeze down false negatives.

# Default GB config and recall knobs: src/models/Gradient_Boosting_Optimization.py:22-54 now instantiates a GradientBoostingClassifier
#  when no model is supplied and lets you record the positive class, preferred scorer, auto-threshold toggle, and beta weight.

# Scoring + fit extras: src/models/Gradient_Boosting_Optimization.py:92-238 introduces _resolve_scoring, manual scorer override,
#  and optimize(scoring=None, fit_params=None) so you can pass recall-weighted scorers or early-stopping kwargs (e.g., XGBoost’s 
# model__early_stopping_rounds).

# Recall-focused thresholding: src/models/Gradient_Boosting_Optimization.py:99-154 adds helper utilities that (optionally) 
# sweep the training probabilities with an F2 objective to select self.decision_threshold, plus set_decision_threshold() for manual tweaks.

# Evaluation updates: src/models/Gradient_Boosting_Optimization.py:271-372 applies any calibrated/manual threshold before scoring, 
# prints recall/F2/false-negative counts, and returns those values alongside the confusion matrix and MCC so you can compare 
# experiments at a glance.

# No automated tests were run (not requested).

# In your convoy notebook, call optimizer = Model_Tester(positive_label='sunk', optimize_scoring='recall'), then optimizer.optimize() and confirm the reported recall/F2 beat the baseline.
# If you tune with XGBoost, pass its params through model__ keys and supply fit_params={'model__eval_set': [(X_val, y_val)], 'model__early_stopping_rounds': 25} to keep the same recall-first workflow.