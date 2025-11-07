#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning Class Number 2 
# (evolution of ML_Class_1 with light hooks designed to work with the 10 algorithms being further tested)

class Model_Tester_V2:
    def __init__(self, model=None, scaler=None, parameter_grid=None, cv_folds:int=5,
                 feature_names:list=None, model_config:dict=None):
        """
        Class Initializer

            model: Model used for machine learning (defaults to random classifier)
            scaler: Scaler used for K-Fold cross-validation
            parameter_grid: Dictionary of hyperparameters for use in Grid Search Cross-Validation
            cv_folds: The number (integer) of folds used in cross-validation (defaults to 5)
            feature_names: Optional list of feature names used in feature importance plot
            model_config: Optional dict with model-specific options (e.g., {'scoring': 'recall',
                           'use_val_split': True, 'validation_size': 0.1, 'notes': 'xgb early stop'})

            Note: When a scaler is provided, the scaler and model are combined into a scikit-learn Pipeline
            for consistent preprocessing across K-Folds, optimization, and evaluation.
        """
        if model is not None and not hasattr(model, "fit"):
            raise ValueError(f"Error: model must be a scikit-learn classifier, but got {type(model)}")
        self.model = model

        self.scaler = scaler
        self.parameter_grid = parameter_grid
        self.cv_folds = cv_folds
        self.feature_names = feature_names

        #Minimal, optional per-model settings
        self.model_config = model_config or {}
        self.scoring = self.model_config.get("scoring", "accuracy")

        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return


    def preprocess_inputs(self, X, y):
        """
        Optional data preprocessing hook.

        Returns (X, y) unchanged by default. Override or supply logic via model_config
        in future if a model requires input checks (e.g., non-negativity).
        """
        return X, y

    def make_estimator(self):
        """
        Build the estimator for training/evaluation.

        If a scaler is provided, returns Pipeline(scaler -> clone(model)); otherwise returns a cloned model.
        This mirrors the prior _build_estimator behavior but is now public for simple overriding.
        """
        if self.scaler is not None and self.model is not None:
            return Pipeline([
                ("scaler", self.scaler),
                ("model", clone(self.model))
            ])
        return clone(self.model) if self.model is not None else None

    def fit_with_hooks(self, estimator, X, y, X_val=None, y_val=None):
        """
        Fit wrapper hook.

        Default is estimator.fit(X, y). If a model uses early stopping or eval sets,
        override this method or attach logic via a future runner; keep simple here.
        """
        estimator.fit(X, y)
        return estimator

    def _coerce_param_grid(self, estimator, param_grid):
        """
        If using a Pipeline, ensure parameters are namespaced (e.g., 'model__C'). Needed for GridSearchCV
        If keys are already namespaced, leave as-is.
        """
        if not isinstance(estimator, Pipeline): #If a pipeline is NOT used, no need to edit param_grid
            return param_grid
        if param_grid is None: #For empty param grid
            return None
        coerced = {}
        for k, v in param_grid.items(): #Construct {"model__C": [0.1, 1, 10]} (example)
            if "__" in k:
                coerced[k] = v
            else:
                coerced[f"model__{k}"] = v
        return coerced

    def train_test_split(self, X, y, train_size=0.8, random_state=1945):
        """
        Perform Train Test Split
        """
        #Optional preprocessing hook (no-ops by default)
        X_in, y_in = self.preprocess_inputs(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_in, y_in, train_size=train_size, random_state=random_state, stratify=y_in)

        if self.feature_names is None:
            if hasattr(X_in, "columns"):
                self.feature_names = list(X_in.columns)
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]

        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return

    def k_folds(self, K=None, random_state=1945, stratified: bool=True):
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
            estimator = self.make_estimator()
            estimator.fit(X_train_fold, y_train_fold)
            train_scores.append(estimator.score(X_train_fold, y_train_fold))
            test_scores.append(estimator.score(X_test_fold, y_test_fold))

        self.k_fold_results = {'train_scores': train_scores, 'test_scores': test_scores}
        print(f"Average Train Score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"Average Test Score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
        return train_scores, test_scores

    def optimize(self, scoring=None):
        """
        Optimization of model/classifier through Grid Search Cross-Validation

        scoring: Optional metric string for GridSearchCV (defaults to self.scoring; 'accuracy' if not set).
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize().")

        estimator = self.make_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        # Choose scoring (kept simple; user can override per call)
        scoring_to_use = scoring if scoring is not None else self.scoring

        if self.parameter_grid:
            #Ensure parameter grid works with a Pipeline
            param_grid = self._coerce_param_grid(estimator, self.parameter_grid)

            #Simple GridSearchCV as in ML_Class_1.py
            grid_search = GridSearchCV(estimator, param_grid, cv=self.cv_folds,
                                       scoring=scoring_to_use, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            print(f"\nBest Hyperparameters Found:\n{grid_search.best_params_}")
            print(f"Best Cross-Validation {scoring_to_use.capitalize()}: {grid_search.best_score_:.4f}")

            #Optional: re-fit best model with a small validation split if the user requests
            if self.model_config.get("use_val_split", False):
                val_size = float(self.model_config.get("validation_size", 0.1))
                X_fit, X_val, y_fit, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=val_size, stratify=self.y_train, random_state=1945)
                #Rebuild with best params (ensures clean state)
                best = self.make_estimator()
                #Apply best params onto the correct step (pipeline-safe)
                if isinstance(best, Pipeline):
                    best.set_params(**{k: v for k, v in self._coerce_param_grid(best, grid_search.best_params_).items()})
                else:
                    for k, v in grid_search.best_params_.items():
                        setattr(best, k, v)
                #Fit using hook (default is plain fit)
                self.best_model = self.fit_with_hooks(best, X_fit, y_fit, X_val, y_val)

        else:
            #Fit the estimator directly when no grid is provided
            est = estimator
            if self.model_config.get("use_val_split", False):
                val_size = float(self.model_config.get("validation_size", 0.1))
                X_fit, X_val, y_fit, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=val_size, stratify=self.y_train, random_state=1945)
                self.best_model = self.fit_with_hooks(est, X_fit, y_fit, X_val, y_val)
            else:
                est.fit(self.X_train, self.y_train)
                self.best_model = est
        return

    def evaluate(self, show_plots: bool=False):
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

        #Use underlying model's name when using a Pipeline (otherwise model name = Pipeline)
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
                positive_index = list(class_labels).index(class_labels[1])
                y_predict_probability = probas[:, positive_index]
            elif hasattr(self.best_model, "decision_function"):
                scores = self.best_model.decision_function(self.X_test)
                if scores.ndim == 1:
                    y_predict_probability = scores

        print(f"\n{model_name} Evaluation:")

        #Classification Report
        print('\nClassification Report:')
        print(classification_report(self.y_test, y_predict))

        #ROC Curve and AUC Score (binary classification only)
        if y_predict_probability is not None:
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

        cm = confusion_matrix(self.y_test, y_predict, labels=class_labels) if class_labels is not None else confusion_matrix(self.y_test, y_predict)

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
            "notes": self.model_config.get("notes", None)
        }
        return results

    #Plots (unchanged from ML_Class_1.py)
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


# --------------------------------------------
# Summary of Changes (from ML_Class_1 to V2)
# --------------------------------------------
#
# 1) Class renamed to Model_Tester_V2 (kept imports, plots, and evaluation flow the same).
# 2) Added model_config (dict) in __init__ with optional keys:
#       - 'scoring' (default 'accuracy' for GridSearchCV)
#       - 'use_val_split' (bool) and 'validation_size' (float) to enable a simple train/val refit stage
#       - 'notes' (free text; echoed in evaluate() results)
# 3) Added three light hooks for model-specific behavior:
#       - preprocess_inputs(X, y): default pass-through; future models can check/modify inputs.
#       - make_estimator(): replaces prior _build_estimator (same behavior, public for easy override).
#       - fit_with_hooks(est, X, y, X_val=None, y_val=None): default est.fit; placeholder for early-stopping, etc.
# 4) optimize(scoring=None): now lets you pass a scoring string; otherwise uses self.model_config['scoring'] or 'accuracy'.
# 5) Optional validation refit: if model_config['use_val_split'] is True, refits best params on a small train/val split
#    using fit_with_hooks. If not set, behavior matches ML_Class_1.
# 6) evaluate(): unchanged except it returns 'notes' from model_config for easy experiment tracking.
# 7) Kept naming, docstring tone, prints, and plotting style identical to your original for maximum continuity.