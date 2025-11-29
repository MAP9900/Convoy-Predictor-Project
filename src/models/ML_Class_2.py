#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Machine Learning Class Number 2 ---
# --- (evolution of ML_Class_1 with light hooks designed to work with the 10 algorithms being further tested) ---


#Recent Addition: 
#    Optional validation refit: if model_config['use_val_split'] is True, refits best params on a small train/val split
#    using fit_with_hooks. If not set, behavior matches ML_Class_1.

class Model_Tester_V2:
    def __init__(self, model=None, scaler=None, parameter_grid=None, cv_folds:int=5,
                 threshold_beta: float = 2.0, feature_names:list=None, model_config:dict=None,
                 random_state:int=1945):
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
        self.scoring = self.model_config.get("scoring", "recall")
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.k_fold_results = {"train_scores": [], "test_scores": []}
        self.decision_threshold = 0.5
        self.threshold_metric = None
        self.threshold_beta = float(self.model_config.get("threshold_beta", threshold_beta))
        self.positive_label = self.model_config.get("positive_label", 1)
        self.random_state = random_state
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

    def _resolve_cv(self):
        """Return a StratifiedKFold splitter matching Gradient_Boosting_Optimization defaults."""
        cv = self.cv_folds
        if cv is None:
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        if isinstance(cv, int):
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        return cv

    def _get_classes(self):
        """Return class labels from the fitted estimator (handles Pipeline wrappers)."""
        estimator = self.best_model
        if estimator is None:
            return None
        if isinstance(estimator, Pipeline):
            estimator = estimator.named_steps.get("model", estimator)
        return getattr(estimator, "classes_", None)

    def _calibrate_threshold(self):
        """
        Calibrate the model's decision threshold based on training data. Set up to focus on the positive class (1) (Use case is for Convoy Porject)

        This method finds an optimal probability cutoff for binary classification
        by maximizing the F-beta score (weighted harmonic mean of precision and recall) on the training set. 
        It works by:

        - Ensuring the model and data are valid (must support `predict_proba`,
          must be binary classification, and must contain the designated positive label). (Checks)
        - Computing predicted probabilities for the positive class.
        - Evaluating precision and recall across candidate thresholds.
        - Calculating F-beta scores (default F1 if beta=1).
        - Selecting the threshold that maximizes this score.

        The chosen threshold is stored in `self.decision_threshold`,
        and the corresponding F-beta score is stored in `self.threshold_metric`.

        If conditions are not met or no valid thresholds exist, defaults to 0.5
        or sets values to None.
        """
        #Checks to make sure model is setup properly:
        if self.best_model is None or self.X_train is None or self.y_train is None: #Check to prevent invalid calibration attempts if model has yet to be 
            self.decision_threshold = None
            self.threshold_metric = None
            return
        if not hasattr(self.best_model, "predict_proba"): #Check to ensure model has "predict_proba" as an output. Will not work without "predict_proba"
            self.decision_threshold = None
            self.threshold_metric = None
            return
        class_labels = self._get_classes()
        if class_labels is None or len(class_labels) != 2: #Check to ensure binary classification is being performed
            self.decision_threshold = None
            self.threshold_metric = None
            return
        if self.positive_label not in class_labels: #Check to ensure model has postive labels 
            self.decision_threshold = None
            self.threshold_metric = None
            return
        #Start threshold optimzation: 
        probas = self.best_model.predict_proba(self.X_train)
        pos_index = list(class_labels).index(self.positive_label) 
        pos_proba = probas[:, pos_index] #Gets probas for the postive class (Focus on postive class is for use on Convoy Prject)
        y_true = np.asarray(self.y_train)
        precision, recall, thresholds = precision_recall_curve(y_true, pos_proba, pos_label=self.positive_label)
        if thresholds.size == 0: #Edge case if model only only predicts one unique probability so default back to 0.5 decision_threshold
            self.decision_threshold = 0.5
            self.threshold_metric = None
            return
        beta = self.threshold_beta if self.threshold_beta and self.threshold_beta > 0 else 1.0 #Default to 1 = F1 Score unless threshold_beta is given
        #beta decides how much to weigh recall vs precision here. beta = 1, weigh equally. beta > 1, recall focused. beta < 1, precision focused
        precision = precision[:-1] #Drop last point as it doesn't correspond to a real threshold 
        recall = recall[:-1] #Drop last point as it doesn't correspond to a real threshold 
        if precision.size == 0 or recall.size == 0 or thresholds.size == 0: #Double checks emptiness after trimming last point
            self.decision_threshold = 0.5
            self.threshold_metric = None
            return
        #Compute F-beta Scores:
        numerator = (1 + beta ** 2) * precision * recall
        denominator = (beta ** 2 * precision) + recall + 1e-12
        fbeta_scores = numerator / denominator
        best_idx = int(np.nanargmax(fbeta_scores))
        #Save optimal cutoff (decision_threshold) and its associated F-beta score (threshold_metric)
        self.decision_threshold = float(thresholds[best_idx])
        self.threshold_metric = float(fbeta_scores[best_idx])
        # print(f"Optimized Decision Threshold: {self.decision_threshold:.4f} with F-beta score: {self.threshold_metric:.4f}")
        

    def set_decision_threshold(self, threshold = None):
        """
        Manual way to set the decision_threshold. 
        Way to err on the side of false-positives so use 0.3 for example
        """
        if threshold is None:
            self.decision_threshold = 0.5 #Defaults to 0.5 
        else:
            self.decision_threshold = float(threshold)
            # print(f"Using Decision Threshold: {self.decision_threshold:.4f}") #No longer needed, since evals prints decision threshold 

    def train_test_split(self, X, y, train_size=0.8, random_state=None):
        """
        Perform Train Test Split
        """
        #Optional preprocessing hook (no-ops by default)
        X_in, y_in = self.preprocess_inputs(X, y)

        split_seed = random_state if random_state is not None else self.random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_in, y_in, train_size=train_size, random_state=split_seed, stratify=y_in)

        if self.feature_names is None:
            if hasattr(X_in, "columns"):
                self.feature_names = list(X_in.columns)
            else:
                self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]

        self.k_fold_results = {"train_scores": [], "test_scores": []}
        return

    def k_folds(self, K=None, random_state=None, stratified: bool=True):
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
        seed = random_state if random_state is not None else self.random_state
        if stratified:
            kf = StratifiedKFold(n_splits=K, random_state=seed, shuffle=True)
        else:
            kf = KFold(n_splits=K, random_state=seed, shuffle=True)
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

    def optimize(self, scoring=None, method="grid", n_iter=50):
        """
        Optimization of model/classifier through Cross-Validation search.

        method: Optimization method to use. Options:
                'grid'   - GridSearchCV (default)
                'random' - RandomizedSearchCV (faster, samples parameter grid)
                'halving' - HalvingGridSearchCV (progressively narrows candidates)
                'bayes'  - BayesSearchCV (requires scikit-optimize)
        n_iter: Number of parameter settings sampled for 'random' or 'bayes' search.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize().")

        estimator = self.make_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        scoring_to_use = scoring if scoring is not None else self.scoring
        cv_splitter = self._resolve_cv()

        if self.parameter_grid:
            #Ensure parameter grid works with a Pipeline
            param_grid = self._coerce_param_grid(estimator, self.parameter_grid)
            #Select optimization method
            search_method = method.lower()
            if search_method == "grid":
                searcher = GridSearchCV(
                    estimator, param_grid,
                    cv=cv_splitter, scoring=scoring_to_use, n_jobs=-1,
                    return_train_score=False)

            elif search_method == "random":
                searcher = RandomizedSearchCV(
                    estimator, param_distributions=param_grid,
                    n_iter=n_iter, cv=cv_splitter,
                    scoring=scoring_to_use, n_jobs=-1,
                    random_state=self.random_state, return_train_score=False)
            elif search_method == "halving":
                try:
                    from sklearn.model_selection import HalvingGridSearchCV
                    y_arr = np.asarray(self.y_train)
                    minority_share = max((y_arr == self.positive_label).mean(), 1e-6)
                    n_splits = cv_splitter.get_n_splits(self.X_train, self.y_train)
                    min_resources = max(int(np.ceil(n_splits / minority_share)), 200)
                    searcher = HalvingGridSearchCV(
                        estimator, param_grid,
                        cv=cv_splitter, scoring=scoring_to_use,
                        n_jobs=-1, factor=3,
                        verbose=False, aggressive_elimination=True,
                        min_resources=min_resources, random_state=self.random_state)
                except ImportError:
                     raise ImportError("HalvingGridSearchCV not available in this sklearn version.")
            elif search_method == "bayes":
                try:
                    from skopt import BayesSearchCV
                    searcher = BayesSearchCV(
                        estimator, search_spaces=param_grid,
                        n_iter=n_iter, cv=cv_splitter,
                        scoring=scoring_to_use, n_jobs=-1,
                        random_state=self.random_state)
                except ImportError:
                    raise ImportError("BayesSearchCV requires scikit-optimize (skopt).")
            else:
                raise ValueError(f"Unknown optimization method: {method}") #Fall back 

            #Run Search
            searcher.fit(self.X_train, self.y_train)
            self.best_model = searcher.best_estimator_
            print(f"\nOptimization Method: {method.capitalize()}")
            print(f"Best Hyperparameters Found:\n{searcher.best_params_}")
            print(f"Best Cross-Validation {scoring_to_use.capitalize()}: {searcher.best_score_:.4f}")

            #Optional: re-fit with validation split if requested
            if self.model_config.get("use_val_split", False):
                val_size = float(self.model_config.get("validation_size", 0.1))
                X_fit, X_val, y_fit, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=val_size,
                    stratify=self.y_train, random_state=self.random_state)
                best = self.make_estimator()
                #Apply best params (pipeline-safe)
                if isinstance(best, Pipeline):
                    best.set_params(**{k: v for k, v in self._coerce_param_grid(best, searcher.best_params_).items()})
                else:
                    for k, v in searcher.best_params_.items():
                        setattr(best, k, v)
                self.best_model = self.fit_with_hooks(best, X_fit, y_fit, X_val, y_val)

        else:
            #Fit estimator directly (no grid)
            est = estimator
            if self.model_config.get("use_val_split", False):
                val_size = float(self.model_config.get("validation_size", 0.1))
                X_fit, X_val, y_fit, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=val_size,
                    stratify=self.y_train, random_state=self.random_state)
                self.best_model = self.fit_with_hooks(est, X_fit, y_fit, X_val, y_val)
            else:
                est.fit(self.X_train, self.y_train)
                self.best_model = est
        self._calibrate_threshold()
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

        #Determine class labels for downstream reporting
        class_labels = self._get_classes()
        if class_labels is None:
            class_labels = getattr(self.best_model, "classes_", None)
            if class_labels is None and isinstance(self.best_model, Pipeline):
                model_step = self.best_model.named_steps.get("model")
                if model_step is not None:
                    class_labels = getattr(model_step, "classes_", None)

        #Probabilities/scores for ROC (only for binary problems)
        probas = None
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

        #Apply calibrated threshold when available, fallback to model predictions otherwise
        if (self.decision_threshold is not None and probas is not None and class_labels is not None
                and len(class_labels) == 2 and self.positive_label in class_labels):
            pos_index = list(class_labels).index(self.positive_label)
            negative_label = [label for label in class_labels if label != self.positive_label][0]
            y_predict = np.where(probas[:, pos_index] >= self.decision_threshold,
                                 self.positive_label,
                                 negative_label)
            print(f"Applied decision threshold: {self.decision_threshold:.4f}") #To keep track of threshold used
        else:
            y_predict = self.best_model.predict(self.X_test)

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
            print(f"{model_name} Confusion Matrix:\n{cm}")

        results = {
            "model_name": model_name,
            "classification_report": classification_report(self.y_test, y_predict, output_dict=True),
            "confusion_matrix": cm,
            "roc_auc": ROC_AUC,
            "mcc": mcc,
            "balanced_accuracy": balanced_acc,
            "notes": self.model_config.get("notes", None)}
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
        plt.savefig(f"/Users/matthewplambeck/Desktop/Convoy Predictor/Plots/{self.model}_PR_Curve.png")
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
        plt.savefig(f"/Users/matthewplambeck/Desktop/Convoy Predictor/Plots/{self.model}_CM.png")
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
        plt.savefig(f"/Users/matthewplambeck/Desktop/Convoy Predictor/Plots/{self.model}_Feature_Importance.png")
        plt.show()
        return
