#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, confusion_matrix, 
                             recall_score, fbeta_score, precision_recall_curve, make_scorer) 
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


class CNB_Class:

    def __init__(self, model=None, scaler=None, parameter_grid=None, cv_folds: int = 5, feature_names: list = None, random_state: int = 1945):
        if model is not None and not hasattr(model, "fit"):
            raise ValueError(f"Error: model must be a scikit-learn classifier, but got {type(model)}")
        self.model = model if model is not None else ComplementNB()
        self.scaler = scaler if scaler is not None else MaxAbsScaler()
        self.parameter_grid = parameter_grid 
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
        self.threshold_metric = None
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
    
    def _resolve_scoring(self, scoring):
        """
        Helper function to determine what scoring function to use when evalualting the model. Defaults to "recall"
        Scoring Options: "accuracy", "precision", "recall", "f1", "roc_auc"
        Can use make_scorer() to add extra kwargs (like what is done for recall)

        Can also pass in a dictionary of scoring metrics for use with GridSearchCV
        ex: scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, pos_label=1), "recall": "recall",}

        """
        if scoring is not None: 
            return scoring #Evaluate based off given scor
        if self.optimize_scoring == 'recall':
            return make_scorer(recall_score, pos_label=self.positive_label) #Calculate recall in regards to the postive label (1) --(This is specifcally for the Convoy Project)--
        return self.optimize_scoring

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

    def optimize(self, scoring=None, refit=None, fit_params=None, use_sample_weight=True):
        """
        Hyperparameter optimization with Stratified CV, optional multi-metric scoring,
        class-imbalance weighting, and optional threshold calibration. Modified from optimzation function in 
        Gradient_Boosting_Optimization.py 
        """
        #Intial Checks 
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize().")
        estimator = self._build_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        #Cross-Validation
        cv_strategy = self.cv_folds
        if cv_strategy is None:
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        elif isinstance(cv_strategy, int):
            cv_strategy = StratifiedKFold(n_splits=cv_strategy, shuffle=True, random_state=self.random_state)

        #Primary scoring metric, defaults to recall
        primary_metric = self._resolve_scoring(scoring) 
        scoring_dict = {
            "recall_macro": "recall_macro",
            "recall_weighted": "recall_weighted",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }

        if isinstance(primary_metric, dict):
            scoring_dict.update(primary_metric)
            default_refit = next(iter(primary_metric.keys()), "f1_weighted")
        else:
            default_refit = primary_metric if isinstance(primary_metric, str) else "f1_weighted"
            #Check if a single scoring metric was passed in. If so, adds to scoring dictionary 
            if isinstance(primary_metric, str) and primary_metric not in scoring_dict:
                scoring_dict[primary_metric] = primary_metric

        refit_metric = refit or default_refit
        #param-grid
        if self.parameter_grid:
            if hasattr(self, "_coerce_param_grid"):
                param_grid = self._coerce_param_grid(estimator, self.parameter_grid)
            else:
                param_grid = self.parameter_grid
            fit_kwargs = {}
            if use_sample_weight and hasattr(self, "_supports_sample_weight") and self._supports_sample_weight(estimator):
                weights = compute_sample_weight(class_weight="balanced", y=self.y_train)
                fit_kwargs["model__sample_weight"] = weights
            if fit_params:
                fit_kwargs.update(fit_params)

            grid_search = GridSearchCV(estimator, param_grid, cv=cv_strategy, scoring=scoring_dict, refit=refit_metric, n_jobs=-1 )
            grid_search.fit(self.X_train, self.y_train, **fit_kwargs)
            self.best_model = grid_search.best_estimator_

            print(f"\nBest Hyperparameters Found:\n{grid_search.best_params_}")
            print(f"Best CV ({refit_metric}): {grid_search.best_score_:.4f}")

        else:
            #If no grid, fit model directly 
            self.best_model = estimator
            fit_kwargs = {}
            if use_sample_weight and hasattr(self, "_supports_sample_weight") and self._supports_sample_weight(estimator):
                weights = compute_sample_weight(class_weight="balanced", y=self.y_train)
                fit_kwargs["model__sample_weight"] = weights
            if fit_params:
                fit_kwargs.update(fit_params)
            self.best_model.fit(self.X_train, self.y_train, **fit_kwargs)

        if getattr(self, "auto_calibrate_threshold", False):
            self._calibrate_threshold()
        else:
            self.threshold_metric = None
            if getattr(self, "decision_threshold", None) is None:
                self.decision_threshold = 0.5
        return 

    def _calibrate_threshold(self):
        """
        *Pulled directly from Gradient_Boosting_Optimization.py*

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

    def evaluate(self, show_plots: bool = False, print_results: bool = True):
            """
            Evaluate the fitted model using the held-out test data and generate reports/plots.
            Returns a dictionary containing key metrics for later comparison.
            show_plots: defaults to false so will not run any plotting functions. Useful when testing to not clutter notebook with plots
            Legacy of ML_Class_1.py with some changes (Removed Pipeline aspects)
            """
            #Check steps prior to evaluation:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Call train_test_split() before evaluate().")
            if self.best_model is None:
                raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.")
            try:
                check_is_fitted(self.best_model)
            except NotFittedError as exc:
                raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.") from exc

            model_name = type(self.best_model).__name__ #For use in plotting
            y_predict = self.best_model.predict(self.X_test) #y value predictions
            #Determine class labels for downstream reporting
            class_labels = getattr(self.best_model, "classes_", None) #Should be [0, 1] for Convoy Project

            #Probabilities/scores for ROC (only for binary classification which this is meant for) (legacy of ML_Class_1.py)
            y_predict_probability = None
            if class_labels is not None and len(class_labels) == 2 and hasattr(self.best_model, "predict_proba"):
                probas = self.best_model.predict_proba(self.X_test)
                if self.positive_label in class_labels:
                    positive_index = list(class_labels).index(self.positive_label)
                else:
                    positive_index = 1
                y_predict_probability = probas[:, positive_index]
            #When custom threshold is defined, override default 0.5 threshold.  
            if y_predict_probability is not None and class_labels is not None and len(class_labels) == 2:
                if self.decision_threshold is not None:
                    if self.positive_label in class_labels:
                        positive_label = self.positive_label
                    else:
                        positive_label = class_labels[1]
                    negative_label = class_labels[0] if class_labels[0] != positive_label else class_labels[1] #Simply picks other class as the negative label
                    #Recompute predictions using the custom decision threshold:
                    y_predict = np.where(y_predict_probability >= self.decision_threshold, positive_label, negative_label)
                    if print_results:
                        message = f"Applied decision threshold: {self.decision_threshold:.4f}" #Print decision threshold to keep track of models
                        if self.threshold_metric is not None:
                            message += f" (F-beta: {self.threshold_metric:.4f})"
                        print(message)


            #Start of Model Evaluation 
            if print_results ==True:
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
                if print_results:
                    print(f"\nROC AUC Score: {ROC_AUC:.4f}")
            else:
                fpr, tpr, ROC_AUC = None, None, None
                if print_results:
                    print("\nROC AUC Score: N/A (requires binary classification with probability or decision scores)")

            #Matthews Correlation Coefficient
            mcc = matthews_corrcoef(self.y_test, y_predict)
            if print_results:
                print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

            #Balanced Accuracy Score
            balanced_acc = balanced_accuracy_score(self.y_test, y_predict)
            if print_results:
                print(f"Balanced Accuracy: {balanced_acc:.4f}")

            #Confusion Matrix
            cm = confusion_matrix(self.y_test, y_predict, labels=class_labels) if class_labels is not None else confusion_matrix(self.y_test, y_predict)
            if print_results:
                print("Confusion Matrix:")
                if class_labels is not None: #prints a nicer looking cm that just print(cm)
                    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_labels], columns=[f"Predicted {c}" for c in class_labels])
                    print(cm_df)
                else:
                    print(cm)

            #Recall and F2 Scores (F2 score is a weighted harmonic mean of precision and recall)
            recall_pos = None
            f2_score = None
            if class_labels is not None and self.positive_label in class_labels:
                recall_pos = recall_score(self.y_test, y_predict, pos_label=self.positive_label)
                f2_score = fbeta_score(self.y_test, y_predict, beta=2.0, pos_label=self.positive_label) #beta of 2 means recall gets 4x the importance of precision
                if print_results:
                    print(f"Recall (positive={self.positive_label}): {recall_pos:.4f}")
                    print(f"F2 Score: {f2_score:.4f}")

            #Examine False Negatives
            if class_labels is not None and self.positive_label in class_labels and len(class_labels) == 2:
                pos_idx = list(class_labels).index(self.positive_label)
                neg_idx = 0 if pos_idx == 1 else 1
                false_negatives = cm[pos_idx, neg_idx] #Extract false negative cases
                if print_results:
                    print(f"False Negatives: {false_negatives}")
            else:
                false_negatives = None

            if show_plots:
                print(f"\n{model_name} Plots:\n")

                if fpr is not None and tpr is not None:
                    self.plot_roc_curve(fpr, tpr, ROC_AUC)

                final_estimator = self.best_model

                if hasattr(final_estimator, "feature_importances_") or hasattr(final_estimator, "coef_"):
                    print(f"{model_name} Feature Importance Plot:")
                    self.plot_feature_importance()
                else:
                    print('Model has no attribute: Feature Importances')


            #Save results in a dictionary for later use if needed
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
                "false_negatives": false_negatives,}
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
