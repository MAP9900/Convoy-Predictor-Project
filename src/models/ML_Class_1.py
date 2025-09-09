#Imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier

#Machine Learning Class Number 1:

class Model_Tester:

    def __init__(self, model = None, scaler = None, parameter_grid = None, cv_folds:int = 5, feature_names:list = None):
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
        self.model = model

        self.scaler = scaler
        self.parameter_grid = parameter_grid
        self.cv_folds = cv_folds
        self.feature_names = feature_names
        self.best_model = None
        self.feature_importance = None
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

    def train_test_split(self, X, y, train_size = 0.8, random_state = 1945):
        """
        Perform Train Test Split
        """
        self.X_train, self. X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y)
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        return
        
    def k_folds(self, K = None, random_state= 1945, stratified: bool = True):
        """
        Perform K-Fold Cross-Validation
        
        If stratified=True (default), uses StratifiedKFold to preserve High/Low Risk class ratios
        in each fold — recommended for imbalanced WWII convoy risk data.
        """
        K = K if K else self.cv_folds

        X_train_array = np.array(self.X_train)
        y_train_array = np.array(self.y_train)
        if stratified:
            kf = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
        else:
            kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
        train_scores, test_scores = [], []
        
        for idxTrain, idxTest in kf.split(X_train_array):
            X_train_fold, X_test_fold = X_train_array[idxTrain], X_train_array[idxTest]
            y_train_fold, y_test_fold = y_train_array[idxTrain], y_train_array[idxTest]

            #Build estimator (Pipeline if scaler is provided) for each fold
            estimator = self._build_estimator()
            estimator.fit(X_train_fold, y_train_fold)
            train_scores.append(estimator.score(X_train_fold, y_train_fold))
            test_scores.append(estimator.score(X_test_fold, y_test_fold))

        self.k_fold_results_ = {'Train Scores': train_scores, 'Test Scores': test_scores}
        print(f"Average Train Score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"Average Test Score: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

        return train_scores, test_scores
    
    def optimize_with_early_stopping(self, early_stopping_rounds: int = 50, validation_size: float = 0.2,
                                     random_state: int = 1945, eval_metric: str = 'auc', verbose: bool = False):
        """
        Optimization using Early Stopping (especially effective for XGBoost).
        
        This method splits the training data into train/validation, finds the optimal number
        of boosting rounds via early stopping, and then refits the final estimator on the
        full training data using the discovered best iteration.
        
        Parameters:
            early_stopping_rounds: Number of rounds with no improvement to trigger stop
            validation_size: Fraction of training data used as validation for early stopping
            random_state: Random seed for reproducibility
            eval_metric: Metric for XGBoost during early stopping (e.g., 'auc' for risk ranking)
            verbose: Whether to print early stopping progress from the underlying model
        
        Notes:
            - Works best with XGBoost (`XGBClassifier`). If a scaler is provided, scaling is
              applied consistently. After determining the best iteration, the model is refit
              on the full training data (with the scaler) to produce `self.best_model`.
            - For non-XGBoost estimators, this will simply fit the pipeline/model on the full
              training data (no early stopping available).
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before optimize_with_early_stopping().")

        base_estimator = self._build_estimator()
        if base_estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        #If not XGBoost, fall back to standard fit on the full data
        underlying = self.model
        is_xgb = isinstance(underlying, XGBClassifier)

        if not is_xgb:
            #Fit directly (Pipeline if present)
            base_estimator.fit(self.X_train, self.y_train)
            self.best_model = base_estimator
            return

        #Early stopping path for XGBoost
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=self.y_train)

        #Prepare data according to presence of scaler
        if self.scaler is not None:
            # Fit scaler on the early-stopping training split only
            scaler = clone(self.scaler)
            X_tr_t = scaler.fit_transform(X_tr)
            X_val_t = scaler.transform(X_val)
            #Clone the underlying XGBClassifier and fit with early stopping
            xgb_model = clone(underlying)
            xgb_model.set_params(eval_metric=eval_metric)
            xgb_model.fit(
                X_tr_t, y_tr,
                eval_set=[(X_val_t, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
            best_iter = getattr(xgb_model, 'best_iteration', None)
            if best_iter is None:
                #Fallback if attribute not present
                best_iter = xgb_model.get_booster().best_ntree_limit if hasattr(xgb_model, 'get_booster') else xgb_model.n_estimators

            #Refit on full training with best n_estimators
            final_estimator = self._build_estimator()
            if isinstance(final_estimator, Pipeline):
                final_estimator.set_params(model__n_estimators=best_iter)
            else:
                final_estimator.set_params(n_estimators=best_iter)
            final_estimator.fit(self.X_train, self.y_train)
            self.best_model = final_estimator
            return
        else:
            #No scaler: fit XGB directly with early stopping
            xgb_model = clone(underlying)
            xgb_model.set_params(eval_metric=eval_metric)
            xgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
            best_iter = getattr(xgb_model, 'best_iteration', None)
            if best_iter is None:
                best_iter = xgb_model.get_booster().best_ntree_limit if hasattr(xgb_model, 'get_booster') else xgb_model.n_estimators

            #Refit on full training with best n_estimators
            final_model = clone(underlying).set_params(n_estimators=best_iter, eval_metric=eval_metric)
            final_model.fit(self.X_train, self.y_train)
            self.best_model = final_model
            return
        
    def optimize(self):
        """
        Optimization of model/classifier through Grid Search Cross-Validation
        """

        estimator = self._build_estimator()
        if estimator is None:
            raise ValueError("No model provided. Please initialize with a valid scikit-learn classifier.")

        if self.parameter_grid:
            #Ensure parameter grid works with a Pipeline
            param_grid = self._coerce_param_grid(estimator, self.parameter_grid)
            grid_search = GridSearchCV(estimator, param_grid, cv=self.cv_folds, scoring='accuracy', n_jobs=-1)
            #n_jobs = -1 uses all available CPUs (Parallel Execution). Switch to -2 to avoid freezing (ex: running server side)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            print(f"\nBest Hyperparameters Found:\n{grid_search.best_params_}")
            print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
        else:
            #Fit the estimator directly when no grid is provided
            self.best_model = estimator
            self.best_model.fit(self.X_train, self.y_train)
        return

    def evaluate(self):
        """
        Evaluates the model/classifier using the test data. 
        
        Returns a classification report, ROC AUC score, Matthews Correlation Coefficient, Balanced Accuracy Score, ROC Curve Plot, 
        Confusion Matrix Plot, and Feature Importance Plot. 
        """

        #Ensure the model is fitted (works for all estimators, incl. Pipelines)
        try:
            check_is_fitted(self.best_model)
        except NotFittedError:
            raise RuntimeError("Model is not fitted before evaluation. Call optimize() or fit the model first.")

        #Use underlying model's name when using a Pipeline
        _name_estimator = self.best_model
        if isinstance(_name_estimator, Pipeline):
            _name_estimator = _name_estimator.named_steps.get("model", _name_estimator)
        model_name = type(_name_estimator).__name__
        y_predict = self.best_model.predict(self.X_test)
        #Probabilities for ROC & Log Loss (fall back to decision_function if needed)
        y_predict_probability = None
        if hasattr(self.best_model, "predict_proba"):
            y_predict_probability = self.best_model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.best_model, "decision_function"):
            y_predict_probability = self.best_model.decision_function(self.X_test)

        #Print Model Name
        print(f"\n{model_name} Evaluation:")

        #Classification Report
        print('\nClassification Report:')
        print(classification_report(self.y_test, y_predict))

        #ROC Curve and AUC Score (Receiver Operating Characteristic - Area Under the Curve)
            #Measures ability to distinguish high vs. low risk
            #The ROC Curve plots True Positive Rate (tpr) vs. False Positive Rate (fpr) at various threshold values
        if y_predict_probability is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_predict_probability)
            ROC_AUC = auc(fpr, tpr)
            print(f"\nROC AUC Score: {ROC_AUC:.4f}") #If AUC is >0.8, the model is performing well; closer to 0.5 means random guessing.
        else:
            fpr, tpr, ROC_AUC = None, None, None
            print("\nROC AUC Score: N/A (model lacks probability/decision scores)")
        

        #Matthews Correlation Coefficient
            #Used due to imablance within the data (more low risk than high risk)
            #A single score that accounts for true/false positives & negatives
        mcc = matthews_corrcoef(self.y_test, y_predict)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}") #Closer to 1 means better classification

        #Balanced Accuracy Score
            #Used due to imablance within the data (more low risk than high risk)
            #Adjusted accuracy for class imbalance
        balanced_acc = balanced_accuracy_score(self.y_test, y_predict)
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        #Plots
        print(f"\n {model_name} Plots: \n\n")

        #ROC Curve
        if fpr is not None and tpr is not None:
            self.plot_roc_curve(fpr, tpr, ROC_AUC)

        #Confusion Matrix
        print(f"{model_name} Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_predict)
        self.plot_confusion_matrix(cm)

        #Feature Importance Plot
        final_estimator = self.best_model
        if isinstance(final_estimator, Pipeline):
            #If using a Pipeline, the final estimator is the 'model' step
            final_estimator = final_estimator.named_steps.get("model", final_estimator)

        if hasattr(final_estimator, "feature_importances_") or hasattr(final_estimator, "coef_"):
            print(f"{model_name} Feature Importance Plot:")
            self.plot_feature_importance()
        else:
            print('Model has no attribute: Feature Importances')

        #Return metrics to support testing and downstream use
        results = {
            "model_name": model_name,
            "classification_report": classification_report(self.y_test, y_predict, output_dict=True),
            "confusion_matrix": cm,
            "roc_auc": ROC_AUC,
            "mcc": mcc,
            "balanced_accuracy": balanced_acc,}
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

    def plot_confusion_matrix(self, cm):
        """
        Plot Confusion Matrix
        """
        plt.figure(figsize=(6,4), facecolor='lightgrey')
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], \
                    yticklabels=['Low Risk', 'High Risk'])
        # ax.set_facecolor('lightgrey')
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

        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            importances = np.abs(coef.ravel())

        if importances is None:
            print('Model has no attribute: Feature Importances')
            return

        feature_importance_df = pd.DataFrame({
                                'Feature': self.feature_names,  
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


