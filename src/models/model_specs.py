#Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#Helpers for imbalance knobs on GBMs
def _class_ratio(y):
    """
    Compute neg/pos ratio for binary labels. Returns None if cannot compute.
    """
    try:
        import numpy as np
        y = np.asarray(y)
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        if pos == 0:
            return None
        return float(neg) / float(pos)
    except Exception:
        return None

def _with_spw(grid, y):
    """
    If y is provided and binary, add a small range of scale_pos_weight values.
    Otherwise return grid unchanged.
    """
    r = _class_ratio(y)
    if r is None:
        return grid
    g = dict(grid)  #shallow copy
    g["scale_pos_weight"] = [0.5 * r, 1.0 * r, 1.5 * r]
    return g


#Dictionary entries (two grids per model: grid_small, grid_large)

MODEL_SPECS = {
    #Decision Tree
    "dt": {
        "name": "DecisionTreeClassifier",
        "estimator": DecisionTreeClassifier(random_state=1945),
        "grid_small": {
            "max_depth": [3, 6, None],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", None],
            "class_weight": [None, "balanced"],
        },
        "grid_large": {
            "max_depth": [3, 6, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.6, None],
            "class_weight": [None, "balanced"],
        },
        "config": {"scoring": "recall", "notes": "Shallow depth + leaf sizes help recall."},
    },

    #Random Forest
    "rf": {
        "name": "RandomForestClassifier",
        "estimator": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=1945),
        "grid_small": {
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 4],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", 0.6, None],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "grid_large": {
            "max_depth": [None, 8, 14],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", 0.6, None],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "config": {"scoring": "recall", "notes": "Use class_weight for imbalance; parallel fit."},
    },

    #Extra Trees
    "et": {
        "name": "ExtraTreesClassifier",
        "estimator": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1945),
        "grid_small": {
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", 0.6, None],
            "class_weight": [None, "balanced"],
        },
        "grid_large": {
            "max_depth": [None, 8, 14],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.6, None],
            "class_weight": [None, "balanced"],
        },
        "config": {"scoring": "recall", "notes": "More randomness can help recall."},
    },

    #Bagging (with a class-weighted tree base estimator)
    "bag": {
        "name": "BaggingClassifier",
        #NOTE: For sklearn >=1.2, use `estimator=`; for older versions, this parameter is called `base_estimator`.
        "estimator": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=1945),
            n_estimators=200,
            random_state=1945
        ),
        "grid_small": {
            "n_estimators": [150, 300],
            "max_samples": [0.7, 1.0],
            "max_features": [0.7, 1.0],
            "bootstrap": [True],
            "bootstrap_features": [False],
        },
        "grid_large": {
            "n_estimators": [200, 400],
            "max_samples": [0.6, 0.9, 1.0],
            "max_features": [0.6, 0.9, 1.0],
            "bootstrap": [True],
            "bootstrap_features": [False, True],
        },
        "config": {"scoring": "recall", "notes": "Importance not exposed at wrapper level."},
    },

    #Gradient Boosting (sklearn)
    "gb": {
        "name": "GradientBoostingClassifier",
        "estimator": GradientBoostingClassifier(random_state=1945),
        "grid_small": {
            "learning_rate": [0.03, 0.05],
            "n_estimators": [400, 800],
            "max_depth": [3, 4],
            "min_samples_leaf": [1, 4],
            "subsample": [0.75, 1.0],
            "max_features": ["sqrt", None],
        },
        "grid_large": {
            "learning_rate": [0.03, 0.05, 0.08],
            "n_estimators": [400, 800, 1200],
            "max_depth": [3, 4, 5],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.7, 0.85, 1.0],
            "max_features": ["sqrt", 0.5, None],
        },
        "config": {"scoring": "recall", "notes": "Classic sklearn GBM; no native early stopping."},
    },

    #AdaBoost
    "ada": {
        "name": "AdaBoostClassifier",
        "estimator": AdaBoostClassifier(random_state=1945),
        "grid_small": {
            "n_estimators": [200, 600],
            "learning_rate": [0.05, 0.1],
        },
        "grid_large": {
            "n_estimators": [200, 600, 1000],
            "learning_rate": [0.03, 0.1, 0.3],
        },
        "config": {"scoring": "recall", "notes": "Imbalance via base estimator if customized."},
    },

    #QDA
    "qda": {
        "name": "QuadraticDiscriminantAnalysis",
        "estimator": QuadraticDiscriminantAnalysis(),
        "grid_small": {
            "reg_param": [0.0, 0.01, 0.1],
        },
        "grid_large": {
            "reg_param": [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        },
        "config": {"scoring": "recall", "notes": "Regularization avoids singular covariance."},
    },

    #XGBoost
    "xgb": {
        "name": "XGBClassifier",
        "estimator": XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=1945, n_jobs=-1
        ) if XGBClassifier is not None else None,
        "grid_small": (lambda y=None: _with_spw({
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.75, 0.9],
            "colsample_bytree": [0.75, 0.9],
            "reg_lambda": [0.0, 1.0],
        }, y)),
        "grid_large": (lambda y=None: _with_spw({
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.08],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 1.0],
            "reg_lambda": [0.0, 1.0, 3.0],
            "reg_alpha": [0.0, 1.0],
        }, y)),
        "config": {
            "scoring": "recall",
            "use_val_split": True,
            "validation_size": 0.1,
            "notes": "Enable early stopping via fit_with_hooks; set eval_set to (X_val,y_val)."
        },
    },
    #XGBoost Random Forest
    "xgbrf": {
        "name": "XGBRFClassifier",
        "estimator": XGBRFClassifier(
            n_estimators=800,
            max_depth=6,
            subsample=0.8,
            colsample_bynode=0.8,
            colsample_bytree=1.0,
            random_state=1945,
            n_jobs=-1,
            eval_metric="logloss"  # harmless; keeps metrics consistent in logs
        ) if 'XGBRFClassifier' in globals() and XGBRFClassifier is not None else None,
        "grid_small": (lambda y=None: _with_spw({
            "n_estimators": [300, 600],
            "max_depth": [4, 6],
            "subsample": [0.6, 0.9],
            "colsample_bynode": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
            "min_child_weight": [1, 5],
            "reg_lambda": [0.0, 1.0],
        }, y)),
        "grid_large": (lambda y=None: _with_spw({
            "n_estimators": [400, 800, 1200],
            "max_depth": [3, 6, 9],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bynode": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 6],
            "reg_lambda": [0.0, 1.0, 3.0],
            "reg_alpha": [0.0, 1.0],
        }, y)),

        "config": {
            "scoring": "recall",
            "notes": (
                "XGBRF = bagged (non-boosted) trees with XGBoost splits. "
                "No early-stopping; tune n_estimators directly. "
                "Focus on depth + (subsample, colsample_bynode/bytree) for recall; "
                "inject scale_pos_weight from y."
            ),
        },
    },

    #LightGBM
    "lgbm": {
        "name": "LGBMClassifier",
        "estimator": LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, random_state=1945, verbose=-1 #Stop training print lines
        ) if LGBMClassifier is not None else None,
        "grid_small": (lambda y=None: _with_spw({
            "num_leaves": [31, 63],
            "max_depth": [-1, 8],
            "min_child_samples": [20, 40],
            "feature_fraction": [0.8, 1.0],
            "bagging_fraction": [0.8, 1.0],
            "learning_rate": [0.03, 0.06],
        }, y)),
        "grid_large": (lambda y=None: _with_spw({
            "num_leaves": [31, 63, 95],
            "max_depth": [-1, 10],
            "min_child_samples": [20, 60],
            "feature_fraction": [0.75, 0.95, 1.0],
            "bagging_fraction": [0.75, 0.95, 1.0],
            "learning_rate": [0.03, 0.05, 0.08],
            "lambda_l2": [0.0, 3.0],
        }, y)),
        "config": {
            "scoring": "recall",
            "use_val_split": True,
            "validation_size": 0.1,
            "notes": "Enable early stopping via fit_with_hooks; pass valid_sets with (X_val,y_val)."
        },
    },

    #CatBoost
    "cat": {
        "name": "CatBoostClassifier",
        "estimator": CatBoostClassifier(
            iterations=3000, learning_rate=0.05, depth=6, random_state=1945, verbose=False
        ) if CatBoostClassifier is not None else None,
        "grid_small": (lambda y=None: _with_spw({
            "depth": [4, 6],
            "l2_leaf_reg": [1.0, 4.0],
            "learning_rate": [0.04, 0.07],
        }, y)),
        "grid_large": (lambda y=None: _with_spw({
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1.0, 5.0],
            "learning_rate": [0.03, 0.05, 0.08],
        }, y)),
        "config": {
            "scoring": "recall",
            "use_val_split": True,
            "validation_size": 0.1,
            "notes": "Enable early stopping via fit_with_hooks; add cat_feature indices later if needed."
        },
    },
}


#Convenience: pick a grid size and optionally pass y for GBMs
def get_spec(key, grid_size="small", y=None):
    """
    Retrieve (estimator, grid, config) for a given key and grid size.

    grid_size: 'small' or 'large'
    y: optional labels for GBMs to auto-inject scale_pos_weight
    """
    spec = MODEL_SPECS[key]
    est = spec["estimator"]
    cfg = spec["config"]
    if grid_size == "small":
        g = spec["grid_small"]
    else:
        g = spec["grid_large"]
    #For callables (GBMs) allow y to customize imbalance weight
    grid = g(y) if callable(g) else g
    return est, grid, cfg


"""
model_specs.py
Simple registry of model defaults and param grids for quick tests vs deeper optimization.

Usage (with Model_Tester_V2):
    from model_specs import MODEL_SPECS

    spec = MODEL_SPECS["gb"]
    model = spec["estimator"]
    grid  = spec["grid_small"] 

    #For XGB/LGBM/CatBoost you can optionally do:
    grid = spec["grid_small"](y)   #injects scale_pos_weight if y provided

    tester = Model_Tester_V2(model=model, parameter_grid=grid, cv_folds=5, model_config=spec["config"])
"""