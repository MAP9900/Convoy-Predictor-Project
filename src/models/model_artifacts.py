#Imports
from datetime import datetime
import json
from pathlib import Path
import joblib
from .ML_Class_2 import Model_Tester_V2

"""Lightweight helpers for persisting Model_Tester_V2 runs."""

#Directory setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_SUBDIR = "algorithm_test_3"
DEFAULT_MODEL_NAMES = [ #Adjust As Needed!
    "dt",
    "rf",
    "et",
    "bag",
    "gb",
    "ada",
    "qda",
    "xgb",
    "xgbrf",
    "lgbm",
    "cat",]


def get_artifact_dir(subdir=None):
    """Return (and create) the directory where models are stored."""
    target = ARTIFACTS_DIR / (subdir or DEFAULT_SUBDIR)
    target.mkdir(parents=True, exist_ok=True)
    return target

def _prep_directory(directory=None):
    if directory is None:
        return get_artifact_dir()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def save_model(name, tester, directory=None, metadata=None):
    """Persist a fitted tester to disk."""
    if tester.best_model is None:
        raise ValueError(f"Model '{name}' has not been optimized/fitted yet.")
    directory = _prep_directory(directory)
    artifact_path = directory / f"{name}.joblib"
    joblib.dump(tester.best_model, artifact_path)
    info = {
        "model_name": name,
        "saved_at": _timestamp(),
        "scoring": tester.scoring,
        "notes": tester.model_config.get("notes"),
        "decision_threshold": tester.decision_threshold,
        "threshold_metric": tester.threshold_metric,
    }
    if metadata:
        info.update(metadata)
    (directory / f"{name}.json").write_text(json.dumps(info, indent=2))
    return artifact_path

def save_models(model_map, directory=None):
    """Save every fitted tester we know about."""
    saved = {}
    for name, tester in model_map.items():
        if tester.best_model is None:
            continue
        saved[name] = str(save_model(name, tester, directory=directory))
    return saved

def load_model(name, directory=None, assign_to=None):
    """Load a saved estimator and optionally reattach it."""
    directory = _prep_directory(directory)
    artifact_path = directory / f"{name}.joblib"
    info_path = directory / f"{name}.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No saved model found for '{name}' at {artifact_path}")

    model = joblib.load(artifact_path)
    metadata = json.loads(info_path.read_text()) if info_path.exists() else {}
    if assign_to is not None:
        assign_to.best_model = model
        assign_to.decision_threshold = metadata.get("decision_threshold")
        assign_to.threshold_metric = metadata.get("threshold_metric")
    return model

def load_models(names=None, directory=None, tester_map=None):
    """Load multiple models at once, assigning them when possible."""
    names = names or DEFAULT_MODEL_NAMES
    directory = _prep_directory(directory)
    loaded = {}
    for name in names:
        try:
            if tester_map and name in tester_map:
                model = load_model(name, directory=directory, assign_to=tester_map[name])
            else:
                model = load_model(name, directory=directory)
        except FileNotFoundError:
            continue
        loaded[name] = model
    return loaded

__all__ = [
    "DEFAULT_MODEL_NAMES",
    "get_artifact_dir",
    "load_model",
    "load_models",
    "save_model",
    "save_models",]
