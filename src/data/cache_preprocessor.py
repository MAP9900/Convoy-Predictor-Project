"""
cache_preprocessor.py

Lightweight utility to materialize cached train/test (and optional validation) splits.
Keeps Model_Tester_V2 runs snappy by avoiding repeated feature engineering.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_frame(input_path: Path) -> pd.DataFrame:
    """
    Read a dataframe based on file extension.
    Supports CSV, Parquet, and Pickle since those cover the existing notebooks.
    """
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(input_path)
    return pd.read_csv(input_path)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """
    Persist a DataFrame using Parquet when available, falling back to CSV otherwise.
    """
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def maybe_sample(df: pd.DataFrame, target: str, sample_frac: float, random_state: int) -> pd.DataFrame:
    """
    Optional down-sampling for fast exploratory passes while keeping stratification.
    """
    if sample_frac >= 0.999:
        return df
    sampled, _ = train_test_split(df, train_size=sample_frac, random_state=random_state, stratify=df[target])
    return sampled.reset_index(drop=True)


def persist_cache(X, y, cache_dir: Path, prefix: str) -> None:
    """
    Save features and targets with a shared prefix (e.g., X_train / y_train).
    """
    write_frame(pd.DataFrame(X), cache_dir / f"X_{prefix}.parquet")
    write_frame(pd.DataFrame({"target": y}), cache_dir / f"y_{prefix}.parquet")


def build_cache(input_path: Path, target_col: str, cache_dir: Path, test_size: float,
                val_size: float, sample_frac: float, random_state: int) -> None:
    """
    End-to-end routine: load, optional sample, split, and persist.
    """
    df = load_frame(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")

    df = maybe_sample(df, target_col, sample_frac, random_state)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
        stratify=y, random_state=random_state)

    if val_size > 0:
        #Adjust validation size relative to the current train block
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=relative_val,
            stratify=y_train, random_state=random_state)
    else:
        X_val = y_val = None

    cache_dir.mkdir(parents=True, exist_ok=True)
    persist_cache(X_train, y_train, cache_dir, "train")
    persist_cache(X_test, y_test, cache_dir, "test")
    if X_val is not None:
        persist_cache(X_val, y_val, cache_dir, "val")

    (cache_dir / "feature_names.txt").write_text("\n".join(X.columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache preprocessed train/test splits for faster experiments.")
    parser.add_argument("--input", required=True, help="Path to raw feature table (csv/parquet/pickle).")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--cache-dir", default="data/cache", help="Directory to store cached splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for test data.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="Optional validation fraction (taken from the training block).")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of the dataset to sample for quick exploratory runs." )
    parser.add_argument("--seed", type=int, default=1945, help="Random seed for deterministic splits.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_cache(
        input_path=Path(args.input),
        target_col=args.target,
        cache_dir=Path(args.cache_dir),
        test_size=args.test_size,
        val_size=args.val_size,
        sample_frac=args.sample_frac,
        random_state=args.seed,)
    print(f"Cached splits written to {args.cache_dir}")


if __name__ == "__main__":
    main()



# Example Test Snippet:

# from pathlib import Path

# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# from src.data.cache_preprocessor import build_cache
# from src.models.model_specs import MODEL_SPECS
# from src.models.ML_Class_2 import Model_Tester_V2
# from src.models.perf_utils import track_perf

# CACHE_DIR = Path("data/cache")
# X_train_path = CACHE_DIR / "X_train.parquet"
# y_train_path = CACHE_DIR / "y_train.parquet"

# if not X_train_pat.exists():
#     build_cache(
#         input_path=Path("data/processed/features.parquet"),
#         target_col="target",
#         cache_dir=CACHE_DIR,
#         test_size=0.2,
#         val_size=0.1,
#         sample_frac=0.4,
#         random_state=1945,
#     )

# X_train = pd.read_parquet(X_train_path)
# y_train = pd.read_parquet(y_train_path)["target"]

# spec = MODEL_SPECS["gb"]
# gb_model = Model_Tester_V2(
#     model=spec["estimator"],
#     scaler=StandardScaler(),
#     parameter_grid=spec["grid_small"],
#     cv_folds=3,
#     feature_names=X_train.columns.tolist(),
#     model_config=spec["config"],
# )

# gb_model.X_train = X_train
# gb_model.y_train = y_train
# gb_model.X_test = pd.read_parquet(CACHE_DIR / "X_test.parquet")
# gb_model.y_test = pd.read_parquet(CACHE_DIR / "y_test.parquet")["target"]

# @track_perf("gb_optimize")
# def run_optimize():
#     gb_model.optimize(scoring="recall")

# run_optimize()
# gb_results = gb_model.evaluate(show_plots=False)