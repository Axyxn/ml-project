"""Validate the real dataset, remove identical records, then stratify."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASET_PAGE, FEATURES, ROOT, TARGET, TrainingConfig


class DatasetError(ValueError):
    """A readable error for missing or unsuitable transaction data."""


def load_dataset(path=ROOT / "data" / "creditcard.csv"):
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"Dataset missing: {path}\nRun: python scripts/download_data.py\n"
            f"Or download creditcard.csv from {DATASET_PAGE}, unzip it, "
            "and place it in the project's data/ folder. Synthetic data is not a replacement."
        )
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise DatasetError(f"Dataset is missing columns: {', '.join(missing)}. Use the original ULB CSV.")
    df = df.loc[:, required].copy()
    for col in required:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"Column {col} must contain numeric values.") from exc
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise DatasetError("Class must contain both 0 (legitimate) and 1 (fraud), with no missing targets.")
    if np.isinf(df[FEATURES].to_numpy()).any():
        raise DatasetError("Infinite feature values are invalid. Correct the CSV before training.")
    if (df[["Time", "Amount"]] < 0).any().any():
        raise DatasetError("Time and Amount must be nonnegative.")
    if df[FEATURES].isna().all().any():
        raise DatasetError("A feature is entirely missing. Obtain a complete dataset.")
    df[TARGET] = df[TARGET].astype("int8")
    if df[TARGET].value_counts().min() < 30:
        raise DatasetError("At least 30 examples of each class are needed for stable splitting and SMOTE CV.")
    return df


def audit_and_clean(df):
    """Exact full-row duplicates are removed without learning any statistics."""
    duplicates = int(df.duplicated().sum())
    clean = df.drop_duplicates().copy()  # Preserve original indices for split audits.
    # Identical predictors with conflicting labels cannot safely be split independently.
    if clean.duplicated(subset=FEATURES).any():
        raise DatasetError("Identical features have conflicting labels; resolve labels before splitting.")
    audit = {
        "raw_rows": len(df), "raw_frauds": int(df[TARGET].sum()),
        "raw_fraud_rate": float(df[TARGET].mean()), "duplicates_removed": duplicates,
        "clean_rows": len(clean), "clean_frauds": int(clean[TARGET].sum()),
        "clean_fraud_rate": float(clean[TARGET].mean()),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
        "encoding": "All 30 predictors are numeric; no categorical encoding is needed.",
        "outliers": "Kept: extreme values can be genuine fraud signals. No clipping or IQR deletion.",
    }
    return clean, audit


def split_dataset(clean, config=None):
    config = config or TrainingConfig()
    if not 0 < config.test_size + config.validation_size < 1:
        raise DatasetError("Test and validation fractions must sum to less than one.")
    X, y = clean[FEATURES], clean[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=config.validation_size / (1 - config.test_size),
        stratify=y_dev, random_state=config.seed)
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}


def stratified_sample(X, y, max_rows, seed):
    """Preserve natural prevalence; used only on training data for bounded work."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X_small, y_small
