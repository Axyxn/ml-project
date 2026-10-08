"""One inference path for notebook, app, tests, and batch CSVs."""
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import FEATURES, ROOT


class PredictionError(ValueError):
    """Readable input error suitable for a Streamlit message."""


class ArtifactError(RuntimeError):
    """Missing or inconsistent training artifacts."""


@dataclass
class ModelBundle:
    pipeline: object
    scaler: object
    selected_features: list
    metadata: dict


def load_artifacts(model_dir=ROOT / "models"):
    model_dir = Path(model_dir)
    names = ["best_model.joblib", "scaler.joblib", "selected_features.joblib", "metadata.json"]
    missing = [name for name in names if not (model_dir / name).is_file()]
    if missing:
        raise ArtifactError(f"Model artifacts missing: {', '.join(missing)}. Run python scripts/train.py first.")
    try:
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["versions"]["sklearn"] != sklearn.__version__:
            raise ArtifactError("scikit-learn differs from the training version. Install requirements.txt or retrain.")
        pipeline = joblib.load(model_dir / "best_model.joblib")
        # Inference is small and may run in restricted Windows/Streamlit
        # environments where joblib cannot create its worker pool.
        classifier = pipeline.named_steps.get("classifier")
        if hasattr(classifier, "n_jobs"):
            classifier.set_params(n_jobs=1)
        scaler = joblib.load(model_dir / "scaler.joblib")
        selected = joblib.load(model_dir / "selected_features.joblib")
        if selected != metadata["selected_features"] or selected != pipeline.named_steps["select"].selected_features_:
            raise ArtifactError("Feature artifacts disagree. Regenerate all artifacts together with scripts/train.py.")
        if metadata["raw_features"] != FEATURES or list(pipeline.feature_names_in_) != FEATURES:
            raise ArtifactError("Model input schema does not match the project's 30-feature schema.")
        # Confirm that the separately loaded scaler is the scaler in the inference pipeline.
        embedded = pipeline.named_steps["scaler"]
        for transformer, attr in [("time_amount", "center_"), ("time_amount", "scale_"),
                                  ("pca", "mean_"), ("pca", "scale_")]:
            a = getattr(scaler.named_transformers_[transformer], attr)
            b = getattr(embedded.named_transformers_[transformer], attr)
            if not np.array_equal(a, b):
                raise ArtifactError("Saved scaler differs from the model's fitted scaler. Retrain artifacts together.")
        threshold = metadata["threshold"]
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactError("Saved threshold is invalid. Retrain the model.")
        return ModelBundle(pipeline, scaler, selected, metadata)
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, AttributeError, EOFError) as exc:
        raise ArtifactError(f"Cannot load model artifacts: {exc}. Re-run python scripts/train.py.") from exc


def validate_transactions(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PredictionError("Provide at least one transaction in a CSV with the 30 required numeric columns.")
    if frame.columns.duplicated().any():
        raise PredictionError("Duplicate column names are not allowed.")
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise PredictionError(f"Missing columns: {', '.join(missing)}. Download the CSV template from the app.")
    X = frame.loc[:, FEATURES].copy()
    for name in FEATURES:
        try:
            X[name] = pd.to_numeric(X[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise PredictionError(f"{name} must contain numeric values; check your CSV.") from exc
    bad_columns = X.columns[~np.isfinite(X.to_numpy(dtype=float)).all(axis=0)].tolist()
    if bad_columns:
        raise PredictionError(f"Missing or infinite values in: {', '.join(bad_columns)}. Enter complete finite values.")
    if (X[["Time", "Amount"]] < 0).any().any():
        raise PredictionError("Time and Amount must be nonnegative. PCA components may be negative.")
    return X.astype(float)


def read_transaction_csv(source):
    try:
        frame = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        raise PredictionError(f"Cannot read CSV: {exc}. Use a comma-separated UTF-8 file with a header.") from exc
    validate_transactions(frame)
    return frame


def predict_transactions(frame, bundle=None, threshold=None):
    bundle = bundle or load_artifacts()
    X = validate_transactions(frame)
    threshold = bundle.metadata["threshold"] if threshold is None else threshold
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PredictionError("Decision threshold must be a finite number between 0 and 1.")
    try:
        probabilities = bundle.pipeline.predict_proba(X)[:, 1]
    except ValueError as exc:
        raise PredictionError(f"The model cannot score these values: {exc}. Check their numeric ranges.") from exc
    if not np.isfinite(probabilities).all():
        raise PredictionError("The model produced invalid probabilities; recheck inputs and training artifacts.")
    result = frame.copy()
    result["fraud_probability"] = probabilities
    result["predicted_class"] = (probabilities >= threshold).astype(int)
    result["prediction"] = np.where(result.predicted_class == 1, "Potential fraud", "Legitimate")
    result["decision_threshold"] = threshold
    return result
