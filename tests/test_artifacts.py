"""Real-data checks for the finished experiment; never use fixtures as reported data."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, TrainingConfig
from src.evaluation import compute_metrics
from src.predict import load_artifacts, predict_transactions
from src.training import load_splits

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not (ROOT / "models/metadata.json").is_file(), reason="Train the real ULB models first")]


@pytest.fixture(scope="module")
def bundle():
    return load_artifacts()


def test_saved_model_scaler_and_preprocessor_agree(bundle):
    examples = pd.read_csv(ROOT / "results/example_transactions.csv")
    full = predict_transactions(examples, bundle)
    individual = [predict_transactions(examples.iloc[[i]], bundle).fraud_probability.iloc[0]
                  for i in range(len(examples))]
    np.testing.assert_allclose(full.fraud_probability, individual, atol=1e-12)
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    X = examples[bundle.metadata["raw_features"]]
    manual = bundle.pipeline.named_steps["classifier"].predict_proba(preprocessor.transform(X))[:, 1]
    np.testing.assert_allclose(full.fraud_probability, manual, atol=1e-12)
    assert list(preprocessor.transform(X).columns) == bundle.selected_features


def test_model_choice_matches_validation_and_frozen_record(bundle):
    table = pd.read_csv(ROOT / "results/model_comparison.csv", keep_default_na=False)
    candidates = table[(table.partition == "validation") & table.stage.isin(["baseline", "tuned"])]
    winner = candidates.sort_values(["average_precision", "f1", "experiment"], ascending=[False, False, True]).iloc[0]
    assert winner.experiment == bundle.metadata["experiment"]
    record = json.loads((ROOT / "results/frozen_selection.json").read_text())
    assert record["test_used_for_selection"] is False
    assert record["threshold"] == bundle.metadata["threshold"]
    assert len(candidates) == 12
    assert len(table[(table.partition == "test") & table.stage.isin(["baseline", "tuned"])]) == 12


def test_real_test_metrics_and_training_scaler_reproduce(bundle):
    config = TrainingConfig(**bundle.metadata["config"])
    splits, audit = load_splits(ROOT / "data/creditcard.csv", config)
    assert audit["data_sha256"] == bundle.metadata["data_audit"]["data_sha256"]
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    robust = bundle.scaler.named_transformers_["time_amount"]
    np.testing.assert_allclose(robust.center_, X_train[["Time", "Amount"]].median().to_numpy())
    scores = bundle.pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, scores, bundle.metadata["threshold"])
    for name, expected in bundle.metadata["final_test_metrics"].items():
        assert metrics[name] == pytest.approx(expected)
    indices = [set(X.index) for X, _ in splits.values()]
    assert not indices[0] & indices[1] and not indices[0] & indices[2] and not indices[1] & indices[2]
