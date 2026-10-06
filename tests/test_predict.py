from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES
from src.predict import ArtifactError, PredictionError, load_artifacts, read_transaction_csv, validate_transactions


def valid_row():
    return pd.DataFrame([{name: 0.0 for name in FEATURES}])


def test_input_requires_schema_and_finite_values():
    X = valid_row()
    with pytest.raises(PredictionError, match="V28"):
        validate_transactions(X.drop(columns="V28"))
    X.loc[0, "V28"] = np.inf
    with pytest.raises(PredictionError, match="V28"):
        validate_transactions(X)
    X.loc[0, "V28"] = np.nan
    with pytest.raises(PredictionError, match="Missing or infinite"):
        validate_transactions(X)


def test_negative_amount_rejected_and_class_ignored():
    X = valid_row()
    X["Class"] = 1
    assert "Class" not in validate_transactions(X)
    X.loc[0, "Amount"] = -1
    with pytest.raises(PredictionError, match="nonnegative"):
        validate_transactions(X)


def test_bad_and_valid_csv_messages():
    with pytest.raises(PredictionError, match="Cannot read CSV"):
        read_transaction_csv(StringIO(""))
    frame = read_transaction_csv(StringIO(valid_row().to_csv(index=False)))
    assert frame.shape == (1, 30)


def test_missing_artifacts_explains_training(tmp_path):
    with pytest.raises(ArtifactError, match="scripts/train.py"):
        load_artifacts(tmp_path)
