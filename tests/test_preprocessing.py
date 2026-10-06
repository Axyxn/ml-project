import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURES, TrainingConfig
from src.preprocessing import make_preprocessing_steps


def test_scalers_and_selection_fit_only_training_and_keep_named_columns():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(300, 30)), columns=FEATURES)
    X["Time"] = np.arange(300, dtype=float)
    X["Amount"] = rng.uniform(0, 200, len(X))
    y = pd.Series([0] * 240 + [1] * 60)
    X.loc[4, "V2"] = np.nan
    prep = Pipeline(make_preprocessing_steps(TrainingConfig(selection_max_rows=200))).fit(X, y)
    robust = prep.named_steps["scaler"].named_transformers_["time_amount"]
    assert robust.center_[0] == pytest.approx(X.Time.median())
    assert robust.center_[1] == pytest.approx(X.Amount.median())
    before = robust.center_.copy()
    held_out = X.iloc[:3].copy()
    held_out.Time = 1e9
    result = prep.transform(held_out)
    assert result.shape == (3, 20)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_array_equal(before, robust.center_)
    selector = prep.named_steps["select"]
    assert selector.ranking_rows_ == 200
    assert selector.ranking_frauds_ == 40
    assert list(result.columns) == selector.selected_features_
