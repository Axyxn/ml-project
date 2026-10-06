"""Verify training-only SMOTE behavior using a small synthetic fixture."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import FEATURES, TrainingConfig
from src.training import make_pipeline


def fixture_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(500, 30)), columns=FEATURES)
    X.Time = rng.uniform(0, 1000, 500)
    X.Amount = rng.uniform(0, 100, 500)
    y = pd.Series([0] * 450 + [1] * 50)
    X.loc[y == 1, "V14"] -= 3
    return X, y


def test_smote_changes_training_counts_and_is_skipped_at_inference(tmp_path, monkeypatch):
    X, y = fixture_data()
    config = TrainingConfig(selection_max_rows=200, smote_ratio=0.5, n_jobs=1)
    pipe = make_pipeline("Logistic Regression", True, config, tmp_path).fit(X, y)
    sampler = pipe.named_steps["smote"]
    assert sampler.sampling_strategy_[1] == 175
    def must_not_resample(*args, **kwargs):
        raise AssertionError("Inference must never call fit_resample")
    monkeypatch.setattr(sampler, "fit_resample", must_not_resample)
    scores = pipe.predict_proba(X.iloc[:7])[:, 1]
    assert len(scores) == 7
    assert np.isfinite(scores).all()
    assert ((0 <= scores) & (scores <= 1)).all()


def test_cv_fits_selection_and_scaling_on_fold_rows_only(tmp_path):
    X, y = fixture_data()
    config = TrainingConfig(selection_max_rows=1000, smote_ratio=0.5, n_jobs=1)
    template = make_pipeline("Decision Tree", True, config, tmp_path)
    for train_idx, val_idx in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        fold = clone(template).fit(X.iloc[train_idx], y.iloc[train_idx])
        assert fold.named_steps["select"].ranking_rows_ == len(train_idx)
        center = fold.named_steps["scaler"].named_transformers_["time_amount"].center_[0]
        assert center == X.iloc[train_idx].Time.median()
        assert len(fold.predict_proba(X.iloc[val_idx])) == len(val_idx)
