"""Checks use clearly synthetic fixtures only; reported results use the ULB CSV."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TrainingConfig
from src.data import DatasetError, audit_and_clean, load_dataset, split_dataset


@pytest.fixture
def transactions():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(600, 30)), columns=FEATURES)
    df["Time"] = np.arange(600, dtype=float)
    df["Amount"] = rng.uniform(0, 1000, 600)
    df["Class"] = [0] * 540 + [1] * 60
    return df


def test_missing_file_explains_download(tmp_path):
    with pytest.raises(DatasetError, match="scripts/download_data.py"):
        load_dataset(tmp_path / "creditcard.csv")


def test_loader_rejects_schema_and_bad_labels(tmp_path, transactions):
    path = tmp_path / "bad.csv"
    transactions.drop(columns="V28").to_csv(path, index=False)
    with pytest.raises(DatasetError, match="V28"):
        load_dataset(path)
    transactions.loc[0, "Class"] = 2
    transactions.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="Class"):
        load_dataset(path)


def test_loader_accepts_valid_numeric_data(tmp_path, transactions):
    path = tmp_path / "valid.csv"
    transactions.to_csv(path, index=False)
    result = load_dataset(path)
    assert result.shape == (600, 31)
    assert result.Class.sum() == 60


def test_duplicates_removed_before_disjoint_stratified_split(transactions):
    raw = pd.concat([transactions, transactions.iloc[:4]], ignore_index=True)
    clean, audit = audit_and_clean(raw)
    assert audit["duplicates_removed"] == 4
    splits = split_dataset(clean, TrainingConfig())
    indices = [set(X.index) for X, _ in splits.values()]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert sum(map(len, indices)) == len(clean)
    for X, y in splits.values():
        assert "Class" not in X
        assert y.mean() == pytest.approx(0.1)


def test_conflicting_duplicate_labels_are_rejected(transactions):
    conflict = transactions.iloc[[0]].copy()
    conflict["Class"] = 1
    with pytest.raises(DatasetError, match="conflicting labels"):
        audit_and_clean(pd.concat([transactions, conflict], ignore_index=True))
