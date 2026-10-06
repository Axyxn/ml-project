import numpy as np
import pytest
from src.evaluation import compute_metrics, select_threshold


def test_known_confusion_counts_and_metrics():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.8, 0.3])
    m = compute_metrics(y, scores)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == m["recall"] == m["f1"] == pytest.approx(0.5)
    assert m["average_precision"] == pytest.approx(5 / 6)


def test_validation_threshold_matches_actual_f1():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.8, 0.3])
    threshold, table = select_threshold(y, scores)
    assert threshold == pytest.approx(0.3)
    assert compute_metrics(y, scores, threshold)["f1"] == pytest.approx(table.f1.max())


def test_always_legitimate_exposes_accuracy_trap():
    y = np.array([0] * 999 + [1])
    m = compute_metrics(y, np.zeros(1000))
    assert m["accuracy"] == pytest.approx(0.999)
    assert m["recall"] == m["precision"] == m["f1"] == 0
    assert m["average_precision"] == pytest.approx(0.001)
