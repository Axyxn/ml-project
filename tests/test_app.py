"""Simulated UI interactions using Streamlit AppTest and actual saved ULB artifacts."""
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.predict import load_artifacts, predict_transactions

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not (ROOT / "models/best_model.joblib").exists(), reason="Train real models first")]


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)


def test_single_manual_prediction_and_real_fraud_example():
    at = app().run()
    assert not at.exception
    assert len(at.number_input) == 30
    button = next(item for item in at.button if item.label == "Predict transaction")
    button.click().run()
    assert not at.exception
    assert any(item.label == "Fraud probability (model score)" for item in at.metric)
    at.selectbox(key="example_choice").select("Fraud example").run()
    next(item for item in at.button if item.label == "Predict transaction").click().run()
    assert not at.exception
    row = pd.read_csv(ROOT / "results/example_transactions.csv").query("Class == 1").iloc[[0]]
    expected = predict_transactions(row, load_artifacts()).iloc[0]
    score_metric = next(item for item in at.metric if item.label == "Fraud probability (model score)")
    assert score_metric.value == f"{100*expected.fraud_probability:.2f}%"
    if expected.predicted_class:
        assert any("Potential fraud" in item.value for item in at.error)
    else:
        assert any("Legitimate" in item.value for item in at.success)


def test_batch_example_button_scores_real_rows():
    at = app().run()
    at.button(key="score_examples").click().run()
    assert not at.exception
    assert len(at.dataframe[-1].value) == 6
    assert "fraud_probability" in at.dataframe[-1].value


def test_csv_upload_branch_and_download_contents():
    examples = pd.read_csv(ROOT / "results/example_transactions.csv")
    # Streamlit 1.50 AppTest does not expose a file-uploader setter. Inject file
    # bytes at the uploader boundary, exercising the actual parser/scoring/UI path.
    uploaded = BytesIO(examples.to_csv(index=False).encode("utf-8"))
    with patch("streamlit.file_uploader", return_value=uploaded):
        at = app().run()
    assert not at.exception
    displayed = at.dataframe[-1].value
    expected = predict_transactions(examples, load_artifacts())
    np.testing.assert_allclose(displayed.fraud_probability, expected.fraud_probability)
    assert any(item.label == "Transactions scored" and item.value == "6" for item in at.metric)
    assert any(item.label == "Download batch predictions" for item in at.get("download_button"))


def test_invalid_upload_is_a_readable_error():
    with patch("streamlit.file_uploader", return_value=BytesIO(b"Amount\n10\n")):
        at = app().run()
    assert not at.exception
    assert any("Missing columns" in item.value for item in at.error)
