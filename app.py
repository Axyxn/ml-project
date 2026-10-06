"""Streamlit demonstration: raw transaction inputs → saved pipeline → fraud scores."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FEATURES, ROOT
from src.predict import ArtifactError, PredictionError, load_artifacts, predict_transactions, read_transaction_csv

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def cached_bundle(artifact_signature):
    # Signature invalidates the cache if the trained artifacts change.
    return load_artifacts(ROOT / "models")


def load_examples():
    path = ROOT / "results/example_transactions.csv"
    return pd.read_csv(path) if path.is_file() else None


def render_result(result):
    row = result.iloc[0]
    if row.predicted_class == 1:
        st.error("Potential fraud — review this transaction.")
    else:
        st.success("Legitimate — below the model's fraud threshold.")
    left, right = st.columns(2)
    left.metric("Fraud probability (model score)", f"{100 * row.fraud_probability:.2f}%")
    right.metric("Decision threshold", f"{row.decision_threshold:.6f}")
    st.caption("This score is uncalibrated. A legitimate prediction can still be missed fraud.")
    st.download_button("Download single result", result.to_csv(index=False).encode("utf-8"),
                       file_name="single_prediction.csv", mime="text/csv", key="download_single")


def main():
    st.title("💳 Credit Card Fraud Detection")
    st.write("Score a transaction or a CSV batch with the trained fraud detection model.")
    try:
        names = ["best_model.joblib", "scaler.joblib", "selected_features.joblib", "metadata.json"]
        signature = tuple((name, (ROOT / "models" / name).stat().st_mtime_ns) for name in names)
        bundle = cached_bundle(signature)
    except (ArtifactError, OSError) as exc:
        st.error(f"Model unavailable: {exc}")
        st.code("python scripts/download_data.py\npython scripts/train.py", language="shell")
        st.stop()

    threshold = bundle.metadata["threshold"]
    examples = load_examples()
    with st.sidebar:
        st.header("Saved model")
        st.write(bundle.metadata["algorithm"])
        st.caption(f"Validation-selected threshold: {threshold:.6f}")
        st.caption("ULB / Worldline data · seed 42")
        st.markdown("**Held-out test results**")
        metrics = bundle.metadata["final_test_metrics"]
        st.metric("Fraud recall", f"{metrics['recall']:.1%}")
        st.metric("Alert precision", f"{metrics['precision']:.1%}")
        st.caption(f"{metrics['tp']} frauds detected, {metrics['fn']} missed, {metrics['fp']} false alerts.")
        with st.expander("About these inputs"):
            st.write("Time is seconds since the dataset's first transaction. Amount is the transaction amount. "
                     "V1–V28 are anonymized PCA values from the dataset or a compatible upstream system. "
                     "Ordinary card details cannot be converted to these values here.")
            st.write("The two-day 2013 dataset is a classroom benchmark. These results describe its frozen test split.")

    single_tab, batch_tab = st.tabs(["Single transaction", "Batch CSV"])
    with single_tab:
        st.subheader("Enter transaction values")
        st.caption("Use raw dataset values. The model applies its saved scaling and feature selection.")
        options = ["Manual entry"]
        if examples is not None and "Class" in examples:
            options += ["Legitimate example", "Fraud example"]
        selected_example = st.selectbox("Load an example", options, key="example_choice")
        if st.session_state.get("loaded_example") != selected_example:
            values = dict.fromkeys(FEATURES, 0.0)
            if selected_example != "Manual entry":
                label = 1 if selected_example == "Fraud example" else 0
                values = examples.loc[examples.Class == label, FEATURES].iloc[0].to_dict()
            for name, value in values.items():
                st.session_state[f"feature_{name}"] = float(value)
            st.session_state["loaded_example"] = selected_example
        if selected_example != "Manual entry":
            st.caption("This is a real held-out dataset example. Its known label is for demonstration; "
                       "the model does not receive that label.")

        with st.form("single_transaction"):
            time_col, amount_col = st.columns(2)
            time_col.number_input("Time (seconds)", min_value=0.0, step=1.0, format="%.4f", key="feature_Time")
            amount_col.number_input("Amount", min_value=0.0, step=0.01, format="%.4f", key="feature_Amount")
            with st.expander("PCA components V1–V28", expanded=True):
                columns = st.columns(4)
                for index in range(1, 29):
                    columns[(index - 1) % 4].number_input(
                        f"V{index}", step=0.01, format="%.6f", key=f"feature_V{index}")
            submitted = st.form_submit_button("Predict transaction", type="primary")
        if submitted:
            frame = pd.DataFrame([{name: st.session_state[f"feature_{name}"] for name in FEATURES}])
            try:
                render_result(predict_transactions(frame, bundle))
            except PredictionError as exc:
                st.error(str(exc))

    with batch_tab:
        st.subheader("Upload transactions")
        st.write("Upload a comma-separated CSV containing Time, V1–V28, and Amount. "
                 "An optional Class column is ignored for prediction.")
        download_col, example_col = st.columns(2)
        download_col.download_button(
            "Download CSV template", pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"),
            file_name="transaction_template.csv", mime="text/csv", key="download_template")
        if examples is not None:
            example_col.download_button(
                "Download real example CSV", examples.to_csv(index=False).encode("utf-8"),
                file_name="example_transactions.csv", mime="text/csv", key="download_examples")
            if st.button("Score example batch", key="score_examples"):
                st.session_state["example_batch_result"] = predict_transactions(examples, bundle)
            if "example_batch_result" in st.session_state:
                st.dataframe(st.session_state["example_batch_result"], hide_index=True)
        uploaded = st.file_uploader("Transaction CSV", type=["csv"], key="transaction_csv")
        if uploaded is not None:
            try:
                uploaded.seek(0)
                frame = read_transaction_csv(uploaded)
                with st.spinner(f"Scoring {len(frame):,} transactions..."):
                    result = predict_transactions(frame, bundle)
                total_col, flagged_col = st.columns(2)
                total_col.metric("Transactions scored", f"{len(result):,}")
                flagged_col.metric("Flagged for review", f"{int(result.predicted_class.sum()):,}")
                st.dataframe(result, hide_index=True)
                st.download_button("Download batch predictions", result.to_csv(index=False).encode("utf-8"),
                                   file_name="fraud_predictions.csv", mime="text/csv", key="download_batch")
                st.caption("Fraud probability is an uncalibrated model score; labels use the saved validation threshold.")
            except PredictionError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
