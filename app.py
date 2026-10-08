"""Streamlit interface for scoring credit-card transactions."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURES, ROOT
from src.predict import ArtifactError, PredictionError, load_artifacts, predict_transactions, read_transaction_csv

st.set_page_config(
    page_title="Fraud Signal Studio",
    layout="wide",
    initial_sidebar_state="auto",
)

INK = "#17343B"
TEAL = "#147D72"
CORAL = "#D65D4B"
MUTED = "#63777A"


@st.cache_resource
def cached_bundle(artifact_signature):
    # Signature invalidates the cache if the trained artifacts change.
    return load_artifacts(ROOT / "models")


def load_examples():
    path = ROOT / "results/example_transactions.csv"
    return pd.read_csv(path) if path.is_file() else None


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink: #17343B; --teal: #147D72; --coral: #D65D4B; --muted: #63777A; }
        html, body, .stApp { font-family: 'DM Sans', sans-serif; }
        input, textarea, button, select { font-family: inherit; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: -0.035em; }
        .stApp { background: #F7F9F6; }
        [data-testid="stHeader"] { background: rgba(247, 249, 246, .92); }
        [data-testid="stSidebar"] { background: #EFF4F0; border-right: 1px solid #DCE7E1; }
        [data-testid="stSidebar"] h2 { font-size: 1.1rem; }
        [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DCE7E1; border-radius: 12px; padding: 14px 16px; }
        [data-testid="stMetricLabel"] { color: #63777A; }
        [data-testid="stMetricValue"] { color: #17343B; font-family: 'Manrope', sans-serif; }
        [data-testid="stTabs"] button { font-weight: 600; }
        [data-testid="stForm"] { background: #FFFFFF; border: 1px solid #DCE7E1; border-radius: 14px; padding: 18px 20px; }
        [data-testid="stExpander"] { background: rgba(255,255,255,.76); border-color: #DCE7E1; border-radius: 12px; }
        [data-testid="stFileUploader"] section { background: #FFFFFF; border-color: #A6C8BB; }
        .hero { position: relative; overflow: hidden; background: #17343B; color: #F4FBF7; border-radius: 18px; padding: 30px 34px; margin: 4px 0 28px; }
        .hero:after { content: ''; position: absolute; width: 250px; height: 250px; right: 3%; top: -105px; border: 1px solid rgba(182, 230, 207, .28); border-radius: 50%; box-shadow: 0 0 0 25px rgba(182,230,207,.04), 0 0 0 52px rgba(182,230,207,.035); }
        .hero h1 { color: #F4FBF7; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.06; margin: 0 0 10px; }
        .hero p { color: #C4DDD4; font-size: 1.04rem; max-width: 660px; margin: 0; line-height: 1.6; }
        .section-note { color: #63777A; margin-top: -10px; margin-bottom: 18px; }
        .result-title { font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 750; color: #17343B; }
        .result-copy { color: #63777A; margin-top: 2px; }
        .stButton button, .stFormSubmitButton button { border-radius: 9px; font-weight: 700; }
        button:focus-visible, input:focus-visible, [tabindex="0"]:focus-visible { outline: 3px solid #147D72 !important; outline-offset: 2px; }
        ::selection { background: #BCE8D6; color: #17343B; }
        @media (max-width: 700px) {
          .hero { padding: 23px 22px; margin-bottom: 20px; }
          .hero:after { right: -110px; }
          [data-testid="stForm"] { padding: 12px; }
        }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; transition-duration: .01ms !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def score_figure(probability, threshold):
    """Draw the model score and its saved operating threshold on one track."""
    color = CORAL if probability >= threshold else TEAL
    fig, ax = plt.subplots(figsize=(8, 1.35))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(0, 1, height=0.25, color="#E7EEEA", edgecolor="none", zorder=1)
    ax.barh(0, probability, height=0.25, color=color, edgecolor="none", zorder=2)
    ax.axvline(threshold, color=INK, linewidth=2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, threshold, 1], ["0", f"threshold  {threshold:.1%}", "100%"])
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def show_figure(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def input_profile_figure(row):
    """Visualize the transaction's signed PCA values, without implying causality."""
    values = pd.Series({f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)})
    values = values.reindex(values.abs().sort_values(ascending=False).head(12).index).sort_values()
    colors = [TEAL if value >= 0 else CORAL for value in values]
    height = max(2.8, 0.28 * len(values))
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(values.index, values.values, color=colors, height=0.68)
    ax.axvline(0, color="#AFC0B9", linewidth=1)
    ax.set_xlabel("Raw PCA component value", color=MUTED, fontsize=9)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color="#E7EEEA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


def importance_figure(bundle):
    """Show fitted-forest global importances, not a transaction-level explanation."""
    classifier = bundle.pipeline.named_steps.get("classifier")
    selector = bundle.pipeline.named_steps.get("select")
    if not hasattr(classifier, "feature_importances_"):
        return None
    names = list(selector.selected_features_)
    importance = pd.Series(classifier.feature_importances_, index=names).nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(importance.index, importance.values, color=TEAL, height=0.68)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color="#E7EEEA", linewidth=0.8)
    ax.set_xlabel("Mean decrease in impurity", color=MUTED, fontsize=9)
    ax.set_title("Features the forest relied on most overall", loc="left", color=INK, fontsize=11, pad=10)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


def render_result(result, bundle):
    row = result.iloc[0]
    flagged = row.predicted_class == 1
    title = "Review this transaction" if flagged else "Below the alert threshold"
    st.markdown(f'<div class="result-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="result-copy">A model score to support review, not a confirmed fraud decision.</div>',
        unsafe_allow_html=True,
    )
    if flagged:
        st.error("Potential fraud — review this transaction.")
    else:
        st.success("Legitimate — below the model's fraud threshold.")

    left, right = st.columns(2)
    left.metric("Fraud probability (model score)", f"{100 * row.fraud_probability:.2f}%")
    right.metric("Decision threshold", f"{row.decision_threshold:.2%}")
    show_figure(score_figure(float(row.fraud_probability), float(row.decision_threshold)))

    profile_col, context_col = st.columns([1.1, 0.9], gap="large")
    with profile_col:
        st.markdown("#### This transaction’s input pattern")
        show_figure(input_profile_figure(row))
        st.caption("Largest 12 PCA values by magnitude. Their original meanings are anonymized.")
    with context_col:
        st.markdown("#### Model context")
        fig = importance_figure(bundle)
        if fig is not None:
            show_figure(fig)
            st.caption("Global feature importance across the fitted forest; it does not explain this one score.")
        st.info("Scores are uncalibrated. A low score does not rule out fraud, and the PCA values have no direct business meaning here.")

    st.download_button(
        "Download this result",
        result.to_csv(index=False).encode("utf-8"),
        file_name="single_prediction.csv",
        mime="text/csv",
        key="download_single",
    )


def render_batch_summary(result):
    scores = result.fraud_probability.astype(float)
    flagged_count = int(result.predicted_class.sum())
    total = len(result)
    st.markdown("#### Batch at a glance")
    first, second, third = st.columns(3)
    first.metric("Transactions scored", f"{total:,}")
    second.metric("Flagged for review", f"{flagged_count:,}")
    second.caption(f"{flagged_count / total:.1%} of this batch")
    third.metric("Highest model score", f"{scores.max():.2%}")

    threshold = float(result.decision_threshold.iloc[0])
    alert = scores[scores >= threshold]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    upper = min(1.0, max(0.08, float(scores.max()) * 1.12, threshold * 1.12))
    bins = np.linspace(0, upper, 17)
    ax.hist(scores[scores < threshold], bins=bins, color=TEAL, alpha=0.85, label="Below threshold")
    if not alert.empty:
        ax.hist(alert, bins=bins, color=CORAL, alpha=0.9, label="At or above threshold")
    ax.axvline(threshold, color=INK, linewidth=1.8, linestyle="--", label=f"Threshold {threshold:.1%}")
    ax.set_xlim(0, upper)
    ax.set_xlabel("Fraud probability (uncalibrated model score)", color=MUTED, fontsize=9)
    ax.set_ylabel("Transactions", color=MUTED, fontsize=9)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color="#E7EEEA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    show_figure(fig)
    st.caption("Bars show this batch’s score distribution; the dashed line is the saved validation threshold.")


def main():
    inject_styles()
    st.markdown(
        """
        <section class="hero">
          <h1>Find the signal in the swipe.</h1>
          <p>Explore how a trained fraud model scores one transaction or a whole batch. See the score, its review threshold, and the data pattern behind your input.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown("## Model notes")
        st.write(f"**{bundle.metadata['algorithm']}** · trained benchmark")
        st.caption(f"Saved review threshold: {threshold:.2%}")
        st.markdown("---")
        st.markdown("#### Held-out test snapshot")
        metrics = bundle.metadata["final_test_metrics"]
        st.metric("Fraud recall", f"{metrics['recall']:.1%}")
        st.metric("Alert precision", f"{metrics['precision']:.1%}")
        st.caption(f"{metrics['tp']} detected · {metrics['fn']} missed · {metrics['fp']} false alerts")
        st.caption("ULB / Worldline benchmark · seed 42")
        with st.expander("What these inputs mean"):
            st.write(
                "Time is seconds since the dataset's first transaction. Amount is the transaction amount. "
                "V1–V28 are anonymized PCA values from the dataset or a compatible upstream system. "
                "Ordinary card details cannot be converted into these values here."
            )
            st.write("This two-day 2013 dataset is a classroom benchmark. Results describe its frozen test split.")

    single_tab, batch_tab = st.tabs(["Single transaction", "Batch CSV"])
    with single_tab:
        st.subheader("Score one transaction")
        st.markdown('<p class="section-note">Start with a real example or enter compatible dataset values.</p>', unsafe_allow_html=True)
        options = ["Manual entry"]
        if examples is not None and "Class" in examples:
            options += ["Legitimate example", "Fraud example"]
        selected_example = st.selectbox("Load a starting point", options, key="example_choice")
        if st.session_state.get("loaded_example") != selected_example:
            values = dict.fromkeys(FEATURES, 0.0)
            if selected_example != "Manual entry":
                label = 1 if selected_example == "Fraud example" else 0
                values = examples.loc[examples.Class == label, FEATURES].iloc[0].to_dict()
            for name, value in values.items():
                st.session_state[f"feature_{name}"] = float(value)
            st.session_state["loaded_example"] = selected_example
        if selected_example != "Manual entry":
            st.caption("Real held-out example. Its known label is shown for demonstration; the model does not receive it.")

        with st.form("single_transaction"):
            time_col, amount_col = st.columns(2)
            time_col.number_input("Time (seconds)", min_value=0.0, step=1.0, format="%.4f", key="feature_Time")
            amount_col.number_input("Amount", min_value=0.0, step=0.01, format="%.4f", key="feature_Amount")
            with st.expander("PCA components V1–V28", expanded=True):
                for first_index in range(1, 29, 4):
                    columns = st.columns(4)
                    for column, index in zip(columns, range(first_index, first_index + 4)):
                        column.number_input(
                            f"V{index}", step=0.01, format="%.6f", key=f"feature_V{index}"
                        )
            submitted = st.form_submit_button("Predict transaction", type="primary", width="stretch")
        if submitted:
            frame = pd.DataFrame([{name: st.session_state[f"feature_{name}"] for name in FEATURES}])
            try:
                render_result(predict_transactions(frame, bundle), bundle)
            except PredictionError as exc:
                st.error(str(exc))

    with batch_tab:
        st.subheader("Read a batch")
        st.markdown('<p class="section-note">Upload a CSV with Time, V1–V28, and Amount. An optional Class column is ignored.</p>', unsafe_allow_html=True)
        download_col, example_col = st.columns(2)
        download_col.download_button(
            "Get a blank CSV template", pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"),
            file_name="transaction_template.csv", mime="text/csv", key="download_template"
        )
        if examples is not None:
            example_col.download_button(
                "Get the real example batch", examples.to_csv(index=False).encode("utf-8"),
                file_name="example_transactions.csv", mime="text/csv", key="download_examples"
            )
            if st.button("Score the example batch", key="score_examples"):
                st.session_state["example_batch_result"] = predict_transactions(examples, bundle)
            if "example_batch_result" in st.session_state:
                example_result = st.session_state["example_batch_result"]
                render_batch_summary(example_result)
                st.dataframe(example_result, hide_index=True, width="stretch")
        uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"], key="transaction_csv")
        if uploaded is not None:
            try:
                uploaded.seek(0)
                frame = read_transaction_csv(uploaded)
                with st.spinner(f"Scoring {len(frame):,} transactions…"):
                    result = predict_transactions(frame, bundle)
                render_batch_summary(result)
                st.dataframe(result, hide_index=True, width="stretch")
                st.download_button(
                    "Download batch predictions", result.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv", mime="text/csv", key="download_batch"
                )
                st.caption("The saved validation threshold sets labels. Fraud probability is an uncalibrated model score.")
            except PredictionError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
