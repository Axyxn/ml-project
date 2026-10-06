"""Generate a readable notebook from version-controlled section text and code."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))
    md("""
    # Credit Card Fraud Detection
    A complete college ML mini-project using the real ULB / Worldline transactions.
    Six baseline and six tuned configurations compare Logistic Regression, Decision Tree,
    and Random Forest with and without SMOTE. All learned transformations stay in training.

    ## 1. Problem Definition
    Credit-card fraud causes financial loss and harms trust. Our objective is to identify
    suspicious transactions for review from their numerical characteristics. The positive
    class is fraud. A false negative misses fraud; a false positive flags a genuine customer
    and creates investigation work. The goal is useful fraud detection with manageable alerts.

    Fraud is extremely rare. A classifier predicting legitimate for every record achieves
    about 99.83% accuracy on the original dataset but catches no fraud. We therefore use
    **average precision (AP)** for tuning and model selection, and also report recall,
    precision, F1, ROC-AUC, trapezoidal PR-AUC, accuracy, and confusion counts.

    **Protocol:** exact deduplication → stratified 60/20/20 split → training-fold preprocessing,
    selection and SMOTE → bounded CV tuning → validation model/threshold selection → frozen
    test evaluation. A random split suits this classroom comparison; future-time evaluation
    is needed for deployment. Every random seed is 42.
    """)
    code('''
    # Locate the repository whether Jupyter starts here or in notebooks/.
    from pathlib import Path
    import sys, json
    ROOT = Path.cwd().resolve()
    if not (ROOT / "src").is_dir():
        ROOT = ROOT.parent
    assert (ROOT / "src").is_dir(), "Open this notebook from the project or notebooks directory."
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import joblib
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown, Image
    from src.config import FEATURES, TrainingConfig
    from src.data import load_dataset, audit_and_clean, split_dataset
    from src.training import make_pipeline, run_training
    from src.predict import load_artifacts, predict_transactions
    from src.evaluation import compute_metrics
    config = TrainingConfig()
    RUN_TRAINING = False  # True repeats the complete experiment; expect several minutes.
    pd.set_option("display.max_columns", 35)
    pd.set_option("display.width", 160)
    def table(path):
        # Preserve the string "None" used for no resampling.
        return pd.read_csv(ROOT / path, keep_default_na=False)
    def figure(filename, insight):
        display(Image(filename=str(ROOT / "figures" / filename), width=1000))
        display(Markdown("**Insight:** " + insight))
    print("Project directory:", ROOT.name, "| seed:", config.seed)
    ''')
    md("""
    ## 2. Dataset
    Source: [Kaggle ULB / Worldline](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    The automatic downloader uses the real CSV mirror linked in
    [TensorFlow's official tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
    Original data: **284,807 transactions and 492 frauds**, observed over two days in
    September 2013 from European cardholders. The data has 30 numeric predictors and one target.

    | Attribute | Description |
    |---|---|
    | Time | Seconds since the first transaction |
    | V1–V28 | Anonymized PCA components; business meanings and original transform unavailable |
    | Amount | Transaction amount |
    | Class | 0 legitimate, 1 fraud |

    If the file is missing, run `python scripts/download_data.py`, or extract Kaggle's
    `creditcard.csv` into `data/`. The loader validates column names, finite numeric data,
    nonnegative Time/Amount, and binary labels. No fabricated data is substituted.
    """)
    code('''
    raw = load_dataset(ROOT / "data/creditcard.csv")
    display(raw.head())
    distribution = raw.Class.value_counts().sort_index().rename_axis("Class").to_frame("records")
    distribution["percent"] = 100 * distribution.records / len(raw)
    display(distribution)
    print(f"Raw rows: {len(raw):,}; frauds: {raw.Class.sum():,}; features: {len(FEATURES)}")
    provenance = ROOT / "data/creditcard.source.json"
    if provenance.exists():
        display(Markdown("**Download provenance:** " + str(json.loads(provenance.read_text()))))
    ''')
    md("""
    ## 3. Data Preprocessing
    Inspect missing values and exact duplicates. The original CSV has no missing values;
    a **training-fold median imputer** handles future compatible datasets with partial
    missingness. Missing targets are rejected. All predictors are numeric, so categorical
    encoding is unnecessary.

    Remove exact duplicate full rows before splitting, without estimating any statistics,
    so copies cannot appear in training and test sets. This removes 1,081 rows, including
    19 repeated fraud rows; the remaining dataset has 283,726 records and 473 frauds.
    Identical inputs with conflicting labels require correction rather than random splitting.

    Split into 60% train, 20% validation, and 20% test using stratification. Fit **RobustScaler**
    (median/IQR) on Time and Amount, and **StandardScaler** on V1–V28 so differing component
    variances do not dominate SMOTE distances. Fit these inside each training pipeline.
    **Keep outliers:** extreme transactions may be legitimate or strong fraud signals;
    blind IQR deletion/clipping could remove what we want to detect.
    """)
    code('''
    clean, audit = audit_and_clean(raw)
    splits = split_dataset(clean, config)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    display(pd.DataFrame({"missing": raw.isna().sum()}))
    print("Exact duplicates removed:", audit["duplicates_removed"])
    display(pd.DataFrame([{ "partition": name, "records": len(X), "frauds": int(y.sum()),
                           "fraud_percent": 100*y.mean() } for name, (X, y) in splits.items()]))
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)
    assert "Class" not in X_train
    # This template is unfitted; GridSearch/RandomizedSearch clones it for each fold.
    template = make_pipeline("Logistic Regression", use_smote=True, config=config)
    display(Markdown("**Pipeline order:** " + " → ".join(name for name, _ in template.steps)))
    display(Markdown(f"**SMOTE target ratio:** {config.smote_ratio:.2f} fraud/legitimate; "
                     "resampling runs only during training."))
    ''')
    md("""
    ## 4. Exploratory Data Analysis
    Overall counts above describe the source. The following design-oriented plots and
    correlations use **training data only**. All saved plots have written insights underneath.
    The test partition is held back from design and selection.
    """)
    code('''
    # On a fresh checkout this prepares EDA and trains all required experiments.
    required_outputs = [ROOT / "results/model_comparison.csv", ROOT / "models/metadata.json",
                        ROOT / "results/eda_insights.json", ROOT / "models/best_model.joblib"]
    if RUN_TRAINING or not all(path.exists() for path in required_outputs):
        run_training(ROOT / "data/creditcard.csv", config=config, root=ROOT)
    eda_insights = json.loads((ROOT / "results/eda_insights.json").read_text())
    ''')
    for name in ["class_imbalance", "amount_time_distributions", "correlation_heatmap", "pca_distributions", "time_amount_scatter"]:
        code(f'figure("{name}.png", eda_insights["{name}"])')
    md("""
    ## 5. Model Development
    **Feature selection:** SelectKBest chooses the top 20 of 30 by continuous mutual
    information with Class. Unlike Pearson correlation, MI can capture nonlinear dependence.
    The feature count is fixed in advance for an explicit, manageable experiment. Rankings
    use a stratified sample of up to 60,000 training rows and are refitted inside each CV fold.
    The transformed full training set is then used to fit the classifier. This reduces
    weak predictors and runtime; it is not a claim that 20 is optimal.

    Training-only Random Forest importance provides a second descriptive view. It does not
    change selection after evaluation. Both individual MI and impurity importance have
    limitations: they can miss interactions or favor continuous predictors. A feature-count
    ablation is future work.
    """)
    code('''
    display(table("results/feature_selection.csv"))
    feature_explanation = (ROOT / "results/feature_selection_explanation.md").read_text(encoding="utf-8")
    figure("feature_selection.png", feature_explanation)
    importance = table("results/feature_importance.csv")
    display(importance)
    figure("feature_importance.png", "Forest importance is learned only from training examples. "
           "High importance suggests useful splits, not a causal business explanation. This secondary "
           "view does not replace the predeclared MI selection rule.")
    ''')
    md("""
    | Algorithm | Why include it? | Main limitation |
    |---|---|---|
    | Logistic Regression | Fast, regularized linear baseline with probability output | May underfit nonlinear interactions |
    | Decision Tree | Nonlinear splits, simple rule structure | High variance; can overfit rare frauds |
    | Random Forest | Averages randomized trees to reduce variance | Higher compute and less direct interpretation |

    Each algorithm is fitted **without resampling** and **with SMOTE**, yielding six
    baseline configurations. SMOTE interpolates between nearby minority training points.
    We use a fraud/legitimate ratio of 0.10, which limits synthetic growth and keeps all
    real training records. Validation and test data stay naturally imbalanced.

    Tune all six configurations using **RandomizedSearchCV**, 3 candidate combinations,
    3 stratified folds, and AP scoring. Search at most 60,000 natural-prevalence training
    rows for reasonable runtime, then refit each winning setting on all training records.
    The complete pipeline includes imputation → scaling → selection → SMOTE → classifier,
    so CV never fits these steps on held-out folds. SMOTE is skipped during prediction.
    """)
    code('''
    display(table("results/baseline_validation.csv")[["algorithm", "imbalance", "fit_seconds",
                                                   "precision", "recall", "f1", "average_precision"]])
    display(table("results/cv_fold_summary.csv"))
    tuning = table("results/tuning_summary.csv")
    display(tuning[["algorithm", "imbalance", "search_rows", "search_frauds", "candidates", "folds",
                    "cv_average_precision", "cv_ap_std", "best_params"]])
    display(table("results/before_after_tuning.csv"))
    ''')
    md("""
    **Reading tuning results:** better CV AP need not improve one held-out validation set
    or the F1 at threshold 0.5. Few frauds make rankings noisy; inspect CV standard deviations.
    The bounded sample/search is a runtime choice and a limitation. Baselines remain eligible
    if the tuned setting performs worse on the independent validation partition.

    ## 6. Model Evaluation
    Accuracy, precision, recall, and F1 use a classification threshold. ROC-AUC and PR
    metrics use probability scores across thresholds. TN/FP/FN/TP connect scores to
    genuine customers flagged and frauds missed.

    **AP versus PR-AUC:** AP weights precision by increments in recall; trapezoidal PR-AUC
    interpolates PR points. We report both and tune/select on AP. The constant legitimate
    reference has no-skill AP equal to prevalence; its interpolated PR area can look
    misleadingly high because of endpoints. See [the AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
    """)
    code('''
    comparisons = table("results/model_comparison.csv")
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    metric_columns = ["experiment", "partition", "threshold", "accuracy", "precision", "recall",
                      "f1", "roc_auc", "average_precision", "pr_auc", "tn", "fp", "fn", "tp"]
    display(comparisons[comparisons.partition == "test"][metric_columns])
    ''')
    code('''
    figure("test_confusion_matrices.png", "All twelve frozen configurations use threshold 0.5 "
           "for this comparison. Examine false negatives for missed fraud and false positives "
           "for review workload; total accuracy alone hides these differences.")
    ''')
    code('''
    figure("test_roc_curves.png", "ROC-AUC compares ranking across false-positive and true-positive rates. "
           "A small false-positive rate can still mean many false alerts when legitimate records dominate. "
           "These test curves describe frozen choices and were not used to select the deployed model.")
    ''')
    code('''
    figure("test_pr_curves.png", "Precision–recall curves show alert reliability versus fraud coverage. "
           "The horizontal reference equals test fraud prevalence. AP is the primary ranking metric "
           "because it reflects precision when fraud is rare.")
    ''')
    code('''
    m = metadata["final_test_metrics"]
    figure("final_threshold_and_confusion.png", f"The left plot uses validation only to select threshold "
           f"{metadata['threshold']:.6f}; the right applies it to test records. It catches {m['tp']} frauds, "
           f"misses {m['fn']}, and raises {m['fp']} false alerts. F1 gains on validation need not transfer "
           "to test; the threshold is retained because selection was already frozen.")
    ''')
    md("""
    ## 7. Model Comparison
    The table includes **all models × all required metrics**, with/without SMOTE and
    before/after tuning, plus the dummy and selected operating threshold. Validation guides
    selection; test estimates frozen choices. AP is threshold-independent for a fixed model,
    so changing its decision threshold changes precision/recall/F1 but leaves AP and ROC-AUC fixed.
    """)
    code('''
    display(comparisons[["algorithm", "imbalance", "stage", "partition", "threshold", "accuracy",
                         "precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc",
                         "tn", "fp", "fn", "tp"]])
    display(Markdown((ROOT / "results/findings.md").read_text(encoding="utf-8")))
    ''')
    md("""
    **Business tradeoff:** false negatives risk lost funds and trust; false positives cost
    review effort and inconvenience customers. Maximum validation F1 is a classroom operating
    choice. Real systems should use bank-specific losses and review capacity, without
    inventing costs for this anonymized dataset. Scores are not independently calibrated;
    SMOTE changes training priors and may especially distort probability interpretation.

    ## 8. Deployment / Demonstration
    The full pipeline is saved with joblib, along with its fitted scaler, preprocessor,
    selected feature list, threshold, dataset SHA256, and version metadata. The Streamlit
    app loads these artifacts and accepts single transactions or CSV batches.

    Run from the project directory:

    ```powershell
    python -m streamlit run app.py
    ```

    Open `http://localhost:8501`, load a real example or enter all 30 raw numeric values,
    and submit a prediction. For batch scoring, upload a CSV with Time,V1,…,V28,Amount;
    Class is optional and ignored for prediction. Download the scored result. Missing or
    invalid values receive helpful errors. Ordinary card details cannot be mapped to
    V1–V28 without the unavailable original PCA transform.
    """)
    code('''
    bundle = load_artifacts(ROOT / "models")  # Also loads and checks the saved scaler/features.
    examples = table("results/example_transactions.csv")
    single = predict_transactions(examples.iloc[[0]], bundle)
    batch = predict_transactions(examples, bundle)
    display(single[["Class", "fraud_probability", "prediction", "decision_threshold"]])
    display(batch[["Time", "Amount", "Class", "fraud_probability", "prediction", "decision_threshold"]])
    np.testing.assert_allclose(single.fraud_probability.iloc[0], batch.fraud_probability.iloc[0])
    print("Saved inference pipeline:", metadata["experiment"])
    print("Selected features:", bundle.selected_features)
    print("Scaler:", bundle.scaler)
    print("SMOTE is skipped for inference; raw inputs are scaled exactly once.")
    ''')
    md("""
    **Conclusion:** this complete workflow demonstrates severe imbalance, leakage prevention,
    feature selection, meaningful model comparisons, tuning, threshold selection, and a
    working prediction path. The measured findings above give the actual final result.

    **Limitations:** anonymized PCA, unavailable transform, a two-day dataset from 2013,
    limited held-out frauds, random rather than temporal evaluation, deduplication differences
    from raw-data benchmarks, possible SMOTE artifacts, MI interaction loss, a small tuning
    budget, and uncalibrated scores.

    **Future work:** temporal/group validation, feature-count ablation, class weights and
    balanced forests, larger nested CV, independent calibration, cost-based thresholds,
    drift monitoring, and human review. See the README's viva Q&A for presentation preparation.
    """)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    destination = ROOT / "notebooks/credit_card_fraud_detection.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(nb, destination)
    print(f"Built {destination} with {len(cells)} cells and eight ordered sections.")


if __name__ == "__main__":
    build()
