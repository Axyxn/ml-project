# Credit Card Fraud Detection

A college ML mini-project using the real ULB / Worldline dataset. Logistic Regression,
Decision Tree, and Random Forest are compared with and without SMOTE, before and after
tuning. The notebook follows the eight required components in order.

## Setup and run

Use Python **3.12** and run commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
python -m streamlit run app.py
```

If activation is blocked, use `.\.venv\Scripts\python.exe` in place of `python`.
On macOS/Linux, activate with `source .venv/bin/activate`. Streamlit serves at
`http://localhost:8501`. Pinned libraries and seed 42 support reproducibility.
See [the implementation plan](IMPLEMENTATION_PLAN.md).

## Dataset

[ULB / Worldline on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions including 492 frauds (about 0.173%), recorded over two days in
September 2013 from European cardholders.

| Attribute | Description |
|---|---|
| `Time` | Seconds since the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1`–`V28` | Anonymized PCA components; original meanings and transform unavailable |
| `Class` | Target: 0 legitimate, 1 fraud |

Automatic download uses the real CSV mirror linked in [TensorFlow's official tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
It checks record/fraud counts and records a SHA256. If unavailable, download the Kaggle
ZIP, extract `creditcard.csv`, and place it in **`data/creditcard.csv`**. Missing files
produce helpful instructions. See [data/README.md](data/README.md).

## Notebook

```powershell
python -m notebook notebooks/credit_card_fraud_detection.ipynb
```

Choose the project environment as the kernel. Register it if needed:

```powershell
python -m ipykernel install --prefix .venv --name credit-fraud --display-name "Credit Fraud (.venv)"
```

The notebook uses saved results when available and trains on a fresh checkout. Set
`RUN_TRAINING = True` in the setup cell to repeat the workflow. Its final section loads
the saved model and performs real inference.

## Method and leakage prevention

1. Audit missing values; remove exact duplicates before splitting to keep copies from
   crossing partitions. Reject identical predictors with conflicting labels.
2. Stratify into **60% training / 20% validation / 20% test**, seed 42. EDA used for design
   examines training data. Keep natural fraud prevalence in validation and test sets.
3. Fit median imputation, RobustScaler on Time/Amount, and StandardScaler on PCA
   components inside each training pipeline. Component variances differ, so scaling also
   makes SMOTE distances comparable. No encoding is needed. Keep outliers as possible fraud signals.
4. Select the **top 20** features by continuous mutual information, fixed in advance.
   Estimate ranks on up to 60,000 stratified training rows; recompute inside every CV
   training fold. Save full rankings and training-only Random Forest importance.
5. Compare all three models with and without SMOTE to a minority/majority ratio of **0.10**.
   This limits synthetic growth and preserves all original training records. Sampling is
   inside the pipeline, after scaling and selection, and skipped at prediction time.
6. Tune **all six configurations** using 3 candidate combinations × 3 stratified folds,
   scored on **average precision**, on up to 60,000 natural-prevalence training rows.
   Refit tuned models on the full training partition. This small budget keeps runtime
   manageable but gives noisy estimates when folds contain few frauds. To expand it:

   ```powershell
   python scripts/train.py --tuning-iterations 6 --tuning-max-rows 120000 --n-jobs 2
   ```

7. Choose the highest validation AP (tie: F1, then name); select its threshold by maximum
   validation F1. Freeze both before test evaluation. Keep the original training-only fit.
8. Evaluate all frozen configurations on test data for comparison. Deployment selection
   comes from validation even if test rankings differ. Do not revise choices from test results.

## Evaluation

| Metric | Interpretation |
|---|---|
| Accuracy | Overall fraction correct; dominated by legitimate records |
| Precision | Fraction of fraud alerts that are actual fraud |
| Recall | Fraction of actual fraud detected |
| F1 | Harmonic mean of precision and recall at a threshold |
| Confusion matrix | TN, FP, FN, TP: review workload and missed fraud |
| ROC-AUC | Ranking across true-positive and false-positive rates |
| Average precision (AP) | Recall-weighted precision; tuning and selection metric |
| PR-AUC (trapezoidal) | Interpolated PR curve area, distinct from AP |

Always predicting legitimate yields about **99.83% accuracy and zero fraud recall**.
PR curves expose alert precision under severe imbalance; the no-skill AP equals prevalence.
ROC-AUC can look strong despite many false alerts. [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Logistic Regression is a regularized linear model: fast, low variance, limited for
nonlinear interactions. A Decision Tree learns nonlinear rules but can overfit rare
frauds. Random Forest averages randomized trees to reduce variance, with higher cost.
SMOTE can raise recall and lower precision; it can also introduce unrealistic points.
The generated discussion uses actual measured changes rather than assuming it helps.

False negatives mean missed fraud and potential financial loss. False positives create
review workload and customer friction. Validation F1 is a classroom threshold choice;
a bank would set it using costs and review capacity. `predict_proba` is an **uncalibrated
model score**, especially with SMOTE, and not a verified real-world risk estimate.

## Results

`results/model_comparison.csv` includes six baseline and six tuned configurations,
the selected operating threshold, and an always-legitimate reference, with separate
validation/test rows. `results/tuning_summary.csv` records best parameters and CV variation.
`results/findings.md` explains results and the final model choice.

<!-- RESULTS_START -->
Selected **Random Forest | None | baseline** by validation AP **0.8659**. Operating threshold: **0.259462**, selected on validation.

| Test metric | Value |
|---|---|
| Accuracy | 0.9994 |
| Precision | 0.8795 |
| Recall | 0.7684 |
| F1 | 0.8202 |
| Roc Auc | 0.9538 |
| Average Precision | 0.7792 |
| Pr Auc | 0.7791 |

Test confusion counts: TN=56,641, FP=10, FN=22, TP=73. Deduplicated dataset: 283,726 records / 473 frauds. These metrics describe this frozen split. See [all comparisons](results/model_comparison.csv), [best parameters](results/tuning_summary.csv), and [measured discussion](results/findings.md).
<!-- RESULTS_END -->

## Demonstration and artifacts

The app accepts 30 numeric predictors for single entry or CSV batch upload, shows labels
and probabilities, and downloads scored CSVs. It offers a schema template and real
example transactions. `Class` is optional and ignored for inference. Invalid, missing,
infinite, or negative Time/Amount inputs receive readable errors. Ordinary card details
cannot replace the anonymized V1–V28 inputs.

`models/best_model.joblib` is the complete inference pipeline; `scaler.joblib`,
`preprocessor.joblib`, `selected_features.joblib`, and `metadata.json` support inspection
and consistency checks. The app loads the saved scaler but passes raw values through
the complete pipeline once, avoiding double scaling. Restart after replacing artifacts.

| Path | Purpose |
|---|---|
| `notebooks/credit_card_fraud_detection.ipynb` | Eight sections, plots, tables, explanations |
| `src/` | Modular data, preprocessing, training, evaluation, reporting, prediction |
| `scripts/` | Download, training, notebook execution, server verification |
| `figures/` | EDA, feature rankings, confusion matrices, ROC/PR, threshold analysis |
| `results/` | Tables, audit, CV parameters, findings, example transactions |
| `tests/` | Leakage, metrics, input, saved artifacts, Streamlit checks |

## Screenshots

Analysis figures appear in the executed notebook. Save presentation screenshots in
`figures/screenshots/`: run the app, load an example, capture its single-result panel
and batch table. No mock image is presented as a running application.

## Verification

```powershell
python -m pytest -q
python scripts/execute_notebook.py
python scripts/verify_server.py
```

Synthetic fixtures are isolated from reported experiments. Integration checks use real
ULB transactions and saved artifacts. UI checks exercise single inference and CSV upload;
the server check requests local HTTP health, then shuts its own process down.

The final run executed all 28 notebook cells with zero errors. Streamlit's health and
page endpoints returned HTTP 200; AppTest exercised manual input, a real fraud example,
the batch example button, injected CSV upload bytes, and an invalid CSV. The upload test
uses bytes at the uploader boundary because Streamlit 1.50 AppTest lacks an upload setter.
See [milestone checks](results/milestone_checks.md) and [test results](results/test_results.xml).

## Conclusion, limitations, and future work

This workflow compares linear and nonlinear models under severe imbalance, with learned
transformations and resampling confined to training. Generated findings describe which
changes actually helped and the consequences of the selected threshold.

PCA anonymization limits interpretation and integration. Two days from 2013 cannot
represent modern markets or changing fraud patterns (concept drift). Random splitting
is weaker than future-time evaluation; card/customer identifiers for group splitting
are unavailable. Deduplication affects comparison with published raw-data benchmarks.
Few held-out frauds mean uncertainty; MI can miss feature interactions, SMOTE can create
unrealistic values, and the bounded search can miss better settings.

Future work: temporal evaluation, class-weight and balanced-forest comparisons, larger
nested CV, feature-count ablation, independent probability calibration, cost-based
thresholds, drift monitoring, and human review. This is an educational demonstration.

## Viva Q&A cheat sheet

| Question | Short answer |
|---|---|
| Why not accuracy? | The huge legitimate class hides missed fraud; always-legitimate gets 99.83% accuracy. |
| What is SMOTE? | Interpolation between nearby minority training points to create synthetic fraud examples. |
| Where does SMOTE run? | Only in training folds, after scaling/selection; never on validation/test data. |
| Why recall? | It measures the share of fraud caught; low recall means more missed fraud. |
| Why precision? | It measures alert reliability and investigation workload. |
| Why split before scaling/selection? | Held-out statistics or labels would leak information into training. |
| Why preserve outliers? | Extreme values may be real fraud signals. |
| Why RobustScaler? | Median/IQR scaling reduces extreme values' influence without deleting them. |
| Why select features? | Reduce weak predictors and complexity; independent ranking can miss interactions. |
| Tree versus forest? | A forest averages randomized trees and usually reduces variance. |
| AP versus PR-AUC? | AP weights precision by recall increments; trapezoidal area interpolates points. |
| Why tune on AP? | It assesses fraud ranking across thresholds with precision under imbalance. |
| False negative? | A real fraud classified as legitimate. |
| Why adjust a threshold? | Lowering it usually catches more fraud but increases false alarms. |
| Why avoid test-set selection? | Test performance should estimate frozen choices rather than guide them. |
| Are probabilities calibrated? | No; SMOTE changes priors and calibration needs independent data. |
| Can I enter ordinary card details? | No; PCA inputs need the original transform or compatible existing features. |
| Concept drift? | Fraud and transaction patterns change over time, weakening older models. |

References: [leakage guide](https://imbalanced-learn.org/stable/common_pitfalls.html),
[SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html),
[mutual information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html).
