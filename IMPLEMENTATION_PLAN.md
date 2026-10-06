# Implementation plan

The plan was presented before implementation. Milestones are checked before proceeding.

| Milestone | Implementation | Acceptance check |
|---|---|---|
| 1. Setup and data | Isolated Python 3.12 environment, pinned libraries, real ULB downloader, schema validation, manual fallback | Original 284,807 rows / 492 frauds; loader and duplicate tests |
| 2. Preprocessing and EDA | Audit, exact deduplication, 60/20/20 stratified split, train-only imputation, RobustScaler, mutual-information selection, descriptive plots | Disjoint partitions, natural prevalence, fitted scaler uses training values only, finite transformed features |
| 3. Baselines | Logistic Regression, Decision Tree, Random Forest × no resampling / SMOTE | Six fitted pipelines, valid probability predictions on natural-prevalence validation data |
| 4. Tuning and evaluation | Tune all six configurations with bounded RandomizedSearchCV; freeze best model and threshold using validation; final test comparison | Fold-local scaling/selection/SMOTE, AP scoring, saved parameters, all required metrics and curves |
| 5. Notebook and artifacts | Eight ordered sections, plot insights, comparisons and explanation, joblib artifacts, README/viva sheet | Notebook executes, saved model/scaler/features agree, reports reflect actual runs |
| 6. Streamlit | Single and batch predictions, template and example data, result downloads | UI interaction tests, CSV upload path, inference equivalence, HTTP server health |

## Design decisions

- Dataset: original ULB / Worldline transactions on Kaggle, with the public TensorFlow tutorial mirror for automatic download. Count and SHA256 checks record provenance.
- Numeric schema: `Time`, `V1`–`V28`, `Amount`, binary target `Class`; no raw personal/card identifiers.
- Deduplication: exact full-row duplicates removed before splitting without learning statistics; conflicting labels for identical predictors rejected. Preserve original indices for partition audits.
- Split: 60% training, 20% validation, 20% test, stratified with seed 42. Descriptive full-data counts do not guide feature or model choices; EDA for design uses training only.
- Preprocessing: training-fold median imputation; RobustScaler on Time and Amount, StandardScaler on PCA components to make SMOTE distances comparable. No blind outlier removal.
- Feature selection: top 20 of 30 by continuous mutual information on a stratified training-only sample up to 60,000 rows. The count is fixed in advance for a transparent classroom experiment. Selection is re-fit inside every CV training fold. Save the full ranking and compare train-only Random Forest importance. Recognize that independent ranking can miss interactions.
- Imbalance: SMOTE to a fraud/legitimate ratio of 0.10, retaining all real training examples and limiting synthetic growth. Preserve real prevalence in all validation/test sets. Compare each algorithm with and without SMOTE at threshold 0.5.
- Tuning: 3 candidate combinations × 3 stratified folds for each of six configurations on a natural-prevalence sample of up to 60,000 training rows. Refit each tuned configuration on all training data. Report this sampling limitation and CV standard deviation. Increase the budget via CLI if desired.
- Selection: highest validation average precision; tie break by F1, then experiment name. Select final threshold by maximum validation F1. This illustrates a tradeoff; actual bank costs/capacity would define a production threshold.
- Test: evaluate frozen configurations for a classroom comparison. Do not change selection from test results; only the preselected model's test row is the final deployment estimate.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, AP, trapezoidal PR-AUC, and TN/FP/FN/TP. Distinguish AP from interpolated area. Include a constant legitimate baseline to expose misleading accuracy.
- Deployment: save a complete imbalanced-learn pipeline plus its fitted preprocessing/scaler/feature list for inspection. The app loads them and uses the complete pipeline on raw feature inputs, avoiding double scaling. SMOTE is skipped during inference.
- Probability: expose `predict_proba` as an uncalibrated model score; SMOTE and severe imbalance can make it differ from real fraud risk. Do not describe it as a certified financial decision.
- Libraries: pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib, Streamlit, Jupyter, requests, pytest.

## Files

`src/` contains reusable modules for config, loading, preprocessing, evaluation, training,
reporting, and prediction. `scripts/` contains download/train/notebook utilities.
`notebooks/credit_card_fraud_detection.ipynb` is the presentation and end-to-end runner.
`models/`, `figures/`, `results/`, and `data/` contain saved deliverables and provenance.
`app.py`, `tests/`, `requirements.txt`, and `README.md` support demonstration and reproducibility.
