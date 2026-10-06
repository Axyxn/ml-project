# Experiment outputs

Run `python scripts/train.py` to reproduce results. Counts refer to deduplicated data.

| File | Contents |
|---|---|
| `data_audit.json` | Raw/clean counts, missing values, duplicate and outlier decisions |
| `split_summary.csv`, `split_indices.npz` | Natural-prevalence partition counts and reproducible row indices |
| `training_correlations.csv`, `training_class_statistics.csv` | Training-only descriptive analysis |
| `eda_insights.json` | Written insights for all EDA figures |
| `feature_selection.csv`, `feature_importance.csv` | All MI ranks/selection flags and forest importance |
| `feature_selection_explanation.md` | Training-only selection rule and exact retained columns |
| `baseline_validation.csv` | Six untuned configurations at threshold 0.5 |
| `cv_fold_summary.csv` | Real training/validation counts for each tuning fold |
| `tuning_candidates.csv`, `tuning_summary.csv` | Candidate scores, best parameters, CV variation, timing |
| `before_after_tuning.csv` | Paired validation metrics for tuning comparisons |
| `validation_comparison.csv` | All twelve validation configurations at 0.5 |
| `frozen_selection.json` | Model/threshold selection recorded before test inference |
| `threshold_analysis.csv` | Precision, recall, F1 across validation thresholds |
| `model_comparison.csv` | All twelve configurations on validation/test, final threshold, dummy reference |
| `findings.md` | Measured SMOTE/tuning effects, bias/variance discussion, final result, limitations |
| `example_transactions.csv` | Three legitimate and three fraudulent real test transactions for the app |

Use `pd.read_csv(path, keep_default_na=False)` for comparison tables so the label `None`
(no imbalance intervention) is preserved as text. AP is average precision and `pr_auc`
is trapezoidal curve area. They are not interchangeable. No-skill AP equals prevalence;
constant-score trapezoidal area may look high because of endpoint interpolation.

Deployment was chosen using validation results; this choice is never revised using
the test comparison. A new design informed by test results needs a fresh holdout.
