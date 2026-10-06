# Measured findings

## Model and threshold selection

Selected **Random Forest | None | baseline** using validation AP 0.8659. The frozen threshold is **0.259462**, selected by validation F1. No test labels were used to select either. All configurations were fixed before test evaluation.

## Effect of SMOTE (validation, threshold 0.5)

- Logistic Regression, baseline: AP 0.7972 → 0.8009; precision 0.899 → 0.418; recall 0.660 → 0.894; false alerts 7 → 117. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.
- Decision Tree, baseline: AP 0.7185 → 0.5992; precision 0.796 → 0.237; recall 0.787 → 0.819; false alerts 19 → 248. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.
- Random Forest, baseline: AP 0.8659 → 0.8326; precision 0.907 → 0.842; recall 0.830 → 0.851; false alerts 8 → 15. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.
- Logistic Regression, tuned: AP 0.7949 → 0.8021; precision 0.913 → 0.442; recall 0.670 → 0.894; false alerts 6 → 106. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.
- Decision Tree, tuned: AP 0.8342 → 0.5509; precision 0.871 → 0.333; recall 0.787 → 0.840; false alerts 11 → 158. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.
- Random Forest, tuned: AP 0.8579 → 0.8399; precision 0.916 → 0.871; recall 0.809 → 0.862; false alerts 7 → 12. Synthetic interpolation changes the learned boundary; it can increase overlap as well as improve minority coverage.

## Effect of tuning (validation)

- Logistic Regression, None: AP 0.7972 → 0.7949; F1 0.7607 → 0.7730. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.
- Logistic Regression, SMOTE: AP 0.8009 → 0.8021; F1 0.5695 → 0.5915. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.
- Decision Tree, None: AP 0.7185 → 0.8342; F1 0.7914 → 0.8268. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.
- Decision Tree, SMOTE: AP 0.5992 → 0.5509; F1 0.3675 → 0.4773. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.
- Random Forest, None: AP 0.8659 → 0.8579; F1 0.8667 → 0.8588. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.
- Random Forest, SMOTE: AP 0.8326 → 0.8399; F1 0.8466 → 0.8663. CV optimizes AP on a smaller training sample, so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.

## Bias, variance, and overfitting

Logistic Regression uses a regularized linear boundary and may underfit nonlinear interactions. A tree can represent nonlinear splits but has high variance with few fraud examples. A forest averages many randomized trees to reduce variance. The following gaps are clues to overfitting rather than proofs; training and validation fraud samples differ.

- Logistic Regression | None | baseline: training AP 0.7395, validation AP 0.7972, gap -0.0577.
- Logistic Regression | SMOTE | baseline: training AP 0.7419, validation AP 0.8009, gap -0.0590.
- Decision Tree | None | baseline: training AP 0.9068, validation AP 0.7185, gap +0.1883.
- Decision Tree | SMOTE | baseline: training AP 0.8750, validation AP 0.5992, gap +0.2758.
- Random Forest | None | baseline: training AP 0.9758, validation AP 0.8659, gap +0.1099.
- Random Forest | SMOTE | baseline: training AP 0.9958, validation AP 0.8326, gap +0.1631.
- Logistic Regression | None | tuned: training AP 0.7426, validation AP 0.7949, gap -0.0523.
- Logistic Regression | SMOTE | tuned: training AP 0.7439, validation AP 0.8021, gap -0.0582.
- Decision Tree | None | tuned: training AP 0.9142, validation AP 0.8342, gap +0.0800.
- Decision Tree | SMOTE | tuned: training AP 0.9731, validation AP 0.5509, gap +0.4223.
- Random Forest | None | tuned: training AP 0.9187, validation AP 0.8579, gap +0.0608.
- Random Forest | SMOTE | tuned: training AP 0.9994, validation AP 0.8399, gap +0.1595.

## Final held-out result

On 56,746 untouched test records, the selected operating threshold finds 73 of 95 frauds, misses 22, and raises 10 false alerts. Precision 0.8795; recall 0.7684; F1 0.8202; ROC-AUC 0.9538; AP 0.7792; trapezoidal PR-AUC 0.7791.

At threshold 0.5, this same model has 3 false alerts and 26 missed frauds. Test F1 changes from 0.8263 to 0.8202 at the validation-selected threshold. A validation gain does not guarantee a test gain, so we keep the already frozen threshold. A lower threshold generally exchanges review workload for fraud coverage. The optimal business choice requires bank-specific losses and capacity; no dollar costs are invented.

AP and trapezoidal PR-AUC use different interpolation rules. Constant scores have a no-skill AP equal to prevalence, while their trapezoidal area can be misleadingly high because of the endpoint interpolation. This is why AP is the primary ranking metric.

## Limitations and future work

Anonymized PCA features and an unavailable transform prevent direct use of ordinary card details. Two days from 2013 are a limited benchmark; modern deployment requires concept-drift monitoring. Random stratification can be optimistic compared with a temporal split. There are few held-out frauds, deduplication changes benchmark counts, MI can miss interactions, and the small search is noisy. SMOTE probability scores are not independently calibrated. Future work includes temporal evaluation, feature-count ablation, class-weight comparisons, larger/nested CV, calibration, and thresholds based on costs and review capacity.
