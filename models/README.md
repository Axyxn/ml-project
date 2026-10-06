# Saved artifacts

Created by `python scripts/train.py` using the real dataset:

- `best_model.joblib`: full fitted imbalanced-learn pipeline accepting all 30 raw numeric columns.
- `scaler.joblib`: fitted ColumnTransformer containing RobustScaler (Time/Amount) and StandardScaler (PCA).
- `preprocessor.joblib`: fitted imputation, scaling, and selection, for inspection/reuse.
- `selected_features.joblib`: ordered selected feature names, after scaling.
- `metadata.json`: chosen configuration, validation selection, operating threshold, dataset SHA256,
  versions, feature schema, and held-out test metrics.
- `baseline_checkpoint.joblib`: six baseline fits for resuming the identical data/config experiment.

The classifier consumes scaled selected columns, so prediction must use the complete pipeline.
Do not manually scale raw inputs before passing them to `best_model.joblib`. The separate
scaler artifact documents the learned transformation; it is already part of the pipeline.
SMOTE runs during fit only. Scores are not independently calibrated.

Binary artifacts are kept locally and ignored by Git. To regenerate them on another machine,
install the pinned requirements, obtain the real CSV, and run the training script.
