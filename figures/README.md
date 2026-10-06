# Generated figures

Training-only analysis:

- `class_imbalance.png`
- `amount_time_distributions.png`
- `correlation_heatmap.png`
- `pca_distributions.png`
- `time_amount_scatter.png`
- `feature_selection.png`
- `feature_importance.png`

Frozen model evaluations:

- `baseline_validation_*`: six initial configurations.
- `validation_*` and `test_*`: ROC, precision–recall, and confusion matrices for all twelve configurations.
- `final_threshold_and_confusion.png`: validation threshold analysis and final test confusion counts.

Confusion matrices for comparisons use 0.5; the final matrix uses the frozen validation
threshold. The notebook places written insights beside each figure. EDA insights are
also in `results/eda_insights.json`. Each image is saved at 150 DPI for presentations.
