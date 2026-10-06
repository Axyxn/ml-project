"""Descriptive figures and evidence-based text for a student presentation."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Scripts work without a display server; notebook embeds saved PNGs.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import FEATURES, ROOT, TrainingConfig, make_output_dirs
from src.data import audit_and_clean, load_dataset, split_dataset, stratified_sample
from src.preprocessing import make_preprocessing_steps

COLORS = {0: "#2563eb", 1: "#e54b4b"}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def create_eda(X_train, y_train, root=ROOT):
    """Design-oriented EDA uses the training partition, preserving the test holdout."""
    root = Path(root)
    train = X_train.assign(Class=y_train)
    sns.set_theme(style="whitegrid", context="notebook")
    insights = {}
    counts = y_train.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Legitimate", "Fraud"], counts, color=list(COLORS.values()))
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Training class distribution at natural prevalence")
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=3)
    save_figure(fig, root / "figures/class_imbalance.png")
    insights["class_imbalance"] = (
        f"Training has {counts[0]:,} legitimate and {counts[1]:,} fraudulent transactions "
        f"({100*y_train.mean():.3f}% fraud). The log axis makes the small fraud class visible. "
        "Accuracy is dominated by legitimate records; never balance the evaluation partitions.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    amount = train.assign(log_amount=np.log1p(train.Amount))
    sns.histplot(amount, x="log_amount", hue="Class", stat="density", common_norm=False,
                 bins=45, element="step", fill=False, palette=COLORS, ax=axes[0])
    axes[0].set_xlabel("log(1 + Amount), for display only")
    axes[0].set_title("Amount distributions by class")
    sns.histplot(train.assign(hours=train.Time / 3600), x="hours", hue="Class", stat="density",
                 common_norm=False, bins=48, element="step", fill=False, palette=COLORS, ax=axes[1])
    axes[1].set_xlabel("Hours since first transaction")
    axes[1].set_title("Time distributions by class")
    save_figure(fig, root / "figures/amount_time_distributions.png")
    medians = train.groupby("Class")["Amount"].median()
    insights["amount_time_distributions"] = (
        f"Median Amount is {medians[0]:.2f} for legitimate and {medians[1]:.2f} for fraud. "
        "Each class is normalized separately, so compare shapes rather than counts. "
        "Log amount is a visualization transform only. Time reflects this short observation window, "
        "not a reliable future fraud rule; neither feature alone perfectly separates classes.")

    corr = train.corr(numeric_only=True)
    corr.to_csv(root / "results/training_correlations.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Training-only Pearson correlation, including target")
    save_figure(fig, root / "figures/correlation_heatmap.png")
    strongest = corr.Class.drop("Class").abs().sort_values(ascending=False).head(4)
    insights["correlation_heatmap"] = (
        "Strongest absolute training target correlations: " +
        ", ".join(f"{name} ({score:.3f})" for name, score in strongest.items()) +
        ". Correlation detects linear associations; low correlation does not exclude nonlinear "
        "predictive value. PCA components are anonymized, so correlation cannot reveal business causes.")

    components = [c for c in strongest.index if c.startswith("V")][:2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col, ax in zip(components, axes):
        sns.boxplot(train, x="Class", y=col, hue="Class", palette=COLORS, legend=False,
                    showfliers=False, ax=ax)
        ax.set_title(f"{col} distribution by class (fliers hidden for display)")
    save_figure(fig, root / "figures/pca_distributions.png")
    insights["pca_distributions"] = (
        f"{', '.join(components)} were chosen for display from training target correlations. "
        "Different medians or spreads suggest predictive signal, but overlap creates errors. "
        "Hiding boxplot fliers improves readability; no rows were removed from training.")

    normal = train[train.Class == 0].sample(min(3000, int(counts[0])), random_state=42)
    sample = pd.concat([normal, train[train.Class == 1]])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(sample.assign(hours=sample.Time / 3600), x="hours", y="Amount", hue="Class",
                    palette=COLORS, alpha=0.6, s=18, ax=ax)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Time and amount: sampled legitimate + all training frauds")
    save_figure(fig, root / "figures/time_amount_scatter.png")
    insights["time_amount_scatter"] = (
        "This combines all training frauds with at most 3,000 legitimate records for readable plotting. "
        "It does not represent class prevalence. Fraud can occur at small or large amounts; "
        "a high-amount-only rule would miss some fraud. The symlog axis retains zero amounts.")
    write_json(root / "results/eda_insights.json", insights)
    train.groupby("Class")[FEATURES].agg(["mean", "median", "std"]).to_csv(
        root / "results/training_class_statistics.csv")
    return insights


def create_feature_report(X_train, y_train, config=None, root=ROOT):
    config = config or TrainingConfig()
    root = Path(root)
    prep = Pipeline(make_preprocessing_steps(config)).fit(X_train, y_train)
    selector = prep.named_steps["select"]
    names = selector.feature_names_in_
    ranking = pd.DataFrame({"feature": names, "mutual_information": selector.scores_,
                            "selected": selector.get_support()}).sort_values(
                                ["mutual_information", "feature"], ascending=[False, True])
    ranking.to_csv(root / "results/feature_selection.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(ranking.feature[::-1], ranking.mutual_information[::-1],
            color=["#2563eb" if keep else "#a1a1aa" for keep in ranking.selected[::-1]])
    ax.set_xlabel("Mutual information with Class (nats)")
    ax.set_title(f"Training feature ranking: blue = selected top {config.selected_k}")
    save_figure(fig, root / "figures/feature_selection.png")
    X_small, y_small = stratified_sample(X_train, y_train, config.selection_max_rows, config.seed)
    transformed = prep.transform(X_small)
    forest = RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=2,
                                    n_jobs=config.n_jobs, random_state=config.seed).fit(transformed, y_small)
    importance = pd.DataFrame({"feature": selector.selected_features_,
                               "importance": forest.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(root / "results/feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance.feature[::-1], importance.importance[::-1], color="#0d9488")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Training-only forest importance among selected features")
    save_figure(fig, root / "figures/feature_importance.png")
    explanation = (
        f"We keep the top {config.selected_k} of 30 features by mutual information with Class, "
        f"estimated on {selector.ranking_rows_:,} stratified training rows containing "
        f"{selector.ranking_frauds_} frauds. Kept: {', '.join(selector.selected_features_)}. "
        "The count was fixed before test evaluation to reduce weak predictors and runtime, not claimed "
        "to be optimal. Mutual information can capture nonlinear dependence. The forest importance "
        "is a secondary descriptive view, not a new test-driven selection rule. Individual rankings "
        "can miss interactions; impurity importance can favor continuous variables. Every CV fold "
        "relearns selection using its own training data, so its selected columns may differ.")
    (root / "results/feature_selection_explanation.md").write_text(explanation, encoding="utf-8")
    return ranking, prep


def prepare_project(data_path=ROOT / "data/creditcard.csv", config=None, root=ROOT):
    config = config or TrainingConfig()
    root = Path(root)
    make_output_dirs(root)
    raw = load_dataset(data_path)
    clean, audit = audit_and_clean(raw)
    with Path(data_path).open("rb") as handle:
        audit["data_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    splits = split_dataset(clean, config)
    summary = []
    for name, (X, y) in splits.items():
        summary.append({"partition": name, "rows": len(X), "legitimate": int((y == 0).sum()),
                        "frauds": int(y.sum()), "fraud_rate": float(y.mean())})
    pd.DataFrame(summary).to_csv(root / "results/split_summary.csv", index=False)
    np.savez_compressed(root / "results/split_indices.npz", **{k: X.index.to_numpy() for k, (X, _) in splits.items()})
    audit["config"] = config.to_dict()
    write_json(root / "results/data_audit.json", audit)
    X_train, y_train = splits["train"]
    create_eda(X_train, y_train, root)
    ranking, prep = create_feature_report(X_train, y_train, config, root)
    return splits, audit, ranking, prep


def write_findings(comparison, metadata, root=ROOT):
    """Use actual validation effects and test error counts, without changing selection."""
    root = Path(root)
    validation = comparison[(comparison.partition == "validation") & comparison.stage.isin(["baseline", "tuned"])].copy()
    # CSV reloads with keep_default_na=False preserve "None" but blank reference
    # values can make training diagnostics object columns. Normalize the filtered rows.
    for name in ["train_average_precision", "train_f1"]:
        validation[name] = pd.to_numeric(validation[name], errors="raise")
    lines = ["# Measured findings", "", "## Model and threshold selection", "",
             f"Selected **{metadata['experiment']}** using validation AP {metadata['validation_ap']:.4f}. "
             f"The frozen threshold is **{metadata['threshold']:.6f}**, selected by validation F1. "
             "No test labels were used to select either. All configurations were fixed before test evaluation.",
             "", "## Effect of SMOTE (validation, threshold 0.5)", ""]
    for stage in ["baseline", "tuned"]:
        for algorithm in validation.algorithm.unique():
            pair = validation[(validation.algorithm == algorithm) & (validation.stage == stage)].set_index("imbalance")
            a, b = pair.loc["None"], pair.loc["SMOTE"]
            lines.append(f"- {algorithm}, {stage}: AP {a.average_precision:.4f} → {b.average_precision:.4f}; "
                         f"precision {a.precision:.3f} → {b.precision:.3f}; recall {a.recall:.3f} → {b.recall:.3f}; "
                         f"false alerts {int(a.fp)} → {int(b.fp)}. Synthetic interpolation changes the learned "
                         "boundary; it can increase overlap as well as improve minority coverage.")
    lines += ["", "## Effect of tuning (validation)", ""]
    for algorithm in validation.algorithm.unique():
        for imbalance in ["None", "SMOTE"]:
            pair = validation[(validation.algorithm == algorithm) & (validation.imbalance == imbalance)].set_index("stage")
            a, b = pair.loc["baseline"], pair.loc["tuned"]
            lines.append(f"- {algorithm}, {imbalance}: AP {a.average_precision:.4f} → {b.average_precision:.4f}; "
                         f"F1 {a.f1:.4f} → {b.f1:.4f}. CV optimizes AP on a smaller training sample, "
                         "so tuning is not guaranteed to improve a particular validation split or its 0.5-threshold F1.")
    lines += ["", "## Bias, variance, and overfitting", "",
              "Logistic Regression uses a regularized linear boundary and may underfit nonlinear interactions. "
              "A tree can represent nonlinear splits but has high variance with few fraud examples. "
              "A forest averages many randomized trees to reduce variance. The following gaps are clues "
              "to overfitting rather than proofs; training and validation fraud samples differ.", ""]
    for _, row in validation.iterrows():
        lines.append(f"- {row.experiment}: training AP {row.train_average_precision:.4f}, "
                     f"validation AP {row.average_precision:.4f}, gap {row.train_average_precision-row.average_precision:+.4f}.")
    m = metadata["final_test_metrics"]
    base = metadata["default_threshold_test_metrics"]
    lines += ["", "## Final held-out result", "",
              f"On {m['rows']:,} untouched test records, the selected operating threshold finds "
              f"{m['tp']} of {m['tp']+m['fn']} frauds, misses {m['fn']}, and raises {m['fp']} false alerts. "
              f"Precision {m['precision']:.4f}; recall {m['recall']:.4f}; F1 {m['f1']:.4f}; "
              f"ROC-AUC {m['roc_auc']:.4f}; AP {m['average_precision']:.4f}; trapezoidal PR-AUC {m['pr_auc']:.4f}.", "",
              f"At threshold 0.5, this same model has {base['fp']} false alerts and {base['fn']} missed frauds. "
              f"Test F1 changes from {base['f1']:.4f} to {m['f1']:.4f} at the validation-selected threshold. "
              "A validation gain does not guarantee a test gain, so we keep the already frozen threshold. "
              "A lower threshold generally exchanges review workload for fraud coverage. The optimal business "
              "choice requires bank-specific losses and capacity; no dollar costs are invented.", "",
              "AP and trapezoidal PR-AUC use different interpolation rules. Constant scores have a "
              "no-skill AP equal to prevalence, while their trapezoidal area can be misleadingly high "
              "because of the endpoint interpolation. This is why AP is the primary ranking metric.", "",
              "## Limitations and future work", "",
              "Anonymized PCA features and an unavailable transform prevent direct use of ordinary card details. "
              "Two days from 2013 are a limited benchmark; modern deployment requires concept-drift monitoring. "
              "Random stratification can be optimistic compared with a temporal split. There are few held-out "
              "frauds, deduplication changes benchmark counts, MI can miss interactions, and the small search "
              "is noisy. SMOTE probability scores are not independently calibrated. Future work includes "
              "temporal evaluation, feature-count ablation, class-weight comparisons, larger/nested CV, "
              "calibration, and thresholds based on costs and review capacity."]
    (root / "results/findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = (f"Selected **{metadata['experiment']}** by validation AP **{metadata['validation_ap']:.4f}**. "
               f"Operating threshold: **{metadata['threshold']:.6f}**, selected on validation.\n\n"
               "| Test metric | Value |\n|---|---|\n" +
               "\n".join(f"| {name.replace('_', ' ').title()} | {m[name]:.4f} |" for name in
                         ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc"]) +
               f"\n\nTest confusion counts: TN={m['tn']:,}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}. "
               f"Deduplicated dataset: {metadata['data_audit']['clean_rows']:,} records / "
               f"{metadata['data_audit']['clean_frauds']} frauds. These metrics describe this frozen split. "
               "See [all comparisons](results/model_comparison.csv), [best parameters](results/tuning_summary.csv), "
               "and [measured discussion](results/findings.md).")
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in content and end in content:
            content = content.split(start)[0] + start + "\n" + summary + "\n" + end + content.split(end, 1)[1]
            readme.write_text(content, encoding="utf-8")
