"""Fraud metrics, operating threshold, and comparison figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from src.reporting import save_figure


def compute_metrics(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if y.shape != scores.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require aligned binary labels containing both classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities between zero and one.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, scores)
    return {
        "threshold": float(threshold), "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "pr_auc": auc(recall, precision),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "rows": len(y), "fraud_rate": float(y.mean()),
    }


def select_threshold(y_validation, probabilities):
    """Maximize validation F1, choosing the highest threshold on exact ties."""
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    best = np.flatnonzero(f1 == np.max(f1))[-1]
    table = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1],
                          "recall": recall[:-1], "f1": f1})
    return float(thresholds[best]), table


def plot_evaluation(y, predictions, root, prefix="test"):
    """Predictions maps frozen experiment names to score arrays; do not tune here."""
    root = Path(root)
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(9, 6))
    for name, scores in predictions.items():
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, scores):.3f})")
        ax_pr.plot(recall, precision, label=f"{name} (AP {average_precision_score(y, scores):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="No skill")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{prefix.title()} ROC curves")
    ax_pr.axhline(np.mean(y), color="black", linestyle="--", label=f"Prevalence {np.mean(y):.4f}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"{prefix.title()} precision–recall curves", ylim=(0, 1.03))
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_pr.legend(fontsize=7, loc="lower left")
    save_figure(fig_roc, root / f"figures/{prefix}_roc_curves.png")
    save_figure(fig_pr, root / f"figures/{prefix}_pr_curves.png")
    ncols = 3
    nrows = int(np.ceil(len(predictions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.7*nrows), squeeze=False)
    for ax, (name, scores) in zip(axes.flat, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y, scores >= 0.5, labels=[0, 1], display_labels=["Legit", "Fraud"],
            values_format="d", cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name, fontsize=9)
    for ax in list(axes.flat)[len(predictions):]:
        ax.set_visible(False)
    save_figure(fig, root / f"figures/{prefix}_confusion_matrices.png")


def plot_final_threshold(y, scores, threshold, threshold_table, root):
    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    view = threshold_table[threshold_table.threshold.between(0.001, 0.999)]
    for metric in ["precision", "recall", "f1"]:
        axes[0].plot(view.threshold, view[metric], label=metric)
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Selected {threshold:.4f}")
    axes[0].set(xlabel="Threshold", ylabel="Validation metric", title="Threshold selected on validation only")
    axes[0].legend()
    ConfusionMatrixDisplay.from_predictions(
        y, scores >= threshold, labels=[0, 1], display_labels=["Legit", "Fraud"],
        values_format="d", cmap="Blues", colorbar=False, ax=axes[1])
    axes[1].set_title("Final test confusion matrix at frozen threshold")
    save_figure(fig, root / "figures/final_threshold_and_confusion.png")
