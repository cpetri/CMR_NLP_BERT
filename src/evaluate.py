"""
Evaluation utilities: classification reports, ROC curves, calibration curves,
training history plots, and prediction probability visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list) -> None:
    """Print per-class precision, recall, and F1-score."""
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def plot_training_history(history: dict, output_path: str | None = None) -> None:
    """
    Plot F1-score and loss curves for training and validation.

    Args:
        history: dict with keys "train_f1", "val_f1", "train_loss", "val_loss".
        output_path: if provided, save the figure to this path instead of showing it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_f1"]) + 1)

    axes[0].plot(epochs, history["train_f1"], label="Train F1")
    axes[0].plot(epochs, history["val_f1"], label="Val F1")
    axes[0].set_title("F1 Score (micro)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss")
    axes[1].set_title("BCE Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, label_names: list, output_path: str | None = None) -> None:
    """
    Plot one ROC curve per diagnostic class.

    Args:
        y_true: (n_samples, n_classes) ground-truth binary array.
        y_prob: (n_samples, n_classes) predicted probability array.
        label_names: list of class names.
        output_path: if provided, save the figure here.
    """
    n_classes = len(label_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_title(name)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")

    plt.suptitle("ROC Curves per Diagnostic Class", y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_calibration_curves(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: list, output_path: str | None = None
) -> None:
    """
    Plot reliability diagrams (calibration curves) for each diagnostic class.

    Args:
        y_true: (n_samples, n_classes) ground-truth binary array.
        y_prob: (n_samples, n_classes) predicted probability array.
        label_names: list of class names.
        output_path: if provided, save the figure here.
    """
    n = len(label_names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(label_names):
        prob_true, prob_pred = calibration_curve(y_true[:, i], y_prob[:, i], n_bins=10)
        axes[i].plot(prob_pred, prob_true, marker="o", label="Model")
        axes[i].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
        axes[i].set_title(name)
        axes[i].set_xlabel("Mean predicted probability")
        axes[i].set_ylabel("Fraction of positives")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Calibration Curves", y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_prediction_probabilities(
    probs: np.ndarray, label_names: list, sample_idx: int = 0, output_path: str | None = None
) -> None:
    """
    Bar plot of predicted probabilities for a single sample.

    Args:
        probs: (n_samples, n_classes) probability array.
        label_names: list of class names.
        sample_idx: index of the sample to visualise.
        output_path: if provided, save the figure here.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=label_names, y=probs[sample_idx], ax=ax)
    ax.axhline(0.5, linestyle="--", color="red", label="Decision threshold (0.5)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Predicted probabilities — sample {sample_idx}")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
