"""Plotting utilities for evaluation results."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils.paths import GRAPHS_DIR, CONFUSION_DIR, ROC_DIR


def plot_confusion_matrix(y_true, y_pred, model_name: str = "model"):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Synthetic"],
        yticklabels=["Real", "Synthetic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    out = CONFUSION_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved confusion matrix to {out}")


def plot_roc_curve(y_true, y_probs, model_name: str = "model"):
    """Save a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = ROC_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved ROC curve to {out}")