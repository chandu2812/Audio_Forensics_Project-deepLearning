"""Evaluation metrics including EER."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).

    Args:
        y_true: Binary labels (0 = real, 1 = fake).
        y_scores: Probability of the positive (fake) class.

    Returns:
        EER as a percentage.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # EER is where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    return float(eer * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict:
    """
    Compute all requested metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_probs: Probability of the positive (fake) class.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "eer": compute_eer(y_true, y_probs),
    }