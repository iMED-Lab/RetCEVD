import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels, predictions, scores):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores)
    auc = float("nan") if np.unique(labels).size < 2 else roc_auc_score(labels, scores)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall": recall_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "kappa": cohen_kappa_score(labels, predictions),
        "auc": auc,
    }
