"""
Evaluation and Metrics Computation for Eye State Classifier.
Computes real classification metrics, confusion matrices, and training curves.
"""

from typing import Dict, Any, Optional
import os
import json
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics from real true labels and predicted probabilities.
    
    Args:
        y_true: Array of binary ground truth labels (0 = CLOSED, 1 = OPEN)
        y_pred_prob: Array of predicted probabilities for class 1 (OPEN)
        threshold: Decision threshold
        
    Returns:
        Dictionary of real measured metrics
    """
    # Binary classification ground truth and predictions
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

    total = max(1, len(y_true))
    acc = float((tp + tn) / total)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Mann-Whitney U test for ROC-AUC
    pos = y_pred_prob[y_true_bin == 1]
    neg = y_pred_prob[y_true_bin == 0]
    if len(pos) > 0 and len(neg) > 0:
        auc = float(np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos]))
    else:
        auc = None

    cm = [[tn, fp], [fn, tp]]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "num_samples": len(y_true),
        "threshold": threshold,
        "status": "Measured"
    }


def plot_and_save_confusion_matrix(cm: list, output_path: str = "models/confusion_matrix.png"):
    """Renders and saves a confusion matrix visualization."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(x=j, y=i, s=f"{cm_arr[i, j]}", va='center', ha='center', size='large', weight='bold')

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['CLOSED', 'OPEN'])
    ax.set_yticklabels(['CLOSED', 'OPEN'])
    ax.set_xlabel('Predicted Label', labelpad=10, weight='bold')
    ax.set_ylabel('True Label', labelpad=10, weight='bold')
    ax.set_title('DriverGuard AI - Eye State Confusion Matrix', pad=15, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_and_save_training_curves(history: Dict[str, list], output_path: str = "models/training_curves.png"):
    """Plots training and validation loss/accuracy curves."""
    import matplotlib.pyplot as plt

    if not history or "loss" not in history:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    # Loss Curve
    ax1.plot(epochs, history["loss"], label="Train Loss", color="#EF4444", lw=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F59E0B", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", weight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Crossentropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy Curve
    if "accuracy" in history:
        ax2.plot(epochs, history["accuracy"], label="Train Acc", color="#10B981", lw=2)
        if "val_accuracy" in history:
            ax2.plot(epochs, history["val_accuracy"], label="Val Acc", color="#3B82F6", lw=2, linestyle="--")
        ax2.set_title("Training & Validation Accuracy", weight="bold")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_model_metrics(metadata_path: str = "models/model_metadata.json") -> Dict[str, Any]:
    """
    Loads real measured metrics from disk.
    If metrics have not yet been measured, returns explicit unmeasured placeholders.
    """
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("evaluation", _get_unmeasured_metrics())
        except Exception:
            pass

    return _get_unmeasured_metrics()


def _get_unmeasured_metrics() -> Dict[str, Any]:
    """Returns explicit 'Not measured yet' dictionary."""
    return {
        "accuracy": "Not measured yet",
        "precision": "Not measured yet",
        "recall": "Not measured yet",
        "f1_score": "Not measured yet",
        "roc_auc": "Not measured yet",
        "confusion_matrix": None,
        "num_samples": "Not measured yet",
        "status": "Not measured yet"
    }
