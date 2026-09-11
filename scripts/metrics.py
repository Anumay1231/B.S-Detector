"""
metrics.py
----------

Metrics and evaluation utilities for audio deepfake detection.

This module provides standard evaluation metrics for audio deepfake classification,
centered around the Equal Error Rate (EER), accuracy, precision, recall, F1 score,
and confusion matrices, along with reporting and comparison utilities.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

__all__ = [
    "compute_eer",
    "compute_all_metrics",
    "print_metrics",
    "print_comparison_table",
]


def compute_eer(
    bonafide_scores: Sequence[float] | np.ndarray,
    spoof_scores: Sequence[float] | np.ndarray,
) -> float:
    """Compute Equal Error Rate (EER) from bonafide and spoof score arrays.

    bonafide_scores are scores for genuine samples (higher = more bonafide).
    spoof_scores are scores for spoofed samples.
    EER is the operating point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR) (i.e. FPR == 1 - TPR).

    Parameters
    ----------
    bonafide_scores : Sequence[float] | np.ndarray
        Scores assigned to genuine/bonafide samples.
    spoof_scores : Sequence[float] | np.ndarray
        Scores assigned to spoofed samples.

    Returns
    -------
    float
        Equal Error Rate (EER) as a float in the range [0.0, 1.0].

    Raises
    ------
    ValueError
        If either bonafide_scores or spoof_scores is empty.
    """
    bonafide = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spoof = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if len(bonafide) == 0 or len(spoof) == 0:
        raise ValueError("Both bonafide_scores and spoof_scores must be non-empty.")

    # Ground truth labels: 1 = bonafide, 0 = spoof
    y_true = np.concatenate(
        [np.ones(len(bonafide), dtype=np.int32), np.zeros(len(spoof), dtype=np.int32)]
    )
    y_scores = np.concatenate([bonafide, spoof])

    # Compute ROC curve using sklearn
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)

    # Handle duplicate FPR values to guarantee strictly increasing x coordinates
    # for scipy.interpolate.interp1d. For duplicate FPRs, keep the last occurrence
    # which has the highest TPR (the upper envelope of the ROC curve).
    mask = np.concatenate([fpr[:-1] != fpr[1:], [True]])
    fpr = fpr[mask]
    tpr = tpr[mask]

    # Ensure endpoints cover [0.0, 1.0]
    if fpr[0] > 0.0:
        fpr = np.concatenate([[0.0], fpr])
        tpr = np.concatenate([[0.0], tpr])
    if fpr[-1] < 1.0:
        fpr = np.concatenate([fpr, [1.0]])
        tpr = np.concatenate([tpr, [1.0]])

    # Build interpolator for TPR as a function of FPR
    f_interp = interp1d(fpr, tpr, kind="linear")

    # Define f(x) = (1 - x) - tpr(x) where x is FPR and (1 - tpr) is FNR.
    # EER occurs where FPR == FNR, i.e. 1.0 - x - tpr(x) == 0.
    def f(x: float) -> float:
        return float(1.0 - x - f_interp(x))

    # Fast boundary checks for perfect or inverse separation
    if abs(f(0.0)) < 1e-12 or f(0.0) <= 0.0:
        return 0.0
    if abs(f(1.0)) < 1e-12 or f(1.0) >= 0.0:
        return 1.0

    try:
        eer = float(brentq(f, 0.0, 1.0))
    except ValueError:
        # Fallback if brentq fails to bracket: find discrete point minimizing |FPR - (1 - TPR)|
        diff = np.abs(fpr - (1.0 - tpr))
        eer = float(fpr[np.nanargmin(diff)])

    return float(np.clip(eer, 0.0, 1.0))


def compute_all_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute all evaluation metrics for audio deepfake detection.

    Parameters
    ----------
    labels : Sequence[int] | np.ndarray
        Ground truth labels (1 = bonafide, 0 = spoof).
    scores : Sequence[float] | np.ndarray
        Bonafide scores (log-probabilities, probabilities, or logits;
        higher indicates more bonafide).
    predictions : Sequence[int] | np.ndarray
        Predicted class labels (1 = bonafide, 0 = spoof).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'accuracy': float
        - 'eer': float
        - 'precision': float
        - 'recall': float
        - 'f1': float
        - 'confusion_matrix': np.ndarray of shape (2, 2)

    Raises
    ------
    ValueError
        If labels, scores, and predictions do not have identical length.
    """
    labels_arr = np.asarray(labels, dtype=np.int32).ravel()
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    preds_arr = np.asarray(predictions, dtype=np.int32).ravel()

    if not (len(labels_arr) == len(scores_arr) == len(preds_arr)):
        raise ValueError(
            f"Input arrays must have identical length: "
            f"labels={len(labels_arr)}, scores={len(scores_arr)}, predictions={len(preds_arr)}."
        )

    accuracy = float(accuracy_score(labels_arr, preds_arr))
    precision = float(
        precision_score(
            labels_arr,
            preds_arr,
            average="binary",
            pos_label=1,
            zero_division=0.0,
        )
    )
    recall = float(
        recall_score(
            labels_arr,
            preds_arr,
            average="binary",
            pos_label=1,
            zero_division=0.0,
        )
    )
    f1 = float(
        f1_score(
            labels_arr,
            preds_arr,
            average="binary",
            pos_label=1,
            zero_division=0.0,
        )
    )

    bonafide_scores = scores_arr[labels_arr == 1]
    spoof_scores = scores_arr[labels_arr == 0]
    eer = compute_eer(bonafide_scores, spoof_scores)

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    return {
        "accuracy": accuracy,
        "eer": eer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def print_metrics(metrics_dict: Mapping[str, Any], title: str = "") -> None:
    """Pretty-print the metrics dictionary.

    Displays scalar performance metrics formatted as percentages to 2 decimal places,
    and displays the 2x2 confusion matrix formatted with class labels.

    Parameters
    ----------
    metrics_dict : Mapping[str, Any]
        Dictionary containing metric values (typically returned by `compute_all_metrics`).
    title : str, optional
        Optional title displayed above the metrics.
    """
    if title:
        header_text = f"=== {title} ==="
        print(header_text)

    def _fmt_pct(key: str) -> str:
        if key not in metrics_dict or metrics_dict[key] is None:
            return "N/A"
        val = float(metrics_dict[key])
        pct = val * 100.0 if abs(val) <= 1.0 else val
        return f"{pct:.2f}%"

    print(f"Accuracy:         {_fmt_pct('accuracy')}")
    print(f"EER:              {_fmt_pct('eer')}")
    print(f"Precision:        {_fmt_pct('precision')}")
    print(f"Recall:           {_fmt_pct('recall')}")
    print(f"F1:               {_fmt_pct('f1')}")

    # Extra scalar metrics if present
    standard_keys = {"accuracy", "eer", "precision", "recall", "f1", "confusion_matrix"}
    for k, v in metrics_dict.items():
        if k not in standard_keys:
            if isinstance(v, (int, float)):
                pct = v * 100.0 if abs(v) <= 1.0 else v
                print(f"{k.capitalize()}: {pct:.2f}%")
            else:
                print(f"{k.capitalize()}: {v}")

    # Confusion matrix formatting with labels
    if "confusion_matrix" in metrics_dict and metrics_dict["confusion_matrix"] is not None:
        cm = np.asarray(metrics_dict["confusion_matrix"])
        if cm.shape == (2, 2):
            print("\nConfusion Matrix:")
            print("                     Pred Spoof (0)    Pred Bonafide (1)")
            print(f"  True Spoof (0)     {int(cm[0, 0]):>14d}    {int(cm[0, 1]):>17d}")
            print(f"  True Bonafide (1)  {int(cm[1, 0]):>14d}    {int(cm[1, 1]):>17d}")
        else:
            print(f"\nConfusion Matrix:\n{cm}")


def print_comparison_table(
    results_dict: Mapping[str, Sequence[Mapping[str, Any]] | Mapping[str, Any]],
) -> None:
    """Print a formatted comparison table of experimental results.

    Parameters
    ----------
    results_dict : Mapping[str, Sequence[Mapping[str, Any]] | Mapping[str, Any]]
        Mapping of experiment name to a list of metrics dicts (or a single metrics dict).
        - For single-entry lists (zero-shot), displays the value.
        - For multi-entry lists (few-shot seeds), displays mean +/- std.
        Columns: Experiment | Accuracy | EER | Precision | Recall | F1
    """
    headers = ["Experiment", "Accuracy", "EER", "Precision", "Recall", "F1"]
    metric_keys = ["accuracy", "eer", "precision", "recall", "f1"]

    rows: List[List[str]] = []

    for exp_name, metrics in results_dict.items():
        if isinstance(metrics, (list, tuple)):
            metrics_list = list(metrics)
        else:
            metrics_list = [metrics]

        if not metrics_list:
            continue

        row: List[str] = [str(exp_name)]

        if len(metrics_list) == 1:
            single_m = metrics_list[0]
            for k in metric_keys:
                if k in single_m and single_m[k] is not None:
                    val = float(single_m[k])
                    pct = val * 100.0 if abs(val) <= 1.0 else val
                    row.append(f"{pct:.2f}%")
                else:
                    row.append("N/A")
        else:
            for k in metric_keys:
                vals = [float(m[k]) for m in metrics_list if k in m and m[k] is not None]
                if vals:
                    vals_pct = [v * 100.0 if abs(v) <= 1.0 else v for v in vals]
                    mean_val = float(np.mean(vals_pct))
                    std_val = float(np.std(vals_pct))
                    row.append(f"{mean_val:.2f} +/- {std_val:.2f}%")
                else:
                    row.append("N/A")

        rows.append(row)

    if not rows:
        print("No results to display.")
        return

    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)

    print(header_str)
    print(sep_str)
    for row in rows:
        row_str = " | ".join(val.ljust(w) for val, w in zip(row, col_widths))
        print(row_str)
