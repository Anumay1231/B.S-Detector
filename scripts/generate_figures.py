#!/usr/bin/env python3
"""
Generate figures for Audio Deepfake Detection
using XLS-R 300M + SLS architecture on Hindi Speech.

Visualizations produced:
1. `fewshot_adaptation_curve.png`- EER and Accuracy reduction curves across sample regimes.
2. `confusion_matrices.png`      - Zero-shot vs Few-shot Error Distribution / CM Heatmaps.
3. `tts_model_distribution.png`  - 5-Model Stratification Breakdown.

Outputs saved to `outputs/figures/`.
"""

import os
import json
from pathlib import Path

# Ensure writable matplotlib cache in sandboxed/restricted environments
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent.parent / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional scientific styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrices():
    """Figure: Confusion Matrix Heatmaps."""
    cm_zeroshot = np.array([[48, 152], [1, 199]])
    cm_n500_seed2 = np.array([[194, 6], [2, 198]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    classes = ["Spoof (0)", "Bona fide (1)"]

    sns.heatmap(cm_zeroshot, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[0], annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Zero-Shot Checkpoint\n(Accuracy: 61.75% | EER: 32.25%)", fontweight="bold")
    axes[0].set_ylabel("True Class", fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontweight="bold")

    sns.heatmap(cm_n500_seed2, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[1], annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title("Few-Shot N=500 (Seed 2)\n(Accuracy: 98.00% | EER: 2.12%)", fontweight="bold")
    axes[1].set_ylabel("True Class", fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontweight="bold")

    plt.suptitle("Confusion Matrix Heatmaps on Held-Out Hindi Test Set", fontweight="bold", y=1.03)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_adaptation_curve():
    """Few-Shot Adaptation Trajectory."""
    n_samples = [0, 10, 50, 500]
    eer_means = [32.25, 27.64, 18.18, 2.66]
    eer_stds = [0.0, 2.27, 1.18, 0.52]
    acc_means = [61.75, 64.33, 75.83, 96.83]
    acc_stds = [0.0, 5.04, 10.49, 0.82]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    color_eer = "#C0392B"
    color_acc = "#27AE60"

    x_indices = np.arange(len(n_samples))

    line1 = ax1.errorbar(x_indices, eer_means, yerr=eer_stds, fmt="-o", color=color_eer,
                         linewidth=2.5, markersize=8, capsize=5, label="Equal Error Rate (% - Lower is Better)")
    ax1.set_xlabel("Supervision Regime (Samples per Class N)", fontweight="bold")
    ax1.set_ylabel("Equal Error Rate (%)", color=color_eer, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_eer)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(["0 (Zero-Shot)", "10", "50", "500"])
    ax1.set_ylim(0, 36)

    ax2 = ax1.twinx()
    line2 = ax2.errorbar(x_indices, acc_means, yerr=acc_stds, fmt="--s", color=color_acc,
                         linewidth=2.5, markersize=8, capsize=5, label="Accuracy (% - Higher is Better)")
    ax2.set_ylabel("Accuracy (%)", color=color_acc, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(55, 102)
    ax2.grid(False)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, facecolor="white")

    plt.title("Few-Shot Adaptation Scaling Curve (Frozen XLS-R 300M)", fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fewshot_adaptation_curve.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_stratification():
    """Spoof Generator Stratification (Equal 20% per TTS)."""
    models = {"IndicTTS": 20, "Edge-TTS": 20, "XTTS-v2": 20, "VITS-MMS": 20, "FastSpeech": 20}
    colors = ["#3498DB", "#9B59B6", "#E74C3C", "#2ECC71", "#F1C40F"]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, _, autotexts = ax.pie(models.values(), labels=models.keys(), autopct="%1.0f%%",
                                  startangle=90, colors=colors, wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Stratified 5-Model Generator Distribution\n(40 Clips Each in Test Set)", fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "tts_model_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print(f"Generating figures in: {OUTPUT_DIR}\n")
    plot_confusion_matrices()
    plot_adaptation_curve()
    plot_dataset_stratification()
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
