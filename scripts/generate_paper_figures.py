#!/usr/bin/env python3
"""
Generate publication-quality figures for the Audio Deepfake Detection paper.

Produces:
  1. fig1_dataset_bias_v1_vs_v2.png: The Dataset Bias Phenomenon (V1 Shortcut vs V2 Realistic Benchmark)
  2. fig2_fewshot_adaptation_curve.png: Accuracy and EER across N in {0, 10, 50} with standard deviation
  3. fig3_backbone_unfreezing_comparison.png: Frozen Backbone vs Unfrozen Top-2 Layers vs Top-4 Layers
  4. fig4_generator_stratification.png: Balanced 5-Generator and Multi-Source Bonafide Dataset Architecture
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_dataset_bias():
    """Fig 1: The Dataset Bias Phenomenon (V1 Shortcut vs V2 Realistic Benchmark)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    categories = ["Zero-Shot\n(N=0)", "Few-Shot\n(N=10)", "Few-Shot\n(N=50)"]
    x = np.arange(len(categories))
    width = 0.35

    # EER values
    v1_eer = [10.60, 1.97, 0.40]
    v2_eer = [28.90, 26.88, 19.70]

    rects1 = ax.bar(x - width/2, v1_eer, width, label="V1 (Biased, Single-Source)", color="#e74c3c", alpha=0.88, edgecolor="#c0392b", linewidth=1.2)
    rects2 = ax.bar(x + width/2, v2_eer, width, label="V2 (Balanced, 5 Generators)", color="#2980b9", alpha=0.88, edgecolor="#1c5980", linewidth=1.2)

    ax.set_ylabel("Equal Error Rate (EER %)", fontweight="bold")
    ax.set_title("Cross-Lingual Deepfake Detection: The Dataset Bias Phenomenon", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="upper left")

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c0392b")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1c5980")

    # Add explanatory annotation
    ax.annotate("V1 exploited generator\nshortcuts (artificially low EER)",
                xy=(1.82, 0.6), xytext=(1.05, 8.5),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                fontsize=9.5, color="#c0392b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", lw=1))

    ax.annotate("V2 tests true cross-generator\ngeneralization across 5 TTS models",
                xy=(2.18, 20.0), xytext=(0.85, 30.5),
                arrowprops=dict(arrowstyle="->", color="#1c5980", lw=1.5),
                fontsize=9.5, color="#1c5980", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#2980b9", lw=1))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_dataset_bias_v1_vs_v2.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig2_adaptation_curve():
    """Fig 2: Few-Shot Adaptation Curve on Balanced Benchmark."""
    fig, ax1 = plt.subplots(figsize=(7, 4.2))

    n_samples = [0, 10, 50]
    eer_mean = [28.90, 26.88, 19.70]
    eer_std = [0.0, 1.83, 0.57]

    acc_mean = [60.00, 65.83, 79.67]
    acc_std = [0.0, 5.87, 0.47]

    color_eer = "#d35400"
    color_acc = "#27ae60"

    ax1.errorbar(n_samples, eer_mean, yerr=eer_std, fmt="-o", color=color_eer,
                 ecolor=color_eer, elinewidth=2, capsize=5, capthick=1.5,
                 linewidth=2.5, markersize=8, label="EER (Lower is better)")
    ax1.set_xlabel("Number of Few-Shot Samples per Class ($N$)", fontweight="bold")
    ax1.set_ylabel("Equal Error Rate (EER %)", color=color_eer, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_eer)
    ax1.set_ylim(15, 35)

    for x, y, std in zip(n_samples, eer_mean, eer_std):
        lbl = f"{y:.1f}%" if std == 0 else f"{y:.1f}±{std:.1f}%"
        ax1.annotate(lbl, xy=(x, y), xytext=(-5, 10), textcoords="offset points",
                     fontsize=9.5, fontweight="bold", color=color_eer)

    ax2 = ax1.twinx()
    ax2.errorbar(n_samples, acc_mean, yerr=acc_std, fmt="-s", color=color_acc,
                 ecolor=color_acc, elinewidth=2, capsize=5, capthick=1.5,
                 linewidth=2.5, markersize=8, label="Accuracy (Higher is better)")
    ax2.set_ylabel("Accuracy (%)", color=color_acc, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(50, 90)
    ax2.grid(False)

    for x, y, std in zip(n_samples, acc_mean, acc_std):
        lbl = f"{y:.1f}%" if std == 0 else f"{y:.1f}±{std:.1f}%"
        ax2.annotate(lbl, xy=(x, y), xytext=(-10, -18), textcoords="offset points",
                     fontsize=9.5, fontweight="bold", color=color_acc)

    plt.title("Few-Shot SLS Adaptation on Frozen XLS-R 300M (Hindi)", fontweight="bold", pad=12)
    ax1.set_xticks(n_samples)
    ax1.set_xticklabels(["0 (Zero-Shot)", "10", "50"], fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig2_fewshot_adaptation_curve.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig3_unfreezing():
    """Fig 3: Frozen Backbone vs Unfrozen Top-2 Layers vs Unfrozen Top-4 Layers (N=50)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    regimes = ["Frozen Backbone\n(SLS Head Only)", "Unfreeze Top-2\nLayers (Top 8%)", "Unfreeze Top-4\nLayers (Top 17%)"]
    x = np.arange(len(regimes))
    width = 0.35

    # Metrics at N=50
    eer_vals = [19.70, 24.14, 25.31]
    acc_vals = [79.67, 76.46, 67.12]

    rects1 = ax.bar(x - width/2, eer_vals, width, label="EER % (Lower is better)", color="#e67e22", alpha=0.9, edgecolor="#d35400", linewidth=1.2)
    rects2 = ax.bar(x + width/2, acc_vals, width, label="Accuracy % (Higher is better)", color="#27ae60", alpha=0.9, edgecolor="#1e8449", linewidth=1.2)

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Few-Shot Adaptation ($N=50$): Frozen Backbone vs Unfreezing", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#bdc3c7")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d35400")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1e8449")

    # Add insight box
    ax.annotate("Catastrophic Overfitting:\nUnfreezing backbone worsens EER by +5.6%",
                xy=(2, 25.5), xytext=(1.05, 42),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                fontsize=9.5, color="#c0392b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", lw=1))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig3_backbone_unfreezing_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig4_generator_stratification():
    """Fig 4: Architecture of the Stratified Balanced Dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    # Pie 1: Class split
    labels_class = ["Bonafide (Real)", "Spoof (Synthetic)"]
    sizes_class = [200, 200]
    colors_class = ["#2ecc71", "#e74c3c"]
    explode_class = (0.04, 0.04)

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes_class, explode=explode_class, labels=labels_class, colors=colors_class,
        autopct="%1.0f%%", startangle=90, textprops=dict(fontweight="bold", fontsize=10.5)
    )
    for at in autotexts1:
        at.set_color("white")
        at.set_fontsize(12)
    ax1.set_title("50/50 Binary Class Balance\n(Total: 400 Test Samples)", fontweight="bold", pad=10)

    # Pie 2: Spoof generator stratification
    labels_spoof = ["Edge-TTS\n(40 clips)", "VITS-MMS\n(40 clips)", "Indic-TTS\n(40 clips)", "FastSpeech\n(40 clips)", "XTTS-v2\n(40 clips)"]
    sizes_spoof = [40, 40, 40, 40, 40]
    colors_spoof = ["#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    explode_spoof = (0.02, 0.02, 0.02, 0.02, 0.02)

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes_spoof, explode=explode_spoof, labels=labels_spoof, colors=colors_spoof,
        autopct="%1.0f%%", startangle=90, textprops=dict(fontsize=9.5)
    )
    for at in autotexts2:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Stratified Spoof Generators\n(20% per TTS Architecture)", fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_generator_stratification.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating publication figures...")
    plot_fig1_dataset_bias()
    plot_fig2_adaptation_curve()
    plot_fig3_unfreezing()
    plot_fig4_generator_stratification()
    print("All publication figures successfully created in outputs/figures/!")


if __name__ == "__main__":
    main()
