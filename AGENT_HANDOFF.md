# Agent Handoff — B.S-Detector (Audio Deepfake Detection)

> **Last updated**: October 2026  
> **Branch**: `base` → `origin/base` at `https://github.com/Anumay1231/B.S-Detector.git`  
> **Status**: Few-Shot Scaling Benchmark Complete ($N=500$) ✅

---

## 1. What This Project Is

We benchmark **parameter-efficient audio deepfake detection on low-resource Hindi speech** using:
- **Model**: Sensitive Layer Selection (SLS) classifier head (~23.4M trainable parameters) on a **frozen XLS-R 300M** backbone (`facebook/wav2vec2-xls-r-300m`, 315.4M parameters).
- **Pretrained Checkpoint**: `sukhdeveyash/XLS-R-SLS-Deepfake-Detection` (`sls_head_english_pretrained.pth` locally).
- **Task**: Cross-lingual transfer and few-shot adaptation from English pretrained representations to Hindi speech.
- **Dataset**: `Jack-ppkdczgx/SEA-Spoof` (HuggingFace), Hindi subset.

---

## 2. Key Benchmark Findings ($N_{\text{test}} = 400$)

All evaluations are conducted on a held-out, 50/50 balanced test set of 400 Hindi clips (200 bona fide / 200 spoof) stratified across 5 TTS architectures:

| Supervision Setting | Train Clips per Class | Accuracy (%) | Equal Error Rate (EER %) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** (English $\to$ Hindi) | 0 | $61.75\%$ | $32.25\%$ | $56.70\%$ | $99.50\%$ | $72.23\%$ |
| **Few-Shot $N=10$** (3 seeds) | 10 | $64.33 \pm 5.04\%$ | $27.64 \pm 2.27\%$ | $58.72 \pm 3.53\%$ | $99.00 \pm 1.41\%$ | $73.63 \pm 2.51\%$ |
| **Few-Shot $N=50$** (3 seeds) | 50 | $75.83 \pm 10.49\%$ | $18.18 \pm 1.18\%$ | $71.69 \pm 10.98\%$ | $93.17 \pm 4.87\%$ | $80.16 \pm 5.81\%$ |
| **Few-Shot $N=500$** (3 seeds) | 500 | $\mathbf{96.83 \pm 0.82\%}$ | $\mathbf{2.66 \pm 0.52\%}$ | $\mathbf{95.78 \pm 1.16\%}$ | $\mathbf{98.00 \pm 1.08\%}$ | $\mathbf{96.87 \pm 0.81\%}$ |

### Seed Highlights for $N=500$:
- **Seed 2 (Best)**: **98.00% Accuracy**, **2.12% EER** (Only 8 errors out of 400 test clips).
- **Seed 1**: **96.25% Accuracy**, **2.50% EER**.
- **Seed 0**: **96.25% Accuracy**, **3.36% EER**.

---

## 3. Dataset & Stratification Architecture

- **Test Set**: 400 clips (200 bona fide, 200 spoof).
  - Bona fide: Multi-source balanced between studio acoustic environments (IndicTTS) and in-the-wild crowdsourced speech (Mozilla Common Voice Hindi).
  - Spoof: 40 clips each from 5 state-of-the-art TTS architectures (`edge_tts`, `vits_mms`, `indic_tts`, `fastspeech`, `xtts-v2`).
- **Disjointness**: 100% strictly disjoint ($\text{Test} \cap \text{Train} = \emptyset$).
- **Nested Hierarchy**: $\mathcal{D}_{\text{train}}^{(N=10, s)} \subset \mathcal{D}_{\text{train}}^{(N=50, s)} \subset \mathcal{D}_{\text{train}}^{(N=500, s)}$ for each seed $s$.

---

## 4. Codebase Map

```
B.S-Detector/
├── docs/
│   ├── HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.pdf  # Comprehensive technical dossier
│   ├── HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.tex  # Dossier LaTeX source
│   └── EXPERIMENT_PROTOCOL_AND_RESULTS.md            # Experimental protocol
├── figures/                                          # Publication-ready plots
│   ├── fig2_fewshot_adaptation_curve.png
│   ├── fig3_backbone_unfreezing_comparison.png
│   └── fig4_generator_stratification.png
├── models/                                           # Saved PyTorch SLS head weights
│   ├── sls_head_n500_seed2_best.pth                  # 98.00% Acc checkpoint
│   ├── sls_head_n50_seed2.pth                        # N=50 checkpoint
│   └── sls_head_english_pretrained.pth               # Pretrained English checkpoint
├── outputs/
│   ├── benchmark_splits_n500.json                    # Exact audio IDs
│   ├── experiment_n500.log                           # Cloud execution log
│   └── results_full_10_50_500.json                   # Consolidated results
├── scripts/
│   ├── infer_local.py                                # Local CLI/Python inference
│   ├── streaming.py                                  # Targeted Parquet shard streaming
│   ├── features.py                                   # XLS-R 300M feature extractor
│   ├── fewshot.py                                    # Few-shot adaptation trainer
│   ├── experiment.py                                 # Zero-shot and few-shot evaluation
│   ├── splits.py                                     # Stratified split generator
│   └── metrics.py                                    # EER, Accuracy, CM calculation
└── README.md
```

---

## 5. Local Inference

```bash
# CPU inference
python scripts/infer_local.py --audio sample.wav --device cpu

# Apple Silicon GPU acceleration
python scripts/infer_local.py --audio sample.wav --device mps
```
