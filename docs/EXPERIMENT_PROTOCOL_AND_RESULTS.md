# Audio Deepfake Detection: SLS on Frozen XLS-R 300M (Hindi Audio Benchmark)

## 1. Executive Summary

This document freezes the experimental protocol, benchmark dataset definition, and final empirical results for audio deepfake detection on **Hindi speech** using the **Sensitive / Selective Layer Summarization (SLS)** architecture on top of a frozen **Wav2Vec2 XLS-R 300M** backbone (`facebook/wav2vec2-xls-r-300m`).

The investigation evaluated cross-lingual transfer from a model trained on English deepfakes to Hindi in two settings:
1. **Zero-Shot Transfer**: Direct evaluation of English-trained SLS weights on held-out Hindi audio.
2. **Few-Shot Parameter-Efficient Adaptation**: Fine-tuning only the lightweight SLS classifier head on $N \in \{10, 50\}$ samples per class across 3 random seeds while keeping all 315.4M parameters of the XLS-R backbone frozen.

---

## 2. Frozen Experimental Protocol

### 2.1 Dataset & Sampling
- **Source**: `Jack-ppkdczgx/SEA-Spoof` (HF dataset), subset `language == 'hi'`.
- **Targeted Sample Extraction**:
  - Shard 0 (`data/train/train-00000.parquet`, Row Groups 0 & 1): 400 Hindi spoof samples.
  - Shard 2 (`data/train/train-00002.parquet`, Row Groups 2 & 3): 400 Hindi bonafide samples.
  - Total Hindi dataset size: **800 clips** (400 bonafide, 400 spoof), cached locally in `data/sea_spoof_en_hi_metadata.parquet`.
- **Audio Preprocessing**:
  - Sampling rate: 16 kHz.
  - Fixed input duration: 64,600 samples (~4.04 seconds).
  - Audio clips shorter than 64,600 samples are repeated/padded; clips longer are center-cropped.

### 2.2 Split Protocol & Disjointness Guarantees
- **Held-Out Test Set**:
  - Balanced: **250 bonafide + 250 spoof = 500 test samples**.
  - Sampled using `RandomState(seed=42)`.
- **Few-Shot Candidate Pool**:
  - Formed strictly by removing `test_ids` from the 800-sample pool: $800 - 500 = 300$ candidate samples (150 bonafide, 150 spoof).
  - **Test/Train Overlap: 0 samples (100% strictly disjoint)**.
- **Few-Shot Pools**:
  - Evaluated at $N=10$ (10 bonafide + 10 spoof = 20 clips) and $N=50$ (50 bonafide + 50 spoof = 100 clips).
  - Evaluated across 3 seeds: `seed_idx = 0` (seed 1000), `seed_idx = 1` (seed 1001), `seed_idx = 2` (seed 1002).
  - **Nested Supervision Property**: For each seed $s$, the 20 samples of $N=10$ are a **100% strict subset** of the 100 samples in $N=50$.

### 2.3 Model Architecture
- **Backbone**: `facebook/wav2vec2-xls-r-300m` (315.4M parameters).
  - **100% Frozen**: Zero gradients propagated to the backbone.
  - Extracts 24 transformer hidden layer representations of shape $(24, T=201, 1024)$.
- **SLS Classifier Head**:
  - `fc0`: `nn.Linear(1024, 1)` followed by Sigmoid gating to generate layer-importance weights.
  - Weighted summation across all 24 layers $\to$ representation $(B, T, 1024)$.
  - `first_bn`: `nn.BatchNorm2d(1)`.
  - `selu`: `nn.SELU()`.
  - `max_pool`: `nn.MaxPool2d(kernel_size=3, stride=3)` $\to$ flattened dim: 22,847.
  - `fc1`: `nn.Linear(22847, 1024)` $\to$ `SELU()`.
  - `fc3`: `nn.Linear(1024, 2)` $\to$ `LogSoftmax(dim=-1)`.
- **Pretrained Checkpoint**: `v1/epoch_2.pth` from `sukhdeveyash/XLS-R-SLS-Deepfake-Detection`.

### 2.4 Training Hyperparameters (Few-Shot)
- **Optimizer**: Adam (`lr = 1e-5`, `weight_decay = 1e-4`).
- **Batch Size**: 4.
- **Epochs**: 10.
- **Loss Function**: `nn.NLLLoss(weight=torch.tensor([0.1, 0.9]))` (class 0 = spoof, class 1 = bonafide).
- **Execution**: On-the-fly batched feature loading to guarantee low memory overhead (<100 MB RAM).

---

## 3. Benchmark Results (500 Test Samples: 250 Bonafide / 250 Spoof)

### 3.1 Consolidated Comparison Table

| Experiment | Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** (English $\to$ Hindi) | **69.20%** | **10.60%** | **62.18%** | **98.00%** | **76.09%** |
| **Few-Shot $N=10$** (mean $\pm$ std) | **97.53 $\pm$ 0.68%** | **1.97 $\pm$ 0.67%** | **96.39 $\pm$ 1.55%** | **98.80 $\pm$ 0.86%** | **97.57 $\pm$ 0.65%** |
| **Few-Shot $N=50$** (mean $\pm$ std) | **99.20 $\pm$ 0.00%** | **0.40 $\pm$ 0.01%** | **100.00 $\pm$ 0.00%** | **98.40 $\pm$ 0.00%** | **99.19 $\pm$ 0.00%** |

*(Note: Precision, Recall, and F1 are computed with `pos_label=1` [Bonafide] following the project convention).*

---

### 3.2 Per-Seed Breakdown

#### Zero-Shot Evaluation
- **Accuracy**: 69.20%
- **EER**: 10.60%
- **Precision**: 62.18% | **Recall**: 98.00% | **F1**: 76.09%
- **Confusion Matrix**:
  ```
                       Pred Spoof (0)    Pred Bonafide (1)
    True Spoof (0)                101                  149
    True Bonafide (1)               5                  245
  ```
- **Error Analysis**: The model correctly recognizes 245 of 250 bonafide samples (98% recall), but misses 149 spoof samples (59.6% false acceptance rate under threshold 0.5) due to cross-lingual domain shift.

#### Few-Shot $N=10$ (20 training clips total)
- **Seed 0**: Acc: **98.20%** | EER: **1.07%** | Prec: **96.89%** | Rec: **99.60%** | F1: **98.22%**
  - Spoof: 242/250 correct (8 missed) | Bonafide: 249/250 correct (1 false alarm)
- **Seed 1**: Acc: **96.60%** | EER: **2.67%** | Prec: **94.30%** | Rec: **99.20%** | F1: **96.69%**
  - Spoof: 235/250 correct (15 missed) | Bonafide: 248/250 correct (2 false alarms)
- **Seed 2**: Acc: **97.80%** | EER: **2.17%** | Prec: **97.99%** | Rec: **97.60%** | F1: **97.80%**
  - Spoof: 245/250 correct (5 missed) | Bonafide: 244/250 correct (6 false alarms)

#### Few-Shot $N=50$ (100 training clips total)
- **Seed 0**: Acc: **99.20%** | EER: **0.40%** | Prec: **100.00%** | Rec: **98.40%** | F1: **99.19%**
  - Spoof: 250/250 correct (0 missed) | Bonafide: 246/250 correct (4 false alarms)
- **Seed 1**: Acc: **99.20%** | EER: **0.39%** | Prec: **100.00%** | Rec: **98.40%** | F1: **99.19%**
  - Spoof: 250/250 correct (0 missed) | Bonafide: 246/250 correct (4 false alarms)
- **Seed 2**: Acc: **99.20%** | EER: **0.40%** | Prec: **100.00%** | Rec: **98.40%** | F1: **99.19%**
  - Spoof: 250/250 correct (0 missed) | Bonafide: 246/250 correct (4 false alarms)

---

## 4. Key Scientific Insights

1. **The Cross-Lingual Zero-Shot Gap**:
   - Audio deepfake detectors trained on high-resource English datasets exhibit an asymmetric error mode when deployed on Indic languages (Hindi): genuine speech is rarely rejected, but synthetic speech frequently escapes detection.
   - However, the underlying XLS-R 300M representations contain strong latent discriminative signals, evidenced by an EER of 10.60% despite the sub-70% argmax accuracy.
2. **Extreme Few-Shot Sample Efficiency**:
   - Providing just **10 bonafide and 10 spoof Hindi samples** ($N=10$) for 10 epochs (taking <3 seconds of compute) reduces the EER by over **81%** (from 10.60% to 1.97%) and increases accuracy to 97.53%.
   - With **50 samples per class** ($N=50$), the detector achieves near-perfect classification: **0.40% EER**, **99.20% accuracy**, and **100% precision on bonafide speech** (0 false positives on spoof).
3. **Parameter Efficiency**:
   - Zero modifications or fine-tuning of the 315M parameter XLS-R backbone are needed. Only the ~23M parameter SLS head is adapted, ensuring minimal risk of catastrophic forgetting and rapid adaptation capability.

---

## 5. Artifacts and Reproduction

- **Metadata**: `data/sea_spoof_en_hi_metadata.parquet` (800 rows).
- **Extracted Features**: `data/deepfake_cache/features/` (709 pre-extracted tensors).
- **Pre-cached Audio**: `data/deepfake_cache/audio/` (800 `.pt` clips).
- **JSON Results**: `outputs/results_full_250.json`.
- **Run Command**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\run_deepfake_detection.py --device cuda --test-per-class 250 --skip-streaming --skip-features
  ```
