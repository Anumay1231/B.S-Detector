# Agent Handoff: Audio Deepfake Detection (SLS on XLS-R 300M) — Full Benchmark Completed

## 1. Project & Status Overview
This project benchmarks audio deepfake detection on **Hindi speech** using the **Sensitive Layer Summarization (SLS)** architecture on top of a frozen **Wav2Vec2 XLS-R 300M** backbone (`facebook/wav2vec2-xls-r-300m`).

The full benchmark protocol (**250 bonafide / 250 spoof = 500 test samples**) has been completely executed, verified, and documented.

### Current Git State
- **Branch**: `base` (tracked as `origin/base` on `https://github.com/Anumay1231/B.S-Detector.git`).
- `main` branch remains untouched.
- All code, protocols, outputs, and paper drafts are version-controlled on `base`.

---

## 2. Final Frozen Benchmark Results (500 Test Samples)

| Experiment | Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** (English $\to$ Hindi) | **69.20%** | **10.60%** | **62.18%** | **98.00%** | **76.09%** |
| **Few-Shot $N=10$** (mean $\pm$ std) | **97.53 $\pm$ 0.68%** | **1.97 $\pm$ 0.67%** | **96.39 $\pm$ 1.55%** | **98.80 $\pm$ 0.86%** | **97.57 $\pm$ 0.65%** |
| **Few-Shot $N=50$** (mean $\pm$ std) | **99.20 $\pm$ 0.00%** | **0.40 $\pm$ 0.01%** | **100.00 $\pm$ 0.00%** | **98.40 $\pm$ 0.00%** | **99.19 $\pm$ 0.00%** |

### Per-Seed Metrics
- **Zero-Shot**: Acc: 69.20%, EER: 10.60%, CM: `[[101, 149], [5, 245]]`
- **Few-Shot $N=10$**:
  - Seed 0 (1000): Acc: 98.20%, EER: 1.07%, Prec: 96.89%, Rec: 99.60%, F1: 98.22%
  - Seed 1 (1001): Acc: 96.60%, EER: 2.67%, Prec: 94.30%, Rec: 99.20%, F1: 96.69%
  - Seed 2 (1002): Acc: 97.80%, EER: 2.17%, Prec: 97.99%, Rec: 97.60%, F1: 97.80%
- **Few-Shot $N=50$**:
  - Seed 0 (1000): Acc: 99.20%, EER: 0.40%, Prec: 100.00%, Rec: 98.40%, F1: 99.19%
  - Seed 1 (1001): Acc: 99.20%, EER: 0.39%, Prec: 100.00%, Rec: 98.40%, F1: 99.19%
  - Seed 2 (1002): Acc: 99.20%, EER: 0.40%, Prec: 100.00%, Rec: 98.40%, F1: 99.19%
  - Confusion Matrix for all 3 seeds: `[[250, 0], [4, 246]]` (0 false alarms on spoof!).

---

## 3. Key Technical Implementations & Architectural Details

### Codebase Organization (`scripts/`)
1. **`scripts/model.py`**:
   - `SLSHead`: Complete replication of Zhang et al. ACM MM 2024 / Yash Sukhdeve architecture.
   - `fc0`: `Linear(1024, 1)` + Sigmoid layer importance gating.
   - 2D downsampling block: `BatchNorm2d(1)` $\to$ `SELU` $\to$ `MaxPool2d(3, 3)` (dim 22,847) $\to$ `Linear(22847, 1024)` $\to$ `SELU` $\to$ `Linear(1024, 2)` $\to$ `LogSoftmax`.
   - `load_backbone()`: Loads `facebook/wav2vec2-xls-r-300m` on GPU/CPU and freezes all parameters.
   - `load_sls_checkpoint()`: Loads `v1/epoch_2.pth` from `sukhdeveyash/XLS-R-SLS-Deepfake-Detection`, automatically stripping `module.` prefix and filtering `ssl_model.*` keys.
2. **`scripts/splits.py`**:
   - `create_test_split()`: Generates balanced test set (250 bonafide, 250 spoof = 500 samples).
   - `create_fewshot_pools()`: Disjoint candidate pool sampling across seeds 0, 1, 2 for $N \in \{10, 50\}$.
   - **Strict Disjointness**: `test_ids` are subtracted before few-shot sampling (0 overlap verified).
   - **Nested Supervision**: For each seed, $N=10$ is a strict subset of $N=50$.
3. **`scripts/features.py`**:
   - Batch feature extractor on GPU (`batch_size=2`), immediately moving hidden layer states to CPU float16 to eliminate GPU VRAM and system RAM pressure.
   - Caches to `data/deepfake_cache/features/{row_id}.pt`.
4. **`scripts/experiment.py`**:
   - `evaluate_zeroshot()`: Evaluates on 500 test samples using pretrained checkpoint.
   - `train_fewshot()`: Fine-tunes **only** the `SLSHead` with on-the-fly batch loading (<100 MB RAM footprint).
   - Evaluates test set under `torch.no_grad()`.
5. **`scripts/metrics.py`**:
   - Exact EER calculation via `scipy.optimize.brentq` on linear ROC interpolation.
   - Accuracy, Precision, Recall, F1 (binary with `pos_label=1`), and 2x2 confusion matrix.
6. **`scripts/run_deepfake_detection.py`**:
   - End-to-end CLI orchestrator script.
   - Automatically exports results to `outputs/results_full_250.json`.

---

## 4. Key Artifacts on Disk
- **Metadata Parquet**: `data/sea_spoof_en_hi_metadata.parquet` (800 rows: 400 bonafide, 400 spoof).
- **Audio Cache**: `data/deepfake_cache/audio/` (800 `.pt` files).
- **Feature Cache**: `data/deepfake_cache/features/` (709 `.pt` files).
- **Results JSON**: `outputs/results_full_250.json`.
- **Protocol Document**: `docs/EXPERIMENT_PROTOCOL_AND_RESULTS.md`.
- **Full Academic Paper Draft**: `paper/sls_hindi_deepfake_detection.md`.

---

## 5. How to Reproduce / Re-run
To re-run the entire pipeline in under 30 seconds using cached features on GPU:
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py --device cuda --test-per-class 250 --skip-streaming --skip-features
```

Or on CPU:
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py --device cpu --test-per-class 250 --skip-streaming --skip-features
```

---

## 6. Next Steps for Next Session
1. **Paper Polishing**: Review and refine `paper/sls_hindi_deepfake_detection.md` for target conference formatting (e.g. converting markdown to LaTeX / IEEEtran / Interspeech template).
2. **Figures Generation**: Generate publication figures (ROC curves, layer-weight importance bar charts from `fc0.weight`, score distribution histograms).
3. **Cross-Language Expansion** (Optional): Apply the exact same frozen pipeline to Tamil, Bengali, or Telugu samples from SEA-Spoof.
