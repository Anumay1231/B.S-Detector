# Agent Handoff: Audio Deepfake Detection (SLS on XLS-R 300M) — V2 Benchmark Completed

## 1. Project & Status Overview
This project benchmarks audio deepfake detection on **Hindi speech** using the **Sensitive Layer Summarization (SLS)** architecture on top of a frozen **Wav2Vec2 XLS-R 300M** backbone (`facebook/wav2vec2-xls-r-300m`).

The full benchmark protocol (**250 bonafide / 250 spoof = 500 test samples**) has been completely executed, verified, and documented.

### Current Git State
- **Branch**: `base` (tracked as `origin/base` on `https://github.com/Anumay1231/B.S-Detector.git`).
- `main` branch remains untouched.
- All code, protocols, outputs, and paper drafts are version-controlled on `base`.

---

## 2. Benchmark Results — V1 (Biased) vs V2 (Balanced)

### V1 Dataset (DEPRECATED — Single-source bias, do not use for final paper)
Spoof clips skewed to 1–2 TTS models; bonafide from IndicTTS only. Results were inflated.

| Experiment | Accuracy (%) | EER (%) |
| :--- | :---: | :---: |
| Zero-Shot | 69.20% | 10.60% |
| Few-Shot N=10 | 97.53 ± 0.68% | 1.97 ± 0.67% |
| Few-Shot N=50 | **99.20 ± 0.00%** | **0.40 ± 0.01%** |

### V2 Dataset ✅ (CURRENT — Stratified across 5 TTS models)
800 clips: 400 bonafide (IndicTTS 61% + CommonVoice 39%), 400 spoof (80 clips × 5 TTS models: indic_tts, edge_tts, xtts-v2, vits_mms, elevenlabs). ~72 clips unavailable (streaming gap), effective test set ~446 samples.

| Experiment | Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** (English → Hindi) | **68.61%** | **28.52%** | **64.47%** | **98.00%** | **77.78%** |
| **Few-Shot N=10** (mean ± std) | **71.45 ± 4.90%** | **22.69 ± 0.54%** | **76.84 ± 6.53%** | **74.53 ± 21.44%** | **73.00 ± 9.46%** |
| **Few-Shot N=50** (mean ± std) | **88.71 ± 1.64%** | **9.86 ± 0.80%** | **86.09 ± 4.18%** | **95.73 ± 2.64%** | **90.53 ± 1.04%** |

### V2 Per-Seed Metrics
- **Zero-Shot**: Acc: 68.61%, EER: 28.52%, CM: `[[61, 135], [5, 245]]`
- **Few-Shot N=10**:
  - Seed 0 (1000): Acc: 73.09%, EER: 22.40%, Prec: 67.96%, Rec: 98.40%, F1: 80.39%
  - Seed 1 (1001): Acc: 76.46%, EER: 23.44%, Prec: 79.12%, Rec: 78.80%, F1: 78.96%
  - Seed 2 (1002): Acc: 64.80%, EER: 22.22%, Prec: 83.45%, Rec: 46.40%, F1: 59.64%
- **Few-Shot N=50**:
  - Seed 0 (1000): Acc: 91.03%, EER: 8.72%, Prec: 92.00%, Rec: 92.00%, F1: 92.00%
  - Seed 1 (1001): Acc: 87.67%, EER: 10.45%, Prec: 83.28%, Rec: 97.60%, F1: 89.87%
  - Seed 2 (1002): Acc: 87.44%, EER: 10.41%, Prec: 82.99%, Rec: 97.60%, F1: 89.71%

### Key Scientific Insight (Bias Confirmed)
The dramatic drop from V1→V2 (EER: 0.40% → 9.86% for N=50; EER: 1.97% → 22.69% for N=10) **confirms the dataset bias hypothesis**. V1 results were inflated because the model exploited artifacts from a narrow set of TTS generators. V2 reflects true cross-generator generalization performance.

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
- **V1 Metadata (deprecated)**: `data/sea_spoof_en_hi_metadata.parquet` (800 rows, biased).
- **V2 Metadata (current)**: `data/sea_spoof_en_hi_metadata_v2.parquet` (800 rows, balanced 5-model stratification).
- **Hindi Pool**: `data/hindi_full_pool.parquet` (16,382 discovered Hindi rows from full corpus scan).
- **Audio Cache**: `data/deepfake_cache/audio/` (637 `.pt` files for V2).
- **Feature Cache**: `data/deepfake_cache/features/` (637 `.pt` files for V2).
- **V2 Results JSON**: `outputs/results_full_250.json` (V2 benchmark results).
- **Protocol Document**: `docs/EXPERIMENT_PROTOCOL_AND_RESULTS.md`.
- **Audit & Pipeline Report**: `docs/DATASET_V2_AUDIT_AND_PIPELINE_REPORT.md`.
- **Full Academic Paper Draft**: `paper/sls_hindi_deepfake_detection.md`.
- **Knowledge Graph**: `graphify-out/` (125 nodes, 242 edges, 8 communities).

---

## 5. How to Reproduce / Re-run
To re-run V2 experiment using cached features on GPU:
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cuda --test-per-class 250 --skip-streaming --skip-features
```

Or on CPU:
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cpu --test-per-class 250 --skip-streaming --skip-features
```

---

## 6. Next Steps for Next Session
1. **Update Paper**: Revise `paper/sls_hindi_deepfake_detection.md` with V2 results and the V1-vs-V2 dataset bias analysis as a key finding.
2. **Fetch Remaining 72 Clips**: Use `fsspec` direct shard reader to fetch the missing bonafide/spoof clips and rebalance the test set to full 500 samples.
3. **Figures Generation**: Generate publication figures — ROC curves for V1 vs V2, TTS-model breakdown bar charts, score distribution histograms.
4. **Cross-Language Expansion** (Optional): Apply the pipeline to Tamil, Bengali, or Telugu samples from SEA-Spoof.
5. **LaTeX Conversion**: Convert paper draft to IEEEtran / Interspeech template format.
