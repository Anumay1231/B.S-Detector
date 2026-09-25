# Agent Handoff — B.S-Detector (Audio Deepfake Detection)

> **Last updated**: 2026-09-25  
> **Branch**: `base` → `origin/base` at `https://github.com/Anumay1231/B.S-Detector.git`  
> **Status**: V2 benchmark complete ✅ — results pushed to GitHub

---

## 1. What This Project Is

We are benchmarking **audio deepfake detection on Hindi speech** using:
- **Model**: SLS (Sensitive Layer Summarization) head on a **frozen XLS-R 300M** backbone (`facebook/wav2vec2-xls-r-300m`, 315.4M params)
- **Pretrained checkpoint**: `sukhdeveyash/XLS-R-SLS-Deepfake-Detection` → `v1/epoch_2.pth` (trained on English)
- **Research question**: Can an English-trained deepfake detector transfer to Hindi, and how much does few-shot adaptation help?
- **Dataset**: `Jack-ppkdczgx/SEA-Spoof` (HuggingFace), Hindi subset

---

## 2. Key Results

### V1 Dataset — DEPRECATED (Single-source bias)
Spoof clips were skewed to 1–2 TTS models; bonafide from IndicTTS only. **Results were inflated.**

| Experiment | Accuracy | EER |
|---|---|---|
| Zero-Shot | 69.20% | 10.60% |
| Few-Shot N=10 | 97.53 ± 0.68% | 1.97 ± 0.67% |
| Few-Shot N=50 | **99.20 ± 0.00%** | **0.40 ± 0.01%** |

### V2 Dataset ✅ — CURRENT (Stratified, 5 TTS models)
800 clips: 400 bonafide (IndicTTS 61% + CommonVoice 39%), 400 spoof (80 clips × 5 TTS models).  
~72 clips could not be streamed → effective test set ~446 samples.

| Experiment | Accuracy | EER | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Zero-Shot** (English → Hindi) | 68.61% | 28.52% | 64.47% | 98.00% | 77.78% |
| **Few-Shot N=10** (mean ± std) | 71.45 ± 4.90% | 22.69 ± 0.54% | 76.84 ± 6.53% | 74.53 ± 21.44% | 73.00 ± 9.46% |
| **Few-Shot N=50** (mean ± std) | **88.71 ± 1.64%** | **9.86 ± 0.80%** | **86.09 ± 4.18%** | **95.73 ± 2.64%** | **90.53 ± 1.04%** |

#### V2 Per-Seed Breakdown
**Zero-Shot** — CM: `[[61, 135], [5, 245]]` (135 spoof clips escape detection)

**Few-Shot N=10:**
| Seed | Accuracy | EER | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 (seed=1000) | 73.09% | 22.40% | 67.96% | 98.40% | 80.39% |
| 1 (seed=1001) | 76.46% | 23.44% | 79.12% | 78.80% | 78.96% |
| 2 (seed=1002) | 64.80% | 22.22% | 83.45% | 46.40% | 59.64% |

**Few-Shot N=50:**
| Seed | Accuracy | EER | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 (seed=1000) | 91.03% | 8.72% | 92.00% | 92.00% | 92.00% |
| 1 (seed=1001) | 87.67% | 10.45% | 83.28% | 97.60% | 89.87% |
| 2 (seed=1002) | 87.44% | 10.41% | 82.99% | 97.60% | 89.71% |

### Key Scientific Insight — Bias Confirmed
The dramatic drop from V1→V2 (EER: 0.40%→9.86% for N=50; EER: 1.97%→22.69% for N=10) **confirms the dataset bias hypothesis**. V1 exploited artifacts from a narrow TTS set. V2 reveals true cross-generator generalization.

---

## 3. Dataset Details

### V2 Composition (`data/sea_spoof_en_hi_metadata_v2.parquet`)
| Class | Source | Count |
|---|---|---|
| Bonafide | IndicTTS | ~245 |
| Bonafide | CommonVoice | ~155 |
| Spoof | indic_tts | 80 |
| Spoof | edge_tts | 80 |
| Spoof | xtts-v2 | 80 |
| Spoof | vits_mms | 80 |
| Spoof | elevenlabs | 80 |

- **Full Hindi pool scanned**: 16,382 rows discovered from 439,362-row SEA-Spoof corpus
- **Bonafide coverage**: Rows 0–174,999 (shard 0–4, IndicTTS + CommonVoice)
- **Spoof coverage**: Rows 380,000–439,362 (shards 12–15, 5 TTS models)
- **72 clips missing**: Streaming was interrupted at shard 14; these are ElevenLabs spoof clips near row 386,6xx–386,7xx

---

## 4. Codebase Map

```
BS-Detector/
├── scripts/
│   ├── model.py              # SLSHead + load_backbone() + load_sls_checkpoint()
│   ├── features.py           # XLS-R feature extractor (GPU batched, CPU float16 cache)
│   ├── experiment.py         # evaluate_zeroshot() + train_fewshot()
│   ├── splits.py             # create_test_split() + create_fewshot_pools()
│   ├── metrics.py            # EER, Accuracy, Precision, Recall, F1, confusion matrix
│   ├── streaming.py          # HuggingFace dataset streamer + load_cached_audio()
│   └── run_deepfake_detection.py  # End-to-end CLI orchestrator
├── data/
│   ├── sea_spoof_en_hi_metadata.parquet      # V1 (DEPRECATED, biased)
│   ├── sea_spoof_en_hi_metadata_v2.parquet   # V2 (CURRENT, stratified)
│   ├── hindi_full_pool.parquet               # Full 16,382-row Hindi scan
│   └── deepfake_cache/
│       ├── audio/    # 637 .pt waveform tensors (V2)
│       └── features/ # 637 .pt XLS-R feature tensors (V2) [shape: (24, 201, 1024)]
├── outputs/
│   └── results_full_250.json   # V2 benchmark results (current)
├── docs/
│   ├── EXPERIMENT_PROTOCOL_AND_RESULTS.md
│   └── DATASET_V2_AUDIT_AND_PIPELINE_REPORT.md
├── paper/
│   └── sls_hindi_deepfake_detection.md   # Full academic paper draft (V1 results, needs update)
└── graphify-out/
    ├── graph.html        # Interactive knowledge graph (open in browser)
    ├── GRAPH_REPORT.md   # Architecture highlights and surprising connections
    └── graph.json        # Full graph (125 nodes, 242 edges, 8 communities)
```

### Model Architecture
- **Backbone**: `facebook/wav2vec2-xls-r-300m` — 315.4M params, **100% frozen**
  - Outputs 24 transformer hidden states, shape `(24, T=201, 1024)` per clip
- **SLS Head** (only part that trains):
  - `fc0`: `Linear(1024,1)` + Sigmoid → per-layer importance weights
  - Weighted sum across 24 layers → `(B, T, 1024)`
  - `BatchNorm2d(1)` → `SELU` → `MaxPool2d(3,3)` → flatten (22,847 dim)
  - `fc1`: `Linear(22847, 1024)` → `SELU`
  - `fc3`: `Linear(1024, 2)` → `LogSoftmax`

### Training Hyperparameters (Few-Shot)
- **Optimizer**: Adam, lr=1e-5, weight_decay=1e-4
- **Batch size**: 4 | **Epochs**: 10
- **Loss**: `NLLLoss(weight=[0.1, 0.9])` — class 0=spoof, class 1=bonafide
- **RAM**: <100 MB (on-the-fly feature loading from disk)

---

## 5. How to Re-run

### Quick Re-run (All Features Cached — ~30 seconds)
```powershell
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cuda --test-per-class 250 --skip-streaming --skip-features
```

### Full Pipeline (Streaming + Feature Extraction)
```powershell
# Step 1: Stream missing audio (slow — ~hours for full dataset scan)
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cuda --test-per-class 250

# Or skip streaming if audio already cached:
.\.venv\Scripts\python.exe scripts\run_deepfake_detection.py `
  --parquet-path data\sea_spoof_en_hi_metadata_v2.parquet `
  --device cuda --test-per-class 250 --skip-streaming
```

---

## 6. Known Issues & Gotchas

| Issue | Description | Status |
|---|---|---|
| **72 missing clips** | ElevenLabs spoof clips (row ~386,6xx–386,7xx in shard 14) were not streamed before task was killed | Open — use `fsspec` direct shard read to fix |
| **Streaming is slow** | Sequential scan at 3 clips/sec through 439k rows takes ~40 hours total | Workaround: use `fsspec` + parquet row group seek |
| **libtorchcodec DLL** | Windows DLL load error for `torchcodec` — fixed by using `datasets.Audio(decode=False)` + `soundfile` in `streaming.py` | Fixed ✅ |
| **features.py crash on missing audio** | Used to `FileNotFoundError` crash — now gracefully skips with warning | Fixed ✅ |
| **run_deepfake_detection.py filter** | Now filters `test_ids` and `fewshot_pools` to only IDs with cached features post-extraction | Fixed ✅ |

---

## 7. Next Steps (Priority Order)

1. **🔴 Fetch remaining 72 ElevenLabs clips** — Write a `fsspec`-based direct parquet row-group reader to grab clips from shard 14 at row indices 386,6xx–386,7xx. This will bring the V2 test set to the full 500 samples.

2. **🔴 Update paper** — `paper/sls_hindi_deepfake_detection.md` still contains V1 results. Needs updating with V2 numbers and a new section on the V1→V2 bias analysis as a key methodological finding.

3. **🟡 Generate publication figures**:
   - ROC curves: V1 vs V2 zero-shot comparison
   - TTS-model breakdown bar chart (per-model detection rate)
   - Layer importance bar chart (from `fc0.weight` sigmoid activations)
   - Score distribution histograms (bonafide vs spoof score separation)

4. **🟡 LaTeX conversion** — Convert paper draft to IEEEtran or Interspeech 2025 template.

5. **🟢 Cross-language expansion** — Apply identical pipeline to Tamil, Bengali, or Telugu samples from SEA-Spoof for a multi-lingual comparison table.

---

## 8. Git History (Recent)

```
f8ae857  feat: V2 benchmark - stratified 5-model dataset confirms bias hypothesis
e858ffd  feat: push graphify knowledge graph and comprehensive dataset audit report
...
```

Branch: `base` | Remote: `https://github.com/Anumay1231/B.S-Detector.git`
