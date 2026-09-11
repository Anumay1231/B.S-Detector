# Agent Handoff: Audio Deepfake Detection (SLS on XLS-R 300M)

## Project Overview
Build a prototype audio deepfake detector using the **SLS (Sensitive / Selective Layer Summarization)** architecture on top of a frozen **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`) backbone, evaluated specifically on **Hindi audio** in zero-shot and few-shot settings.

### Key Constraints & Requirements
1. **Branch**: Work on branch `base`. Do **NOT** push to remote yet.
2. **Backbone**: Frozen throughout — only the SLS head receives gradients.
3. **Data Source**:
   - Metadata: `/Users/yashgarg/Documents/bsdetector/B.S-Detector/data/sea_spoof_en_hi_metadata.parquet`
   - Audio: Streamed from HuggingFace dataset `Jack-ppkdczgx/SEA-Spoof` with `streaming=True` (do NOT download full dataset).
   - Audio is joined against local parquet via `row_id` / `utterance_id`.
   - Streaming strategy: Single streaming pass capturing the union of all needed IDs, cached locally.
4. **Pretrained Model**: Checkpoint weights from HuggingFace `sukhdeveyash/XLS-R-SLS-Deepfake-Detection` (`v1/epoch_2.pth`).
5. **Experiments to run**:
   - **Zero-shot**: Pretrained English SLS head evaluated directly on held-out Hindi test set. Report Accuracy, EER, Confusion Matrix.
   - **Few-shot (N=10)**: Fine-tune pretrained SLS head (backbone frozen) on $N=10$ per class (10 bonafide, 10 spoof) across 3 seeds. Report mean ± std.
   - **Few-shot (N=50)**: Fine-tune on $N=50$ per class across 3 seeds. Report mean ± std.
   - **Final comparison table**: Accuracy, EER, Precision, Recall, F1 across Zero-shot vs $N=10$ vs $N=50$.

---

## Current Status & Completed Work

### 1. Parquet Metadata Findings
- File: `data/sea_spoof_en_hi_metadata.parquet` (170,930 total rows, all `split='train'`).
- Hindi (`language == 'hi'`) subset: **54,938 rows** (`38,756 spoof`, `16,182 bonafide`).
- `row_id` format example: `sea_bench_clean_v2_train_000000000`.
- Verified columns include `row_id`, `utterance_id`, `language`, `label`, `spoof_type`, `category`.

### 2. Implemented Codebase (`src/deepfake_detection/`)
All files have been created and verified:
1. `src/deepfake_detection/__init__.py`: Package initialization.
2. `src/deepfake_detection/splits.py`:
   - `load_hindi_metadata()`: Loads parquet and filters for Hindi.
   - `create_test_split()`: Creates balanced test set (default: 250 bonafide, 250 spoof = 500 total).
   - `create_fewshot_pools()`: Disjoint few-shot sets for $N \in \{10, 50\}$ across 3 seeds (`seed=1000 + i`).
   - `get_all_needed_ids()`: Unions all required test and few-shot IDs for single-pass streaming.
   - `get_labels_for_ids()`: Maps row IDs to integer labels (1=bonafide, 0=spoof).
3. `src/deepfake_detection/model.py`:
   - `SLSHead`: Exact reproduction of Zhang et al. ACM MM 2024 / Yash Sukhdeve's checkpoint architecture:
     - `fc0`: `nn.Linear(1024, 1)` $\to$ Sigmoid gated layer weights.
     - Weighted sum of 24 transformer hidden layers.
     - `first_bn`: `nn.BatchNorm2d(1)`.
     - `selu`: `nn.SELU()`.
     - `maxpool`: `nn.MaxPool2d(kernel_size=3, stride=3)` $\to$ flattened dim: $22,847$.
     - `fc1`: `nn.Linear(22847, 1024)` $\to$ `SELU`.
     - `fc3`: `nn.Linear(1024, 2)`.
     - `logsoftmax`: `nn.LogSoftmax(dim=-1)`.
   - `load_backbone()`: Loads `facebook/wav2vec2-xls-r-300m` via HuggingFace `transformers`, sets `eval()`, freezes parameters.
   - `extract_hidden_states()`: Extracts hidden states from layers 1..24 ($24 \times (B, T=201, 1024)$).
   - `load_sls_checkpoint()`: Downloads `v1/epoch_2.pth` from `sukhdeveyash/XLS-R-SLS-Deepfake-Detection` and loads only head keys, filtering out `ssl_model.*`.
   - `pad_or_truncate()`: Fixed padding/tiling to 64,600 audio samples (16 kHz, ~4.04s).
4. `src/deepfake_detection/streaming.py`:
   - `stream_and_cache_audio()`: Single-pass streaming over `Jack-ppkdczgx/SEA-Spoof`, matching against target ID set, saves audio `.pt` tensors to `data/deepfake_cache/audio/`.
   - `load_cached_audio()`: Loads cached audio tensor by `row_id`.
5. `src/deepfake_detection/features.py`:
   - `extract_and_cache_features()`: Extracts 24-layer hidden states using frozen XLS-R backbone and saves as float16 `.pt` tensors in `data/deepfake_cache/features/`.
   - `load_cached_features()`, `load_features_batch()`, `collate_features()`: Utilities to load and collate pre-extracted features into batches without touching raw audio or the backbone.
6. `src/deepfake_detection/metrics.py`:
   - `compute_eer()`: Calculates Equal Error Rate using ROC interpolation (`scipy.optimize.brentq`).
   - `compute_all_metrics()`: Accuracy, EER, precision, recall, F1, and confusion matrix.
   - `print_metrics()`, `print_comparison_table()`: Formatted CLI outputs.

---

## Remaining Work to Complete

### 1. Implement `src/deepfake_detection/experiment.py`
Needs functions to run:
- **Zero-shot evaluation**:
  - Load cached features for test set.
  - Load pretrained `SLSHead` weights (`load_sls_checkpoint(sls_head)`).
  - Run inference with `torch.no_grad()`.
  - Compute and log zero-shot metrics (Acc, EER, Confusion Matrix, Prec, Rec, F1).
- **Few-shot training loop**:
  - Function: `train_fewshot(sls_head, train_row_ids, labels_dict, epochs=10, lr=1e-5, batch_size=4, weight=[0.1, 0.9])`.
  - Important: Train **only** the `SLSHead` using the pre-cached features (no forward pass through XLS-R needed during training!).
  - Criterion: `nn.NLLLoss(weight=torch.tensor([0.1, 0.9]))` (index 0=spoof, index 1=bonafide).
  - Optimizer: `Adam(sls_head.parameters(), lr=1e-5, weight_decay=1e-4)`.
  - For each few-shot condition ($N=10$, $N=50$):
    - Run over 3 seeds.
    - Before each seed, re-initialize `SLSHead` with `load_sls_checkpoint()`.
    - Evaluate fine-tuned head on the same held-out test set.
    - Collect metrics per seed and calculate `mean ± std`.

### 2. Implement Orchestrator Script `scripts/run_deepfake_detection.py`
A single runnable script that ties everything together:
1. Initialize logger and directories.
2. Load Hindi metadata and generate splits (`test_ids`, `fewshot_pools`, `all_needed_ids`).
3. Stream and cache audio for `all_needed_ids` (with progress reporting & ID verification).
4. Extract and cache XLS-R features for `all_needed_ids`.
5. Run zero-shot experiment on test set.
6. Run few-shot experiments ($N=10$, $N=50$ across 3 seeds).
7. Print final comparison table (`print_comparison_table`).

### 3. Update Project Files
- **`requirements.txt`**: Add `transformers`, `datasets`, `huggingface_hub`, `pandas`, `pyarrow`.
- **`.gitignore`**: Ensure `data/deepfake_cache/` is ignored.

---

## Execution Instructions for Next Agent
1. **Do not modify working modules** (`splits.py`, `model.py`, `features.py`, `metrics.py`, `streaming.py`) unless a bug is uncovered.
2. Ensure you have the Python dependencies installed in the environment (`pandas`, `pyarrow`, `transformers`, `datasets`, `huggingface_hub`).
3. When running `stream_and_cache_audio()`, check if the user is authenticated with HuggingFace Hub (`huggingface-cli whoami` or token) as `Jack-ppkdczgx/SEA-Spoof` is a gated dataset.
4. Keep git operations restricted to the `base` branch and do not push to remote.
