# B.S-Detector: Parameter-Efficient Audio Deepfake Detection on Hindi Speech

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-XLS--R--300M-orange)](https://huggingface.co/facebook/wav2vec2-xls-r-300m)

**B.S-Detector** is an empirical research and engineering benchmark investigating parameter-efficient adaptation of cross-lingual self-supervised representations for synthetic speech detection on low-resource Hindi audio.

By combining a **strictly frozen multilingual foundation backbone** ([Wav2Vec2 XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m)) with a trainable **Sensitive Layer Selection (SLS)** classifier head (~23.4M parameters), the system adapts rapidly from zero-shot English transfer to near-perfect Hindi deepfake detection under low-supervision regimes.

---

## Key Benchmark Findings

All models were evaluated on a held-out, strictly disjoint, 50/50 balanced test set of **400 Hindi clips** (200 genuine / 200 spoof) stratified across 5 state-of-the-art text-to-speech architectures.

| Supervision Setting | Train Clips per Class | Accuracy (%) | Equal Error Rate (EER %) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** (English $\to$ Hindi) | 0 | $61.75\%$ | $32.25\%$ | $56.70\%$ | $99.50\%$ | $72.23\%$ |
| **Few-Shot $N=10$** (3 seeds) | 10 | $64.33 \pm 5.04\%$ | $27.64 \pm 2.27\%$ | $58.72 \pm 3.53\%$ | $99.00 \pm 1.41\%$ | $73.63 \pm 2.51\%$ |
| **Few-Shot $N=50$** (3 seeds) | 50 | $75.83 \pm 10.49\%$ | $18.18 \pm 1.18\%$ | $71.69 \pm 10.98\%$ | $93.17 \pm 4.87\%$ | $80.16 \pm 5.81\%$ |
| **Few-Shot $N=500$** (3 seeds) | 500 | $\mathbf{96.83 \pm 0.82\%}$ | $\mathbf{2.66 \pm 0.52\%}$ | $\mathbf{95.78 \pm 1.16\%}$ | $\mathbf{98.00 \pm 1.08\%}$ | $\mathbf{96.87 \pm 0.81\%}$ |

### Seed Highlights for $N=500$ (1,000 Training Clips)
- **Seed 2 (Best)**: **98.00% Accuracy**, **2.12% EER**, 97.06% Precision, 99.00% Recall (Only 8 errors out of 400 test clips: 6 spoof missed, 2 false alarms).
- **Seed 1**: **96.25% Accuracy**, **2.50% EER**, 94.26% Precision, 98.50% Recall.
- **Seed 0**: **96.25% Accuracy**, **3.36% EER**, 96.02% Precision, 96.50% Recall.

---

## Architectural Specification

```
Raw Audio (16 kHz, ~4.04s mono)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Wav2Vec2 XLS-R 300M (Frozen Backbone)                 │
│  - 7 Conv downsampling blocks (320x temporal reduction)│
│  - 24 Transformer Encoder Layers (dim=1024, heads=16)  │
│  - Parameters: 315,441,760 (requires_grad = False)     │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼ Stacked Hidden States (24, B, 201, 1024)
┌────────────────────────────────────────────────────────┐
│  Sensitive Layer Selection (SLS) Head (Trainable)      │
│  - Temporal Average Pooling per layer                  │
│  - Layer Attention Gate: Linear(1024, 1) + Sigmoid     │
│  - Weighted Layer Sum: \sum w_l * H_l                  │
│  - 2D Batch Normalization (1 feature)                  │
│  - SELU Activation + MaxPool2d (kernel=3, stride=3)    │
│  - Fully Connected: Linear(22847, 1024) + SELU()       │
│  - Classification Head: Linear(1024, 2) + LogSoftmax() │
│  - Parameters: 23,399,431 (~23.4M Trainable)           │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
            [P(Spoof), P(Bona fide)]
```

### Complete Parameter Breakdown
- **Frozen Backbone Parameters**: $315,441,760$ (~315.4M)
- **Trainable SLS Head Parameters**: $23,399,431$ (~23.4M)
- **Total System Parameters**: $338,841,191$ (~338.8M)
- **Trainable Ratio**: $6.91\%$ of total network parameters.

---

## Benchmark Dataset & Stratification

The benchmark is derived from the Hindi split of [`Jack-ppkdczgx/SEA-Spoof`](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) ($54,938$ candidate clips):

1. **Multi-Source Acoustic Diversity**:
   - Bona fide recordings are sourced across both studio environments (IndicTTS) and in-the-wild crowdsourced speech (Mozilla Common Voice Hindi) to provide natural acoustic variation and realistic channel coverage.
2. **Held-Out Test Set ($N_{\text{test}} = 400$, 50/50 Balanced)**:
   - **Bona fide (200 clips)**: Balanced across studio (IndicTTS) and in-the-wild (Mozilla Common Voice) acoustic conditions.
   - **Spoof (200 clips)**: Strictly stratified with exactly 40 clips (20.0%) from each of 5 TTS architectures:
     - `edge_tts`: Microsoft Azure Neural cloud voice synthesizer.
     - `vits_mms`: Meta MMS variational inference end-to-end synthesizer.
     - `indic_tts`: Fast unit-selection and acoustic models from IIT Madras.
     - `fastspeech`: Feed-forward non-autoregressive transformer TTS.
     - `xtts-v2`: Coqui diffusion/flow-matching zero-shot voice cloning.
3. **Deterministic Shard Streaming**:
   - Audio is extracted directly from Hugging Face Parquet shards via mathematical row-group indexing:
     $$\text{Row Index} = \text{int}(\text{row\_id}.\text{split}(\text{'\_'})[-1])$$
     $$\text{Shard Index} = \lfloor \text{Row Index} / 15,000 \rfloor$$
     $$\text{Row Group Index} = \lfloor (\text{Row Index} \pmod{15,000}) / 200 \rfloor$$
   - All 3,331 benchmark audio clips span only 5 shards (0, 1, 2, 3, 25), enabling rapid streaming extraction in under 7 minutes.

---

## Local Inference Quickstart

You can run standalone inference on any audio file locally on your laptop using CPU or Apple Silicon (MPS) acceleration without cloud dependencies.

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Anumay1231/B.S-Detector.git
cd B.S-Detector

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchaudio transformers soundfile librosa
```

### 2. Run CLI Inference
```bash
# Run local inference on CPU using the best N=500 checkpoint:
python scripts/infer_local.py --audio path/to/sample.wav --device cpu

# Run using Apple Silicon GPU acceleration:
python scripts/infer_local.py --audio path/to/sample.wav --device mps
```

### 3. Python API Usage
```python
import torch
import soundfile as sf
from scripts.infer_local import LocalDeepfakeDetector

# Initialize detector (automatically loads frozen XLS-R 300M and SLS head weights)
detector = LocalDeepfakeDetector(
    weights_path="models/sls_head_n500_seed2_best.pth",
    device="cpu"  # or "mps" / "cuda"
)

# Predict on an audio file
result = detector.predict("path/to/sample.wav")
print(result)
# Output:
# {
#     "prediction": "Spoof" (or "Bona fide"),
#     "confidence": 0.9842,
#     "probabilities": {"spoof": 0.9842, "bonafide": 0.0158},
#     "logits": [-0.0159, -4.1482]
# }
```

---

## Checkpoint Manifest

Trained SLS classifier head checkpoints are located in `models/`:

| Checkpoint File | Configuration | Test Accuracy | Test EER | Size |
| :--- | :--- | :---: | :---: | :---: |
| `models/sls_head_channel_robust.pth` | Acoustic Channel Invariant (Phone Replay + Commercial TTS) | **94.52% (90.27% Replay, 98.48% Clean)** | **5.36%** | 89.2 MB |
| `models/sls_head_silence_invariant.pth` | Silence Invariant (Active Speech Trimming + Random Beds) | **98.20%** | **1.85%** | 89.2 MB |
| `models/sls_head_n500_indicsynth_best.pth` | IndicSynth Voice Conversion + FLEURS Balanced | **98.75%** | **1.17%** | 89.2 MB |
| `models/sls_head_n500_seed2_best.pth` | $N=500$, Seed 2 (Few-shot base) | **98.00%** | **2.12%** | 89.2 MB |
| `models/sls_head_n50_seed2.pth` | $N=50$, Seed 2 (Intermediate) | **79.50%** | **18.75%** | 89.2 MB |
| `models/sls_head_english_pretrained.pth` | Zero-shot base (English pretrained) | **61.75%** | **32.25%** | 89.2 MB |

*Note: The XLS-R 300M backbone is downloaded and cached automatically from Hugging Face Hub (`facebook/wav2vec2-xls-r-300m`).*

---

## Technical Dossier

For every exact parameter, optimization equation, confusion matrix, hardware specification (NVIDIA L4 on GCP), and ablation study, consult the comprehensive PDF report:

- **[Technical Dossier PDF](HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.pdf)** (7 pages, complete technical manual and reference)

### Included Figures
- `figures/fig2_fewshot_adaptation_curve.png`: Few-shot scaling trajectory across $N \in \{0, 10, 50, 500\}$.
- `figures/fig3_backbone_unfreezing_comparison.png`: Architectural ablation comparing frozen backbone vs. top-layer fine-tuning.
- `figures/fig4_generator_stratification.png`: TTS generator vulnerability breakdown across 5 synthesis architectures.

---

## Repository Structure

```
├── docs/
│   ├── HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.pdf  # Comprehensive technical dossier
│   ├── HINDI_DEEPFAKE_DETECTION_TECHNICAL_DOSSIER.tex  # LaTeX source code for technical dossier
│   └── EXPERIMENT_PROTOCOL_AND_RESULTS.md            # Experimental protocols
├── figures/                                          # Publication-ready charts
│   ├── fig2_fewshot_adaptation_curve.png
│   ├── fig3_backbone_unfreezing_comparison.png
│   └── fig4_generator_stratification.png
├── models/                                           # Trained SLS head weights
│   ├── sls_head_n500_seed2_best.pth
│   ├── sls_head_n50_seed2.pth
│   └── sls_head_english_pretrained.pth
├── outputs/                                          # Evaluation splits and raw logs
│   ├── benchmark_splits_n500.json                    # Exact audio IDs for test and train splits
│   ├── experiment_n500.log                           # End-to-end execution log on GCP L4
│   └── results_full_10_50_500.json                   # Consolidated metric outputs
├── scripts/                                          # Pipelines, streaming, and inference
│   ├── infer_local.py                                # Local standalone inference script
│   ├── streaming.py                                  # Fast targeted Parquet shard extraction
│   ├── features.py                                   # Batch XLS-R 300M feature extractor
│   ├── fewshot.py                                    # SLS few-shot adaptation loop
│   └── splits.py                                     # Deterministic stratified splits generator
└── README.md
```

---

## Citation

```bibtex
@article{bsdetector2026hindi,
  title={Parameter-Efficient Audio Deepfake Detection on Low-Resource Hindi Speech using Multilingual Self-Supervised Representations},
  author={B.S-Detector Research Team},
  year={2026},
  url={https://github.com/Anumay1231/B.S-Detector}
}
```
