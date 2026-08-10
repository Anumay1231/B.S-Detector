# Speaker Verification

Speaker verification module for the B.S. Detector project. Determines whether
a test audio sample and a reference audio sample were spoken by the same
person, producing a `MATCH` / `NON_MATCH` / `UNCERTAIN` verdict.

**Status: project foundation and environment are set up (Phases 1–2
complete).** No audio processing, embedding model, similarity scoring, or
calibration has been implemented yet. See
[Status / Roadmap](#status--roadmap) below.

## Architecture

```
Reference Audio
       ↓
Audio Preprocessing
       ↓
ECAPA-TDNN
       ↓
Reference Embedding

Test Audio
       ↓
Audio Preprocessing
       ↓
ECAPA-TDNN
       ↓
Test Embedding

Reference + Test Embeddings
       ↓
Cosine Similarity
       ↓
Threshold Calibration
       ↓
MATCH / NON_MATCH / UNCERTAIN
```

Reference audio and test audio are each independently preprocessed and
passed through an ECAPA-TDNN speaker encoder to produce fixed-size speaker
embeddings. The two embeddings are compared using cosine similarity, and the
resulting score is passed through a calibrated threshold (see
`src/speaker_verification/calibration.py`) to produce a final verdict of
`MATCH`, `NON_MATCH`, or `UNCERTAIN`.

For the baseline architecture, `calibration.py` is scoped strictly to
genuine/impostor trial score handling, threshold calibration, FAR, FRR, EER,
and ROC/ROC-AUC computation. Fuzzy logic is not part of the baseline and is
not implemented here — see [Status / Roadmap](#status--roadmap).

## Relationship to the B.S. Detector pipeline

This module is a component of the larger B.S. Detector project. It is
intended to be called **conditionally**, by the larger B.S. Detector
pipeline, **after the deepfake detection stage** — i.e. speaker verification
runs on audio that has already been screened for deepfake/synthetic
manipulation, as an additional identity-consistency check.

## Project layout

```
speaker-verification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── speaker_verification/
│       ├── __init__.py
│       ├── config.py
│       ├── audio.py
│       ├── encoder.py
│       ├── similarity.py
│       ├── verifier.py
│       ├── calibration.py
│       ├── evaluation.py
│       └── types.py
│
├── scripts/
│   ├── verify.py
│   ├── calibrate.py
│   └── evaluate.py
│
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_similarity.py
│   └── test_verifier.py
│
├── data/
│   ├── reference/
│   ├── test/
│   └── trials/
│
├── outputs/
│
└── docs/
    └── architecture.md
```

- `src/speaker_verification/` — the installable Python package containing
  all module logic.
- `scripts/` — CLI entry points for verification, calibration, and
  evaluation workflows.
- `tests/` — unit tests for the package.
- `data/reference/` — reference (enrollment) audio samples.
- `data/test/` — test (query) audio samples to verify against a reference.
- `data/trials/` — labeled trial pairs/metadata used for calibration and
  evaluation.
- `outputs/` — generated artifacts (results, calibrated thresholds, reports).
- `docs/` — design and architecture documentation.

## Status / Roadmap

- Phase 1 — Project foundation: **COMPLETE**
- Phase 2 — Environment setup: **COMPLETE**
- Phase 3 — Audio preprocessing: not started
- Phase 4 — ECAPA-TDNN embedding extraction: not started
- Phase 5 — Cosine similarity scoring: not started
- Phase 6 — Threshold calibration (genuine/impostor scores, FAR, FRR, EER,
  ROC/ROC-AUC): not started
- Deepfake detector integration: not started

**Not part of the baseline** — a separate, optional, experimental module to
be considered only after the ECAPA-TDNN baseline is fully evaluated:

- Fuzzy logic decision boundaries

## Hardware / device support

This project is designed to run on **CPU by default** and to **automatically
use an available NVIDIA GPU via CUDA** when one is present (`torch.cuda.is_available()`).
No code path requires a GPU; a GPU is used opportunistically for speed once
the ECAPA-TDNN encoder is implemented in a later phase.

Two environments are relevant to this project, and they are not the same:

- **This development/verification sandbox** — a Linux container with no
  GPU. `torch.cuda.is_available()` correctly reports `False` here, and
  `scripts/check_environment.py` was run and verified in this environment.
- **The developer's actual machine** (an NVIDIA RTX 3060) — a real CUDA
  GPU is present here. `torch.cuda.is_available()` should report `True` on
  that machine once PyTorch is installed there, but this has not been
  verified by this sandbox and must be re-checked locally (see below).

## Setup

### Requirements

- Python 3.10+
- ~2 GB free disk space for dependencies (PyTorch's standard Linux/Windows
  wheel includes CUDA runtime support, used automatically if a compatible
  GPU + driver is present, and otherwise simply unused on CPU-only
  machines)

### Create and activate a virtual environment

**Linux / macOS:**

```bash
cd speaker-verification
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
cd speaker-verification
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the exact versions listed in `requirements.txt`
(torch 2.5.1, torchaudio 2.5.1 — versions matched to avoid ABI mismatches —
plus speechbrain, numpy, scipy, scikit-learn, matplotlib, and soundfile).
On a machine with an NVIDIA GPU and a compatible driver (such as the
developer's RTX 3060), pip will automatically pull in the CUDA runtime
libraries torch needs (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) as
transitive dependencies — no extra flags or separate GPU install step are
required.

### Verify the environment

```bash
python scripts/check_environment.py
```

This prints installed versions of every dependency, reports whether CUDA
is available, runs a basic CPU tensor operation, and (if CUDA is available)
a basic CUDA tensor operation. It does **not** download the ECAPA-TDNN
model or any dataset.

**Important:** run this script on your own machine (not just in a sandbox)
to confirm whether it reports `CUDA available: True` there. It was run and
verified in this development sandbox, where it correctly and expectedly
reports `CUDA available: False` (no GPU present in that environment) —
that result says nothing about whether your RTX 3060 will be detected;
that must be checked locally.
