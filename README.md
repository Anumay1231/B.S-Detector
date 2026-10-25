# B.S. Detector

Check whether a voice clip is real or synthetic. Drop in a recording and a frozen **XLS-R 300M** speech model with a **selective layer summarization (SLS)** head gives a verdict. It also shows the 4-second window it heard and which of its 24 layers it relied on. Everything runs on your own computer.

### [▶ Download and run B.S. Detector](https://github.com/Anumay1231/B.S-Detector/archive/refs/heads/main.zip)

1. Unzip the download.
2. Double-click **`start.bat`** (Windows) or **`start.command`** (macOS). On Linux, run `./start.command`.
3. Your browser opens at `http://127.0.0.1:8000` when the app is ready.

Or from a terminal:

```bash
git clone https://github.com/Anumay1231/B.S-Detector && cd B.S-Detector && python run.py
```

You need **Python 3.9 or newer**. The first run creates a `.venv`, installs PyTorch and the other dependencies, and downloads about 1.3 GB of model weights from Hugging Face. Later starts take a few seconds. Node.js isn't needed: the built interface ships in `frontend/dist`.

## What the app does

- **Analyze:** upload a file (WAV, MP3, FLAC, OGG, M4A) or record from your microphone. The browser converts the audio to 16 kHz mono and draws its spectrogram. You can drag the 4.04-second analysis window to any part of the clip.
- **Verdict:** the chance that the window is synthetic, plus the learned gate weight for each of XLS-R's 24 layers.
- **Findings:** the Hindi benchmark results behind the model, including where it fails.
- **Session log:** every check you run, exportable as JSON.

## How well it works

We tested on a balanced Hindi set of 200 real voices (IndicTTS, Mozilla Common Voice) and 200 fakes, with 40 from each of five generators: Edge-TTS, VITS-MMS, Indic-TTS, FastSpeech and XTTS-v2.

| Setup | Fakes caught (of 200) | Real voices passed (of 200) | Accuracy | EER |
| :-- | --: | --: | --: | --: |
| English-trained, no Hindi | 44 | 196 | 60.00% | 28.90% |
| Adapted on 10 Hindi clips per class | 74 | 189 | 65.83 ± 5.87% | 26.88 ± 1.83% |
| Adapted on 50 Hindi clips per class | 155 | 164 | 79.67 ± 0.47% | 19.70 ± 0.57% |

Unfreezing the top layers of XLS-R made things worse: 24.14% EER with 2 layers unfrozen and 25.31% with 4. The app runs the English-trained checkpoint, so on Hindi speech a "genuine" verdict is weak evidence. Full numbers are in [`outputs/`](outputs/) and [`paper/`](paper/).

## Project layout

This repository holds the two stages of B.S. Detector:

- **Deepfake detection** (stage 1): the web app, plus the XLS-R + SLS research pipeline.
- **Speaker verification** (stage 2): the ECAPA-TDNN module, documented [below](#speaker-verification).

```
run.py              one-command launcher (venv, dependencies, server, browser)
start.bat           double-click launcher for Windows
start.command       double-click launcher for macOS / Linux
server/
  app.py            FastAPI: /api/status, /api/analyze, serves the frontend
  inference.py      XLS-R 300M + SLS head inference
frontend/           React + TypeScript (Vite); dist/ is the built app
scripts/            deepfake benchmark pipeline and speaker-verification CLIs
src/                speaker_verification package (ECAPA-TDNN, calibration)
tests/              speaker-verification unit tests
outputs/            benchmark results (JSON) and figures
paper/              deepfake-detection paper draft and research dossier
docs/               design notes for both stages
```

## Developing the interface

```bash
python run.py --no-browser        # API on :8000
cd frontend && npm install && npm run dev   # UI on :5173, proxies /api to :8000
npm run build                     # rebuild frontend/dist before committing
```

`BSD_DEVICE` (`cuda`, `mps` or `cpu`) and `BSD_CHECKPOINT` (`v1` or `v2`) override the defaults.

## Credits

SLS checkpoint: [sukhdeveyash/XLS-R-SLS-Deepfake-Detection](https://huggingface.co/sukhdeveyash/XLS-R-SLS-Deepfake-Detection). Backbone: [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m). Data: [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof).

This is a research tool. Its verdicts are evidence, not proof.

---

## Speaker Verification

Speaker verification module for the B.S. Detector project. Given a test
audio sample and a reference audio sample, it decides whether the same
person spoke both and returns a `MATCH` / `NON_MATCH` / `UNCERTAIN`
verdict.

**Status: Phases 1–7 are complete** (project foundation, environment,
audio preprocessing, the ECAPA-TDNN speaker encoder, cosine similarity,
the calibration/verification decision layer and the large-scale VoxCeleb
calibration pipeline). **The pipeline has since been run on real
VoxCeleb1 data (Phases 8B and 8C)**, which produced the calibrated
threshold this project uses. See [Status / Roadmap](#status--roadmap) and
[docs/calibration.md](docs/calibration.md) for details.

### Architecture

```
Reference Audio                          Test Audio
      ↓                                        ↓
Preprocessing                          Preprocessing
      ↓                                        ↓
ECAPA-TDNN                              ECAPA-TDNN
      ↓                                        ↓
Embedding A                            Embedding B
      │                                        │
      └──────────── Cosine Similarity ─────────┘
                          ↓
                   Similarity Score
                          ↓
                     Calibration
                (genuine + impostor trials
                    → FAR / FRR / EER)
                          ↓
                      Threshold
                          ↓
                  MATCH / NON_MATCH
```

The reference and test audio are preprocessed separately and passed
through an ECAPA-TDNN speaker encoder, which turns each into a fixed-size
speaker embedding. Cosine similarity then compares the two embeddings and
gives a single score, roughly in `[-1.0, 1.0]`.

**A raw similarity score is not a verification decision on its own.**
`src/speaker_verification/verifier.py` only turns a score into `MATCH` /
`NON_MATCH` when it is given an explicit threshold (`score >= threshold ->
MATCH`). It never makes one up or falls back to a default.
`src/speaker_verification/calibration.py` computes FAR, FRR, EER and
ROC-AUC from labeled genuine/impostor trial scores, so the threshold comes
from data instead of a guess. The threshold now in use was calibrated on
real VoxCeleb1 trials. See
[Calibration & verification decision](#calibration--verification-decision)
below and [docs/calibration.md](docs/calibration.md) for the methodology
and data requirements.

Fuzzy logic and ML-based classification are not part of the baseline and
are not implemented here (see [Status / Roadmap](#status--roadmap)).

### Relationship to the B.S. Detector pipeline

This module is one part of the larger B.S. Detector project. The main
pipeline calls it **conditionally, after the deepfake detection stage**.
In other words, speaker verification only runs on audio that has already
been screened for deepfake or synthetic manipulation, as an extra
identity check.

### Project layout

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
│       ├── embedding_cache.py
│       ├── evaluation.py
│       ├── types.py
│       └── datasets/
│           ├── __init__.py
│           ├── trials.py
│           └── voxceleb.py
│
├── scripts/
│   ├── verify.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── check_environment.py
│   ├── test_audio_pipeline.py
│   ├── test_encoder.py
│   ├── test_similarity.py
│   ├── build_trials.py
│   ├── extract_embeddings.py
│   ├── score_trials.py
│   ├── run_voxceleb_calibration.py
│   └── score_personal_recording.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_audio.py
│   ├── test_encoder.py
│   ├── test_similarity.py
│   ├── test_calibration.py
│   ├── test_verifier.py
│   ├── test_datasets_trials.py
│   ├── test_datasets_voxceleb.py
│   ├── test_embedding_cache.py
│   ├── test_extract_embeddings.py
│   ├── test_score_trials.py
│   ├── test_run_voxceleb_calibration.py
│   ├── test_build_trials.py
│   └── test_voxceleb_pipeline_integration.py
│
├── data/
│   ├── reference/
│   ├── test/
│   ├── trials/
│   └── voxceleb/
│       ├── raw/
│       ├── metadata/
│       ├── trials/
│       └── processed/
│
├── outputs/
│   ├── embeddings/
│   ├── scores/
│   ├── calibration/
│   └── evaluation/
│
└── docs/
    ├── architecture.md
    ├── model.md
    └── calibration.md
```

- `src/speaker_verification/`: the installable Python package with all
  the module logic.
- `scripts/`: command-line entry points for verification, calibration and
  evaluation.
- `tests/`: unit tests for the package.
- `data/reference/`: reference (enrollment) audio samples.
- `data/test/`: test (query) audio samples to check against a reference.
- `data/trials/`: labeled trial pairs and metadata for calibration and
  evaluation.
- `outputs/`: generated results, calibrated thresholds and reports,
  including `embeddings/`, `scores/`, `calibration/` and `evaluation/`
  (Phase 7).
- `data/voxceleb/`: VoxCeleb data (Phase 7), with `raw/` audio,
  `metadata/`, sampled `trials/` CSVs and `processed/` artifacts. All of
  it is excluded from Git.
- `docs/`: design and architecture notes (`architecture.md`, `model.md`,
  `calibration.md`).

### Status / Roadmap

- Phase 1, project foundation: **COMPLETE**
- Phase 2, environment setup: **COMPLETE**
- Phase 3, audio preprocessing: **COMPLETE**
- Phase 4, ECAPA-TDNN embedding extraction: **COMPLETE** (checked on an
  NVIDIA GeForce RTX 3060 Laptop GPU with CUDA 12.4; 192-dimensional
  embeddings)
- Phase 5, cosine similarity scoring: **COMPLETE**
- Phase 6, calibration (FAR/FRR/EER/ROC-AUC) and verification decision
  layer: **COMPLETE**. Since Phase 8B it uses a **real calibrated
  threshold of 0.252784**, derived from 28,280 VoxCeleb1 trials (EER
  0.87%, ROC-AUC 0.9986; speaker-disjoint evaluation: FAR 1.15% / FRR
  0.48%). See
  [docs/calibration.md, "Phase 8B"](docs/calibration.md#phase-8b--first-real-calibration-results-voxceleb1)
  for the full results and three caveats on how to quote them.
- Phase 7, large-scale VoxCeleb calibration pipeline: **COMPLETE**, and
  run on real VoxCeleb1 data in Phases 8B and 8C (see
  [Large-scale VoxCeleb calibration](#large-scale-voxceleb-calibration-phase-7)).
- Deepfake detector integration: not started

Phase 6 note: `calibration.py` and `verifier.py` are covered by 46
deterministic unit tests (27 + 19) on synthetic, hand-checked data (see
[Calibration & verification decision](#calibration--verification-decision)).
The real-data scripts (`scripts/calibrate.py`, `scripts/verify.py`) were
first exercised end to end with a stub encoder, because the pretrained
model could not be downloaded in the first dev environment
(see [Encoder limitations](#encoder-limitations)). Running them on real
audio needs a machine with normal internet access, as with Phases 4–5.

Phase 3 note: the unit tests check that preprocessing is correct
(validation, mono conversion, resampling; see
[Audio preprocessing](#audio-preprocessing)). They say nothing about how
preprocessing affects speaker-verification accuracy. That can only be
measured with the encoder and evaluation pipeline (Phases 4–6) running on
real trial data.

Phase 4 note: the pretrained model could not be downloaded in the container the project was built in, because its network policy blocks the
Hugging Face Hub (see [Encoder limitations](#encoder-limitations)). It
was later run and checked on the RTX 3060 machine mentioned above.

Phase 5 note: the cosine similarity maths and input validation are
checked by deterministic unit tests on hand-built vectors (see
[Cosine similarity](#cosine-similarity)). These don't need the pretrained
model and pass anywhere. The real-embeddings integration test
(`scripts/test_similarity.py`) couldn't run in that container for the same network reason as Phase 4. **A raw cosine
similarity score is not a verification result** (see
[Architecture](#architecture)).

**Not part of the baseline:** a separate, optional, experimental module,
to be considered only once the ECAPA-TDNN baseline is fully evaluated:

- Fuzzy logic decision boundaries

### Hardware / device support

The project runs on **CPU by default** and **uses an NVIDIA GPU through
CUDA automatically** when one is available (`torch.cuda.is_available()`).
No code path needs a GPU. From Phase 4, the ECAPA-TDNN encoder
(`src/speaker_verification/encoder.py`) is the first component that
actually uses the GPU, with the same use-CUDA-if-available pattern
(`encoder.resolve_device()`). The GPU name and index are always read from
`torch.cuda` at runtime, never hard-coded.

Two environments matter here, and they behave differently:

- **The original development environment**: a Linux container with no
  GPU. `torch.cuda.is_available()` correctly reports `False` there, and
  `scripts/check_environment.py` was run and checked in it.
- **The RTX 3060 machine**: a real CUDA GPU is present.
  `torch.cuda.is_available()` should report `True` once PyTorch is
  installed, but confirm this locally (see below).

### Setup

#### Requirements

- Python 3.10+
- About 2 GB of free disk space for dependencies. The standard PyTorch
  wheel for Linux/Windows includes CUDA runtime support, which is used
  automatically when a compatible GPU and driver are present and ignored
  on CPU-only machines.

#### Create and activate a virtual environment

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

#### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the exact versions pinned in `requirements.txt`: torch
2.5.1 and torchaudio 2.5.1 (matched to avoid ABI mismatches), plus
speechbrain, numpy, scipy, scikit-learn, matplotlib and soundfile. On a
machine with an NVIDIA GPU and a compatible driver, such as the RTX 3060,
pip pulls in the CUDA runtime libraries torch needs
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12` and so on) as dependencies.
There is no separate GPU install step or extra flag.

#### Verify the environment

```bash
python scripts/check_environment.py
```

This prints the installed version of every dependency, reports whether
CUDA is available, and runs a basic CPU tensor operation, plus a CUDA one
if CUDA is available. It does **not** download the ECAPA-TDNN model or
any dataset.

**Important:** run this on your own machine to confirm it reports
`CUDA available: True` there. In the development container it
reports `CUDA available: False`, which is expected because that container
has no GPU. That result says nothing about whether your RTX 3060 will be
detected.

### Audio preprocessing

`src/speaker_verification/audio.py` is the input pipeline that the
ECAPA-TDNN encoder (Phase 4) builds on:

```
Audio File
    ↓
Validation (exists, supported extension, decodable)
    ↓
Load (native sample rate, native channel count)
    ↓
Mono conversion (average across channels, if more than one)
    ↓
Resample to 16,000 Hz (skipped if already 16 kHz)
    ↓
Preprocessed waveform: torch.FloatTensor, shape (1, num_samples)
```

**Supported input formats:** WAV and FLAC. Any other extension raises
`UnsupportedAudioFormatError`.

**What it deliberately does not do in this phase:** no denoising or
band-pass filtering, and no loudness or amplitude normalization beyond
the lossless integer-to-float PCM scaling every decoder does. It never
modifies, moves or overwrites the source file.

**Error handling:** a missing file, a corrupt or undecodable file and
zero-length audio each raise their own exception
(`AudioFileNotFoundError`, `AudioDecodeError`, `EmptyAudioError`). Very
short audio is processed rather than rejected, with a `UserWarning` if it
is under about 100 ms.

**Main functions:** `load_audio(path)` (raw load, no resampling or mono
conversion), `preprocess_audio(path)` (the full pipeline above) and
`get_audio_info(path)` (metadata only: sample rate, channels, frame
count and duration, without necessarily decoding the whole waveform).

#### Running the audio preprocessing tests

You need `pytest`, which isn't in `requirements.txt` because that file
only lists runtime dependencies:

```bash
pip install pytest
pytest tests/test_audio.py -v
```

The tests generate their own temporary WAV/FLAC fixtures (mono, stereo,
multi-channel, different sample rates, corrupt, empty, very short), so no
external dataset is needed.

#### Manually inspecting a single file

```bash
python scripts/test_audio_pipeline.py path/to/audio.wav
```

This prints the file's original sample rate, channel count and duration,
then the processed (16 kHz, mono) sample rate, channel count, duration
and waveform shape. It says so explicitly when a stereo or multi-channel
input was converted to mono.

### ECAPA-TDNN speaker encoder

`src/speaker_verification/encoder.py` wraps a pretrained ECAPA-TDNN
speaker-embedding model:

```
Audio
    ↓
Audio preprocessing
    ↓
16 kHz mono waveform
    ↓
Pretrained ECAPA-TDNN
    ↓
Speaker embedding
    ↓
[Cosine similarity — NEXT PHASE]
```

**Model:** `speechbrain/spkrec-ecapa-voxceleb`, used pretrained as-is
(not trained or fine-tuned in this project). Model documentation, the
verified API details and the cache location are in
[`docs/model.md`](docs/model.md).

**API:** `SpeakerEncoder()` loads the model once.
`encoder.encode(waveform)` takes a waveform already preprocessed by
`audio.py` and returns a 1-D embedding tensor, and
`encoder.encode_file(path)` does both steps for one file. The rest of the
project depends on this interface, not on SpeechBrain directly.

**Device handling:** CUDA is used when available
(`torch.cuda.is_available()`), otherwise CPU, the same as everywhere else
in the project. You can ask for a device with
`SpeakerEncoder(device="cuda")`. If CUDA isn't available, that raises a
clear error instead of quietly falling back to CPU.

#### Running the encoder tests

```bash
pip install pytest
pytest tests/test_encoder.py -v
```

Device-selection tests always run. Tests that need the real pretrained
model are skipped with a clear reason, not failed, if the model can't be
downloaded (see [Encoder limitations](#encoder-limitations)).

#### Manually inspecting a file (or a pair of files)

```bash
python scripts/test_encoder.py path/to/audio.wav
python scripts/test_encoder.py reference.wav test.wav   # two-file mode
```

Single-file mode prints the device, original and processed audio
metadata, the embedding's shape, dtype, device and norm, and the
inference time. Two-file mode extracts an embedding for each file and
shows their shapes and statistics side by side. It does **not** compute
similarity or a MATCH/NON_MATCH decision; that belongs to a later phase.

#### Encoder limitations

The pretrained model is downloaded from the Hugging Face Hub the first
time it is used and then cached locally (see `docs/model.md`), so network
access is required. The container blocks
`huggingface.co` at the network-policy level, so the model couldn't be
downloaded there, and no real embedding output (values, shape, inference
time) could be produced in it. That is a limit of the environment, not a
bug in `encoder.py`. Run `pytest tests/test_encoder.py -v` and
`python scripts/test_encoder.py <file>` on a machine with normal internet
access to get real results. (This has since been done; see
[Status / Roadmap](#status--roadmap).)

### Cosine similarity

`src/speaker_verification/similarity.py` answers one question only: how
similar are these two speaker embeddings?

```
embedding_a
embedding_b
      ↓
cosine similarity = dot(A, B) / (||A|| · ||B||)
      ↓
single scalar score, range ≈ [-1.0, 1.0]
```

**API:** `cosine_similarity(embedding_a, embedding_b)`. Pass two 1-D
embeddings (for example from `SpeakerEncoder.encode()`) to get a Python
`float`, or two 2-D batches of shape `(batch, embedding_dim)` to get a
`(batch,)` tensor of per-pair scores. It uses
`torch.nn.functional.cosine_similarity` rather than a hand-written
version.

**Validation:** it raises a specific, clearly named error for `None`,
non-tensor, non-floating-point, wrong-dimensionality or empty input; for
mismatched embedding dimension, batch size or device; for NaN/Inf; and
for zero-norm or near-zero-norm input. A zero-norm embedding raises an
error instead of returning an arbitrary score, because cosine similarity
is undefined for a zero vector.

**Device handling:** both embeddings must already be on the same device
(CPU or CUDA), and the computation runs there with no extra transfers.
Neither input tensor is modified.

**Scope:** this module has no threshold, no MATCH/NON_MATCH decision, no
FAR/FRR/EER/ROC and no calibration (see [Architecture](#architecture)).
**A raw cosine similarity score says nothing about whether two
recordings come from the same speaker until it is compared against a
threshold calibrated on representative genuine/impostor trial data
(Phase 6).**

#### Running the similarity tests

```bash
pip install pytest
pytest tests/test_similarity.py -v
```

These are deterministic tests on hand-built vectors (identical, scaled,
orthogonal, opposite, mismatched, NaN/Inf, zero-vector, batched) with
known expected scores. They don't need the pretrained model and run
anywhere, on CPU or CUDA.

#### Real ECAPA integration test (two audio files)

```bash
python scripts/test_similarity.py reference.wav test.wav
```

This runs both files through the audio preprocessing and
`SpeakerEncoder`, then reports their cosine similarity. It shows the
device, the GPU (if any), both embeddings' shapes and devices, the score,
and how long the similarity step took, timed separately from ECAPA
inference. It prints a raw score only. **It does not give a
MATCH/NON_MATCH verdict.**

**Same-speaker / different-speaker testing:** any personal recordings
used for this (for example under `tests/recording test/`, which
`.gitignore` excludes) are only for local, manual testing of the
pipeline. A same-speaker score is not an accuracy result, and a single
score shouldn't be called "good" or "bad". Judging that needs the
threshold calibration from Phase 6.

#### Similarity limitations

Like the encoder test, `scripts/test_similarity.py` downloads the
pretrained model the first time it runs, which the dev container's network policy blocks (see
[Encoder limitations](#encoder-limitations)). So no real
embedding-based score could be produced there. The logic in
`similarity.py` is fully covered by unit tests that don't need the model
(see above). Re-run the end-to-end script on a machine with normal
internet access, such as the one Phase 4 was checked on, to get a real
score.

### Calibration & verification decision

The full methodology is in [docs/calibration.md](docs/calibration.md):
FAR/FRR/EER definitions, how the EER point is chosen, threshold
direction, data requirements, and the data limits this project ran into.
In short:

- `src/speaker_verification/calibration.py`: `calibrate(genuine_scores,
  impostor_scores)` computes FAR and FRR at every candidate threshold,
  the EER operating point and ROC-AUC. It raises
  `InsufficientCalibrationDataError` if either class is empty. If either
  class has fewer than `MIN_RELIABLE_TRIALS_PER_CLASS` (30) trials, it
  sets `is_statistically_reliable=False` and adds explicit warnings. It
  still returns a result rather than failing silently or blocking.
- `src/speaker_verification/verifier.py`: `SpeakerVerifier(encoder,
  threshold, is_calibrated=False)` applies `score >= threshold -> MATCH`
  (otherwise `NON_MATCH`). `threshold` has **no default**, so the caller
  always has to supply it. `is_calibrated` defaults to `False`, so an
  ad-hoc threshold can't be mistaken for a validated one.

**Current data status (updated in Phase 8B): calibrated on 28,280 real
VoxCeleb1 trials, threshold = 0.252784.** Before that, there was only one
genuine trial (cosine similarity 0.635386) and no impostor trials, and
`calibrate()` correctly refused to calibrate on that. The single score
was never used as a threshold anywhere in the code. The real calibrated
value turned out to be much lower, which is exactly why guessing would
have been a mistake.

#### Running the calibration tests

```bash
pip install pytest
pytest tests/test_calibration.py tests/test_verifier.py -v
```

There are 46 deterministic tests (27 for calibration, 19 for the
verifier) on synthetic, hand-checked data: perfect separation, full
overlap, boundary and tie cases, empty or one-sided data, NaN/Inf and
out-of-range scores, exact threshold equality and so on. Both test files
say in their module docstrings that the data is synthetic and is not
real speaker-verification evidence.

#### Calibrating against real trial data

```bash
python scripts/calibrate.py --trials data/trials.csv
```

The script expects a CSV with columns `reference,test,label`, where
`label` is `genuine` or `impostor`. If `data/trials.csv` doesn't exist
(it doesn't in this repository), the script explains the format it needs
and exits cleanly with code 0. It doesn't error out or invent a result.

#### Running a verification decision

```bash
python scripts/verify.py reference.wav test.wav --threshold 0.6123 --calibrated
```

`--threshold` is required and has no default. Leave out `--calibrated`
for an ad-hoc or example threshold, and the output is marked
`UNCALIBRATED`.

### Large-scale VoxCeleb calibration (Phase 7)

The Phase 6 calibration engine works and is tested, but this project
only had one genuine trial and no impostor trials of its own, far too few
for a threshold anyone could trust. Phase 7 adds a pipeline that feeds
real genuine and impostor scores from
[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), a standard
public speaker-verification dataset, into the same unmodified
`calibration.py`, which gives a real data-driven baseline threshold.
**ECAPA-TDNN is not fine-tuned or changed at any point.** The
methodology (calibration/evaluation split, speaker-leakage checks,
embedding cache design, and the Hindi/Hinglish domain limitation) is in
[docs/calibration.md, "Phase 7"](docs/calibration.md#phase-7-large-scale-calibration-using-voxceleb).

**Status: run on real data (Phases 8B and 8C).** The full pipeline, with
4,715 utterances embedded on an RTX 3060, was run two ways:

| Run | Trials | EER | Threshold | Purpose |
|---|---|---|---|---|
| **8B** speaker-disjoint | 28,280 | 0.87% | **0.252784** | Chooses the threshold this project ships; held-out FAR 1.15% / FRR 0.48% is the unbiased estimate |
| **8C** full official protocol | 37,720 | 1.04% | 0.272217 | Literature-comparable benchmark only (ROC-AUC 0.998945) |

Both runs are written up in
[docs/calibration.md, "Phase 8B"](docs/calibration.md#phase-8b--first-real-calibration-results-voxceleb1)
and
["Phase 8C"](docs/calibration.md#phase-8c--full-voxceleb1-official-protocol-benchmark),
including why 8C scores worse than 8B and why 1.04% is above the
published figure of about 0.80%. The container
still can't reach VoxCeleb or download the pretrained model (see
[Encoder limitations](#encoder-limitations)), so these results were
produced on the RTX 3060 machine.

#### Running the Phase 7 tests

```bash
pip install pytest
pytest tests/test_datasets_trials.py tests/test_datasets_voxceleb.py \
       tests/test_embedding_cache.py tests/test_extract_embeddings.py \
       tests/test_score_trials.py tests/test_run_voxceleb_calibration.py \
       tests/test_build_trials.py tests/test_voxceleb_pipeline_integration.py -v
```

Everything here is synthetic (generated trial-list text, tiny WAV
fixtures, stub encoders), so no real VoxCeleb data or pretrained model is
needed.

#### Running the pipeline on real data

Once you have VoxCeleb audio and an official trial-list file (see
docs/calibration.md for exactly what to download and where to put it):

```bash
python scripts/build_trials.py \
    --trials <official_trial_list.txt> \
    --audio-root <path_to_voxceleb_audio> \
    --max-genuine 1000 --max-impostor 1000 --seed 42

python scripts/extract_embeddings.py --trials data/voxceleb/trials/subset.csv
python scripts/score_trials.py --trials data/voxceleb/trials/subset.csv
python scripts/run_voxceleb_calibration.py --scores outputs/scores/voxceleb_scores.csv
```

If your trial list identifies speakers by name
(`Eartha_Kitt/x6uYqmx31kE_0000001.wav`) instead of by VoxCeleb id
(`id10270/x6uYqmx31kE/00001.wav`), also pass VGG's speaker-metadata
table so names can be mapped to ids:

```bash
python scripts/build_trials.py \
    --trials data/raw/voxceleb1/trials/voxceleb1_test.txt \
    --audio-root data/raw/voxceleb1/vox1/test_wav/wav \
    --meta data/raw/voxceleb1/vox1/vox1/vox1_meta.csv \
    --output data/trials/voxceleb1_trials.csv \
    --seed 42 --calibration-fraction 0.5 --split-seed 42
```

Both forms are converted to the same canonical utterance id internally.
See [docs/calibration.md, "Trial-list formats"](docs/calibration.md#trial-list-formats-numeric-ids-vs-speaker-names).

What each script does:

- `build_trials.py` samples a deterministic subset and splits it into
  speaker-disjoint calibration and evaluation sets.
- `extract_embeddings.py` loads the encoder once and caches embeddings,
  skipping any utterance that is already cached.
- `score_trials.py` computes cosine similarity with the existing
  `similarity.py`.
- `run_voxceleb_calibration.py` calibrates on the calibration split only,
  reports the FAR and FRR of that threshold on the held-out evaluation
  split, and runs a speaker-leakage check.

#### Personal recordings stay separate

```bash
python scripts/score_personal_recording.py
```

This scores the project's two personal recordings (kept out of Git) with
the same pipeline and labels the result "Personal same-speaker
validation (OUT-OF-DOMAIN)". They are never mixed into VoxCeleb
calibration and never used to change the VoxCeleb-derived threshold
automatically.
