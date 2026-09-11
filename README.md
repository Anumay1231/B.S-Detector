# Speaker Verification

Speaker verification module for the B.S. Detector project. Determines whether
a test audio sample and a reference audio sample were spoken by the same
person, producing a `MATCH` / `NON_MATCH` / `UNCERTAIN` verdict.

**Status: project foundation, environment, audio preprocessing, the
ECAPA-TDNN speaker encoder, cosine similarity comparison, the
calibration/verification decision layer, and a large-scale VoxCeleb
calibration pipeline are complete (Phases 1–7).** The VoxCeleb pipeline
(code + tests) is fully built and verified against synthetic fixtures,
but **has not yet been run on real VoxCeleb data in this environment**
(VoxCeleb requires its own registration-gated download, and the
pretrained model itself is blocked from downloading in this sandbox —
see [Phase 7 dataset/access limitations](#calibration--verification-decision)).
See [Status / Roadmap](#status--roadmap) and
[docs/calibration.md](docs/calibration.md) below.

## Architecture

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

Reference audio and test audio are each independently preprocessed and
passed through an ECAPA-TDNN speaker encoder to produce fixed-size speaker
embeddings. The two embeddings are compared using cosine similarity,
producing a single scalar score in roughly `[-1.0, 1.0]`.

**A raw similarity score is not itself a verification decision.**
`src/speaker_verification/verifier.py` turns a score into `MATCH` /
`NON_MATCH` only once given an explicit threshold (`score >= threshold ->
MATCH`); it never invents or defaults one. `src/speaker_verification/calibration.py`
computes FAR, FRR, EER, and ROC-AUC from labeled genuine/impostor trial
scores so that threshold can be data-derived rather than guessed. **This
project does not currently have enough trial data to produce a real
calibrated threshold** — see
[Calibration & verification decision](#calibration--verification-decision)
below and [docs/calibration.md](docs/calibration.md) for the full
methodology and honest data-requirement discussion.

Fuzzy logic and ML-based classification are not part of the baseline and
are not implemented here — see [Status / Roadmap](#status--roadmap).

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

- `src/speaker_verification/` — the installable Python package containing
  all module logic.
- `scripts/` — CLI entry points for verification, calibration, and
  evaluation workflows.
- `tests/` — unit tests for the package.
- `data/reference/` — reference (enrollment) audio samples.
- `data/test/` — test (query) audio samples to verify against a reference.
- `data/trials/` — labeled trial pairs/metadata used for calibration and
  evaluation.
- `outputs/` — generated artifacts (results, calibrated thresholds, reports),
  including `embeddings/`, `scores/`, `calibration/`, `evaluation/` (Phase 7).
- `data/voxceleb/` — VoxCeleb-specific data (Phase 7): `raw/` audio,
  `metadata/`, sampled `trials/` CSVs, and `processed/` artifacts. All
  excluded from Git.
- `docs/` — design and architecture documentation (`architecture.md`,
  `model.md`, `calibration.md`).

## Status / Roadmap

- Phase 1 — Project foundation: **COMPLETE**
- Phase 2 — Environment setup: **COMPLETE**
- Phase 3 — Audio preprocessing: **COMPLETE**
- Phase 4 — ECAPA-TDNN embedding extraction: **COMPLETE** (empirically
  validated on the developer's NVIDIA GeForce RTX 3060 Laptop GPU, CUDA
  12.4: 192-dimensional embeddings)
- Phase 5 — Cosine similarity scoring: **COMPLETE**
- Phase 6 — Calibration (FAR/FRR/EER/ROC-AUC) and verification decision
  layer: **COMPLETE** — and, as of Phase 8B, backed by a **real
  calibrated threshold of 0.252784**, derived from 28,280 VoxCeleb1
  trials (EER 0.87%, ROC-AUC 0.9986, speaker-disjoint evaluation:
  FAR 1.15% / FRR 0.48%). See
  [docs/calibration.md, "Phase 8B"](docs/calibration.md#phase-8b--first-real-calibration-results-voxceleb1)
  for the full results and the three caveats on how to quote them.
- Phase 7 — Large-scale VoxCeleb calibration pipeline: **COMPLETE (code +
  tests)** — **not yet run on real VoxCeleb data.** See
  [Calibration & verification decision](#calibration--verification-decision)
  below and docs/calibration.md, "Phase 7", for what needs to be
  obtained manually (VoxCeleb access is registration-gated, and this
  sandbox cannot download the pretrained model either).
- Deepfake detector integration: not started

Phase 6 note: `calibration.py` and `verifier.py` are fully implemented
and covered by 46 deterministic unit tests (27 + 19) using synthetic,
hand-verified data — see
[Calibration & verification decision](#calibration--verification-decision).
The real-data path (`scripts/calibrate.py`, `scripts/verify.py`) is
exercised end-to-end with a stub encoder in this sandbox (the pretrained
model remains unreachable here — see
[Encoder limitations](#encoder-limitations)); running them against real
audio requires a machine with normal internet access, same as Phases 4–5.

Phase 3 note: audio preprocessing correctness (validation, mono conversion,
resampling) has been verified by unit tests (see
[Audio preprocessing](#audio-preprocessing) below). No claim is made about
how preprocessing affects downstream speaker-verification accuracy — that
can only be measured once the ECAPA-TDNN encoder and evaluation pipeline
(Phases 4–6) exist and are run against real trial data.

Phase 4 note: the pretrained model could not be downloaded and run in
this development sandbox itself (the Hugging Face Hub is blocked by this
sandbox's network policy — see [Encoder limitations](#encoder-limitations)
below); it has since been verified on the developer's own machine (see
above).

Phase 5 note: cosine similarity's mathematics and input validation are
verified by deterministic unit tests using hand-constructed vectors (see
[Cosine similarity](#cosine-similarity) below) — those do not require the
pretrained model and pass in any environment. The real-embeddings
integration test (`scripts/test_similarity.py`) could not be executed in
this sandbox for the same network-access reason as Phase 4. **A raw
cosine similarity score is not a verification result** — see
[Architecture](#architecture) above.

**Not part of the baseline** — a separate, optional, experimental module to
be considered only after the ECAPA-TDNN baseline is fully evaluated:

- Fuzzy logic decision boundaries

## Hardware / device support

This project is designed to run on **CPU by default** and to **automatically
use an available NVIDIA GPU via CUDA** when one is present (`torch.cuda.is_available()`).
No code path requires a GPU. As of Phase 4, the ECAPA-TDNN encoder
(`src/speaker_verification/encoder.py`) is the first component that
actually uses a GPU when one is available, via the same
CUDA-if-available-else-CPU pattern (`encoder.resolve_device()`); GPU
name/index are always read from `torch.cuda` at runtime and never
hard-coded.

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

## Audio preprocessing

`src/speaker_verification/audio.py` implements the input pipeline that the
ECAPA-TDNN encoder (Phase 4) will build on:

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

**Supported input formats:** WAV and FLAC. Other extensions raise a clear
`UnsupportedAudioFormatError`.

**What it does NOT do (by design, this phase):** no denoising or band-pass
filtering, no loudness/amplitude normalization beyond the lossless
integer-to-float PCM scaling every decoder performs, and it never modifies,
moves, or overwrites the source file.

**Error handling:** missing files, corrupt/undecodable files, and
zero-length audio each raise a distinct, clearly-named exception
(`AudioFileNotFoundError`, `AudioDecodeError`, `EmptyAudioError`). Very
short audio is processed rather than rejected, with a `UserWarning` if it
falls under ~100 ms.

**Main functions:** `load_audio(path)` (raw load, no resampling/mono
conversion), `preprocess_audio(path)` (full pipeline above), and
`get_audio_info(path)` (metadata only — sample rate, channels, frame
count, duration — without necessarily decoding the full waveform).

### Running the audio preprocessing tests

Requires `pytest` (not part of `requirements.txt`, which is scoped to
runtime dependencies only):

```bash
pip install pytest
pytest tests/test_audio.py -v
```

Tests generate their own temporary WAV/FLAC fixtures (mono, stereo,
multi-channel, various sample rates, corrupt, empty, very short) — no
external dataset is required.

### Manually inspecting a single file

```bash
python scripts/test_audio_pipeline.py path/to/audio.wav
```

Prints the file's original sample rate, channel count, and duration,
followed by the processed (16 kHz, mono) sample rate, channel count,
duration, and waveform shape — explicitly noting when a stereo/multi-channel
input was converted to mono.

## ECAPA-TDNN speaker encoder

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

**Model:** `speechbrain/spkrec-ecapa-voxceleb` (pretrained only — not
trained or fine-tuned in this project). Full model documentation,
verified API details, and caching location are in
[`docs/model.md`](docs/model.md).

**API:** `SpeakerEncoder()` loads the model once;
`encoder.encode(waveform)` takes an already-preprocessed waveform (from
`audio.py`) and returns a 1-D embedding tensor; `encoder.encode_file(path)`
does both steps for a single file. The rest of the project depends on
this interface, not on SpeechBrain directly.

**Device handling:** CUDA is used automatically when available
(`torch.cuda.is_available()`), otherwise CPU — same pattern as the rest
of this project. An explicit device can be requested via
`SpeakerEncoder(device="cuda")`; requesting CUDA when it's unavailable
raises a clear error instead of silently falling back.

### Running the encoder tests

```bash
pip install pytest
pytest tests/test_encoder.py -v
```

Device-selection tests always run. Tests that need the real pretrained
model are skipped with a clear reason (not failed) if the model can't be
downloaded — see [Encoder limitations](#encoder-limitations).

### Manually inspecting a file (or a pair of files)

```bash
python scripts/test_encoder.py path/to/audio.wav
python scripts/test_encoder.py reference.wav test.wav   # two-file mode
```

Single-file mode prints device, original/processed audio metadata,
embedding shape/dtype/device/norm, and inference time. Two-file mode
extracts an embedding for each file independently and reports their
shapes/statistics side by side — it does **not** compute similarity or a
MATCH/NON_MATCH decision (that's a later phase).

### Encoder limitations

The pretrained model is downloaded from the Hugging Face Hub on first use
and cached locally (see `docs/model.md`). This requires network access.
It could not be downloaded in this project's development sandbox (that
sandbox blocks `huggingface.co` at the network-policy level), so
runtime-verified embedding output (exact values, confirmed shape,
measured inference time) could not be produced there. This is a sandbox
limitation, not a defect in `encoder.py` — re-run
`pytest tests/test_encoder.py -v` and `python scripts/test_encoder.py
<file>` on a machine with normal internet access (such as your own) to
get real, verified results. (This has since been done — see
[Status / Roadmap](#status--roadmap) above.)

## Cosine similarity

`src/speaker_verification/similarity.py` answers exactly one question:
"how similar are these two speaker embeddings?" — nothing more.

```
embedding_a
embedding_b
      ↓
cosine similarity = dot(A, B) / (||A|| · ||B||)
      ↓
single scalar score, range ≈ [-1.0, 1.0]
```

**API:** `cosine_similarity(embedding_a, embedding_b)` — pass two 1-D
embeddings (e.g. from `SpeakerEncoder.encode()`) to get back a Python
`float`; pass two 2-D batches (`(batch, embedding_dim)`) to get back a
`(batch,)` tensor of per-pair scores. Built on
`torch.nn.functional.cosine_similarity` rather than a manual
reimplementation.

**Validation:** raises a specific, clearly-named error for `None`,
non-tensor, non-floating, wrong-dimensionality, empty, mismatched
embedding dimension, mismatched batch size, mismatched device, NaN/Inf,
or zero-(or near-zero-)norm input — a zero-norm embedding raises rather
than silently returning an arbitrary score, since cosine similarity is
undefined for a zero vector.

**Device handling:** both embeddings must already be on the same device
(CPU or CUDA); the computation runs there without unnecessary transfers.
Neither input tensor is modified.

**Scope:** this module does not implement a threshold, a MATCH/NON_MATCH
decision, FAR/FRR/EER/ROC, or calibration — see
[Architecture](#architecture) above. **A raw cosine similarity score has
no established meaning about whether two recordings are from the same
speaker until it is interpreted against a threshold calibrated on
representative genuine/impostor trial data (Phase 6).**

### Running the similarity tests

```bash
pip install pytest
pytest tests/test_similarity.py -v
```

These are deterministic tests against hand-constructed vectors (identical,
scaled, orthogonal, opposite, mismatched, NaN/Inf, zero-vector, batched)
with known expected scores — they do not require the pretrained model and
run in any environment, CPU or CUDA.

### Real ECAPA integration test (two audio files)

```bash
python scripts/test_similarity.py reference.wav test.wav
```

Runs both files through the existing audio preprocessing and
`SpeakerEncoder`, then reports their cosine similarity — device, GPU (if
any), both embeddings' shapes/devices, the similarity score, and the
similarity computation's own timing (measured separately from ECAPA
inference). It prints a raw score only — **no MATCH/NON_MATCH verdict is
produced.**

**Same-speaker / different-speaker testing:** any personal recordings
used for this (e.g. under `tests/recording test/`, which is excluded via
`.gitignore`) are for local, manual pipeline testing only. A same-speaker
score is not an accuracy result, and an observed score should not be
described as "good" or "bad" — that requires the threshold calibration
in Phase 6.

### Similarity limitations

Like the encoder test, `scripts/test_similarity.py` requires downloading
the pretrained model on first use, which this development sandbox's
network policy blocks (see [Encoder limitations](#encoder-limitations)
above) — so a real embeddings-based similarity score could not be
produced here. `similarity.py`'s own logic is fully covered by
deterministic unit tests that don't depend on the model (see above); the
end-to-end integration script should be re-run on a machine with normal
internet access (such as your own, where Phase 4 was already verified)
to get a real score.

## Calibration & verification decision

Full methodology (FAR/FRR/EER definitions, EER selection criterion,
threshold direction, data requirements, and an honest discussion of this
project's current data limitation) lives in
[docs/calibration.md](docs/calibration.md). Summary:

- `src/speaker_verification/calibration.py` — `calibrate(genuine_scores,
  impostor_scores)` computes FAR/FRR at every candidate threshold, the
  EER operating point, and ROC-AUC. It raises
  `InsufficientCalibrationDataError` if either class is empty, and flags
  `is_statistically_reliable=False` with explicit warnings when either
  class has fewer than `MIN_RELIABLE_TRIALS_PER_CLASS` (30) trials —
  never a silent/blocking failure, always a truthful result.
- `src/speaker_verification/verifier.py` — `SpeakerVerifier(encoder,
  threshold, is_calibrated=False)` applies `score >= threshold -> MATCH`
  (else `NON_MATCH`). `threshold` has **no default**; it must always be
  supplied by the caller. `is_calibrated` defaults to `False` so ad-hoc
  thresholds are never mistaken for validated ones.

**This project's real data status (updated in Phase 8B): calibrated on
28,280 real VoxCeleb1 trials, threshold = 0.252784.** Earlier phases had
only one genuine trial (cosine similarity 0.635386) and zero impostor
trials, which `calibrate()` correctly refused to calibrate on. That
single score was never used as a threshold anywhere in this codebase —
and the real calibrated value turned out to be much lower than it, which
is precisely why it was right not to guess.

### Running the calibration tests

```bash
pip install pytest
pytest tests/test_calibration.py tests/test_verifier.py -v
```

46 deterministic tests (27 calibration + 19 verifier) against synthetic,
hand-verified data (perfect separation, full overlap, boundary/tie
conditions, empty/one-sided data, NaN/Inf/out-of-range scores, exact
threshold equality, etc.) — explicitly labeled as synthetic in both test
files' module docstrings, not real speaker-verification evidence.

### Calibrating against real trial data

```bash
python scripts/calibrate.py --trials data/trials.csv
```

Expects a CSV with columns `reference,test,label` (`label` is `genuine`
or `impostor`). If `data/trials.csv` doesn't exist — which is the current
state of this project — the script explains the exact format needed and
exits cleanly (exit code 0), rather than erroring or fabricating a
result.

### Running a verification decision

```bash
python scripts/verify.py reference.wav test.wav --threshold 0.6123 --calibrated
```

`--threshold` is required — there is no default. Omit `--calibrated` for
an ad-hoc/example threshold; the output is then clearly marked
`UNCALIBRATED`.

## Large-scale VoxCeleb calibration (Phase 7)

Phase 6's calibration engine is real and tested, but this project only
has one genuine trial and zero impostor trials of its own — nowhere
near enough for a trustworthy threshold. Phase 7 adds a pipeline that
feeds the same, unmodified `calibration.py` real genuine/impostor
scores from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/),
a standard public speaker-verification dataset, producing an actual
data-driven baseline threshold. **ECAPA-TDNN is not fine-tuned or
modified anywhere in this process.** Full methodology (calibration vs.
evaluation split, speaker-leakage checks, embedding cache design,
Hindi/Hinglish domain limitation) is in
[docs/calibration.md, "Phase 7"](docs/calibration.md#phase-7-large-scale-calibration-using-voxceleb).

**Status: RUN ON REAL DATA (Phases 8B and 8C).** The full pipeline —
4,715 utterances embedded on an RTX 3060 — has been run two ways:

| Run | Trials | EER | Threshold | Purpose |
|---|---|---|---|---|
| **8B** speaker-disjoint | 28,280 | 0.87% | **0.252784** | Chooses the threshold this project ships; held-out FAR 1.15% / FRR 0.48% is the unbiased estimate |
| **8C** full official protocol | 37,720 | 1.04% | 0.272217 | Literature-comparable benchmark only (ROC-AUC 0.998945) |

Both are documented in
[docs/calibration.md, "Phase 8B"](docs/calibration.md#phase-8b--first-real-calibration-results-voxceleb1)
and
["Phase 8C"](docs/calibration.md#phase-8c--full-voxceleb1-official-protocol-benchmark),
including why 8C scores worse than 8B and why 1.04% sits above the
published ~0.80%.
The development sandbox itself still cannot reach VoxCeleb or download
the pretrained model (see [Encoder limitations](#encoder-limitations)),
so these results were produced on the developer's own machine.

### Running the Phase 7 tests

```bash
pip install pytest
pytest tests/test_datasets_trials.py tests/test_datasets_voxceleb.py \
       tests/test_embedding_cache.py tests/test_extract_embeddings.py \
       tests/test_score_trials.py tests/test_run_voxceleb_calibration.py \
       tests/test_build_trials.py tests/test_voxceleb_pipeline_integration.py -v
```

All synthetic (generated trial-list text, tiny WAV fixtures, stub
encoders) — no real VoxCeleb data or pretrained model required.

### Running the pipeline on real data

Once you have obtained VoxCeleb audio and an official trial-list file
yourself (see docs/calibration.md for exactly what to get and where):

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
table so the names can be mapped to ids:

```bash
python scripts/build_trials.py \
    --trials data/raw/voxceleb1/trials/voxceleb1_test.txt \
    --audio-root data/raw/voxceleb1/vox1/test_wav/wav \
    --meta data/raw/voxceleb1/vox1/vox1/vox1_meta.csv \
    --output data/trials/voxceleb1_trials.csv \
    --seed 42 --calibration-fraction 0.5 --split-seed 42
```

Both forms are normalized to the same canonical utterance id internally
— see [docs/calibration.md, "Trial-list formats"](docs/calibration.md#trial-list-formats-numeric-ids-vs-speaker-names).

`build_trials.py` samples a deterministic subset and splits it into
speaker-disjoint calibration/evaluation sets;
`extract_embeddings.py` loads the encoder once and caches embeddings
(skipping any utterance already cached); `score_trials.py` computes
cosine similarity via the existing `similarity.py`;
`run_voxceleb_calibration.py` calibrates on the calibration split only
and reports FAR/FRR of that threshold on the held-out evaluation split,
plus a speaker-leakage check.

### Personal recordings stay separate

```bash
python scripts/score_personal_recording.py
```

Scores this project's own two personal recordings (kept outside Git)
using the same pipeline, explicitly labeled "Personal same-speaker
validation (OUT-OF-DOMAIN)" — never mixed into VoxCeleb calibration and
never used to change the VoxCeleb-derived threshold automatically.
