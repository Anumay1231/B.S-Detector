# Architecture

Status: Phases 1–7 (foundation, environment, audio preprocessing,
ECAPA-TDNN speaker encoder, cosine similarity, calibration and the
verification decision layer, and a large-scale VoxCeleb calibration
pipeline) are implemented. This document will continue to describe the
detailed design of the speaker verification module as later phases are
implemented.

## Pipeline overview

```
Reference Audio                          Test Audio
      ↓                                        ↓
Preprocessing (audio.py)              Preprocessing (audio.py)
      ↓                                        ↓
ECAPA-TDNN encoder                    ECAPA-TDNN encoder
      ↓                                        ↓
Embedding A (192-D)                  Embedding B (192-D)
      │                                        │
      └──────────── Cosine Similarity ─────────┘
                          ↓
                   Similarity Score
                          ↓
                     Calibration
          (genuine + impostor trials → FAR/FRR/EER)
                          ↓
                      Threshold
                          ↓
                  MATCH / NON_MATCH
```

**As of Phase 5, the pipeline stopped at "Similarity Score."** Phase 6
added the remaining stages in code
(`src/speaker_verification/calibration.py`,
`src/speaker_verification/verifier.py`), and **as of Phase 8B the
pipeline is complete end to end with a real calibrated threshold of
0.252784**, derived from 28,280 VoxCeleb1 trials (EER 0.87%). Until
then this project had only one genuine trial and zero impostor trials,
which was insufficient data to calibrate (see
[Calibration & verification decision](#calibration--verification-decision-phase-6--implemented)
below and [docs/calibration.md](calibration.md) for the full
methodology). A raw similarity score is still not, by itself, a
verification decision — it only becomes one once paired with a threshold
that the caller supplies explicitly, and that threshold is only
meaningful if it came from real calibration on representative data.

## Audio preprocessing (Phase 3 — implemented)

Expanding the "Audio Preprocessing" step above, as implemented in
`src/speaker_verification/audio.py`:

```
Audio File
    ↓
Validation
    ↓
Load
    ↓
Mono Conversion
    ↓
16 kHz Resampling
    ↓
Preprocessed Waveform
    ↓
[ECAPA-TDNN will be added in Phase 4]
```

- **Validation**: confirms the file exists, is a regular file, has a
  supported extension (WAV/FLAC), and can actually be decoded — each
  failure mode raises a distinct exception (`AudioFileNotFoundError`,
  `UnsupportedAudioFormatError`, `AudioDecodeError`, `EmptyAudioError`).
- **Load**: reads native sample rate and channel count via `soundfile`,
  returning a `torch.FloatTensor` of shape `(channels, num_samples)`.
- **Mono Conversion**: multi-channel audio is averaged across channels
  into a single channel.
- **16 kHz Resampling**: `torchaudio.functional.resample` is used to
  resample to exactly 16,000 Hz; skipped when the source is already
  16 kHz.
- **Preprocessed Waveform**: final output is a CPU `torch.FloatTensor` of
  shape `(1, num_samples)` at 16,000 Hz — ready to be moved to a GPU
  (`.to(device)`) once the encoder exists.

No filtering or loudness normalization is applied. The source file is only
ever opened for reading.

Device selection (CPU vs CUDA) is intentionally **not** handled in
`audio.py`. Hardware selection will be handled by the ECAPA-TDNN encoder in
Phase 4, following a "use CUDA if `torch.cuda.is_available()`, else CPU"
pattern so the same code runs unmodified in this GPU-less development
sandbox and on the developer's RTX 3060 machine.

## ECAPA-TDNN speaker encoder (Phase 4 — implemented)

Expanding the "ECAPA-TDNN" step in the pipeline overview above, as
implemented in `src/speaker_verification/encoder.py`:

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

- **Model**: `speechbrain/spkrec-ecapa-voxceleb`, a pretrained (not
  fine-tuned here) ECAPA-TDNN checkpoint. Full details, verified API, and
  known limitations are documented separately in `docs/model.md`.
- **API used**: `speechbrain.inference.speaker.EncoderClassifier` — the
  current, non-deprecated SpeechBrain 1.x interface (verified against the
  installed `speechbrain==1.1.0` package; the older
  `speechbrain.pretrained` path now just redirects here with a
  deprecation warning).
- **Our interface**: the rest of the project depends on
  `speaker_verification.encoder.SpeakerEncoder` (`encode(waveform)` /
  `encode_file(path)`), not on SpeechBrain types directly — SpeechBrain
  is an implementation detail confined to `encoder.py`.
- **Separation of concerns**: `encoder.py` assumes its input has already
  been validated, mono-converted, and resampled to 16 kHz by `audio.py`;
  it does not repeat any of that.
- **Model loaded once**: `SpeakerEncoder.__init__` loads the model a
  single time; `encode()`/`encode_file()` reuse the loaded model rather
  than reloading per call.
- **Device handling**: CUDA is used automatically when
  `torch.cuda.is_available()`, otherwise CPU — see `resolve_device()` in
  `encoder.py`. An explicit device can be requested; requesting CUDA
  explicitly when unavailable raises a clear `DeviceError` rather than
  silently falling back. GPU identity (name/index) is always read from
  `torch.cuda` at runtime, never hard-coded.
- **Inference guarantees**: model in eval mode, `torch.no_grad()` during
  inference, returned embedding detached from the computation graph.
- **Output**: a 1-D `torch.Tensor` of shape `(192,)` per input waveform
  (see `docs/model.md` for how this shape was determined and its
  verification status).
- **Model caching**: SpeechBrain's normal fetch/cache mechanism is used
  (Hugging Face Hub cache, typically `~/.cache/huggingface/hub`) — no
  model files are stored inside this repository.

This phase does **not** implement cosine similarity, threshold
calibration, FAR/FRR/EER/ROC, MATCH/NON_MATCH/UNCERTAIN decisions, fuzzy
logic, or deepfake detection — those remain later phases (see below).

## Cosine similarity (Phase 5 — implemented)

Expanding the "Cosine Similarity" step in the pipeline overview above, as
implemented in `src/speaker_verification/similarity.py`:

```
embedding_a
embedding_b
      ↓
cosine similarity = dot(A, B) / (||A|| · ||B||)
      ↓
single scalar score, range ≈ [-1.0, 1.0]
```

- **API**: `cosine_similarity(embedding_a, embedding_b)` — accepts a
  single pair of 1-D embeddings (returns a Python `float`) or a batch of
  2-D embeddings (returns a `(batch,)` tensor of per-pair scores).
- **Implementation**: uses `torch.nn.functional.cosine_similarity`
  (PyTorch's established, numerically-stable implementation) rather than
  a manual re-implementation of the dot-product/norm formula.
- **Validation**: rejects `None`, non-tensor, non-floating-point,
  wrong-dimensionality, empty, mismatched-dimension, mismatched-device,
  NaN/Inf, and zero-(or near-zero-)norm inputs with a specific exception
  type and a clear message. A zero-norm embedding raises
  `InvalidEmbeddingError` rather than silently returning an arbitrary
  score — cosine similarity is mathematically undefined for a zero
  vector.
- **Purity**: neither input tensor is modified; no unnecessary device
  transfers are performed (both embeddings must already be on the same
  device — CPU or CUDA — and the computation runs there; only the final
  scalar is copied to host memory, unavoidable for a Python `float`
  return value).
- **Scope**: this module answers only "how similar are these two
  embeddings?" — it does not threshold, calibrate, or decide anything.
  `speaker_verification.verifier` (Phase 6, see below) is where a
  similarity score, once calibrated, feeds into an actual decision.

## Calibration & verification decision (Phase 6 — implemented)

Expanding the "Calibration," "Threshold," and "MATCH / NON_MATCH" steps
in the pipeline overview above. Full methodology (FAR/FRR/EER math, EER
interpolation method, threshold direction, statistical-reliability rule
of thumb, and this project's real data status) is documented in
[docs/calibration.md](calibration.md); this section covers the
architectural contract.

**`src/speaker_verification/calibration.py`** — pure score-in,
statistics-out. Takes labeled genuine and impostor cosine-similarity
scores (produced upstream by `similarity.py`; this module never touches
audio or embeddings directly) and computes:

- `compute_far(impostor_scores, threshold)` / `compute_frr(genuine_scores,
  threshold)` — single-threshold error rates.
- `compute_far_frr_curve(genuine_scores, impostor_scores)` — the full
  FAR/FRR curve across all candidate thresholds (also usable as an ROC
  curve).
- `compute_eer(genuine_scores, impostor_scores)` — the Equal Error Rate
  operating point, via exact match → linear interpolation across the
  FAR−FRR sign change → nearest-point fallback, in that order of
  preference (never a naive average of two arbitrary points).
- `compute_roc_auc(operating_points)` — trapezoidal-rule AUC.
- `calibrate(genuine_scores, impostor_scores) -> CalibrationResult` — the
  main entry point, bundling all of the above plus a statistical
  reliability flag (`MIN_RELIABLE_TRIALS_PER_CLASS = 30` per class, a
  documented rule of thumb) and human-readable warnings.

**Threshold direction is fixed and explicit everywhere:**
`score >= threshold -> MATCH`, `score < threshold -> NON_MATCH`. This is
the same convention used by `compute_far`/`compute_frr` and by
`SpeakerVerifier`.

**`calibrate()` never fabricates a result.** If either the genuine or
impostor score list is empty, it raises
`InsufficientCalibrationDataError` rather than returning a partial or
default threshold — this is the exact code path exercised by this
project's real current data (one genuine trial, zero impostor trials).

**`src/speaker_verification/verifier.py`** — the decision layer.
`SpeakerVerifier(encoder, threshold, is_calibrated=False)` composes the
existing `encoder.py` and `similarity.py` modules (`verify_files()`
preprocesses → encodes → compares, exactly like the earlier phases'
scripts did manually) and applies the threshold rule above to produce a
`VerificationResult` (`MATCH`/`NON_MATCH`, the raw score, the threshold
used, and whether that threshold `is_calibrated`). `threshold` has **no
default value** — `SpeakerVerifier` cannot be constructed without the
caller supplying one explicitly, by design, so a real or ad-hoc value is
never silently assumed. `is_calibrated` defaults to `False` so that
ad-hoc/example thresholds are never confused with ones derived from real
`calibrate()` output.

**Real-data integration**: `scripts/calibrate.py` reads a labeled
`reference,test,label` trial CSV, scores every pair with the real
encoder + `cosine_similarity`, and calls `calibrate()`; `scripts/verify.py`
runs a single reference/test pair through `SpeakerVerifier` given an
explicit `--threshold`. Neither script has a default/fallback threshold.

This phase does not implement fuzzy logic, ML-based classification, or
ECAPA-TDNN fine-tuning — see [Out of scope for the baseline](#out-of-scope-for-the-baseline)
below.

## Large-scale VoxCeleb calibration pipeline (Phase 7 — implemented)

Phase 7 does not add a new stage to the core pipeline above -- it adds
a parallel, offline **data pipeline** whose only job is to produce real
genuine/impostor scores at scale for the EXISTING "Calibration" stage,
using the EXISTING encoder/similarity/calibration code unmodified:

```
VoxCeleb trial-list file (official format, obtained by the user)
        |
src/speaker_verification/datasets/voxceleb.py  (parse only, no audio touched)
        |
src/speaker_verification/datasets/trials.py    (validate -> sample -> speaker-disjoint split)
        |
scripts/build_trials.py  -->  standardized trials CSV
        |
scripts/extract_embeddings.py
        |   (encoder.py, audio.py -- UNCHANGED from Phase 3/4)
        |   (src/speaker_verification/embedding_cache.py -- new, caches by utterance id)
        v
scripts/score_trials.py
        |   (similarity.py -- UNCHANGED from Phase 5)
        v
scripts/run_voxceleb_calibration.py
        |   (calibration.py -- UNCHANGED from Phase 6, called only on the
        |    calibration split; evaluation split measures FAR/FRR of the
        |    resulting threshold)
        v
Calibration report
```

- **Two trial-list conventions**: `voxceleb.py` accepts both the
  canonical numeric-id form (`id10270/x6uYqmx31kE/00001.wav`) and the
  speaker-name form (`Eartha_Kitt/x6uYqmx31kE_0000001.wav`) used by the
  mirrored `voxceleb1_test.txt`. The latter is resolved via VGG's
  `vox1_meta.csv` (passed as `--meta`), never by guessing; both are
  normalized to the canonical id before any downstream stage sees them.
  See docs/calibration.md, "Trial-list formats".
- **Dataset adapter pattern**: `src/speaker_verification/datasets/`
  splits into a dataset-agnostic `trials.py` (the `Trial` dataclass,
  sampling, speaker-disjoint calibration/evaluation splitting,
  file-existence validation) and a VoxCeleb-specific `voxceleb.py`
  (trial-list text parsing, VoxCeleb path/speaker-id conventions). A
  future second dataset would add a sibling adapter module, not touch
  `trials.py` or the pipeline scripts.
- **No embeddings are computed during parsing**: `voxceleb.py` and
  `trials.py` do pure text/metadata handling; embedding extraction is a
  fully separate stage (`extract_embeddings.py`), per Phase 7A's
  explicit separation-of-concerns requirement.
- **Embedding cache**: `src/speaker_verification/embedding_cache.py`
  avoids recomputing an embedding for any utterance already processed
  (including one referenced by multiple trials). See its module
  docstring for the format choice (consolidated `torch.save()` dict +
  JSON-Lines manifest) and its documented scale ceiling.
- **Calibration/evaluation split**: threshold selection
  (`calibration.calibrate()`) and threshold evaluation
  (`calibration.compute_far()`/`compute_frr()`) are always run on
  disjoint trial sets, split at the speaker level
  (`trials.split_trials_by_speaker()`), with an independent
  leakage re-check (`trials.speaker_overlap()`). See
  docs/calibration.md, "Phase 7", for the full methodology and the
  distinction from VGG's own (unsplit) official protocol.
- **Personal recordings**: `scripts/score_personal_recording.py` reuses
  this same pipeline for this project's own two recordings, but keeps
  that score entirely separate from VoxCeleb calibration (labeled
  out-of-domain/project-specific validation) -- see docs/calibration.md.

This phase does not implement fuzzy logic, ML-based classification,
ECAPA-TDNN fine-tuning, distributed computing, a vector database, or
FAISS -- see [Out of scope for the baseline](#out-of-scope-for-the-baseline)
and docs/calibration.md, "Phase 7", "Explicitly out of scope for Phase 7".

## Integration with B.S. Detector

This module is invoked conditionally by the larger B.S. Detector pipeline,
after the deepfake detection stage, as an identity-consistency check on
audio that has already passed deepfake screening.

## Sections to be filled in as implementation proceeds

- Audio preprocessing details (`audio.py`): **done, see above**
- ECAPA-TDNN encoder details (`encoder.py`): **done, see above and docs/model.md**
- Similarity scoring details (`similarity.py`): **done, see above**
- Threshold calibration and verification decision (`calibration.py`,
  `verifier.py`): **done, see above and docs/calibration.md** — code,
  tests, and a real calibrated threshold (0.252784) from 28,280
  VoxCeleb1 trials
- Evaluation methodology (`evaluation.py`)
- B.S. Detector pipeline integration contract

## Out of scope for the baseline

Fuzzy logic decision boundaries and ML-based classification are
explicitly out of scope for the ECAPA-TDNN baseline, including
`calibration.py` and `verifier.py`. Fuzzy logic will be considered only
as a separate, optional, experimental module introduced later, after the
baseline has been fully evaluated. ECAPA-TDNN fine-tuning is likewise not
part of this baseline — only the pretrained checkpoint is used.
