# Architecture

Status: Phases 1–3 (foundation, environment, audio preprocessing) are
implemented. This document will continue to describe the detailed design
of the speaker verification module as later phases are implemented.

## Pipeline overview

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

## Integration with B.S. Detector

This module is invoked conditionally by the larger B.S. Detector pipeline,
after the deepfake detection stage, as an identity-consistency check on
audio that has already passed deepfake screening.

## Sections to be filled in as implementation proceeds

- Audio preprocessing details (`audio.py`): **done, see above**
- ECAPA-TDNN encoder details (`encoder.py`)
- Similarity scoring details (`similarity.py`)
- Threshold calibration methodology (`calibration.py`): genuine/impostor
  trial scores, threshold calibration, FAR, FRR, EER, ROC/ROC-AUC
- Evaluation methodology (`evaluation.py`)
- B.S. Detector pipeline integration contract

## Out of scope for the baseline

Fuzzy logic decision boundaries are explicitly out of scope for the
ECAPA-TDNN baseline and for `calibration.py`. Fuzzy logic will be considered
only as a separate, optional, experimental module introduced later, after
the baseline has been fully evaluated.
