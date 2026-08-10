# Architecture

Status: Phases 1–4 (foundation, environment, audio preprocessing,
ECAPA-TDNN speaker encoder) are implemented. This document will continue
to describe the detailed design of the speaker verification module as
later phases are implemented.

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

## Integration with B.S. Detector

This module is invoked conditionally by the larger B.S. Detector pipeline,
after the deepfake detection stage, as an identity-consistency check on
audio that has already passed deepfake screening.

## Sections to be filled in as implementation proceeds

- Audio preprocessing details (`audio.py`): **done, see above**
- ECAPA-TDNN encoder details (`encoder.py`): **done, see above and docs/model.md**
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
