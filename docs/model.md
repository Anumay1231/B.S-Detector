# Pretrained model: speechbrain/spkrec-ecapa-voxceleb

This document records what was verified about the pretrained speaker
embedding model used in Phase 4, where that information came from, and
what remains to be confirmed on a machine with normal internet access.

## Model identity and source

- **Model id / source:** `speechbrain/spkrec-ecapa-voxceleb`
- **Hosted at:** Hugging Face Hub
  (https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Architecture:** ECAPA-TDNN (Emphasized Channel Attention, Propagation
  and Aggregation - Time Delay Neural Network), trained for speaker
  recognition on VoxCeleb1 + VoxCeleb2.
- **Why this model:** it is the model explicitly requested for this
  project, and it is SpeechBrain's standard, widely used pretrained
  ECAPA-TDNN speaker-embedding checkpoint. No substitute model was used.
- **Usage here:** pretrained, frozen weights only. This project does not
  train or fine-tune this model in any phase so far (including Phase 7 —
  see docs/calibration.md, "Phase 7": the VoxCeleb calibration pipeline
  only generates scores with this unmodified model, it never updates its
  weights).

## Phase 7 caveat: the model was itself trained on VoxCeleb

`speechbrain/spkrec-ecapa-voxceleb` was trained on VoxCeleb1 + VoxCeleb2
(see "Architecture" above). Phase 7 calibrates a threshold using
VoxCeleb trial data. This means VoxCeleb-based calibration/evaluation
numbers are **not** an independent, out-of-domain test of the model —
the model has already seen VoxCeleb-domain speech (though the official
VoxCeleb1 test-trial protocol's speakers are held out from VoxCeleb1's
own training/dev split, so it is not literally training-on-test-set,
depending on which VoxCeleb2 training recipe was used to produce this
specific checkpoint, which has not been independently re-verified
here). Treat any VoxCeleb-derived EER/threshold as a best-case,
in-domain baseline, not a guarantee of the same performance on
genuinely novel audio (such as this project's actual Hindi/Hinglish
target domain — see docs/calibration.md, "Phase 7M").

## Verified SpeechBrain API (installed version: speechbrain==1.1.0)

The interface was verified by introspecting the actually-installed
package (not assumed from older tutorials):

- **Current interface (used by this project):**
  ```python
  from speechbrain.inference.speaker import EncoderClassifier
  model = EncoderClassifier.from_hparams(
      source="speechbrain/spkrec-ecapa-voxceleb",
      run_opts={"device": "cpu"},  # or "cuda", "cuda:0", ...
  )
  embeddings = model.encode_batch(wavs)  # wavs: [batch, time], 16 kHz
  ```
- **Deprecated interface (NOT used):** `from speechbrain.pretrained import
  EncoderClassifier`. Importing `speechbrain.pretrained` in the installed
  1.1.0 package raises `UserWarning: Module 'speechbrain.pretrained' was
  deprecated, redirecting to 'speechbrain.inference'` — confirmed by
  direct import in this environment. `speechbrain.inference.speaker` is
  the current, non-deprecated path.
- `EncoderClassifier.encode_batch(wavs, wav_lens=None, normalize=False)`
  (signature and docstring read directly from the installed package):
  expects `wavs` of shape `[batch, time]` (or `[batch, time, channels]`
  depending on the model) at **16,000 Hz**, and returns the encoded batch
  as a `torch.Tensor`.

## Expected input

- **Sample rate:** 16,000 Hz (matches `TARGET_SAMPLE_RATE` in
  `audio.py` / `EXPECTED_SAMPLE_RATE` in `encoder.py` — no mismatch).
- **Channel format:** mono. `audio.py`'s `preprocess_audio()` already
  guarantees mono, 16 kHz output, so `encoder.py` performs no further
  preprocessing.
- **Tensor shape into `encode_batch`:** `[batch, time]`. Our
  `preprocess_audio()` output has shape `(1, num_samples)` — i.e. "1
  channel, N samples" — which, once reinterpreted as "1 waveform in the
  batch, N samples," is exactly the `[batch=1, time=N]` shape
  `encode_batch` expects. `encoder.py` passes it through directly (after
  validating it) rather than duplicating audio.py's preprocessing.

## Embedding output

- **Documented/expected dimensionality:** 192-dimensional, per the
  published ECAPA-TDNN VoxCeleb checkpoint design (a widely cited value
  for this specific SpeechBrain checkpoint).
- **Raw output shape:** `encode_batch` returns a batched tensor; for this
  architecture the commonly reported raw shape is `[batch, 1, 192]` (an
  extra singleton dimension before the final pooling squeeze). `encoder.py`
  does not hard-code this — it squeezes the raw output and then verifies
  the result is 1-D at runtime, raising `EncoderError` if it isn't (see
  `SpeakerEncoder.encode`).
- **This project's documented representation:** a 1-D tensor of shape
  `(192,)` for a single input waveform.
- **Verification status:** the exact runtime shape and values could not
  be executed and confirmed in the sandbox this code was developed in —
  see [Known limitation](#known-limitation-model-could-not-be-downloaded-in-this-sandbox)
  below. `tests/test_encoder.py::test_embedding_has_expected_dimension`
  asserts `embedding.shape[0] == 192` and should be run on a machine with
  normal network access to confirm this against the real model.
  **Update:** this has since been empirically confirmed by the developer
  on their own machine (NVIDIA GeForce RTX 3060 Laptop GPU, CUDA 12.4):
  the model loads successfully and produces 192-dimensional embeddings,
  matching what was documented here ahead of time.

## Device handling

- CUDA is used automatically when `torch.cuda.is_available()` is `True`;
  otherwise CPU is used. See `encoder.resolve_device()`.
- An explicit device can be requested (`SpeakerEncoder(device="cuda")`);
  if CUDA is explicitly requested but unavailable, a clear `DeviceError`
  is raised rather than silently falling back (auto mode, i.e.
  `device=None`, is what silently falls back to CPU).
- The GPU name/index are always read from `torch.cuda` at runtime and are
  never hard-coded.

## Model caching

- Loading uses SpeechBrain's normal `fetch()`/cache mechanism. When no
  explicit `savedir` is given (the default in `SpeakerEncoder`),
  SpeechBrain's own docs state it will "just use cache" — backed by the
  Hugging Face Hub's standard local cache, typically
  `~/.cache/huggingface/hub` (Linux/macOS) or
  `%USERPROFILE%\.cache\huggingface\hub` (Windows).
- This is **outside** the project repository, so nothing is committed to
  Git and no repo-local cache directory needed to be added. `.gitignore`
  additionally excludes `pretrained_models/` and `model_cache/` as a
  defensive measure in case a caller ever passes an explicit local
  `savedir`.
- The model is downloaded once on first use and reused from cache on
  subsequent runs/processes.

## Speaker verification API (not used)

SpeechBrain also exposes `speechbrain.inference.speaker.SpeakerRecognition`,
which wraps `EncoderClassifier` plus its own built-in similarity/threshold
decision. This project deliberately does **not** use `SpeakerRecognition`
anywhere: `encoder.py` only extracts embeddings via `EncoderClassifier`.
Cosine similarity (`similarity.py`, Phase 5) and the calibration/decision
layer (`calibration.py`, `verifier.py`, Phase 6) are this project's own
implementations, built so the threshold is always caller-supplied and
traceable to real calibration data rather than an opaque library default
— see `docs/architecture.md` and `docs/calibration.md`.

## Known limitation: model could not be downloaded in this sandbox

The development sandbox this project was built in blocks network access
to `huggingface.co` at the proxy/firewall level (confirmed directly:
`curl -I https://huggingface.co` returns `403 Forbidden` /
`X-Proxy-Error: blocked-by-allowlist`, and
`EncoderClassifier.from_hparams(...)` fails with `ProxyError: 403
Forbidden` for the same reason). This means:

- The pretrained model could not actually be downloaded or run in this
  sandbox, so embedding shape/values/inference time could not be
  empirically measured here.
- `encoder.py` was written and validated against the **actual installed
  SpeechBrain 1.1.0 API** (via direct introspection of the installed
  package's classes/docstrings/source, not assumed from memory or older
  tutorials), and its waveform-validation/device-selection/error-handling
  logic was tested directly (see `tests/test_encoder.py`).
- Tests that require the real model download are written to call
  `pytest.skip(...)` with a clear, specific reason when this happens,
  rather than fail or fabricate results.
- This is an environment limitation, not a code defect, and should not be
  treated as a Phase 4 failure. **Re-run `tests/test_encoder.py` and
  `scripts/test_encoder.py` on a machine with normal internet access**
  (such as the developer's own machine) to get the first real,
  empirically-verified embedding output from this model.

## Cross-lingual accuracy caveat

`speechbrain/spkrec-ecapa-voxceleb` is trained and evaluated on VoxCeleb,
which is predominantly English speech. This project involves
Hindi/Hinglish (bilingual) speech. **No claim is made here that this
model performs comparably well on Hindi/Hinglish audio just because it
performs well on VoxCeleb** — cross-lingual/cross-accent performance for
this use case has not been measured and must be evaluated experimentally
once representative Hindi/Hinglish genuine and impostor trial data exists
(see `docs/calibration.md` for the calibration data requirement this
also blocks — the same missing data prevents both a valid threshold and
a cross-lingual accuracy claim).
