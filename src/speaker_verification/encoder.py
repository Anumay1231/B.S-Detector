"""
encoder.py
----------

A clean wrapper around a pretrained ECAPA-TDNN speaker-embedding model.

This module is Phase 4 of the speaker-verification pipeline:

    Audio
        |
    Existing audio preprocessing (audio.py)
        |
    Pretrained ECAPA-TDNN (this module)
        |
    Speaker embedding

Model used: ``speechbrain/spkrec-ecapa-voxceleb`` (see docs/model.md for
the source, rationale, and verified API details). The model is used
exactly as published — it is not trained or fine-tuned here.

This module deliberately does NOT implement:
    - cosine similarity or any other embedding comparison
    - threshold calibration or MATCH/NON_MATCH/UNCERTAIN decisions
    - fuzzy logic
    - deepfake detection
Those belong to later phases. This module's only job is:
``preprocessed waveform in -> speaker embedding out``.

Design goals:
    - The model is loaded once per ``SpeakerEncoder`` instance, not on
      every call to ``encode()``/``encode_file()``.
    - CPU is always supported; CUDA is used automatically when available
      (``torch.cuda.is_available()``), with an explicit override available.
    - SpeechBrain-specific types are not exposed outside this module — the
      rest of the project depends on ``SpeakerEncoder`` / ``encode`` /
      ``encode_file``, not on ``speechbrain.inference`` directly.
    - Inference always runs under ``torch.no_grad()`` with the model in
      eval mode, and the returned embedding is detached from the
      computation graph.

Model caching: loading uses SpeechBrain's normal fetch/cache mechanism
(backed by the Hugging Face Hub cache, typically
``~/.cache/huggingface/hub``) unless an explicit ``savedir`` is given.
Nothing is downloaded into this repository, and the standard cache
location is outside of any Git-tracked directory. See docs/model.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

# SpeechBrain-specific import is confined to this module. As of SpeechBrain
# 1.0+, the modern inference interface lives under `speechbrain.inference`;
# the older `speechbrain.pretrained` path is deprecated (it now just
# redirects here with a UserWarning). This was verified against the
# installed speechbrain==1.1.0 package before writing this module.
from speechbrain.inference.speaker import EncoderClassifier

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Hugging Face Hub model id for the pretrained ECAPA-TDNN speaker encoder.
#: See docs/model.md for why this specific model was chosen.
DEFAULT_MODEL_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"

#: Sample rate this model expects its input waveform to already be at.
#: This module does not resample; audio.py's preprocess_audio() guarantees
#: this rate.
EXPECTED_SAMPLE_RATE: int = 16_000


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class EncoderError(Exception):
    """Base class for all errors raised by this module."""


class ModelLoadError(EncoderError):
    """Raised when the pretrained model fails to download or initialize.

    This commonly wraps network errors (e.g. the model could not be
    fetched from the Hugging Face Hub because of no network access, a
    proxy/firewall block, or the Hub being unreachable) as well as
    local errors (corrupt cache, invalid source name).
    """


class DeviceError(EncoderError):
    """Raised when an explicitly requested compute device is invalid or
    unavailable (e.g. ``device="cuda"`` requested but no CUDA-capable GPU
    is visible to PyTorch). Auto-selection never raises this — it silently
    falls back to CPU when CUDA is unavailable.
    """


class InvalidWaveformError(EncoderError):
    """Raised when the input to encode() is not a usable waveform tensor
    (wrong type, wrong shape/dimensionality, empty, or an unsupported
    dtype)."""


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """Describes the compute device a SpeakerEncoder is using.

    Attributes:
        device: The resolved ``torch.device``.
        device_str: String form passed to SpeechBrain (e.g. "cpu", "cuda:0").
        gpu_name: CUDA device name (e.g. "NVIDIA GeForce RTX 3060"), or
            None on CPU.
        gpu_index: CUDA device index, or None on CPU.
    """

    device: torch.device
    device_str: str
    gpu_name: Optional[str]
    gpu_index: Optional[int]


def resolve_device(device: Optional[str] = None) -> DeviceInfo:
    """Resolve the compute device to use.

    Args:
        device: Explicit override, e.g. "cpu", "cuda", or "cuda:0". If
            None (default), CUDA is used automatically when
            ``torch.cuda.is_available()`` is True, otherwise CPU.

    Returns:
        A DeviceInfo describing the resolved device. GPU identity
        (name/index) is never hard-coded — it is always read from
        ``torch.cuda`` at runtime, so this reports whatever GPU is
        actually present (e.g. an RTX 3060 on the developer's machine),
        or None on a CPU-only machine such as this development sandbox.

    Raises:
        DeviceError: If ``device`` explicitly requests CUDA (e.g. "cuda"
            or "cuda:0") but no CUDA-capable GPU is available. This is a
            hardware/environment failure and is never silently hidden;
            callers that want automatic CPU fallback should leave
            ``device=None`` instead.
    """
    if device is None:
        if torch.cuda.is_available():
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise DeviceError(
                f"Device '{device}' was explicitly requested, but no "
                f"CUDA-capable GPU is available in this environment "
                f"(torch.cuda.is_available() is False). Pass device=None "
                f"for automatic CPU fallback instead."
            )

    if torch_device.type == "cuda":
        index = torch_device.index if torch_device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
        return DeviceInfo(
            device=torch_device,
            device_str=f"cuda:{index}",
            gpu_name=gpu_name,
            gpu_index=index,
        )

    return DeviceInfo(
        device=torch_device, device_str="cpu", gpu_name=None, gpu_index=None
    )


# --------------------------------------------------------------------------
# SpeakerEncoder
# --------------------------------------------------------------------------


class SpeakerEncoder:
    """Loads a pretrained ECAPA-TDNN model once and extracts speaker
    embeddings from preprocessed waveforms.

    Example:
        >>> encoder = SpeakerEncoder()  # doctest: +SKIP
        >>> waveform, sr = preprocess_audio("reference.wav")  # doctest: +SKIP
        >>> embedding = encoder.encode(waveform)  # doctest: +SKIP
        >>> embedding.shape  # doctest: +SKIP
        torch.Size([192])
    """

    def __init__(
        self,
        model_source: str = DEFAULT_MODEL_SOURCE,
        device: Optional[str] = None,
        savedir: Optional[str] = None,
    ) -> None:
        """Load the pretrained model. This performs (or reuses a cached)
        model download and should be done once, not per-file.

        Args:
            model_source: Hugging Face Hub model id (or local path)
                understood by SpeechBrain's fetch mechanism. Defaults to
                DEFAULT_MODEL_SOURCE (speechbrain/spkrec-ecapa-voxceleb).
            device: Explicit device override ("cpu", "cuda", "cuda:0").
                If None (default), CUDA is used automatically when
                available, otherwise CPU.
            savedir: Optional explicit local cache directory. If None
                (default), SpeechBrain uses its normal cache mechanism
                (the Hugging Face Hub cache, typically
                ``~/.cache/huggingface/hub``) rather than any
                project-local directory. See docs/model.md.

        Raises:
            DeviceError: If ``device`` explicitly requests CUDA but none
                is available.
            ModelLoadError: If the model cannot be downloaded or
                initialized (e.g. no network access, blocked by a
                firewall/proxy, invalid source, or corrupt cache).
        """
        self.model_source = model_source
        self.device_info = resolve_device(device)

        try:
            self._model: EncoderClassifier = EncoderClassifier.from_hparams(
                source=model_source,
                savedir=savedir,
                run_opts={"device": self.device_info.device_str},
            )
        except DeviceError:
            raise
        except Exception as exc:  # noqa: BLE001 - want to wrap any failure
            raise ModelLoadError(
                f"Failed to load pretrained model '{model_source}': "
                f"{type(exc).__name__}: {exc}. This is commonly caused by "
                f"no network access to the Hugging Face Hub (the model is "
                f"downloaded on first use and then cached). If you are "
                f"behind a firewall/proxy or in an offline/sandboxed "
                f"environment, verify connectivity to huggingface.co, or "
                f"pre-populate the local cache from a machine that has "
                f"network access."
            ) from exc

        # Explicitly ensure eval mode (no dropout/batchnorm updates) even
        # though SpeechBrain's Pretrained already freezes parameters by
        # default — this is a deliberate, documented guarantee rather than
        # relying solely on library defaults.
        self._model.mods.eval()

        # Embedding dimensionality is confirmed empirically on first call
        # to encode(), not assumed here (see encode()).
        self.embedding_dim: Optional[int] = None

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def _validate_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Validate and normalize a waveform to shape (1, num_samples).

        Accepts either (num_samples,) or (1, num_samples). Raises
        InvalidWaveformError for anything else (wrong type, multi-channel,
        empty, non-floating dtype).
        """
        if not isinstance(waveform, torch.Tensor):
            raise InvalidWaveformError(
                f"waveform must be a torch.Tensor, got {type(waveform)!r}."
            )

        if not torch.is_floating_point(waveform):
            raise InvalidWaveformError(
                f"waveform must be a floating-point tensor (got dtype "
                f"{waveform.dtype}). audio.py's preprocess_audio() returns "
                f"float32 as expected; if you constructed this tensor "
                f"yourself, cast it with waveform.float()."
            )

        if waveform.ndim == 1:
            wav = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise InvalidWaveformError(
                    f"Expected a mono waveform (1 channel), got shape "
                    f"{tuple(waveform.shape)} with {waveform.shape[0]} "
                    f"channels. Convert to mono first (see "
                    f"speaker_verification.audio.preprocess_audio)."
                )
            wav = waveform
        else:
            raise InvalidWaveformError(
                f"Expected a 1-D (num_samples,) or 2-D (1, num_samples) "
                f"waveform tensor, got {waveform.ndim}-D shape "
                f"{tuple(waveform.shape)}."
            )

        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise InvalidWaveformError("waveform is empty (zero samples).")

        if not torch.isfinite(wav).all():
            raise InvalidWaveformError(
                "waveform contains non-finite values (NaN or Inf)."
            )

        return wav

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a speaker embedding from a single preprocessed waveform.

        Args:
            waveform: A mono, 16 kHz waveform as produced by
                ``speaker_verification.audio.preprocess_audio`` — a
                ``torch.FloatTensor`` of shape ``(1, num_samples)`` (a
                plain ``(num_samples,)`` tensor is also accepted). This
                function assumes preprocessing (validation, mono
                conversion, resampling) has already happened; it does not
                repeat it.

        Returns:
            A 1-D ``torch.Tensor`` of shape ``(embedding_dim,)`` (192 for
            the standard ECAPA-TDNN VoxCeleb checkpoint, confirmed
            empirically from the model's actual output rather than
            assumed — see docs/model.md), detached from the computation
            graph, on the same device as the model.

        Raises:
            InvalidWaveformError: If ``waveform`` is not a usable mono
                floating-point tensor.
            EncoderError: If the underlying model raises during inference.
        """
        wav = self._validate_waveform(waveform)
        wav = wav.to(self.device_info.device)

        try:
            with torch.no_grad():
                raw_embedding = self._model.encode_batch(wav)
        except Exception as exc:  # noqa: BLE001
            raise EncoderError(
                f"Inference failed on waveform of shape {tuple(wav.shape)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # The model returns embeddings with a batch dimension (and, for
        # this architecture, an extra singleton dimension before pooling,
        # e.g. [batch, 1, embedding_dim]). We only ever pass a batch of 1
        # (a single file's waveform), so we squeeze down to a clean 1-D
        # (embedding_dim,) vector rather than assuming the exact rank.
        embedding = raw_embedding.detach().squeeze()

        if embedding.ndim != 1:
            raise EncoderError(
                f"Unexpected embedding shape after squeezing: "
                f"{tuple(embedding.shape)} (raw shape was "
                f"{tuple(raw_embedding.shape)}). Expected a 1-D vector; "
                f"the model's output layout may differ from what this "
                f"wrapper assumes."
            )

        if self.embedding_dim is None:
            self.embedding_dim = embedding.shape[0]

        return embedding

    def encode_file(self, path: str) -> torch.Tensor:
        """Load and preprocess an audio file (via
        ``speaker_verification.audio.preprocess_audio``), then extract its
        speaker embedding.

        Args:
            path: Path to a WAV or FLAC file.

        Returns:
            Same as ``encode()``: a 1-D embedding tensor.

        Raises:
            speaker_verification.audio.AudioError: For any audio
                loading/preprocessing failure (missing file, unsupported
                format, corrupt/empty audio) — see audio.py.
            InvalidWaveformError, EncoderError: For encoding failures.
        """
        # Local import to avoid a hard import-time dependency cycle and to
        # keep this module's top-level imports focused on the encoder
        # itself; audio.py has no knowledge of this module.
        from speaker_verification.audio import preprocess_audio

        waveform, _sample_rate = preprocess_audio(path)
        return self.encode(waveform)

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    def describe_device(self) -> str:
        """Human-readable description of the device this encoder is
        running on, e.g. "cuda:0 (NVIDIA GeForce RTX 3060)" or "cpu"."""
        if self.device_info.device.type == "cuda":
            return f"{self.device_info.device_str} ({self.device_info.gpu_name})"
        return self.device_info.device_str


# --------------------------------------------------------------------------
# Convenience timing helper (used by scripts/test_encoder.py)
# --------------------------------------------------------------------------


def timed_encode(encoder: SpeakerEncoder, waveform: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Run ``encoder.encode(waveform)`` and return ``(embedding,
    elapsed_seconds)``. CUDA calls are synchronized before stopping the
    timer so the measurement reflects actual GPU compute time, not just
    kernel-launch time.
    """
    if encoder.device_info.device.type == "cuda":
        torch.cuda.synchronize(encoder.device_info.device)
    start = time.perf_counter()
    embedding = encoder.encode(waveform)
    if encoder.device_info.device.type == "cuda":
        torch.cuda.synchronize(encoder.device_info.device)
    elapsed = time.perf_counter() - start
    return embedding, elapsed
