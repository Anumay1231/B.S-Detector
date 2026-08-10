"""
audio.py
--------

Audio loading and preprocessing for the speaker-verification pipeline.

This module is the input layer that later phases (ECAPA-TDNN embedding
extraction) will build on. It is intentionally self-contained: it does not
import torch's CUDA machinery, does not select a compute device, and does
not know anything about embeddings, similarity, or calibration. Device
placement (CPU vs GPU) is the responsibility of the encoder module added
in a later phase — this module only guarantees that its output is a plain
CPU `torch.Tensor` that can be trivially moved to a GPU later via
`.to(device)`.

Supported input formats (Phase 3 baseline): WAV and FLAC.

Pipeline implemented here:

    Audio File
        |
    Validation (exists, supported extension, decodable)
        |
    Load (native sample rate, native channel count)
        |
    Mono conversion (average channels, if more than one)
        |
    Resample to 16,000 Hz (skipped if already 16 kHz)
        |
    Preprocessed waveform: torch.FloatTensor of shape (1, num_samples)

Design notes / explicit non-goals for this phase:
    - No aggressive filtering (no denoising, no band-pass filtering).
    - No loudness/amplitude normalization beyond the lossless integer ->
      float PCM scaling that soundfile performs on read (mapping full-scale
      PCM samples to the [-1.0, 1.0] float range). This is a format
      conversion, not a gain/normalization step.
    - The original source file is only ever opened for reading; it is
      never modified, moved, or overwritten by any function in this
      module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import soundfile as sf
import torch
import torchaudio

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Target sample rate that preprocess_audio() always resamples to.
TARGET_SAMPLE_RATE: int = 16_000

#: File extensions accepted by this module in the Phase 3 baseline.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac"})

#: Below this duration, audio is still processed (never silently dropped)
#: but a warning is attached to the AudioInfo / raised via UserWarning,
#: since very short clips may not carry enough signal for reliable speaker
#: verification later. This is informational only and never blocks
#: processing.
MIN_RECOMMENDED_DURATION_SECONDS: float = 0.10


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class AudioError(Exception):
    """Base class for all audio-preprocessing errors raised by this module."""


class AudioFileNotFoundError(AudioError):
    """Raised when the given audio path does not exist or is not a file."""


class UnsupportedAudioFormatError(AudioError):
    """Raised when the file extension is not one of SUPPORTED_EXTENSIONS."""


class AudioDecodeError(AudioError):
    """Raised when a file cannot be read/decoded as audio (corrupt file,
    truncated header, or content that does not match its extension)."""


class EmptyAudioError(AudioError):
    """Raised when a file decodes successfully but contains zero audio
    frames (e.g. a valid WAV header with no sample data)."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioInfo:
    """Metadata about an audio file, without necessarily loading the full
    waveform into memory.

    Attributes:
        path: Path to the audio file, as given.
        sample_rate: Native sample rate of the file, in Hz.
        num_channels: Number of channels in the file (1 = mono).
        num_frames: Number of sample frames (per channel).
        duration_seconds: Duration of the audio, in seconds.
        subtype: Underlying sample encoding reported by libsndfile, e.g.
            "PCM_16" or "FLOAT". Useful for diagnostics.
    """

    path: str
    sample_rate: int
    num_channels: int
    num_frames: int
    duration_seconds: float
    subtype: str


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _validate_path(path: str) -> None:
    """Validate that ``path`` exists, is a regular file, and has a
    supported extension. Raises AudioFileNotFoundError or
    UnsupportedAudioFormatError with a clear message; does not attempt to
    open or decode the file.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a string or os.PathLike, got {type(path)!r}")

    path_str = os.fspath(path)

    if not os.path.exists(path_str):
        raise AudioFileNotFoundError(f"Audio file not found: '{path_str}'")

    if not os.path.isfile(path_str):
        raise AudioFileNotFoundError(
            f"Path exists but is not a regular file: '{path_str}'"
        )

    ext = os.path.splitext(path_str)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}' for file '{path_str}'. "
            f"Supported extensions: {supported}."
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def get_audio_info(path: str) -> AudioInfo:
    """Inspect an audio file and return its metadata without loading the
    full waveform into memory.

    Args:
        path: Path to a WAV or FLAC file.

    Returns:
        An AudioInfo describing the file's native sample rate, channel
        count, frame count, duration, and sample subtype.

    Raises:
        AudioFileNotFoundError: If the path does not exist or is not a file.
        UnsupportedAudioFormatError: If the file extension is not WAV/FLAC.
        AudioDecodeError: If the file cannot be parsed as audio (corrupt
            file, truncated/invalid header, or content that does not
            match its extension).
    """
    _validate_path(path)
    path_str = os.fspath(path)

    try:
        info = sf.info(path_str)
    except Exception as exc:  # soundfile raises RuntimeError/LibsndfileError
        raise AudioDecodeError(
            f"Could not decode audio file '{path_str}': {exc}"
        ) from exc

    return AudioInfo(
        path=path_str,
        sample_rate=info.samplerate,
        num_channels=info.channels,
        num_frames=info.frames,
        duration_seconds=(info.frames / info.samplerate) if info.samplerate else 0.0,
        subtype=info.subtype,
    )


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file exactly as stored, with no preprocessing.

    The source file is opened read-only and is never modified.

    Args:
        path: Path to a WAV or FLAC file.

    Returns:
        A tuple ``(waveform, sample_rate)`` where ``waveform`` is a
        ``torch.FloatTensor`` of shape ``(num_channels, num_samples)``
        with values in the ``[-1.0, 1.0]`` range (standard PCM-to-float
        scaling, not loudness normalization), and ``sample_rate`` is the
        file's native sample rate in Hz.

    Raises:
        AudioFileNotFoundError: If the path does not exist or is not a file.
        UnsupportedAudioFormatError: If the file extension is not WAV/FLAC.
        AudioDecodeError: If the file cannot be decoded.
        EmptyAudioError: If the file decodes successfully but contains zero
            audio frames.
    """
    _validate_path(path)
    path_str = os.fspath(path)

    try:
        # always_2d guarantees shape (num_frames, num_channels) even for
        # mono files, so downstream shape handling is uniform.
        samples, sample_rate = sf.read(path_str, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioDecodeError(
            f"Could not decode audio file '{path_str}': {exc}"
        ) from exc

    if samples.shape[0] == 0:
        raise EmptyAudioError(
            f"Audio file '{path_str}' contains zero audio frames."
        )

    # samples: (num_frames, num_channels) -> torch convention (channels, num_frames)
    waveform = torch.from_numpy(samples.T).contiguous()
    return waveform, sample_rate


def preprocess_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file and prepare it for the speaker encoder: convert
    to mono and resample to TARGET_SAMPLE_RATE (16,000 Hz).

    This is the main entry point later phases (the ECAPA-TDNN encoder)
    should call. It performs no filtering, no loudness normalization, and
    never modifies the source file.

    Args:
        path: Path to a WAV or FLAC file.

    Returns:
        A tuple ``(waveform, sample_rate)`` where ``waveform`` is a
        ``torch.FloatTensor`` of shape ``(1, num_samples)`` (mono,
        16,000 Hz) and ``sample_rate`` is always ``TARGET_SAMPLE_RATE``.
        The tensor lives on CPU; callers may move it to a GPU later with
        ``waveform.to(device)``.

    Raises:
        AudioFileNotFoundError: If the path does not exist or is not a file.
        UnsupportedAudioFormatError: If the file extension is not WAV/FLAC.
        AudioDecodeError: If the file cannot be decoded, or if resampling
            fails (e.g. on pathologically short audio).
        EmptyAudioError: If the file contains zero audio frames.

    Warns:
        UserWarning: If the resulting audio is shorter than
            MIN_RECOMMENDED_DURATION_SECONDS. Processing still proceeds;
            this is informational only.
    """
    waveform, sample_rate = load_audio(path)

    # --- Mono conversion ---
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # --- Resample to TARGET_SAMPLE_RATE (skip if already correct) ---
    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
            )
        except Exception as exc:
            raise AudioDecodeError(
                f"Failed to resample '{path}' from {sample_rate} Hz to "
                f"{TARGET_SAMPLE_RATE} Hz: {exc}"
            ) from exc

    duration_seconds = waveform.shape[-1] / TARGET_SAMPLE_RATE
    if duration_seconds < MIN_RECOMMENDED_DURATION_SECONDS:
        import warnings

        warnings.warn(
            f"Preprocessed audio '{path}' is very short "
            f"({duration_seconds * 1000:.1f} ms). This may not contain "
            f"enough signal for reliable downstream speaker verification.",
            UserWarning,
            stacklevel=2,
        )

    return waveform, TARGET_SAMPLE_RATE
