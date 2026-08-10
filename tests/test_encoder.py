"""
test_encoder.py
----------------

Unit tests for src/speaker_verification/encoder.py.

Loading the pretrained ECAPA-TDNN model requires network access to the
Hugging Face Hub on first use (it is then cached locally — see
docs/model.md). Some execution environments (including the sandbox this
project was originally developed in) block that network access entirely.
Tests that require the real pretrained model use the `speaker_encoder`
fixture below, which attempts to load the model once per test session and
calls `pytest.skip(...)` with a clear, specific reason if that fails for
network/availability reasons — this is treated as an environment
limitation, not a test failure or a code defect. Device-selection logic
that does not require the model is tested unconditionally.

No real speaker dataset is used. `_make_sine_wave()` generates a
deterministic synthetic waveform purely to exercise the encode() pipeline
end-to-end (shapes, dtypes, determinism, gradient behavior). A synthetic
sine wave says nothing about real speaker-verification accuracy; these
tests only verify that the pipeline runs correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speaker_verification.encoder import (
    DeviceError,
    EncoderError,
    InvalidWaveformError,
    ModelLoadError,
    SpeakerEncoder,
    resolve_device,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_sine_wave(seconds: float = 1.0, sample_rate: int = 16_000, freq: float = 220.0) -> torch.Tensor:
    """Deterministic synthetic mono waveform, shape (1, num_samples).

    For pipeline testing only — NOT a substitute for real speech and
    makes no claim about speaker-verification accuracy.
    """
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate
    data = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0)


# --------------------------------------------------------------------------
# Fixture: real pretrained model, skipped gracefully if unavailable
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def speaker_encoder() -> SpeakerEncoder:
    try:
        return SpeakerEncoder(device="cpu")
    except ModelLoadError as exc:
        pytest.skip(
            f"Pretrained model 'speechbrain/spkrec-ecapa-voxceleb' could "
            f"not be loaded (likely no network access to the Hugging Face "
            f"Hub in this environment). This is an environment limitation, "
            f"not a code defect. Original error: {exc}"
        )


# --------------------------------------------------------------------------
# Device selection (no model download required)
# --------------------------------------------------------------------------


def test_resolve_device_auto_returns_cpu_or_cuda() -> None:
    info = resolve_device(None)
    assert info.device.type in ("cpu", "cuda")
    if torch.cuda.is_available():
        assert info.device.type == "cuda"
        assert info.gpu_name is not None
    else:
        assert info.device.type == "cpu"
        assert info.gpu_name is None


def test_resolve_device_explicit_cpu() -> None:
    info = resolve_device("cpu")
    assert info.device.type == "cpu"
    assert info.device_str == "cpu"
    assert info.gpu_name is None
    assert info.gpu_index is None


def test_resolve_device_explicit_cuda_raises_if_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment; nothing to test here.")
    with pytest.raises(DeviceError):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this environment.")
def test_resolve_device_explicit_cuda_when_available() -> None:
    info = resolve_device("cuda")
    assert info.device.type == "cuda"
    assert info.gpu_name is not None
    assert info.gpu_index is not None


# --------------------------------------------------------------------------
# 1. Encoder initialization / 2. CPU initialization / 4. Model loads
# --------------------------------------------------------------------------


def test_encoder_initializes_on_cpu(speaker_encoder: SpeakerEncoder) -> None:
    assert speaker_encoder.device_info.device.type == "cpu"
    assert speaker_encoder.model_source == "speechbrain/spkrec-ecapa-voxceleb"


# --------------------------------------------------------------------------
# 3. CUDA initialization IF CUDA is available
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this environment.")
def test_encoder_initializes_on_cuda() -> None:
    try:
        encoder = SpeakerEncoder(device="cuda")
    except ModelLoadError as exc:
        pytest.skip(f"Pretrained model could not be loaded: {exc}")
    assert encoder.device_info.device.type == "cuda"
    assert encoder.device_info.gpu_name is not None


# --------------------------------------------------------------------------
# 5-7. Valid waveform -> finite, correctly-shaped embedding
# --------------------------------------------------------------------------


def test_encode_produces_embedding(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()
    embedding = speaker_encoder.encode(waveform)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0


def test_embedding_is_finite(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()
    embedding = speaker_encoder.encode(waveform)
    assert torch.isfinite(embedding).all()


def test_embedding_has_expected_dimension(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()
    embedding = speaker_encoder.encode(waveform)
    # The standard speechbrain/spkrec-ecapa-voxceleb checkpoint is
    # documented to produce a 192-dimensional embedding (see
    # docs/model.md). This assertion verifies that against the actual
    # installed/downloaded model rather than assuming it in encoder.py.
    assert embedding.shape[0] == 192


def test_encode_accepts_1d_waveform(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave().squeeze(0)  # (num_samples,) instead of (1, num_samples)
    assert waveform.ndim == 1
    embedding = speaker_encoder.encode(waveform)
    assert embedding.ndim == 1


# --------------------------------------------------------------------------
# 8. Repeated inference does not create gradients
# --------------------------------------------------------------------------


def test_encode_does_not_create_gradients(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()

    embedding = speaker_encoder.encode(waveform)
    assert embedding.requires_grad is False
    assert embedding.grad_fn is None

    # No parameter should have accumulated a gradient from inference.
    for param in speaker_encoder._model.mods.parameters():
        assert param.grad is None


# --------------------------------------------------------------------------
# 9. Same audio processed twice -> consistent embeddings
# --------------------------------------------------------------------------


def test_encode_is_deterministic_for_same_input(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()
    embedding_1 = speaker_encoder.encode(waveform)
    embedding_2 = speaker_encoder.encode(waveform)
    assert torch.allclose(embedding_1, embedding_2, atol=1e-6)


# --------------------------------------------------------------------------
# 10. Invalid waveform input -> useful error
# --------------------------------------------------------------------------


def test_encode_rejects_non_tensor(speaker_encoder: SpeakerEncoder) -> None:
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode([0.1, 0.2, 0.3])  # type: ignore[arg-type]


def test_encode_rejects_empty_waveform(speaker_encoder: SpeakerEncoder) -> None:
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode(torch.zeros(1, 0))


def test_encode_rejects_multichannel_waveform(speaker_encoder: SpeakerEncoder) -> None:
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode(torch.zeros(2, 16_000))


def test_encode_rejects_wrong_ndim(speaker_encoder: SpeakerEncoder) -> None:
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode(torch.zeros(1, 1, 16_000))


def test_encode_rejects_non_floating_dtype(speaker_encoder: SpeakerEncoder) -> None:
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode(torch.zeros(1, 16_000, dtype=torch.int16))


def test_encode_rejects_non_finite_values(speaker_encoder: SpeakerEncoder) -> None:
    waveform = _make_sine_wave()
    waveform[0, 0] = float("nan")
    with pytest.raises(InvalidWaveformError):
        speaker_encoder.encode(waveform)


# --------------------------------------------------------------------------
# encode_file()
# --------------------------------------------------------------------------


def test_encode_file(tmp_path, speaker_encoder: SpeakerEncoder) -> None:
    import soundfile as sf

    path = tmp_path / "sample.wav"
    num_samples = 16_000
    t = np.arange(num_samples) / 16_000
    data = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), data, 16_000, subtype="PCM_16")

    embedding = speaker_encoder.encode_file(str(path))
    assert embedding.ndim == 1
    assert torch.isfinite(embedding).all()
