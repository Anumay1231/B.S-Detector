"""
test_audio.py
--------------

Unit tests for src/speaker_verification/audio.py.

All test audio is generated on the fly into pytest's tmp_path fixture —
no external dataset or checked-in audio fixture is required.

Covers (per Phase 3 test plan):
    1. Mono WAV
    2. Stereo WAV -> mono
    3. Different sample rate -> resampled to 16 kHz
    4. Missing file
    5. Invalid / corrupt file
    6. Very short audio
    7. Multi-channel (>2 channels) audio -> mono
    plus: metadata via get_audio_info, empty audio, unsupported extension,
    and confirmation that load_audio() does NOT alter sample rate/channels
    (only preprocess_audio() does).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaker_verification.audio import (
    AudioDecodeError,
    AudioFileNotFoundError,
    AudioInfo,
    EmptyAudioError,
    TARGET_SAMPLE_RATE,
    UnsupportedAudioFormatError,
    get_audio_info,
    load_audio,
    preprocess_audio,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _write_wav(path: Path, seconds: float, sample_rate: int, num_channels: int) -> None:
    """Write a simple sine-wave WAV file with the given duration, sample
    rate, and channel count. Each channel gets a slightly different
    frequency so mono-averaging is verifiably non-degenerate.
    """
    num_frames = max(int(seconds * sample_rate), 1)
    t = np.arange(num_frames) / sample_rate
    channels = []
    for ch in range(num_channels):
        freq = 220.0 * (ch + 1)  # 220 Hz, 440 Hz, 660 Hz, ...
        channels.append(0.1 * np.sin(2 * np.pi * freq * t))
    data = np.stack(channels, axis=-1).astype(np.float32)  # (frames, channels)
    if num_channels == 1:
        data = data[:, 0]
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def _write_empty_wav(path: Path, sample_rate: int = 16_000, num_channels: int = 1) -> None:
    """Write a structurally valid WAV file with zero audio frames."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(num_channels)
        f.setsampwidth(2)  # 16-bit PCM
        f.setframerate(sample_rate)
        f.writeframes(b"")  # zero frames


# --------------------------------------------------------------------------
# 1. Mono WAV
# --------------------------------------------------------------------------


def test_mono_wav_preprocess(tmp_path: Path) -> None:
    path = tmp_path / "mono_16k.wav"
    _write_wav(path, seconds=1.0, sample_rate=16_000, num_channels=1)

    waveform, sample_rate = preprocess_audio(str(path))

    assert sample_rate == TARGET_SAMPLE_RATE
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == pytest.approx(16_000, abs=2)


def test_mono_wav_load_raw(tmp_path: Path) -> None:
    path = tmp_path / "mono_22050.wav"
    _write_wav(path, seconds=0.5, sample_rate=22_050, num_channels=1)

    waveform, sample_rate = load_audio(str(path))

    # load_audio() must NOT resample or alter channel count.
    assert sample_rate == 22_050
    assert waveform.shape[0] == 1


# --------------------------------------------------------------------------
# 2. Stereo WAV -> mono
# --------------------------------------------------------------------------


def test_stereo_wav_converts_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo_16k.wav"
    _write_wav(path, seconds=1.0, sample_rate=16_000, num_channels=2)

    # Raw load should still report 2 channels.
    raw_waveform, raw_sr = load_audio(str(path))
    assert raw_waveform.shape[0] == 2
    assert raw_sr == 16_000

    # Preprocessed output must be mono.
    waveform, sample_rate = preprocess_audio(str(path))
    assert waveform.shape[0] == 1
    assert sample_rate == TARGET_SAMPLE_RATE


# --------------------------------------------------------------------------
# 3. Different sample rate -> resampled to 16 kHz
# --------------------------------------------------------------------------


@pytest.mark.parametrize("original_rate", [8_000, 22_050, 44_100, 48_000])
def test_resample_to_16k(tmp_path: Path, original_rate: int) -> None:
    path = tmp_path / f"mono_{original_rate}.wav"
    _write_wav(path, seconds=1.0, sample_rate=original_rate, num_channels=1)

    waveform, sample_rate = preprocess_audio(str(path))

    assert sample_rate == TARGET_SAMPLE_RATE
    # Duration should be preserved to within a couple of samples' worth of time.
    expected_samples = int(round(1.0 * TARGET_SAMPLE_RATE))
    assert waveform.shape[1] == pytest.approx(expected_samples, abs=4)


def test_already_16k_is_not_mutated_in_length(tmp_path: Path) -> None:
    path = tmp_path / "already_16k.wav"
    _write_wav(path, seconds=0.25, sample_rate=16_000, num_channels=1)

    raw_waveform, _ = load_audio(str(path))
    processed_waveform, sample_rate = preprocess_audio(str(path))

    assert sample_rate == TARGET_SAMPLE_RATE
    # No resampling needed, so frame count should match exactly.
    assert processed_waveform.shape[1] == raw_waveform.shape[1]


# --------------------------------------------------------------------------
# 4. Missing file
# --------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.wav"

    with pytest.raises(AudioFileNotFoundError):
        load_audio(str(path))

    with pytest.raises(AudioFileNotFoundError):
        get_audio_info(str(path))

    with pytest.raises(AudioFileNotFoundError):
        preprocess_audio(str(path))


def test_directory_instead_of_file_raises(tmp_path: Path) -> None:
    directory = tmp_path / "a_directory.wav"
    directory.mkdir()

    with pytest.raises(AudioFileNotFoundError):
        load_audio(str(directory))


# --------------------------------------------------------------------------
# 5. Invalid / corrupt file
# --------------------------------------------------------------------------


def test_corrupt_wav_raises_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is not a real wav file, just some bytes \x00\x01\x02")

    with pytest.raises(AudioDecodeError):
        load_audio(str(path))

    with pytest.raises(AudioDecodeError):
        get_audio_info(str(path))


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "song.mp3"
    path.write_bytes(b"\x00" * 128)

    with pytest.raises(UnsupportedAudioFormatError):
        load_audio(str(path))


def test_empty_audio_raises(tmp_path: Path) -> None:
    path = tmp_path / "zero_frames.wav"
    _write_empty_wav(path)

    with pytest.raises(EmptyAudioError):
        load_audio(str(path))

    with pytest.raises(EmptyAudioError):
        preprocess_audio(str(path))


# --------------------------------------------------------------------------
# 6. Very short audio
# --------------------------------------------------------------------------


def test_very_short_audio_processes_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "very_short.wav"
    # ~3 ms at 16 kHz -> well under MIN_RECOMMENDED_DURATION_SECONDS
    _write_wav(path, seconds=0.003, sample_rate=16_000, num_channels=1)

    with pytest.warns(UserWarning):
        waveform, sample_rate = preprocess_audio(str(path))

    assert sample_rate == TARGET_SAMPLE_RATE
    assert waveform.shape[1] > 0


def test_single_sample_audio_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "one_sample.wav"
    _write_wav(path, seconds=0.0001, sample_rate=8_000, num_channels=1)

    # Should not raise an uncaught/cryptic exception; either it processes
    # (possibly with a warning) or raises a clear AudioError subclass.
    try:
        with pytest.warns(UserWarning):
            waveform, sample_rate = preprocess_audio(str(path))
        assert sample_rate == TARGET_SAMPLE_RATE
    except AudioDecodeError:
        pass  # acceptable: a clear, documented error for pathological input


# --------------------------------------------------------------------------
# 7. Multi-channel (>2 channels) -> mono
# --------------------------------------------------------------------------


def test_multichannel_audio_converts_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "four_channel.wav"
    _write_wav(path, seconds=0.5, sample_rate=16_000, num_channels=4)

    raw_waveform, _ = load_audio(str(path))
    assert raw_waveform.shape[0] == 4

    waveform, sample_rate = preprocess_audio(str(path))
    assert waveform.shape[0] == 1
    assert sample_rate == TARGET_SAMPLE_RATE


# --------------------------------------------------------------------------
# get_audio_info()
# --------------------------------------------------------------------------


def test_get_audio_info_metadata(tmp_path: Path) -> None:
    path = tmp_path / "info_check.wav"
    _write_wav(path, seconds=2.0, sample_rate=44_100, num_channels=2)

    info = get_audio_info(str(path))

    assert isinstance(info, AudioInfo)
    assert info.sample_rate == 44_100
    assert info.num_channels == 2
    assert info.duration_seconds == pytest.approx(2.0, abs=0.01)
    assert info.num_frames == pytest.approx(44_100 * 2, abs=4)


# --------------------------------------------------------------------------
# FLAC support
# --------------------------------------------------------------------------


def test_flac_mono_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mono.flac"
    num_frames = 16_000
    t = np.arange(num_frames) / 16_000
    data = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), data, 16_000, format="FLAC")

    waveform, sample_rate = preprocess_audio(str(path))
    assert sample_rate == TARGET_SAMPLE_RATE
    assert waveform.shape[0] == 1
