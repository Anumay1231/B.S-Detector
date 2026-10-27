"""
test_inference_silence.py
-------------------------
Unit tests verifying robust handling of audio with leading silence,
speech onset auto-alignment, and pure silence rejection.
"""

import numpy as np
import pytest
import torch

from server.inference import find_speech_onset_index, pad_or_truncate_audio, INPUT_LENGTH, TARGET_SR
from scripts.model import find_speech_onset_index as model_find_onset, pad_or_truncate


def test_speech_onset_detection_with_leading_silence():
    # 2.5 seconds of silence followed by 5.0 seconds of simulated speech (total 7.5s)
    # Allows full 4.04s window starting at onset without reaching the end of the file
    sr = 16000
    silence = torch.zeros(int(2.5 * sr))
    speech = torch.randn(int(5.0 * sr)) * 0.1
    combo = torch.cat([silence, speech])

    onset = find_speech_onset_index(combo, target_length=INPUT_LENGTH, sample_rate=sr)
    onset_sec = onset / sr

    # Onset should be ~2.35s (2.5s minus 150ms acoustic buffer)
    assert 2.2 <= onset_sec <= 2.5
    assert onset + INPUT_LENGTH <= combo.shape[0]


def test_speech_onset_short_audio():
    # Short audio (< 4.04s) returns onset 0
    short_audio = torch.randn(30000) * 0.1
    onset = find_speech_onset_index(short_audio, target_length=INPUT_LENGTH)
    assert onset == 0


def test_pad_or_truncate_audio_aligns_window():
    sr = 16000
    silence = torch.zeros(int(3.0 * sr))
    speech = torch.randn(int(3.0 * sr)) * 0.2
    combo = torch.cat([silence, speech])

    window = pad_or_truncate_audio(combo, target_length=INPUT_LENGTH)
    assert window.shape[0] == INPUT_LENGTH
    # The extracted window should contain strong speech energy, not silence
    rms = torch.sqrt(torch.mean(window ** 2)).item()
    assert rms > 0.05, f"Window RMS was {rms}, expected > 0.05 (speech energy)"


def test_scripts_model_pad_or_truncate_consistency():
    sr = 16000
    silence = torch.zeros(int(2.0 * sr))
    speech = torch.randn(int(4.0 * sr)) * 0.15
    combo = torch.cat([silence, speech])

    onset_server = find_speech_onset_index(combo, target_length=INPUT_LENGTH, sample_rate=sr)
    onset_model = model_find_onset(combo, target_length=INPUT_LENGTH, sample_rate=sr)
    assert onset_server == onset_model

    win = pad_or_truncate(combo, target_length=INPUT_LENGTH, align_onset=True)
    assert win.shape[0] == INPUT_LENGTH
    assert torch.sqrt(torch.mean(win ** 2)).item() > 0.03


def test_pure_silence_rejection():
    from server.inference import run_forensic_detection
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(64600, dtype=np.float32), 16000, format='WAV')
    wav_bytes = buf.getvalue()

    # Pass dummy None objects for backbone and head because validation occurs before model call
    with pytest.raises(ValueError, match="The audio recording contains silence"):
        run_forensic_detection(wav_bytes, None, None, "cpu")
