"""The web app's stage 2 guard: noise-dominated clips must not reach the encoder."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import speaker  # noqa: E402

SR = speaker.SAMPLE_RATE
rng = np.random.default_rng(0)


def voice_in_room(gain: float) -> np.ndarray:
    """2 s of room noise, 3 s of a voice-like warble at `gain`, 2 s of room noise."""
    t = np.arange(3 * SR) / SR
    voice = np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 3 * t)) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    clip = rng.standard_normal(7 * SR) * 0.003
    clip[2 * SR : 5 * SR] += voice * gain
    return clip.astype(np.float32)


def test_clear_voice_has_high_snr_and_keeps_speech():
    clip = voice_in_room(0.3)
    assert speaker.speech_snr(clip) > speaker.MIN_SNR_DB
    kept = speaker.speech_only(clip)
    assert 2.5 * SR < len(kept) < 4.5 * SR  # the voice, not the 4 s of room


def test_continuous_speech_is_not_trimmed_away():
    t = np.arange(4 * SR) / SR
    assert len(speaker.speech_only((np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32))) > 3.5 * SR


def test_noise_only_clip_is_refused_before_the_encoder():
    noise = (rng.standard_normal(5 * SR) * 0.01).astype(np.float32)
    with pytest.raises(ValueError, match="too quiet"):
        speaker.verify(None, voice_in_room(0.3), noise)
