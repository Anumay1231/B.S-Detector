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


def test_threshold_rises_with_speech_and_is_clamped():
    t = [speaker.threshold_for(s) for s in (1.0, 3.0, 8.0, 15.0, 60.0)]
    assert t[0] == t[1]  # below the minimum speech it never goes lower
    assert t[1] < t[2] < t[3]
    assert t[3] <= t[4] == speaker.threshold_for(20.0)
    # every Phase 9 threshold is stricter than the VoxCeleb one that let friends through
    assert min(t) > speaker.VOXCELEB_THRESHOLD


def test_long_clips_are_embedded_as_the_mean_of_3_s_segments():
    import torch

    seen = []

    class Encoder:
        def encode(self, wav):
            seen.append(wav.shape[-1])
            return torch.ones(192) * (len(seen) + 1)

    emb = speaker.embed(Encoder(), np.zeros(11 * SR, dtype=np.float32))
    assert seen == [3 * SR, 3 * SR, 3 * SR, 2 * SR]  # 3 + 3 + 3 + the 2 s tail
    seen.clear()
    speaker.embed(Encoder(), np.zeros(10 * SR, dtype=np.float32))
    assert seen == [3 * SR] * 3  # a 1 s tail is too short to stand on its own
    assert torch.allclose(emb.norm(), torch.tensor(1.0))
    seen.clear()
    speaker.embed(Encoder(), np.zeros(4 * SR, dtype=np.float32))
    assert seen == [4 * SR]  # short clips go in whole


def test_too_little_speech_is_refused():
    clip = voice_in_room(0.3)
    clip[int(3.5 * SR) : 5 * SR] = rng.standard_normal(int(1.5 * SR)) * 0.003  # only 1.5 s of voice left
    with pytest.raises(ValueError, match="clear speech"):
        speaker.verify(None, clip, clip)
