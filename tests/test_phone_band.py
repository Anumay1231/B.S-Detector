"""Stage 2 does not judge telephone-band audio: its constants were fitted on full-band speech."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
import speaker  # noqa: E402
from ui_path_check import phone_line  # noqa: E402

SR = speaker.SAMPLE_RATE


def voice(seconds: float, seed: int) -> np.ndarray:
    """A warble with breath-like broadband noise, so it has energy above 4 kHz like real speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 3 * t)) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * 0.3
    return (tone + rng.standard_normal(len(t)) * 0.03).astype(np.float32)


def features(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x, n=4096))
    return np.array([b.sum() for b in np.array_split(spec, 16)]) + 1e-3


class Ecapa:
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(features(wav.squeeze(0).numpy()))


class Second:
    def encode(self, wav: np.ndarray) -> np.ndarray:
        return features(wav)[::-1].copy()


STATE = SimpleNamespace(system="fused", lock=threading.Lock(), encoder=Ecapa(), second=Second())


def test_high_band_share_separates_wideband_from_a_phone_line():
    x = voice(6, 0)
    assert speaker.high_band_share(x) > 10 * speaker.PHONE_BAND_SHARE
    assert speaker.high_band_share(phone_line(x)) < speaker.PHONE_BAND_SHARE


def test_wideband_pair_is_judged():
    r = speaker.verify(STATE, voice(9, 0), voice(9, 1))
    assert r["phoneBand"] == []


def test_phone_band_test_clip_is_not_judged():
    r = speaker.verify(STATE, voice(9, 0), phone_line(voice(9, 1)))
    assert r["phoneBand"] == ["voice to check"]
    assert r["outcome"] == "UNCERTAIN"
    assert np.isfinite(r["score"])  # the score is still reported


def test_phone_band_known_clip_is_named():
    r = speaker.verify(STATE, [voice(6, 0), phone_line(voice(6, 2))], voice(9, 1))
    assert r["phoneBand"] == ["known voice 2"]
    assert r["outcome"] == "UNCERTAIN"
