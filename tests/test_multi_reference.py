"""Stage 2 with 1-3 clips of the known voice: one clip behaves as before, several are pooled."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import speaker  # noqa: E402

SR = speaker.SAMPLE_RATE


def voice(seconds: float, pitch: float, seed: int) -> np.ndarray:
    """A voice-like warble over faint noise; `pitch` stands in for who is speaking."""
    t = np.arange(int(seconds * SR)) / SR
    clip = np.sin(2 * np.pi * (pitch + 40 * np.sin(2 * np.pi * 3 * t)) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * 0.3
    return (clip + np.random.default_rng(seed).standard_normal(len(t)) * 0.002).astype(np.float32)


def features(x: np.ndarray) -> np.ndarray:
    """A deterministic stand-in embedding: energy in 16 spectral bands."""
    spec = np.abs(np.fft.rfft(x, n=4096))
    return np.array([b.sum() for b in np.array_split(spec, 16)]) + 1e-3


class Ecapa:
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(features(wav.squeeze(0).numpy()))


class Second:
    def encode(self, wav: np.ndarray) -> np.ndarray:
        return features(wav)[::-1].copy()


STATE = SimpleNamespace(system="fused", lock=threading.Lock(), encoder=Ecapa(), second=Second())


def test_one_clip_as_array_or_list_gives_the_same_answer():
    ref, tst = voice(10, 180, 0), voice(9, 180, 1)
    a = speaker.verify(STATE, ref, tst)
    b = speaker.verify(STATE, [ref], tst)
    for k in ("outcome", "score", "threshold", "margin", "referenceSeconds", "testSeconds", "reliable"):
        assert a[k] == pytest.approx(b[k]), k
    assert "referenceClips" not in a and "referenceClips" not in b


def test_one_clip_keeps_the_old_shorter_side_rule():
    ref, tst = voice(10, 180, 0), voice(6, 180, 1)
    r = speaker.verify(STATE, ref, tst)
    assert r["threshold"] == pytest.approx(speaker.threshold_for(min(r["referenceSeconds"], r["testSeconds"]), "fused"))


def test_several_clips_pool_every_segment():
    a, b = voice(7, 180, 0), voice(5, 180, 2)
    pooled = speaker.embed(Ecapa(), [a, b])
    by_hand = torch.stack(
        [e / e.norm() for e in (Ecapa().encode(torch.from_numpy(p).unsqueeze(0)) for p in speaker.segments(a) + speaker.segments(b))]
    ).mean(dim=0)
    assert torch.allclose(pooled, by_hand)
    assert np.allclose(speaker.embed_second(Second(), [a, b]), np.mean(
        [e / np.linalg.norm(e) for e in (Second().encode(p) for p in speaker.segments(a) + speaker.segments(b))], axis=0))


def test_several_short_clips_add_up_for_the_cut_off():
    refs = [voice(4.5, 180, i) for i in range(2)]
    tst = voice(15, 180, 9)
    r = speaker.verify(STATE, refs, tst)
    assert r["referenceClips"] == 2
    total = r["referenceSeconds"]
    assert total > 8  # each clip alone would be under the reliable 8 s
    assert r["threshold"] == pytest.approx(speaker.threshold_for(min(total, r["testSeconds"]), "fused"))
    assert r["reliable"]


def test_reference_speech_counts_up_to_20_s():
    refs = [voice(9, 180, i) for i in range(3)]
    tst = voice(25, 180, 9)
    r = speaker.verify(STATE, refs, tst)
    assert r["referenceSeconds"] > 20
    assert r["threshold"] == pytest.approx(speaker.threshold_for(20.0, "fused"))
    assert r["margin"] == pytest.approx(speaker.margin_for(20.0, "fused"))


def test_more_than_three_clips_are_refused():
    with pytest.raises(ValueError, match="between 1 and 3"):
        speaker.verify(STATE, [voice(5, 180, i) for i in range(4)], voice(5, 180, 9))


def test_a_bad_extra_clip_is_named():
    noise = (np.random.default_rng(3).standard_normal(5 * SR) * 0.01).astype(np.float32)
    with pytest.raises(ValueError, match="known voice 2 is too quiet"):
        speaker.verify(STATE, [voice(6, 180, 0), noise], voice(6, 180, 1))


# --- the HTTP route -------------------------------------------------------------

app_module = pytest.importorskip("app")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.model, "state", "ready")
    monkeypatch.setattr(app_module.voices, "state", "ready")
    calls = {}

    def fake_detect(wav, full_clip=True):
        audio = speaker.read_wav(wav)
        # a clip whose first sample is exactly 0.5 plays the "synthetic" one
        spoof = bool(np.isclose(audio[0], 0.5, atol=1e-3))
        return SimpleNamespace(verdict="SPOOF" if spoof else "BONAFIDE", prob_spoof=0.9 if spoof else 0.1)

    def fake_verify(state, reference, test):
        calls["reference"] = reference
        return {"outcome": "MATCH"}

    monkeypatch.setattr(app_module, "detect", fake_detect)
    monkeypatch.setattr(app_module, "detection_json", lambda d: {"verdict": d.verdict, "probSpoof": d.prob_spoof})
    monkeypatch.setattr(app_module.speaker, "verify", fake_verify)
    return TestClient(app_module.app), calls


def wav(x: np.ndarray) -> bytes:
    return speaker.wav_bytes(x)


def post(c, refs, test):
    files = [("reference", (f"r{i}.wav", wav(r), "audio/wav")) for i, r in enumerate(refs)]
    files.append(("test", ("t.wav", wav(test), "audio/wav")))
    return c.post("/api/verify", files=files, data={"windowStart": "0"})


def test_route_one_reference_is_unchanged(client):
    c, calls = client
    r = post(c, [voice(6, 180, 0)], voice(6, 180, 1))
    assert r.status_code == 200
    body = r.json()
    assert isinstance(calls["reference"], np.ndarray)
    assert "referenceDetections" not in body
    assert body["verification"] == {"outcome": "MATCH"}


def test_route_passes_every_reference(client):
    c, calls = client
    r = post(c, [voice(6, 180, i) for i in range(3)], voice(6, 180, 9))
    assert r.status_code == 200
    assert len(calls["reference"]) == 3
    assert len(r.json()["referenceDetections"]) == 3


def test_route_interlocks_on_any_synthetic_reference(client):
    c, calls = client
    fake = voice(6, 180, 5)
    fake[0] = 0.5
    r = post(c, [voice(6, 180, 0), fake], voice(6, 180, 9))
    body = r.json()
    assert body["interlocked"] and body["interlockReason"] == "reference"
    assert body["referenceDetection"]["verdict"] == "SPOOF"
    assert "reference" not in calls


def test_route_refuses_four_references(client):
    c, _ = client
    r = post(c, [voice(4, 180, i) for i in range(4)], voice(6, 180, 9))
    assert r.status_code == 400
