"""Full recordings must reach detection, including every reference and the test tail."""
import asyncio
import io

import numpy as np
import pytest
from fastapi import UploadFile

from server import app
from server.inference import ForensicInferenceResult


@pytest.mark.parametrize("fake_clip", [None, "test", "reference"])
def test_verify_scans_complete_inputs_before_identity_check(monkeypatch, fake_clip):
    monkeypatch.setattr(app.model, "state", "ready")
    monkeypatch.setattr(app.voices, "state", "ready")
    sr = app.speaker.SAMPLE_RATE
    clips = {
        "reference": np.full(12 * sr, 0.1, dtype=np.float32),
        "second_reference": np.full(9 * sr, 0.1, dtype=np.float32),
        "test": np.full(30 * sr, 0.1, dtype=np.float32),
    }
    if fake_clip:
        clips[fake_clip][-sr:] = 0.9
    seen = []

    def detect(wav, full_clip=True):
        audio = app.speaker.read_wav(wav)
        seen.append((len(audio), full_clip))
        spoof = float(audio[-1]) > 0.5
        return ForensicInferenceResult(
            verdict="SPOOF" if spoof else "BONAFIDE", confidence=90,
            prob_bonafide=0.1 if spoof else 0.9, prob_spoof=0.9 if spoof else 0.1,
            layer_weights=[0.5] * 24, latency_ms=1, audio_duration_sec=len(audio)/sr,
            device_used="cpu", interlock_triggered=spoof, layer_summary="test",
        )

    verified = []
    monkeypatch.setattr(app, "detect", detect)
    monkeypatch.setattr(app.speaker, "verify", lambda *args: verified.append(True) or {"outcome": "MATCH"})

    def upload(name):
        return UploadFile(filename=name + ".wav", file=io.BytesIO(app.speaker.wav_bytes(clips[name])))

    response = asyncio.run(app.verify(
        reference=[upload("reference"), upload("second_reference")], test=upload("test"),
    ))
    import json
    payload = json.loads(response.body)
    assert seen == [(30 * sr, True), (12 * sr, True), (9 * sr, True)]
    assert payload["interlocked"] is (fake_clip is not None)
    assert payload["interlockReason"] == {None: None, "test": "window", "reference": "reference"}[fake_clip]
    assert bool(verified) is (fake_clip is None)
