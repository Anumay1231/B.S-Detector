"""
Stage 2 for the web app: speaker verification with the project's
speaker_verification package (src/), used as-is.

ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) turns each clip into a
192-dim embedding, cosine similarity compares them, and SpeakerVerifier
applies the calibrated threshold. Nothing here re-implements that logic.
"""

import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("bsdetector.speaker")

#: EER threshold from the Phase 8B speaker-disjoint VoxCeleb1 calibration
#: (28,280 trials, EER 0.87%, held-out FAR 1.15% / FRR 0.48%).
#: See docs/calibration.md, "Phase 8B". score >= threshold -> MATCH.
CALIBRATED_THRESHOLD = 0.252784
THRESHOLD = float(os.environ.get("BSD_SV_THRESHOLD", CALIBRATED_THRESHOLD))
SAMPLE_RATE = 16_000
MIN_SECONDS = 1.0
#: Seconds of actual speech each clip needs after silence is trimmed.
MIN_SPEECH_SECONDS = 1.5
#: Below this the voice is buried in mic noise and ECAPA embeds the room instead:
#: on random LibriSpeech speakers at ~3 dB, 36-45 of 45 impostor pairs crossed the
#: threshold, against 2 of 45 on clean audio. Clean speech measures ~38 dB.
MIN_SNR_DB = 6.0


class VerifierState:
    def __init__(self) -> None:
        self.state = "loading"
        self.message = "Loading the speaker encoder."
        self.verifier: Any = None
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            from speaker_verification.encoder import SpeakerEncoder
            from speaker_verification.verifier import SpeakerVerifier

            t0 = time.perf_counter()
            encoder = SpeakerEncoder(device=os.environ.get("BSD_DEVICE") or None)
            self.verifier = SpeakerVerifier(
                encoder, threshold=THRESHOLD, is_calibrated=THRESHOLD == CALIBRATED_THRESHOLD
            )
            self.state = "ready"
            self.message = f"Ready in {time.perf_counter() - t0:.0f} s."
        except ImportError as exc:
            logger.exception("Speaker verification unavailable")
            self.state = "unavailable"
            self.message = f"Speaker verification needs speechbrain and torchaudio ({exc}). Run `python run.py` to install them."
        except Exception as exc:
            logger.exception("Speaker encoder failed to load")
            self.state = "error"
            self.message = f"The speaker encoder failed to load: {exc}"

    def as_json(self) -> Dict[str, Any]:
        device = None
        if self.verifier is not None:
            device = self.verifier.encoder.device_info.device_str
        return {
            "state": self.state,
            "message": self.message,
            "device": device,
            "model": "speechbrain/spkrec-ecapa-voxceleb",
            "threshold": THRESHOLD,
            "isCalibrated": THRESHOLD == CALIBRATED_THRESHOLD,
        }


def read_wav(data: bytes) -> np.ndarray:
    """Decode WAV bytes (the browser sends 16 kHz mono) to float32 mono."""
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio

        samples = torchaudio.functional.resample(torch.from_numpy(samples), rate, SAMPLE_RATE).numpy()
    return samples


def wav_bytes(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def speech_snr(samples: np.ndarray, frame: int = 480) -> float:
    """Rough SNR in dB: loudest fifth of the frames against the quietest tenth."""
    n = len(samples) // frame
    if n < 10:
        return 0.0
    db = np.sort(10 * np.log10(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10))
    return float(db[int(n * 0.8):].mean() - db[: max(1, n // 10)].mean())


def speech_only(samples: np.ndarray, frame: int = 480) -> np.ndarray:
    """Keep only the frames that carry speech, then normalise the level.

    ECAPA embeds whatever it is given. A quiet microphone recording is mostly room
    noise, and two noise-dominated clips embed almost identically (cosine ~0.98), so
    two different people recorded on the same mic can read as one speaker. Dropping
    the non-speech frames makes the voiceprint about the voice, not the room.
    """
    n = len(samples) // frame
    if n == 0:
        return samples[:0]
    frames = samples[: n * frame].reshape(n, frame)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    floor, peak_db = np.percentile(db, 10), db.max()
    if peak_db - floor < 10.0:
        # no quiet stretch to measure the room against: all speech (or all noise)
        keep = np.ones(n, dtype=bool)
    else:
        keep = db > floor + max(6.0, 0.3 * (peak_db - floor))
    # 150 ms hangover either side so word edges survive
    pad = 5
    keep = np.convolve(keep.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    voiced = frames[keep].reshape(-1)
    peak = float(np.abs(voiced).max()) if len(voiced) else 0.0
    return voiced / peak * 0.9 if peak > 0 else voiced


def verify(state: VerifierState, reference: np.ndarray, test: np.ndarray) -> Dict[str, Any]:
    import torch

    for name, clip in (("known voice", reference), ("voice to check", test)):
        if len(clip) < MIN_SECONDS * SAMPLE_RATE:
            raise ValueError(f"The {name} clip is shorter than {MIN_SECONDS:.0f} second. Use at least a few seconds of speech.")

    snr = {"known voice": speech_snr(reference), "voice to check": speech_snr(test)}
    for name, clip in (("known voice", reference), ("voice to check", test)):
        if snr[name] < MIN_SNR_DB:
            logger.info("speaker check refused: %s SNR %.1f dB", name, snr[name])
            raise ValueError(
                f"The {name} is too quiet next to the background noise to tell voices apart. "
                "Record again closer to the microphone, in a quieter room."
            )
    reference, test = speech_only(reference), speech_only(test)
    for name, clip in (("known voice", reference), ("voice to check", test)):
        if len(clip) < MIN_SPEECH_SECONDS * SAMPLE_RATE:
            raise ValueError(
                f"The {name} has only {len(clip) / SAMPLE_RATE:.1f} s of clear speech. "
                "Record again closer to the microphone and speak for a few seconds."
            )

    t0 = time.perf_counter()
    with state.lock:
        encoder = state.verifier.encoder
        emb_ref = encoder.encode(torch.from_numpy(reference).unsqueeze(0))
        emb_test = encoder.encode(torch.from_numpy(test).unsqueeze(0))
        result = state.verifier.verify_embeddings(emb_ref, emb_test)
    logger.info(
        "speaker check: %s score %.3f (SNR known %.1f dB, test %.1f dB; speech %.1f s / %.1f s)",
        result.outcome.value, result.score, snr["known voice"], snr["voice to check"],
        len(reference) / SAMPLE_RATE, len(test) / SAMPLE_RATE,
    )
    return {
        "outcome": result.outcome.value,
        "score": result.score,
        "threshold": result.threshold,
        "isCalibrated": result.is_calibrated,
        "latencyMs": (time.perf_counter() - t0) * 1000.0,
        "referenceSeconds": len(reference) / SAMPLE_RATE,
        "testSeconds": len(test) / SAMPLE_RATE,
    }
