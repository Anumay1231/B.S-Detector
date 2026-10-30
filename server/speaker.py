"""
Stage 2 for the web app: speaker verification with the project's
speaker_verification package (src/).

ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) turns each clip into a
192-dim embedding and cosine similarity compares them.

Phase 9 (docs/speaker-verification-hindi-english.md) changed four things after
the VoxCeleb threshold let a third or more of different Hindi / Indian-English
speakers through:
  - each clip is embedded as the average of its 3 s speech segments;
  - a second encoder, WeSpeaker ResNet34, is scored alongside ECAPA and the two
    z-normalised cosines are averaged (score fusion);
  - the threshold depends on how much speech the shorter clip has; and
  - scores within a margin of the threshold come back UNCERTAIN instead of a guess.
All constants are fitted on Hindi + English Common Voice speakers
(scripts/sv_fusion.py) and tested on held-out Google SVQ speakers.
"""

import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("bsdetector.speaker")

#: EER threshold from the Phase 8B speaker-disjoint VoxCeleb1 calibration
#: (28,280 trials, EER 0.87%, held-out FAR 1.15% / FRR 0.48%). Kept for reference:
#: on Hindi and Indian-English speakers it accepts 22-76% of same-gender impostors.
VOXCELEB_THRESHOLD = 0.252784
#: Phase 9 decision constants from scripts/sv_fusion.py (outputs/sv_bench/fusion_fit.json).
#: d = seconds of speech in the shorter clip, clamped to [MIN_SPEECH_SECONDS, 20].
#: threshold(d) = a + b ln d; unsure margin(d) = max(floor, c + e ln d);
#: score >= threshold + margin -> MATCH, score < threshold - margin -> NON_MATCH,
#: anything between -> UNCERTAIN.
#: "fused" scores are the mean of each model's cosine z-normalised by the mean / std
#: of its impostor scores; "ecapa" (used if WeSpeaker can't load) is the plain cosine.
#: Fitted on Common Voice hi + en and Google SVQ hi_in + en_in. In 2-fold cross-
#: validation over SVQ speakers, with >= 8 s of speech the fused system was right on
#: 96.9-100% of the pairs it answered (clean and same-room, Hindi and English).
FUSION_NORM = {"ecapa": (0.186006, 0.129773), "wespeaker": (0.220665, 0.138429)}
DECISION = {
    "fused": {"threshold": (0.018941, 1.077648), "margin": (1.400611, -0.604268), "floor": 0.40},
    "ecapa": {"threshold": (0.192616, 0.138964), "margin": (0.236979, -0.099336), "floor": 0.0519},
}
#: Set BSD_SV_THRESHOLD to force one fixed (uncalibrated) cosine threshold, ECAPA only.
FIXED_THRESHOLD: Optional[float] = float(os.environ["BSD_SV_THRESHOLD"]) if os.environ.get("BSD_SV_THRESHOLD") else None
SAMPLE_RATE = 16_000
MIN_SECONDS = 1.0
#: Seconds of actual speech each clip needs after silence is trimmed. Below ~3 s the
#: held-out equal error rate is 7-17%, too unreliable to report.
MIN_SPEECH_SECONDS = 3.0
#: From about this much speech in both clips the held-out accuracy is 95%+.
RELIABLE_SPEECH_SECONDS = 8.0
#: Each clip is embedded as the mean of its speech cut into pieces this long.
SEGMENT_SECONDS = 3.0
#: Below this the voice is buried in mic noise and ECAPA embeds the room instead:
#: on random LibriSpeech speakers at ~3 dB, 36-45 of 45 impostor pairs crossed the
#: threshold, against 2 of 45 on clean audio. Clean speech measures ~38 dB.
MIN_SNR_DB = 6.0
#: The known voice may come as up to this many clips; their speech counts together, up to 20 s.
MAX_REFERENCES = 3
MAX_REFERENCE_SECONDS = 20.0


class VerifierState:
    def __init__(self) -> None:
        self.state = "loading"
        self.message = "Loading the speaker encoder."
        self.encoder: Any = None
        self.second: Any = None  # WeSpeaker ResNet34, for score fusion
        self.lock = threading.Lock()

    @property
    def system(self) -> str:
        return "fused" if self.second is not None and FIXED_THRESHOLD is None else "ecapa"

    def load(self) -> None:
        try:
            from speaker_verification.encoder import SpeakerEncoder

            t0 = time.perf_counter()
            # The original VoxCeleb ECAPA, which FUSION_NORM / DECISION were fitted on. The
            # Indian-adapted weights (models/ecapa_indian_adapted.pth) were not kept: on held-out
            # speakers they were better on Hindi but worse on Indian English in 5 of 9 cells
            # (docs/speaker-verification-hindi-english.md). BSD_ECAPA_WEIGHTS loads other weights.
            self.encoder = SpeakerEncoder(device=os.environ.get("BSD_DEVICE") or None,
                                          adapted_weights=os.environ.get("BSD_ECAPA_WEIGHTS") or False)
            try:
                from speaker_verification.wespeaker import WeSpeakerEncoder

                self.second = WeSpeakerEncoder()
            except Exception:  # fusion is an improvement, not a requirement
                logger.exception("WeSpeaker ResNet34 unavailable; using ECAPA alone")
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
        ecapa_name = "ECAPA-TDNN (Indian-adapted)" if getattr(self.encoder, "is_adapted", False) else "speechbrain/spkrec-ecapa-voxceleb"
        models = [ecapa_name]
        if self.system == "fused":
            models.append("Wespeaker/wespeaker-voxceleb-resnet34-LM")
        return {
            "state": self.state,
            "message": self.message,
            "device": self.encoder.device_info.device_str if self.encoder is not None else None,
            "model": " + ".join(models),
            "system": self.system,
            # the threshold at 8 s of speech; each check reports the one it used
            "threshold": threshold_for(RELIABLE_SPEECH_SECONDS, self.system),
            "isCalibrated": FIXED_THRESHOLD is None,
            # FUSION_NORM and DECISION were fitted on the original ECAPA; true only when
            # BSD_ECAPA_WEIGHTS loads other weights, which those constants do not match
            "isAdapted": bool(getattr(self.encoder, "is_adapted", False)),
            "adaptedLoadError": getattr(self.encoder, "adapted_load_error", None),
            "calibratedOn": "speechbrain/spkrec-ecapa-voxceleb",
        }


def _log_d(speech_seconds: float) -> float:
    return float(np.log(np.clip(speech_seconds, MIN_SPEECH_SECONDS, 20.0)))


def threshold_for(speech_seconds: float, system: str = "ecapa") -> float:
    """The calibrated threshold for a pair whose shorter clip has this much speech."""
    if FIXED_THRESHOLD is not None:
        return FIXED_THRESHOLD
    a, b = DECISION[system]["threshold"]
    return a + b * _log_d(speech_seconds)


def margin_for(speech_seconds: float, system: str = "ecapa") -> float:
    """Half-width of the UNCERTAIN band around the threshold."""
    if FIXED_THRESHOLD is not None:
        return 0.0
    c, e = DECISION[system]["margin"]
    return max(DECISION[system]["floor"], c + e * _log_d(speech_seconds))


def decide(score: float, threshold: float, margin: float) -> str:
    if score >= threshold + margin:
        return "MATCH"
    if score < threshold - margin:
        return "NON_MATCH"
    return "UNCERTAIN"


def fused_score(cos: Dict[str, float]) -> float:
    return float(np.mean([(cos[m] - FUSION_NORM[m][0]) / FUSION_NORM[m][1] for m in ("ecapa", "wespeaker")]))


def segments(speech: np.ndarray) -> List[np.ndarray]:
    """SEGMENT_SECONDS pieces; a tail under half a piece is dropped, a short clip stays whole."""
    n = int(SEGMENT_SECONDS * SAMPLE_RATE)
    if len(speech) < 1.5 * n:
        return [speech]
    k = len(speech) // n
    pieces = [speech[i * n : (i + 1) * n] for i in range(k)]
    if len(speech) - k * n > n // 2:
        pieces.append(speech[k * n :])
    return pieces


def _all_segments(speech: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
    """Segments of one clip, or of several clips of the same voice pooled together."""
    clips = [speech] if isinstance(speech, np.ndarray) else list(speech)
    return [p for clip in clips for p in segments(clip)]


def embed(encoder: Any, speech: Union[np.ndarray, Sequence[np.ndarray]]) -> Any:
    """Mean of the length-normalised ECAPA embeddings of each speech segment.

    On held-out SVQ speakers this beat one pass over the whole clip, most for long
    clips in a shared room (Hindi, 11 s+: EER 16.8% -> 5.8%). Several clips of the
    known voice are averaged the same way, over all their segments.
    """
    import torch

    embs = [encoder.encode(torch.from_numpy(np.ascontiguousarray(p)).unsqueeze(0)) for p in _all_segments(speech)]
    return torch.stack([e / e.norm() for e in embs]).mean(dim=0)


def embed_second(encoder: Any, speech: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
    """The same segment averaging for the WeSpeaker encoder (numpy in and out)."""
    embs = [encoder.encode(p) for p in _all_segments(speech)]
    return np.mean([e / np.linalg.norm(e) for e in embs], axis=0)


def _cos(a: Any, b: Any) -> float:
    a, b = (np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float64).reshape(-1) for x in (a, b))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def verify(state: VerifierState, reference: Union[np.ndarray, Sequence[np.ndarray]], test: np.ndarray) -> Dict[str, Any]:
    """Compare a voice with a known voice given as one clip or up to MAX_REFERENCES clips."""
    refs = [reference] if isinstance(reference, np.ndarray) else list(reference)
    if not 1 <= len(refs) <= MAX_REFERENCES:
        raise ValueError(f"Give between 1 and {MAX_REFERENCES} clips of the known voice.")
    names = ["known voice"] if len(refs) == 1 else [f"known voice {i}" for i in range(1, len(refs) + 1)]
    clips = list(zip(names, refs)) + [("voice to check", test)]
    for name, clip in clips:
        if len(clip) < MIN_SECONDS * SAMPLE_RATE:
            raise ValueError(f"The {name} clip is shorter than {MIN_SECONDS:.0f} second. Use at least a few seconds of speech.")

    snr = {name: speech_snr(clip) for name, clip in clips}
    for name, _ in clips:
        if snr[name] < MIN_SNR_DB:
            logger.info("speaker check refused: %s SNR %.1f dB", name, snr[name])
            raise ValueError(
                f"The {name} is too quiet next to the background noise to tell voices apart. "
                "Record again closer to the microphone, in a quieter room."
            )
    clips = [(name, speech_only(clip)) for name, clip in clips]
    for name, clip in clips:
        if len(clip) < MIN_SPEECH_SECONDS * SAMPLE_RATE:
            raise ValueError(
                f"The {name} has only {len(clip) / SAMPLE_RATE:.1f} s of clear speech. "
                f"It needs at least {MIN_SPEECH_SECONDS:.0f} s, and about {RELIABLE_SPEECH_SECONDS:.0f}-15 s for a reliable answer. "
                "Record again closer to the microphone and keep talking."
            )
    reference = [clip for _, clip in clips[:-1]]
    test = clips[-1][1]

    # d is the shorter side; several known-voice clips count together, up to 20 s
    reference_seconds = sum(len(r) for r in reference) / SAMPLE_RATE
    speech = min(min(reference_seconds, MAX_REFERENCE_SECONDS), len(test) / SAMPLE_RATE)
    system = state.system
    t0 = time.perf_counter()
    with state.lock:
        cos = {"ecapa": _cos(embed(state.encoder, reference), embed(state.encoder, test))}
        if system == "fused":
            cos["wespeaker"] = _cos(embed_second(state.second, reference), embed_second(state.second, test))
    score = fused_score(cos) if system == "fused" else cos["ecapa"]
    threshold, margin = threshold_for(speech, system), margin_for(speech, system)
    outcome = decide(score, threshold, margin)
    logger.info(
        "speaker check: %s %s score %.3f vs %.3f +/- %.3f (cosines %s; SNR %s; speech %.1f s in %d known clip(s) / %.1f s)",
        outcome, system, score, threshold, margin, {k: round(v, 3) for k, v in cos.items()},
        {k: round(v, 1) for k, v in snr.items()}, reference_seconds, len(reference), len(test) / SAMPLE_RATE,
    )
    result = {
        "outcome": outcome,
        "score": score,
        "threshold": threshold,
        "margin": margin,
        "system": system,
        "cosines": cos,
        "isCalibrated": FIXED_THRESHOLD is None,
        "latencyMs": (time.perf_counter() - t0) * 1000.0,
        "referenceSeconds": reference_seconds,
        "testSeconds": len(test) / SAMPLE_RATE,
        # below this the held-out error rate is several times higher; the UI says so
        "reliable": speech >= RELIABLE_SPEECH_SECONDS,
        "reliableSeconds": RELIABLE_SPEECH_SECONDS,
    }
    if len(reference) > 1:
        result["referenceClips"] = len(reference)
    return result
