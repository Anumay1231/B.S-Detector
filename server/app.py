"""
B.S. Detector — local API server.

Serves the built frontend (frontend/dist) and a small JSON API around both
stages of the project:

  Stage 1  XLS-R 300M + SLS deepfake detector (inference.py)
  Stage 2  ECAPA-TDNN speaker verification (speaker.py -> src/speaker_verification)

  GET  /api/status   loading state of both models
  POST /api/analyze  body = WAV bytes (the browser sends 16 kHz mono) -> stage 1 verdict
  POST /api/verify   form: reference (1-3 WAVs of the known voice), test (WAV), windowStart (s)
                     -> stage 1 on the test clip and each known clip, then stage 2 only if
                     all sound genuine (the interlock)

The models load in a background thread so the page is usable (and can show
progress) while the weights download on first run.
"""

import os
import sys
import threading
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
DIST_DIR = REPO_ROOT / "frontend" / "dist"
sys.path.insert(0, str(SERVER_DIR))

import inference as engine  # noqa: E402
import speaker  # noqa: E402

logger = logging.getLogger("bsdetector.server")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHECKPOINT = os.environ.get("BSD_CHECKPOINT", "v1")
DEVICE = os.environ.get("BSD_DEVICE") or None
# Hindi-adapted SLS heads from the team's runs (same layer layout as the English
# checkpoint), best first. The first one present is used; BSD_HEAD forces one.
# The phase 12 head leads: 0.7 x the silence-invariant head + 0.3 x a head retrained on
# Indian phone, studio and accented speech with phone/codec/room augmentation
# (scripts/train_phase12.py, scripts/mix_heads.py). On scripts/ui_path_check.py it gets
# 0 wrong (the silence-invariant head: 0) and catches 7 of 9 fakes on phone-line audio
# (6 of 9); on held-out people and voices, 76.3% balanced accuracy (73.4%). The channel-robust head is not
# listed: it got 17 answers wrong and flagged the clean built-in samples, and its room /
# loudspeaker-replay idea is now part of the phase 12 training.
HINDI_HEADS = {
    "hindi-phase12": REPO_ROOT / "models" / "sls_head_phase12.pth",
    "hindi-robust": REPO_ROOT / "models" / "sls_head_silence_invariant.pth",
    "hindi-indicsynth": REPO_ROOT / "models" / "sls_head_n500_indicsynth_best.pth",
}
if os.environ.get("BSD_HEAD"):
    HINDI_HEADS = {"hindi-custom": Path(os.environ["BSD_HEAD"])}


def load_hindi_head(head: Any, path: Path) -> None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items() if not k.startswith("ssl_model.")}
    head.load_state_dict(state)
    head.eval()


class ModelState:
    def __init__(self) -> None:
        self.state = "loading" if engine.TORCH_AVAILABLE else "unavailable"
        self.message = (
            "Loading the detector. The first run downloads about 1.3 GB of weights."
            if engine.TORCH_AVAILABLE
            else "PyTorch isn't installed. Run `python run.py` to set up the environment."
        )
        self.device: Optional[str] = None
        self.backbone: Any = None
        self.head: Any = None
        self.loaded_at: Optional[float] = None
        self.lock = threading.Lock()
        # "english" (the published English checkpoint) or a key of HINDI_HEADS
        self.head_kind = "english"
        self.head_note: Optional[str] = None

    def load(self) -> None:
        if not engine.TORCH_AVAILABLE:
            return
        try:
            t0 = time.perf_counter()
            self.backbone, self.head, self.device = engine.load_detector_models(
                device=DEVICE, checkpoint_version=CHECKPOINT
            )
            for kind, path in HINDI_HEADS.items():
                if not path.is_file():
                    continue
                try:
                    load_hindi_head(self.head, path)
                    self.head_kind = kind
                    logger.info("Using the Hindi-adapted SLS head from %s", path)
                    break
                except Exception as exc:  # try the next head, else keep the English one
                    logger.exception("Could not load %s", path)
                    self.head_note = f"{path.name} could not be loaded ({exc})."
            self.loaded_at = time.time()
            self.state = "ready"
            self.message = f"Ready in {time.perf_counter() - t0:.0f} s."
        except Exception as exc:  # surfaced to the UI verbatim
            logger.exception("Model load failed")
            self.state = "error"
            self.message = f"The model failed to load: {exc}"

    def as_json(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "device": self.device,
            "checkpoint": CHECKPOINT,
            "head": self.head_kind,
            "headNote": self.head_note,
            "backbone": "facebook/wav2vec2-xls-r-300m",
            "windowSeconds": engine.INPUT_LENGTH / engine.TARGET_SR,
        }


model = ModelState()
voices = speaker.VerifierState()


def load_all() -> None:
    model.load()
    voices.load()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=load_all, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="B.S. Detector", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class HashedAssets(StaticFiles):
    """Vite fingerprints everything in /assets, so it can be cached forever."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return {**model.as_json(), "verifier": voices.as_json()}


@app.post("/api/analyze")
async def analyze(request: Request) -> JSONResponse:
    if model.state != "ready":
        raise HTTPException(status_code=503, detail=model.message)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="The request had no audio in it.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Clips must be under 25 MB.")

    try:
        result = await run_in_threadpool(detect, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=422, detail=f"Couldn't read that audio: {exc}")
    return JSONResponse(detection_json(result))


def detect(wav: bytes, full_clip: bool = True) -> engine.ForensicInferenceResult:
    with model.lock:
        return engine.run_forensic_detection(wav, model.backbone, model.head, model.device, full_clip=full_clip)


def detection_json(result: engine.ForensicInferenceResult) -> Dict[str, Any]:
    return {
        "verdict": result.verdict,
        "probSpoof": result.prob_spoof,
        "probBonafide": result.prob_bonafide,
        "layerWeights": result.layer_weights,
        "latencyMs": result.latency_ms,
        "durationSec": result.audio_duration_sec,
        "device": result.device_used,
        "head": model.head_kind,
        "windows": result.windows,
        "numWindows": result.num_windows,
        "aggregation": result.aggregation,
        "peakSpoofWindow": result.peak_spoof_window,
    }


@app.post("/api/verify")
async def verify(
    reference: List[UploadFile] = File(...),
    test: UploadFile = File(...),
    windowStart: float = Form(0.0),
) -> JSONResponse:
    if model.state != "ready":
        raise HTTPException(status_code=503, detail=model.message)
    if voices.state != "ready":
        raise HTTPException(status_code=503, detail=voices.message)
    if len(reference) > speaker.MAX_REFERENCES:
        raise HTTPException(status_code=400, detail=f"Send at most {speaker.MAX_REFERENCES} clips of the known voice.")

    ref_bytes = [await r.read() for r in reference]
    test_bytes = await test.read()
    if not all(ref_bytes) or not test_bytes:
        raise HTTPException(status_code=400, detail="Both clips are needed: a known voice and a voice to check.")
    if any(len(b) > MAX_UPLOAD_BYTES for b in [*ref_bytes, test_bytes]):
        raise HTTPException(status_code=413, detail="Clips must be under 25 MB.")

    def run() -> Dict[str, Any]:
        refs = [speaker.read_wav(b) for b in ref_bytes]
        tst = speaker.read_wav(test_bytes)
        # Stage 1 hears the same 4.04 s window the person chose on the test clip.
        start = max(0, int(windowStart * speaker.SAMPLE_RATE))
        window = tst[start : start + engine.INPUT_LENGTH]
        detection = detect(speaker.wav_bytes(window if len(window) else tst), full_clip=False)
        # The known voice is checked too: a cloned "known voice" would otherwise
        # vouch for a clone. With several clips, each one is checked.
        reference_checks = [detect(speaker.wav_bytes(r[: engine.INPUT_LENGTH]), full_clip=False) for r in refs]
        reference_check = max(reference_checks, key=lambda d: d.prob_spoof)
        reason = None
        if detection.verdict == "SPOOF":
            reason = "window"
        elif reference_check.verdict == "SPOOF":
            reason = "reference"
        payload = {
            "detection": detection_json(detection),
            "referenceDetection": detection_json(reference_check),
            "interlocked": reason is not None,
            "interlockReason": reason,
            "verification": None,
        }
        if len(refs) > 1:
            payload["referenceDetections"] = [detection_json(d) for d in reference_checks]
        # The interlock: never vouch for the identity of a voice that sounds synthetic.
        if reason is None:
            payload["verification"] = speaker.verify(voices, refs[0] if len(refs) == 1 else refs, tst)
        return payload

    try:
        payload = await run_in_threadpool(run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Verification failed")
        raise HTTPException(status_code=422, detail=f"Couldn't process that audio: {exc}")
    return JSONResponse(payload)


if DIST_DIR.exists():
    app.mount("/assets", HashedAssets(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (DIST_DIR / path).resolve()
        if not (path and candidate.is_file() and DIST_DIR in candidate.parents):
            candidate = DIST_DIR / "index.html"
        if candidate.name == "index.html":
            # revalidate every time, so a rebuilt UI shows up without a hard refresh
            return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
        return FileResponse(candidate)
