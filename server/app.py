"""
B.S. Detector — local API server.

Serves the built frontend (frontend/dist) and a small JSON API around the
XLS-R 300M + SLS detector in inference.py:

  GET  /api/status   model loading state, device, checkpoint
  POST /api/analyze  body = WAV bytes (the browser sends 16 kHz mono) -> verdict

The model loads in a background thread so the page is usable (and can show
progress) while the ~1.3 GB of weights download on first run.
"""

import os
import sys
import threading
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
DIST_DIR = REPO_ROOT / "frontend" / "dist"
sys.path.insert(0, str(SERVER_DIR))

import inference as engine  # noqa: E402

logger = logging.getLogger("bsdetector.server")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHECKPOINT = os.environ.get("BSD_CHECKPOINT", "v1")
DEVICE = os.environ.get("BSD_DEVICE") or None


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

    def load(self) -> None:
        if not engine.TORCH_AVAILABLE:
            return
        try:
            t0 = time.perf_counter()
            self.backbone, self.head, self.device = engine.load_detector_models(
                device=DEVICE, checkpoint_version=CHECKPOINT
            )
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
            "backbone": "facebook/wav2vec2-xls-r-300m",
            "windowSeconds": engine.INPUT_LENGTH / engine.TARGET_SR,
        }


model = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=model.load, name="model-loader", daemon=True).start()
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
    return model.as_json()


@app.post("/api/analyze")
async def analyze(request: Request) -> JSONResponse:
    if model.state != "ready":
        raise HTTPException(status_code=503, detail=model.message)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="The request had no audio in it.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Clips must be under 25 MB.")

    def run() -> engine.ForensicInferenceResult:
        with model.lock:
            return engine.run_forensic_detection(body, model.backbone, model.head, model.device)

    try:
        result = await run_in_threadpool(run)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=422, detail=f"Couldn't read that audio: {exc}")

    return JSONResponse(
        {
            "verdict": result.verdict,
            "probSpoof": result.prob_spoof,
            "probBonafide": result.prob_bonafide,
            "layerWeights": result.layer_weights,
            "latencyMs": result.latency_ms,
            "durationSec": result.audio_duration_sec,
            "device": result.device_used,
        }
    )


if DIST_DIR.exists():
    app.mount("/assets", HashedAssets(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (DIST_DIR / path).resolve()
        if path and candidate.is_file() and DIST_DIR in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
