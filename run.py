#!/usr/bin/env python3
"""
B.S. Detector — one-command local launcher.

    python run.py              set up (first run only), start, open the browser
    python run.py --no-browser start without opening a browser tab
    python run.py --port 9000  use a different port

First run creates ./.venv and installs server/requirements.txt (PyTorch is the
big one). The model weights (~1.3 GB) download from Hugging Face the first time
the server starts and are cached after that. The team's Hindi-adapted detector
head (~94 MB) ships in models/ and downloads once from GitHub if it's missing.
Uses the standard library only.
"""

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "server" / "requirements.txt"
STAMP = VENV / ".bsd-requirements"
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"

# The default Hindi head (first in server/app.py HINDI_HEADS) is committed to the repo;
# this URL fetches it again if models/ lost it.
HEAD_FILE = ROOT / "models" / "sls_head_phase12.pth"
HEAD_URL = "https://raw.githubusercontent.com/Anumay1231/B.S-Detector/main/models/sls_head_phase12.pth"


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def ensure_environment() -> Path:
    py = venv_python()
    if not py.exists():
        say("Creating a virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)

    wanted = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not STAMP.exists() or STAMP.read_text().strip() != wanted:
        say("Installing dependencies. This takes a few minutes the first time ...")
        subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"])
        subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        STAMP.write_text(wanted)
    return py


def ensure_frontend() -> None:
    if (DIST / "index.html").exists():
        return
    npm = shutil.which("npm")
    if not npm:
        sys.exit("frontend/dist is missing and npm isn't installed. Re-download the repo or install Node.js.")
    say("Building the frontend ...")
    subprocess.check_call([npm, "ci"], cwd=FRONTEND)
    subprocess.check_call([npm, "run", "build"], cwd=FRONTEND)


def ensure_hindi_head() -> None:
    """Fetch the Hindi head once. Without it the app falls back to the English checkpoint."""
    models_dir = ROOT / "models"
    if (
        os.environ.get("BSD_HEAD")
        or (models_dir / "sls_head_silence_invariant.pth").exists()
        or (models_dir / "sls_head_channel_robust.pth").exists()
        or (models_dir / "sls_head_n500_indicsynth_best.pth").exists()
        or HEAD_FILE.exists()
    ):
        return
    say("Downloading the Hindi detector head (~94 MB) ...")
    part = HEAD_FILE.with_suffix(".part")
    try:
        HEAD_FILE.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(HEAD_URL, timeout=30) as res, open(part, "wb") as out:
            shutil.copyfileobj(res, out, 1 << 20)
        # a PyTorch checkpoint is a zip archive; anything else is an error page or a cut-off download
        with open(part, "rb") as f:
            if f.read(2) != b"PK" or part.stat().st_size < 10_000_000:
                raise ValueError("the download isn't a PyTorch checkpoint")
        part.replace(HEAD_FILE)
        say("Hindi head ready.")
    except Exception as exc:
        part.unlink(missing_ok=True)
        say(f"Couldn't get the Hindi head ({exc}). Using the English checkpoint, which misses most Hindi fakes.")
        say(f"To add it by hand, put sls_head_phase12.pth in {HEAD_FILE.parent}.")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def open_when_up(url: str) -> None:
    for _ in range(240):
        try:
            urllib.request.urlopen(url + "api/status", timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B.S. Detector locally.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("B.S. Detector needs Python 3.10 or newer.")

    print("\nB.S. Detector\n")
    py = ensure_environment()
    ensure_frontend()
    ensure_hindi_head()

    port = args.port
    while not port_free(port):
        port += 1
    url = f"http://127.0.0.1:{port}/"

    say(f"Starting at {url}  (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Thread(target=open_when_up, args=(url,), daemon=True).start()

    cmd = [str(py), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    try:
        subprocess.call(cmd, cwd=ROOT / "server")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
