#!/usr/bin/env python3
"""
B.S. Detector — one-command local launcher.

    python run.py              set up (first run only), start, open the browser
    python run.py --no-browser start without opening a browser tab
    python run.py --port 9000  use a different port

First run creates ./.venv and installs server/requirements.txt (PyTorch is the
big one). The model weights (~1.3 GB) download from Hugging Face the first time
the server starts and are cached after that. Uses the standard library only.
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
