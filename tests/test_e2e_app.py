"""End to end: start the app on a free port and run scripts/ui_path_check.py against it.

Slow (it loads both stages' models and sends every demo clip), so it only runs when
asked:

    BSD_E2E=1 python -m pytest tests/test_e2e_app.py -s
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READY_TIMEOUT_S = 900  # a first run downloads ~1.3 GB of weights

pytestmark = pytest.mark.skipif(os.environ.get("BSD_E2E") != "1", reason="set BSD_E2E=1 to run the end-to-end app check")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_ready(url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            pytest.fail(f"the server exited early with code {server.returncode}")
        try:
            s = json.load(urllib.request.urlopen(f"{url}/api/status", timeout=5))
            if s.get("state") == "ready" and (s.get("verifier") or {}).get("state") == "ready":
                return
            if s.get("state") == "error" or (s.get("verifier") or {}).get("state") == "error":
                pytest.fail(f"model load failed: {s}")
        except OSError:
            pass
        time.sleep(2)
    pytest.fail(f"the server was not ready after {READY_TIMEOUT_S} s")


def test_ui_path_check_passes():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "server", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_ready(url, server)
        check = subprocess.run(
            [sys.executable, "scripts/ui_path_check.py", "--url", url, "--no-stress"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        print(check.stdout)
        assert check.returncode == 0, f"ui_path_check.py reported wrong answers:\n{check.stdout}\n{check.stderr}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
