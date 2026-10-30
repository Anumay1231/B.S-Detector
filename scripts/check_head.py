"""
Ship check for a candidate detector head, hands-free.

    python scripts/check_head.py models/sls_head_phase12_full.pth [more heads ...]

For each head: start the app on a free port with BSD_HEAD set to it, wait until
both stages are ready, run scripts/ui_path_check.py on normal audio (with the
stress table) and on a phone line, stop the server, and print the "N wrong"
lines and the stress table. Reports go to outputs/check_<head>_{normal,phone}.txt.
Run it with the app's own environment (.venv), like ui_path_check.py.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, server: subprocess.Popen, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"the server exited with code {server.returncode}")
        try:
            s = json.load(urllib.request.urlopen(f"{url}/api/status", timeout=5))
            if s.get("state") == "ready" and (s.get("verifier") or {}).get("state") == "ready":
                return
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit("the server was not ready in time")


def check(head: Path) -> None:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "BSD_HEAD": str(head.resolve())}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "server", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url, server)
        print(f"\n=== {head.name}", flush=True)
        for label, extra in (("normal", []), ("phone", ["--phone", "--no-stress"])):
            report = OUT / f"check_{head.stem}_{label}.txt"
            subprocess.run([sys.executable, "scripts/ui_path_check.py", "--url", url, "--out", str(report), *extra],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            lines = report.read_text(encoding="utf-8").splitlines()
            keep = [ln for ln in lines if "WRONG" in ln or "wrong in" in ln]
            if label == "normal":
                start = next((i for i, ln in enumerate(lines) if ln.startswith("(c)")), None)
                keep += lines[start + 1 : start + 7] if start is not None else []
            print(f"--- {label}")
            print("\n".join(keep), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    heads = [Path(h) for h in sys.argv[1:]]
    if not heads:
        raise SystemExit(__doc__)
    OUT.mkdir(exist_ok=True)
    for h in heads:
        if not h.is_file():
            raise SystemExit(f"no such head: {h}")
        check(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
