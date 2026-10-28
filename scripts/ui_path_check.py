"""
Check the running app the way the web page uses it, on clips with known answers.

    python run.py --no-browser          # in another terminal (or any server on --url)
    python scripts/ui_path_check.py [--url http://127.0.0.1:8000] [--out report.txt]

Three parts, all through the HTTP API, never the models directly:

(a) "Check this voice": every demo/*/*/*.wav and frontend/public/samples/*.wav,
    sent as the browser sends it. frontend/src/lib/audio.ts findSpeechOnset() picks
    the 4.04 s window start, and only that 4.04 s slice is posted to /api/analyze.
    Files whose name starts with "fake" must come back SPOOF, the rest BONAFIDE.
(b) "Run both checks": the demo speaker pairs, the clone interlock pairs and the
    bundled sample pairs, posted to /api/verify with the same onset windowStart.
(c) Stress table, informational: each file whole, first 4.04 s only, silence
    trimmed, 0.5 s of quiet in front, and starting 1 s into the speech.

Exits non-zero if anything in (a) or (b) is wrong.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16_000
WINDOW = 64_600  # what the model hears: ~4.04 s


def find_speech_onset(s: np.ndarray) -> float:
    """Port of findSpeechOnset() in frontend/src/lib/audio.ts, in seconds."""
    if len(s) <= WINDOW:
        return 0.0
    frame, hop = round(0.025 * SR), round(0.010 * SR)
    n = (len(s) - frame) // hop
    if n <= 0:
        return 0.0
    energy = np.array([np.mean(s[i * hop : i * hop + frame] ** 2) for i in range(n)])
    if energy.max() < 1e-5:
        return 0.0
    first = int(np.argmax(energy >= max(1e-5, energy.max() * 0.01)))
    return min(max(0.0, first * hop / SR - 0.15), (len(s) - WINDOW) / SR)


def read(path: Path) -> np.ndarray:
    a, rate = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if rate != SR:
        raise SystemExit(f"{path} is {rate} Hz; the checks expect 16 kHz files")
    return a


def wav(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, a.astype(np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post(url: str, data: bytes, ctype: str) -> dict:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:120]}"}


def form(fields: dict, files: dict) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, data in files.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{k}.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + data + b"\r\n"
    return body + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def trim(a: np.ndarray) -> np.ndarray:
    db = 10 * np.log10(np.convolve(a**2, np.ones(480) / 480, mode="same") + 1e-10)
    on = np.flatnonzero(db > db.max() - 35)
    return a[on[0] : on[-1]] if len(on) else a


def pairs() -> list[tuple[str, str, str]]:
    """(known voice, voice to check, expected): MATCH, NON_MATCH or INTERLOCK."""
    out = []
    for lang in ("hindi", "english"):
        d = f"demo/speaker/{lang}"
        out += [
            (f"{d}/person_A_take1.wav", f"{d}/person_A_take2.wav", "MATCH"),
            (f"{d}/person_A_take1.wav", f"{d}/person_B.wav", "NON_MATCH"),
            (f"{d}/same_room_person_A.wav", f"{d}/same_room_person_B.wav", "NON_MATCH"),
        ]
        d = f"demo/deepfake/{lang}"
        out += [
            (f"{d}/real_person1.wav", f"{d}/fake_xtts_clone_of_person1.wav", "INTERLOCK"),
            (f"{d}/fake_xtts_clone_of_person1.wav", f"{d}/real_person1.wav", "INTERLOCK"),
        ]
    manifest = json.loads((ROOT / "frontend/public/samples/manifest.json").read_text(encoding="utf-8"))
    files = {s["id"]: s["file"] for s in manifest["samples"]}
    kinds = {s["id"]: s["kind"] for s in manifest["samples"]}
    for p in manifest["pairs"]:
        exp = "INTERLOCK" if "generated" in (kinds[p["reference"]], kinds[p["test"]]) else ("MATCH" if p["expected"] == "same" else "NON_MATCH")
        out.append((f"frontend/public/samples/{files[p['reference']]}", f"frontend/public/samples/{files[p['test']]}", exp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, help="also write the report to this file")
    ap.add_argument("--no-stress", action="store_true", help="skip part (c)")
    a = ap.parse_args()
    lines: list[str] = []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731

    try:
        status = json.load(urllib.request.urlopen(f"{a.url}/api/status", timeout=10))
    except Exception as exc:
        raise SystemExit(f"No server at {a.url} ({exc}). Start it with: python run.py --no-browser")
    v = status.get("verifier") or {}
    if status.get("state") != "ready" or v.get("state") != "ready":
        raise SystemExit("The server is up but the models are still loading. Try again in a minute.")
    say(f"server: head={status.get('head')}  verifier={v.get('model')} ({v.get('system')})")

    files = sorted((ROOT / "demo").glob("*/*/*.wav")) + sorted((ROOT / "frontend/public/samples").glob("*.wav"))
    wrong = 0

    say("\n(a) Check this voice: onset-snapped 4.04 s window, as the browser sends it")
    for f in files:
        x = read(f)
        start = find_speech_onset(x)
        r = post(f"{a.url}/api/analyze", wav(x[int(start * SR) : int(start * SR) + WINDOW]), "audio/wav")
        expect = "SPOOF" if f.name.startswith("fake") else "BONAFIDE"
        ok = r.get("verdict") == expect
        wrong += not ok
        got = r.get("error") or f"{r.get('verdict')} p_spoof {r.get('probSpoof', 0):.3f}"
        say(f"  {'ok   ' if ok else 'WRONG'} expect {expect:8s} got {got:28s} window {start:4.2f}s  {f.relative_to(ROOT).as_posix()}")

    say("\n(b) Run both checks: windowStart from the same onset rule")
    for ref, tst, expect in pairs():
        test_audio = read(ROOT / tst)
        body, ctype = form({"windowStart": find_speech_onset(test_audio)}, {"reference": wav(read(ROOT / ref)), "test": wav(test_audio)})
        r = post(f"{a.url}/api/verify", body, ctype)
        if "error" in r:
            got, extra = "ERROR", r["error"]
        else:
            got = "INTERLOCK" if r["interlocked"] else r["verification"]["outcome"]
            extra = f"test p_spoof {r['detection']['probSpoof']:.2f}, known p_spoof {r.get('referenceDetection', {}).get('probSpoof', float('nan')):.2f}"
            if r.get("verification"):
                vv = r["verification"]
                extra += f", score {vv['score']:.2f} vs {vv['threshold']:.2f}"
        ok = got == expect
        wrong += not ok
        say(f"  {'ok   ' if ok else 'WRONG'} expect {expect:9s} got {got:9s} {Path(ref).name} vs {Path(tst).name}  ({extra})")

    if not a.no_stress:
        say("\n(c) Stress table (informational): % of files called SPOOF")
        rng = np.random.default_rng(0)
        conds = {
            "whole file (/api/analyze)": lambda x: x,
            "first 4.04 s from 0 s": lambda x: x[:WINDOW],
            "silence trimmed": trim,
            "0.5 s quiet in front": lambda x: np.concatenate([(rng.standard_normal(SR // 2) * 1e-3).astype(np.float32), trim(x)]),
            "starts 1 s into speech": lambda x: trim(x)[SR:],
        }
        say(f"  {'condition':28s} {'real flagged':>14s} {'fakes caught':>14s}")
        for name, edit in conds.items():
            real, fake = [], []
            for f in files:
                r = post(f"{a.url}/api/analyze", wav(edit(read(f))), "audio/wav")
                (fake if f.name.startswith("fake") else real).append(r.get("verdict") == "SPOOF")
            say(f"  {name:28s} {sum(real):>6d}/{len(real):<7d} {sum(fake):>6d}/{len(fake):<7d}")

    say(f"\n{wrong} wrong in (a) and (b)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
