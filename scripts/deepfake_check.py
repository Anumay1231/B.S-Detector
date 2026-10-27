"""
Run stage 1 (XLS-R 300M + SLS) on folders of real and generated clips, with
both heads, exactly as the web app does: the first 4.04 s window, un-normalised.

    python scripts/deepfake_check.py --real DIR [DIR ...] --fake DIR [DIR ...]

Prints per-clip spoof probabilities and a per-language / per-kind summary,
and writes outputs/deepfake_check.json. `--room` also re-scores every real clip
after the simulated laptop-mic room from sv_benchmark.py, which is what a live
recording looks like to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))

import inference as engine  # noqa: E402

AUDIO = {".wav", ".mp3", ".flac", ".ogg"}


def read(path: Path) -> np.ndarray:
    import sv_benchmark

    return sv_benchmark._decode(path.read_bytes()) if path.suffix != ".wav" else _wav(path)


def _wav(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 16000:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
    return a


def lang_of(path: Path, meta: dict) -> str:
    if path.name in meta:
        return meta[path.name].get("lang", "?")
    for tag in ("_hi_", "-hi", "hindi"):
        if tag in path.name.lower():
            return "hi"
    return "en"


def kind_of(path: Path, meta: dict) -> str:
    return meta.get(path.name, {}).get("kind") or path.name.split("_")[0]


class Detector:
    def __init__(self):
        import torch

        from app import load_hindi_head  # noqa: E402

        self.torch = torch
        self.backbone, self.english, self.device = engine.load_detector_models(device="cpu")
        _, hindi, _ = engine.load_detector_models(device="cpu")
        load_hindi_head(hindi, ROOT / "models" / "sls_head_n500_seed2_best.pth")
        self.heads = {"english": self.english, "hindi": hindi}

    def __call__(self, audio: np.ndarray) -> dict:
        x = engine.pad_or_truncate_audio(self.torch.from_numpy(audio.astype(np.float32))).unsqueeze(0)
        with self.torch.no_grad():
            hs = list(self.backbone(x, output_hidden_states=True).hidden_states[1:])
            return {k: float(self.torch.exp(h(hs)[0])[0, 0]) for k, h in self.heads.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", type=Path, default=[])
    ap.add_argument("--fake", nargs="*", type=Path, default=[])
    ap.add_argument("--room", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "deepfake_check.json")
    a = ap.parse_args()

    det = Detector()
    rows = []
    for label, dirs in (("real", a.real), ("fake", a.fake)):
        for d in dirs:
            meta = {}
            for mf in ("meta.json", "fakes.json"):
                if (d / mf).exists():
                    meta.update({m["file"]: m for m in json.loads((d / mf).read_text(encoding="utf-8"))})
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in AUDIO:
                    continue
                audio = read(p)
                r = {"file": str(p.relative_to(d.parent)), "label": label, "lang": lang_of(p, meta), "kind": kind_of(p, meta), **det(audio)}
                rows.append(r)
                print(f"{label:4s} {r['lang']} {r['kind']:10s} en={r['english']:.3f} hi={r['hindi']:.3f}  {p.name}", flush=True)
                if a.room and label == "real":
                    import sv_benchmark as sb

                    for k in range(sb.ROOMS):
                        wav = sb.in_room(audio, sb.make_room(k), np.random.default_rng(k))
                        rows.append({**r, "kind": f"{r['kind']}+room{k}", **det(wav)})

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["lang"], r["kind"].split("+")[0] + ("+room" if "+room" in r["kind"] else ""))].append(r)
    print("\nlabel lang kind            n   flagged(english head)  flagged(hindi head)")
    summary = []
    for (label, lang, kind), rs in sorted(groups.items()):
        fe = np.mean([r["english"] >= 0.5 for r in rs])
        fh = np.mean([r["hindi"] >= 0.5 for r in rs])
        summary.append({"label": label, "lang": lang, "kind": kind, "n": len(rs), "flaggedEnglishHead": fe, "flaggedHindiHead": fh})
        print(f"{label:5s} {lang:4s} {kind:14s} {len(rs):3d}   {fe*100:6.1f}%                {fh*100:6.1f}%")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
