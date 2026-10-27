"""
Build demo/: real vs generated clips and speaker-check pairs, Hindi and English.

    python scripts/build_demo.py --real DIR --fake DIR      # copy + make the clips
    python scripts/build_demo.py --score                    # run both stages, write README results

Deepfake clips come from the deepfake test set (scripts/make_fakes.py): real
Common Voice speakers (CC0), an XTTS-v2 clone of each of those same speakers, and
Edge-TTS stock voices. Speaker pairs are ~12 s recordings made by joining 2-3
Google SVQ utterances (CC-BY-4.0) of one person, plus a pair of two different
people put through the same simulated room, the case that broke the old cut-off.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "server"))

DEMO = ROOT / "demo"
SR = 16_000


def write(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(audio).max()) or 1.0
    sf.write(path, (audio / peak * 0.89).astype(np.float32), SR, subtype="PCM_16")


def deepfake_clips(real: Path, fake: Path, rows: list[dict]) -> list[dict]:
    """Two real speakers per language, each with their own XTTS clone, plus two Edge-TTS voices."""
    import deepfake_check as dc

    meta = {m["file"]: m for m in json.loads((real / "meta.json").read_text(encoding="utf-8"))}
    fakes = {m["file"]: m for m in json.loads((fake / "fakes.json").read_text(encoding="utf-8"))}
    picks = {
        "hindi": {"real": ["cv_hi_c0c130a042_0.wav", "cv_hi_f398e532c4_0.wav"],
                  "clone": ["xtts_hi_c0c130a042.wav", "xtts_hi_f398e532c4.wav"],
                  "edge": ["edge_hi_2_hi-IN-MadhurNeural.mp3", "edge_hi_5_hi-IN-SwaraNeural.mp3"]},
        "english": {"real": ["cv_en_28590f5e8d_0.wav", "cv_en_5fa722c6eb_0.wav"],
                    "clone": ["xtts_en_28590f5e8d.wav", "xtts_en_5fa722c6eb.wav"],
                    "edge": ["edge_en_2_en-IN-PrabhatNeural.mp3", "edge_en_4_en-IN-PrabhatNeural.mp3"]},
    }
    out = []
    for lang, kinds in picks.items():
        for i, (r, c) in enumerate(zip(kinds["real"], kinds["clone"]), 1):
            name_r = f"deepfake/{lang}/real_person{i}.wav"
            name_c = f"deepfake/{lang}/fake_xtts_clone_of_person{i}.wav"
            write(DEMO / name_r, dc.read(real / r))
            write(DEMO / name_c, dc.read(fake / c))
            m = meta[r]
            out.append({"file": name_r, "kind": "real", "lang": lang, "source": f"Mozilla Common Voice 17 ({m['lang']}), clip {Path(m['cv_path']).name}", "license": "CC0-1.0", "text": m["sentence"]})
            out.append({"file": name_c, "kind": "fake", "lang": lang, "source": f"Coqui XTTS-v2 voice clone of real_person{i} (cloned from that clip)", "license": "CPML (non-commercial)", "text": fakes[c]["text"]})
        for k, e in enumerate(kinds["edge"], 1):
            voice = fakes[e]["voice"]
            name = f"deepfake/{lang}/fake_edge_tts_{voice.split('-')[2].replace('Neural', '').lower()}_{k}.wav"
            write(DEMO / name, dc.read(fake / e))
            out.append({"file": name, "kind": "fake", "lang": lang, "source": f"Microsoft Edge neural TTS, stock voice {voice}", "license": "Generated with Microsoft Edge TTS", "text": fakes[e]["text"]})
    return out


def speaker_clips() -> list[dict]:
    import sv_benchmark as sb

    out = []
    room = sb.make_room(0)
    rng = np.random.default_rng(5)
    for lang, locale in (("hindi", "hi_in"), ("english", "en_in")):
        items = [it for it in sb.load_svq(locale, per_speaker_env=40) if it["env"] == "clean"]
        by: dict = {}
        for it in items:
            by.setdefault((it["speaker"], it["gender"]), []).append(it)
        # enough utterances per take for >= 10 s of speech (SVQ English questions are short)
        per = 3 if lang == "hindi" else 5
        males = sorted([k for k, v in by.items() if k[1] == "male" and len(v) >= 3 * per], key=lambda k: -len(by[k]))[:2]
        (a, _), (b, _) = males
        take = lambda spk, i: np.concatenate([sb._decode(x["audio"]) for x in by[(spk, "male")][per * i : per * i + per]])  # noqa: E731
        clips = {
            "person_A_take1": take(a, 0),
            "person_A_take2": take(a, 1),
            "person_B": take(b, 0),
        }
        # the friend's-house case: two different people, one laptop mic, one room
        clips["same_room_person_A"] = sb.in_room(take(a, 2), room, rng)
        clips["same_room_person_B"] = sb.in_room(take(b, 1), room, rng)
        for name, audio in clips.items():
            f = f"speaker/{lang}/{name}.wav"
            write(DEMO / f, audio)
            who = "A" if "person_A" in name else "B"
            spk = a if who == "A" else b
            out.append({"file": f, "kind": "real", "lang": lang, "person": who,
                        "source": f"Google SVQ {locale} ({spk.split('-')[-1]}), {per} utterances joined" + ("; passed through the same simulated room + laptop mic" if "same_room" in name else ""),
                        "license": "CC-BY-4.0"})
    return out


PAIRS = [
    ("hindi", "person_A_take1", "person_A_take2", "same"),
    ("hindi", "person_A_take1", "person_B", "different"),
    ("hindi", "same_room_person_A", "same_room_person_B", "different"),
    ("english", "person_A_take1", "person_A_take2", "same"),
    ("english", "person_A_take1", "person_B", "different"),
    ("english", "same_room_person_A", "same_room_person_B", "different"),
]


def score() -> dict:
    """Both stages exactly as the server runs them, plus the old VoxCeleb cut-off for comparison."""
    import torch

    import deepfake_check as dc
    import speaker

    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    det = dc.Detector()
    for c in manifest["clips"]:
        audio = dc.read(DEMO / c["file"])
        c["probSpoof"] = det(audio)["hindi"]  # the first 4.04 s, as the app hears it
    state = speaker.VerifierState()
    state.load()
    results = []
    enc = state.encoder
    for lang, ref, tst, expected in PAIRS:
        r, t = (dc.read(DEMO / f"speaker/{lang}/{n}.wav") for n in (ref, tst))
        v = speaker.verify(state, r, t)
        old = float(torch.nn.functional.cosine_similarity(
            enc.encode(torch.from_numpy(speaker.speech_only(r)).unsqueeze(0)),
            enc.encode(torch.from_numpy(speaker.speech_only(t)).unsqueeze(0)), dim=0))
        results.append({"lang": lang, "reference": ref, "test": tst, "expected": expected, **v,
                        "oldScore": old, "oldThreshold": speaker.VOXCELEB_THRESHOLD,
                        "oldOutcome": "MATCH" if old >= speaker.VOXCELEB_THRESHOLD else "NON_MATCH"})
    manifest["pairs"] = results
    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path)
    ap.add_argument("--fake", type=Path)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.real and a.fake:
        for old in (DEMO / "deepfake", DEMO / "speaker"):
            shutil.rmtree(old, ignore_errors=True)
        clips = deepfake_clips(a.real, a.fake, []) + speaker_clips()
        DEMO.mkdir(exist_ok=True)
        (DEMO / "manifest.json").write_text(json.dumps({"clips": clips}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(clips)} clips")
    if a.score:
        m = score()
        for c in m["clips"]:
            print(f"{c['kind']:4s} {c['file']:52s} p_spoof {c['probSpoof']:.3f}")
        for p in m["pairs"]:
            print(f"{p['lang']:7s} {p['reference']:20s} vs {p['test']:20s} expect {p['expected']:9s} new {p['outcome']:9s} {p['score']:.3f}/{p['threshold']:.3f}  old {p['oldOutcome']} {p['oldScore']:.3f}")


if __name__ == "__main__":
    main()
