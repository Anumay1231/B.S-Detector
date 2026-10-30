"""
Make the Edge-TTS and XTTS-v2 fakes for the phase 12 detector head.

Runs in the GPU environment with coqui-tts and edge-tts installed:

    python scripts/phase12_fakes.py [--edge-only] [--xtts-only]

Writes data/raw/phase12/fakes/<kind>/<split>/*.{mp3,wav} and fakes.json there.
- edge: stock voices from phase12_common.EDGE_VOICES, reading Common Voice
  sentences (only the sentence text goes to Microsoft's TTS service).
- xtts: Coqui XTTS-v2 clones of Common Voice speakers, each saying a sentence
  of someone else's. A speaker's clone goes to the same split as the speaker,
  and no demo speaker is ever cloned.
Reruns skip files that already exist.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase12_common import DATA, EDGE_VOICES  # noqa: E402
from phase12_real import common_voice  # noqa: E402

OUT = DATA / "fakes"
COUNTS = {
    "edge": {"train": 80, "val": 50, "test": 80},  # per voice
    "xtts": {"en": {"train": 160, "val": 30, "test": 50}, "hi": {"train": 80, "val": 20, "test": 40}},
}


async def make_edge(cv: dict, rng: random.Random) -> list[dict]:
    import edge_tts

    made: list[dict] = []
    sem = asyncio.Semaphore(4)

    async def one(voice: str, text: str, path: Path, meta: dict) -> None:
        async with sem:
            for attempt in range(4):
                try:
                    if not path.exists():
                        await edge_tts.Communicate(text, voice).save(str(path))
                    made.append(meta)
                    return
                except Exception as exc:  # the service drops requests now and then
                    if attempt == 3:
                        print(f"edge failed {path.name}: {exc}", flush=True)
                    await asyncio.sleep(2 + 3 * attempt)

    jobs = []
    for split, by_lang in EDGE_VOICES.items():
        for lang, voices in by_lang.items():
            sentences = [r["sentence"] for r in cv[lang]]
            for voice in voices:
                d = OUT / "edge" / split
                d.mkdir(parents=True, exist_ok=True)
                for i in range(COUNTS["edge"][split]):
                    text = rng.choice(sentences)
                    p = d / f"{voice}_{i:03d}.mp3"
                    jobs.append(one(voice, text, p, {"file": str(p.relative_to(DATA)), "kind": "edge", "split": split,
                                                     "lang": lang, "voice": voice, "text": text}))
    await asyncio.gather(*jobs)
    print(f"edge: {len(made)} clips", flush=True)
    return made


def make_xtts(cv: dict, rng: random.Random) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    import TTS.tts.models.xtts as xtts_module
    from huggingface_hub import snapshot_download
    from scipy.signal import resample_poly
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    def load_audio(path, sampling_rate):
        """Coqui reads the reference clip through torchcodec, which needs FFmpeg DLLs on
        Windows; soundfile reads the WAV we write just as well."""
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != sampling_rate:
            g = np.gcd(int(sr), int(sampling_rate))
            x = resample_poly(x, int(sampling_rate) // g, int(sr) // g).astype(np.float32)
        return torch.from_numpy(np.clip(x, -1, 1)).unsqueeze(0)

    xtts_module.load_audio = load_audio

    ckpt = snapshot_download("tts-hub/XTTS-v2")
    config = XttsConfig()
    config.load_json(f"{ckpt}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=ckpt, eval=True)
    if torch.cuda.is_available():
        model.cuda()
    made = []
    tmp = OUT / "xtts" / "_ref.wav"
    for lang, per_split in COUNTS["xtts"].items():
        rows = cv[lang]
        sentences = [r["sentence"] for r in rows]
        for split, n in per_split.items():
            d = OUT / "xtts" / split
            d.mkdir(parents=True, exist_ok=True)
            # one clone per speaker, from a clip of at least 3 s
            by_spk: dict[str, list[dict]] = {}
            for r in rows:
                if r["split"] == split:
                    by_spk.setdefault(r["speaker"], []).append(r)
            speakers = sorted(by_spk)
            rng.shuffle(speakers)
            done = 0
            for spk in speakers:
                if done >= n:
                    break
                p = d / f"xtts_{lang}_{spk[:12]}.wav"
                ref = next((r for r in by_spk[spk] if r["seconds"] >= 3.0), None)
                if ref is None:
                    continue
                text = rng.choice([s for s in sentences if s != ref["sentence"]])
                if not p.exists():
                    audio, sr = sf.read(io.BytesIO(ref["audio"]), dtype="float32", always_2d=True)
                    sf.write(tmp, audio.mean(axis=1), sr)
                    try:
                        with torch.no_grad():
                            wav = model.synthesize(text, config, speaker_wav=str(tmp), language=lang)["wav"]
                    except Exception as exc:  # a bad reference clip; try the next speaker
                        print(f"xtts failed {spk[:12]}: {exc}", flush=True)
                        continue
                    sf.write(p, np.asarray(wav, dtype=np.float32), 24000)
                made.append({"file": str(p.relative_to(DATA)), "kind": "xtts", "split": split, "lang": lang,
                             "clonedSpeaker": spk, "text": text})
                done += 1
                if done % 20 == 0:
                    print(f"xtts {lang} {split}: {done}/{n}", flush=True)
            print(f"xtts {lang} {split}: {done} clips", flush=True)
    tmp.unlink(missing_ok=True)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-only", action="store_true")
    ap.add_argument("--xtts-only", action="store_true")
    a = ap.parse_args()
    rng = random.Random(12)
    cv = {lang: common_voice(lang, with_audio=not a.edge_only) for lang in ("hi", "en")}
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    if not a.xtts_only:
        made += asyncio.run(make_edge(cv, rng))
    if not a.edge_only:
        made += make_xtts(cv, rng)
    # read the index only now, so an --edge-only and an --xtts-only run can go side by side
    index = OUT / "fakes.json"
    kinds = {m["kind"] for m in made}
    old = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    made += [m for m in old if m["kind"] not in kinds]
    index.write_text(json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
