"""
Generate synthetic speech for the deepfake test set and the demo folder.

Runs in a separate environment with coqui-tts and edge-tts installed (they pin
library versions the app does not want):

    python scripts/make_fakes.py --real <dir with meta.json> --out <dir>

Two kinds of fake, in Hindi and English:
  - xtts-clone: Coqui XTTS-v2 cloning a real Common Voice speaker (CC0) from one
    of their clips, then saying a different sentence. This is the attack the
    interlock exists for: a cloned voice presented as the real person.
  - edge: Microsoft Edge neural TTS stock voices (hi-IN, en-IN). This sends the
    sentence text, and nothing else, to Microsoft's online TTS service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

HI = [
    "मैं आज शाम को बाज़ार जाकर कुछ ताज़ी सब्ज़ियाँ खरीदूँगा।",
    "कृपया अपना खाता नंबर और पिन मुझे अभी बता दीजिए।",
    "कल सुबह की बैठक दस बजे की जगह ग्यारह बजे होगी।",
    "मेरी आवाज़ पहचानिए, मैं तुम्हारा भाई बोल रहा हूँ, तुरंत पैसे भेजो।",
    "बारिश की वजह से आज सारी ट्रेनें देर से चल रही हैं।",
    "इस परियोजना का अंतिम प्रस्तुतीकरण अगले सोमवार को है।",
]
EN = [
    "Please transfer the money to this account before the end of the day.",
    "The meeting has been moved to eleven tomorrow morning, see you there.",
    "Hi, it's me. I lost my phone, so I'm calling from a friend's number.",
    "The train was late again because of the heavy rain in the city.",
    "Our final project presentation is scheduled for next Monday afternoon.",
    "Can you read me the verification code that just came to your phone?",
]
EDGE = {"hi": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"], "en": ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]}


def xtts(meta: list[dict], real: Path, out: Path, per_lang: int) -> list[dict]:
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    # the XTTS-v2 weights as mirrored on the Hugging Face Hub
    ckpt = snapshot_download("tts-hub/XTTS-v2")
    config = XttsConfig()
    config.load_json(f"{ckpt}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=ckpt, eval=True)
    made = []
    for lang, texts in (("hi", HI), ("en", EN)):
        refs = [m for m in meta if m["lang"] == lang and m["file"].endswith("_0.wav")][:per_lang]
        for i, ref in enumerate(refs):
            text = texts[i % len(texts)]
            name = f"xtts_{lang}_{ref['speaker'][:10]}.wav"
            if not (out / name).exists():  # reruns pick up where they stopped
                with torch.no_grad():
                    wav = model.synthesize(text, config, speaker_wav=str(real / ref["file"]), language=lang)["wav"]
                sf.write(out / name, wav, 24000)
            made.append({"file": name, "lang": lang, "kind": "xtts-clone", "clonedFrom": ref["file"], "speaker": ref["speaker"], "text": text})
            print("xtts", name, flush=True)
    return made


async def edge(out: Path, per_lang: int) -> list[dict]:
    import edge_tts

    made = []
    for lang, texts in (("hi", HI), ("en", EN)):
        for i in range(per_lang):
            voice = EDGE[lang][i % 2]
            text = texts[i % len(texts)]
            mp3 = out / f"edge_{lang}_{i}_{voice}.mp3"
            if not mp3.exists():
                await edge_tts.Communicate(text, voice).save(str(mp3))
            made.append({"file": mp3.name, "lang": lang, "kind": "edge-tts", "voice": voice, "text": text})
            print("edge", mp3.name, flush=True)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xtts", type=int, default=10, help="cloned clips per language")
    ap.add_argument("--edge", type=int, default=8, help="stock-voice clips per language")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.real / "meta.json").read_text(encoding="utf-8"))
    made = asyncio.run(edge(a.out, a.edge)) if a.edge else []
    if a.xtts:
        made += xtts(meta, a.real, a.out, a.xtts)
    (a.out / "fakes.json").write_text(json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
