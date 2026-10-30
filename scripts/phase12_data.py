"""
Download the data for the phase 12 detector head into the Hugging Face cache
(where scripts/sv_benchmark.py also looks for it).

    python scripts/phase12_data.py

Real speech: Google SVQ hi_in/en_in in all four recording environments, Common
Voice 17 Hindi (validated) and English (first 8 train shards), FLEURS Hindi and
IndicTTS Hindi studio speech. Fakes: a few
IndicSynth Hindi shards (XTTS-v2 and FreeVC24). Edge-TTS and English XTTS-v2
fakes are made separately by scripts/phase12_fakes.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
INDICSYNTH_SHARDS = [0, 13, 27, 40, 54, 67, 81, 94]  # spread over the 107 Hindi shards
CV_EN_SHARDS = 8  # Common Voice English: more shards, more Indian-accented speakers


def get(repo: str, name: str) -> Path:
    p = Path(hf_hub_download(repo, name, repo_type="dataset"))
    print(f"{p.stat().st_size / 1e6:8.1f} MB  {repo}/{name}", flush=True)
    return p


def main() -> int:
    for loc in ("hi_in", "en_in"):
        for env in ("clean", "background_speech", "media_noise", "traffic_noise"):
            get("google/svq", f"2.0.0/utts_{loc}_{env}.parquet")
    files = HfApi().list_repo_files("fixie-ai/common_voice_17_0", repo_type="dataset")
    for f in sorted(x for x in files if x.startswith("hi/validated/")):
        get("fixie-ai/common_voice_17_0", f)
    for f in sorted(x for x in files if x.startswith("en/train-"))[:CV_EN_SHARDS]:
        get("fixie-ai/common_voice_17_0", f)
    for s in INDICSYNTH_SHARDS:
        get("vdivyasharma/IndicSynth", f"Hindi/train-{s:05d}-of-00107.parquet")
    # FLEURS Hindi read speech; its validation split holds the bundled real-hindi sample, so it is not used
    for split in ("train", "test"):
        hf_hub_download("google/fleurs", f"hi_in/{split}/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
        print(f"google/fleurs hi_in/{split}", flush=True)
    # studio recordings of real people: TTS voices are trained on audio like this, so the head
    # has to learn that a clean studio voice is not by itself a sign of a fake
    for s in (0, 5):
        get("SPRINGLab/IndicTTS-Hindi", f"data/train-{s:05d}-of-00010.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
