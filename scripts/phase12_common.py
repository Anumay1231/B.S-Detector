"""
Shared by the phase 12 scripts: which speakers and voices go to train, val or test.

Splits are by person (and by TTS voice), never by clip, so the held-out numbers
measure voices the head never heard. Every speaker in demo/ and in the bundled
samples is kept out of all three splits: they are the app check's own test set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "phase12"  # gitignored (data/raw/)

# demo/manifest.json and frontend/public/samples/manifest.json. SVQ uses one id
# per person across hi_in and en_in, so these are excluded in both languages.
DEMO_SVQ = {"speaker_7113944816261864857", "speaker_11298494297948342752", "speaker_8706784997864741703"}
# Common Voice clips used in demo/ (their speakers are looked up by client_id)
DEMO_CV_CLIPS = {"common_voice_hi_27637219.mp3", "common_voice_hi_23976026.mp3",
                 "common_voice_en_26950526.mp3", "common_voice_en_34693.mp3"}

# Edge-TTS stock voices. Neerja and Swara are never trained on (handover rule);
# two more voices are kept for validation so checkpoint choice never sees test voices.
EDGE_VOICES = {
    "train": {"hi": ["hi-IN-MadhurNeural", "mr-IN-ManoharNeural"],
              "en": ["en-IN-PrabhatNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-GB-RyanNeural"]},
    "val": {"hi": ["mr-IN-AarohiNeural"], "en": ["en-AU-NatashaNeural"]},
    "test": {"hi": ["hi-IN-SwaraNeural"], "en": ["en-IN-NeerjaNeural"]},
}


def _h(key: str) -> bytes:
    return hashlib.md5(key.encode("utf-8")).digest()


def split_of(key: str) -> str:
    """70% train, 10% val, 20% test by a stable hash of the speaker key (for large pools)."""
    b = _h(key)[0] % 10
    return "train" if b < 7 else "val" if b < 8 else "test"


def rank_split(keys) -> dict[str, str]:
    """The same 70/10/20 for small pools (SVQ has ~20 people): rank by hash, then cut,
    so every split gets someone. Val and test get at least one each."""
    order = sorted(set(keys), key=_h)
    n = len(order)
    n_test = max(1, round(0.2 * n))
    n_val = max(1, round(0.1 * n))
    return {k: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, k in enumerate(order)}
