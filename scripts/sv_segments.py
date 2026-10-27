"""
Whole-clip vs segment-averaged ECAPA embeddings on the same held-out clips.

    python scripts/sv_segments.py

Each SVQ "clip" is 1-3 utterances of one speaker joined end to end (as in
sv_duration.py). Method "whole" embeds the joined speech in one pass, as the app
did. Method "segments" cuts the speech into SEGMENT_SECONDS pieces, embeds each
and averages the length-normalised embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv_benchmark as sb  # noqa: E402

SEGMENT_SECONDS = 3.0


def segment_embed(model, wav: np.ndarray, seconds: float = SEGMENT_SECONDS) -> np.ndarray:
    n = int(seconds * sb.SR)
    if len(wav) < 1.5 * n:
        return sb.l2n(model(wav))
    k = len(wav) // n
    segs = [wav[i * n : (i + 1) * n] for i in range(k)]
    if len(wav) - k * n > n // 2:
        segs.append(wav[k * n :])
    return sb.l2n(np.mean([sb.l2n(model(s)) for s in segs], axis=0))


def main() -> None:
    import sv_duration as sd

    model = sb.get_model("ecapa")
    rooms = [sb.make_room(r) for r in range(sb.ROOMS)]
    loaders = sb.corpora()
    for name in ("svq_hi", "svq_en"):
        rng = np.random.default_rng(11)
        gs = sd.groups(loaders[name](), rng)
        rec = {k: [] for k in ("speaker", "gender", "cond", "seconds", "whole", "segments")}
        for g in gs:
            audio = np.concatenate([sb._decode(it["audio"]) for it in g])
            for c in ("clean", "room0"):
                wav = audio if c == "clean" else sb.in_room(audio, rooms[0], rng)
                wav = sb.speaker.speech_only(wav)
                if len(wav) < sb.SR:
                    continue
                rec["speaker"].append(g[0]["speaker"]); rec["gender"].append(g[0]["gender"]); rec["cond"].append(c)
                rec["seconds"].append(len(wav) / sb.SR)
                rec["whole"].append(sb.l2n(model(wav))); rec["segments"].append(segment_embed(model, wav))
        d = {k: np.array(v) for k, v in rec.items()}
        np.savez(sb.OUT / f"seg_ecapa__{name}.npz", **d)
        for method in ("whole", "segments"):
            s, dm, y, cond = sd.pair_trials({**d, "emb": d[method]}, np.random.default_rng(0))
            print(f"\n{name} / {method}")
            for lo, hi in [(0, 6), (6, 11), (11, 99)]:
                for c in ("clean", "room"):
                    m = (dm >= lo) & (dm < hi) & ((cond == "clean") if c == "clean" else (cond != "clean"))
                    if (m & (y == 1)).sum() < 15:
                        continue
                    e, t = sb.eer(s[m & (y == 1)], s[m & (y == 0)])
                    print(f"  {lo:>2}-{hi:<2}s {c:5s} gen {int((m & (y == 1)).sum()):4d}  EER {e*100:5.2f}%  thr {t:.3f}")


if __name__ == "__main__":
    main()
