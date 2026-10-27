"""
Duration-aware calibration of the ECAPA cosine threshold on Hindi + English.

    python scripts/sv_duration.py embed     # concatenated clips -> embeddings
    python scripts/sv_duration.py fit       # threshold(seconds) + held-out report

sv_benchmark.py showed two things about the deployed ECAPA model on Indian
speakers: errors fall steeply with the amount of speech (Hindi same-room EER
15.9% at ~5 s per side, 2.8% at ~15 s), and the best threshold rises with it
(cosine ~0.40 at 5 s, ~0.58 at 15 s). A single VoxCeleb threshold (0.2528)
therefore cannot be right for both a 4 s and a 15 s clip.

Here every "clip" is 1-3 utterances of one speaker joined end to end, then run
through the app's own speech_only() and ECAPA, so durations span roughly 3-20 s
of speech. Trials pair clips recorded in the same condition (clean, or the same
simulated room on both sides, as when two people share one laptop mic).

The threshold is fitted only on Common Voice speakers (Hindi + English train
halves). It is then tested on Google SVQ (Hindi and Indian English), a different
corpus with different speakers and phones, so the reported numbers are held out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv_benchmark as sb  # noqa: E402
from sv_segments import segment_embed  # noqa: E402

OUT = sb.OUT
FIT_SETS = ("cv_hi_train", "cv_en_train")
TEST_SETS = ("svq_hi", "svq_en")
CONDS = ("clean", "room0", "room1")
MAX_GROUPS = 450


def groups(items: list[dict], rng) -> list[list[dict]]:
    """Split each speaker's clips into groups of 1, 2 or 3 (a short, medium or long clip)."""
    by: dict = {}
    for it in items:
        by.setdefault(it["speaker"], []).append(it)
    out = []
    for spk, its in by.items():
        its = [its[i] for i in rng.permutation(len(its))]
        i = 0
        while i < len(its):
            n = int(rng.integers(1, 4))
            if i + n > len(its):
                n = len(its) - i
            out.append(its[i : i + n])
            i += n
    return out


def embed(model_name: str, only: list[str] | None = None) -> None:
    model = sb.get_model(model_name)
    rooms = [sb.make_room(r) for r in range(sb.ROOMS)]
    loaders = sb.corpora()
    for name in only or FIT_SETS + TEST_SETS:
        path = OUT / f"dur_{model_name}__{name}.npz"
        if path.exists():
            print("[skip]", path.name)
            continue
        t0 = time.time()
        rng = np.random.default_rng(11)
        gs = groups(loaders[name](), rng)
        if len(gs) > MAX_GROUPS:  # a two-parameter curve does not need more
            gs = [gs[i] for i in sorted(rng.choice(len(gs), MAX_GROUPS, replace=False))]
        rec = {k: [] for k in ("speaker", "gender", "n", "cond", "seconds", "emb", "seg")}
        for gi, g in enumerate(gs):
            try:
                audio = np.concatenate([sb._decode(it["audio"]) for it in g])
            except Exception as exc:
                print("  skip", exc)
                continue
            for c in CONDS:
                wav = audio if c == "clean" else sb.in_room(audio, rooms[int(c[4:])], rng)
                wav = sb.speaker.speech_only(wav)
                if len(wav) < sb.SR:
                    continue
                rec["speaker"].append(g[0]["speaker"])
                rec["gender"].append(g[0]["gender"])
                rec["n"].append(len(g))
                rec["cond"].append(c)
                rec["seconds"].append(len(wav) / sb.SR)
                rec["emb"].append(model(wav))
                rec["seg"].append(segment_embed(model, wav))
            if gi % 50 == 0:
                print(f"  {name}: {gi}/{len(gs)} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(path, **{k: np.array(v) for k, v in rec.items()})
        print(f"[done] {path.name}: {len(rec['emb'])} clips in {time.time() - t0:.0f} s", flush=True)


def load(model: str, name: str) -> dict:
    d = np.load(OUT / f"dur_{model}__{name}.npz")
    return {k: d[k] for k in d.files}


def pair_trials(d: dict, rng, max_each: int = 20000):
    """Same-condition pairs. Impostors are same-gender; returns index arrays + labels."""
    E = sb.l2n(d["seg"] if "seg" in d else d["emb"])
    I, J, Y = [], [], []
    for c in np.unique(d["cond"]):
        idx = np.flatnonzero(d["cond"] == c)
        i, j = np.triu_indices(len(idx), 1)
        i, j = idx[i], idx[j]
        same = d["speaker"][i] == d["speaker"][j]
        imp = ~same & (d["gender"][i] == d["gender"][j])
        for mask, y in ((same, 1), (imp, 0)):
            k = np.flatnonzero(mask)
            k = rng.choice(k, min(len(k), max_each // 3), replace=False)
            I.append(i[k]), J.append(j[k]), Y.append(np.full(len(k), y))
    I, J, Y = map(np.concatenate, (I, J, Y))
    score = np.sum(E[I] * E[J], axis=1)
    dmin = np.minimum(d["seconds"][I], d["seconds"][J])
    return score, dmin, Y, d["cond"][I]


BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 11), (11, 15), (15, 99)]


def fit_threshold(score, dmin, y):
    """EER threshold per duration bucket, then t(d) = a + b*ln(d) by weighted least squares."""
    xs, ts, ws = [], [], []
    for lo, hi in BUCKETS:
        m = (dmin >= lo) & (dmin < hi)
        if (y[m] == 1).sum() < 30 or (y[m] == 0).sum() < 30:
            continue
        _, thr = sb.eer(score[m & (y == 1)], score[m & (y == 0)])
        xs.append(np.log(np.median(dmin[m])))
        ts.append(thr)
        ws.append(np.sqrt(m.sum()))
    b, a = np.polyfit(xs, ts, 1, w=ws)
    return float(a), float(b), list(zip(np.exp(xs).round(1).tolist(), np.round(ts, 3).tolist()))


def report(score, dmin, y, cond, thr_fn, label: str) -> dict:
    out = {}
    print(f"\n{label}")
    print("  speech (shorter clip)  cond    n_gen  n_imp   EER    acc@t(d)  FAR@t(d)  FRR@t(d)  FAR@0.253")
    for lo, hi in BUCKETS:
        for c in ("clean", "room"):
            m = (dmin >= lo) & (dmin < hi) & ((cond == "clean") if c == "clean" else (cond != "clean"))
            g, i = m & (y == 1), m & (y == 0)
            if g.sum() < 20 or i.sum() < 20:
                continue
            e, _ = sb.eer(score[g], score[i])
            t = thr_fn(dmin)
            far, frr = float(np.mean(score[i] >= t[i])), float(np.mean(score[g] < t[g]))
            acc = 1 - (far + frr) / 2
            far_old = float(np.mean(score[i] >= sb.speaker.CALIBRATED_THRESHOLD))
            key = f"{lo}-{hi}s/{c}"
            out[key] = {"genuine": int(g.sum()), "impostor": int(i.sum()), "eer": e, "balancedAccuracy": acc, "far": far, "frr": frr, "farAtVoxCelebThreshold": far_old}
            print(f"  {lo:>2}-{hi:<2} s              {c:6s} {g.sum():6d} {i.sum():6d}  {e*100:5.2f}%   {acc*100:6.2f}%   {far*100:6.2f}%   {frr*100:6.2f}%   {far_old*100:6.2f}%")
    return out


def fit(model: str) -> None:
    rng = np.random.default_rng(0)
    names = [n for n in FIT_SETS if (OUT / f"dur_{model}__{n}.npz").exists()]
    print("fitting on", names)
    parts = [pair_trials(load(model, n), rng) for n in names]
    s, d, y, c = (np.concatenate(z) for z in zip(*parts))
    a, b, points = fit_threshold(s, d, y)
    print(f"threshold(d) = {a:.4f} + {b:.4f} * ln(d)   bucket EER thresholds (median s, thr): {points}")
    thr = lambda dm: a + b * np.log(np.clip(dm, 1.0, 30.0))  # noqa: E731
    result = {"model": model, "a": a, "b": b, "bucketThresholds": points, "fit": report(s, d, y, c, thr, "fit: Common Voice hi+en (train speakers)")}
    for n in TEST_SETS:
        s, d, y, c = pair_trials(load(model, n), rng)
        result[n] = report(s, d, y, c, thr, f"held out: {n}")
    (OUT / f"duration_fit_{model}.json").write_text(json.dumps(result, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "fit"])
    ap.add_argument("--model", default="ecapa")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    embed(a.model, a.only) if a.cmd == "embed" else fit(a.model)


if __name__ == "__main__":
    main()
