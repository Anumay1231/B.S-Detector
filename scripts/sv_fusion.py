"""
Fit the Phase 9 speaker-check decision: ECAPA + ResNet34 score fusion, a
duration-aware threshold and an "unsure" band. Writes the constants the server
uses and a held-out report.

    python scripts/sv_fusion.py            # after sv_duration.py embed for both models

Fitted on Common Voice Hindi + English speakers plus half of the Google SVQ
speakers (hi_in, en_in), and reported on the other half of SVQ; then the halves
swap (2-fold cross-validation by speaker). SVQ records the same people in both
languages, so a person is always wholly on one side, in both languages. Common
Voice alone under-sets the threshold for SVQ's phone recordings (0.47 vs ~0.55
cosine at 9 s), which is why some SVQ audio is in the fitting data. The shipped
constants are fitted on Common Voice + all of SVQ.

Fusion: each model's cosine is z-normalised with the mean / std of same-gender
impostor scores on the fitting set, and the two are averaged.
Threshold: t(d) = a + b ln d, d = speech seconds in the shorter clip, from the
EER point in each duration bucket.
Unsure band: |score - t(d)| < m(d) gives UNCERTAIN; m(d) is the smallest margin
that keeps the error on the remaining (decided) fitting trials at TARGET_ERROR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv_benchmark as sb  # noqa: E402
import sv_duration as sd  # noqa: E402

MODELS = ("ecapa", "wespeaker_r34")
TARGET_ERROR = 0.02
#: Minimum half-width of the unsure band, in fused z units (ECAPA: the same in cosine,
#: 0.4 x its impostor std). The fitted margin reaches 0 by ~8 s, where same-room Hindi
#: still sits at ~91%; this floor lifts every >= 8 s condition above 95% in
#: cross-validation, answering 81-100% of pairs.
MARGIN_FLOOR = {"fused": 0.40, "ecapa": 0.40 * 0.1298}
OUT_JSON = sb.OUT / "fusion_fit.json"


def trial_index(d: dict, rng, max_each: int = 20000):
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
    return np.concatenate(I), np.concatenate(J), np.concatenate(Y)


def person(spk: np.ndarray) -> np.ndarray:
    """SVQ ids look like svq-hi_in-speaker_123; the same person has the same number in hi_in and en_in."""
    return np.array([x.split("-")[-1] for x in spk])


def fold_of(spk: np.ndarray) -> np.ndarray:
    import hashlib

    return np.array([hashlib.md5(p.encode()).digest()[0] & 1 for p in person(spk)])


def load_set(name: str, fold: int | None = None) -> dict:
    """Trials of one set; with `fold`, only trials whose two clips are both in that SVQ fold."""
    ds = {m: sd.load(m, name) for m in MODELS}
    if fold is not None:
        keep = fold_of(ds[MODELS[0]]["speaker"]) == fold
        ds = {m: {k: v[keep] for k, v in d.items()} for m, d in ds.items()}
    a, b = ds[MODELS[0]], ds[MODELS[1]]
    assert (a["speaker"] == b["speaker"]).all() and (a["cond"] == b["cond"]).all(), "sets are not aligned"
    rng = np.random.default_rng(0)
    I, J, Y = trial_index(a, rng)
    cos = {m: np.sum(sb.l2n(ds[m]["seg"])[I] * sb.l2n(ds[m]["seg"])[J], axis=1) for m in MODELS}
    dmin = np.minimum(a["seconds"][I], a["seconds"][J])
    return {"cos": cos, "y": Y, "dmin": dmin, "cond": a["cond"][I]}


def fuse(cos: dict, norm: dict) -> np.ndarray:
    return np.mean([(cos[m] - norm[m][0]) / norm[m][1] for m in MODELS], axis=0)


def fit_margin(score, dmin, y, thr_fn):
    xs, ms, ws = [], [], []
    for lo, hi in sd.BUCKETS:
        m = (dmin >= lo) & (dmin < hi)
        if m.sum() < 100:
            continue
        dist = score[m] - thr_fn(dmin[m])
        ym = y[m]
        best = None
        for margin in np.linspace(0, 3, 301):
            keep = np.abs(dist) >= margin
            g, i = keep & (ym == 1), keep & (ym == 0)
            if g.sum() < 10 or i.sum() < 10:
                break
            err = 0.5 * (np.mean(dist[g] < 0) + np.mean(dist[i] >= 0))
            if err <= TARGET_ERROR:
                best = margin
                break
        if best is not None:
            xs.append(np.log(np.median(dmin[m]))), ms.append(best), ws.append(np.sqrt(m.sum()))
    if len(xs) < 2:
        return float(np.mean(ms) if ms else 0.5), 0.0
    e, c = np.polyfit(xs, ms, 1, w=ws)
    return float(c), float(e)


def evaluate(s: dict, score, thr_fn, margin_fn, label: str) -> dict:
    print(f"\n{label}")
    print("  speech     cond   pairs   forced acc   decided   acc on decided")
    out = {}
    for lo, hi in [(3, 6), (6, 8), (8, 11), (11, 15), (15, 99), (8, 99)]:
        for c in ("clean", "room"):
            m = (s["dmin"] >= lo) & (s["dmin"] < hi) & ((s["cond"] == "clean") if c == "clean" else (s["cond"] != "clean"))
            g, i = m & (s["y"] == 1), m & (s["y"] == 0)
            if g.sum() < 15 or i.sum() < 15:
                continue
            t = thr_fn(s["dmin"])
            forced = 1 - 0.5 * (np.mean(score[g] < t[g]) + np.mean(score[i] >= t[i]))
            sure = np.abs(score - t) >= margin_fn(s["dmin"])
            gd, idd = g & sure, i & sure
            decided = (gd.sum() + idd.sum()) / (g.sum() + i.sum())
            acc_d = 1 - 0.5 * (np.mean(score[gd] < t[gd]) + np.mean(score[idd] >= t[idd])) if gd.sum() and idd.sum() else float("nan")
            out[f"{lo}-{hi}s/{c}"] = {"genuine": int(g.sum()), "impostor": int(i.sum()), "forcedAccuracy": forced, "decidedFraction": decided, "accuracyOnDecided": acc_d}
            print(f"  {lo:>2}-{hi:<2} s    {c:5s} {g.sum():4d}/{i.sum():5d}   {forced*100:6.2f}%    {decided*100:5.1f}%     {acc_d*100:6.2f}%")
    return out


def merge(parts: list) -> dict:
    return {k: (np.concatenate([f[k] for f in parts]) if k != "cos" else {m: np.concatenate([f["cos"][m] for f in parts]) for m in MODELS}) for k in parts[0]}


def fit_constants(s_fit: dict) -> dict:
    imp = s_fit["y"] == 0
    norm = {m: (float(s_fit["cos"][m][imp].mean()), float(s_fit["cos"][m][imp].std())) for m in MODELS}
    out = {"norm": norm}
    for system in ("ecapa", "fused"):
        sc = s_fit["cos"]["ecapa"] if system == "ecapa" else fuse(s_fit["cos"], norm)
        a, b, pts = sd.fit_threshold(sc, s_fit["dmin"], s_fit["y"])
        thr = lambda dm, a=a, b=b: a + b * np.log(np.clip(dm, 3.0, 20.0))  # noqa: E731
        c, e = fit_margin(sc, s_fit["dmin"], s_fit["y"], thr)
        out[system] = {"threshold": [a, b], "margin": [c, e], "marginFloor": MARGIN_FLOOR[system], "buckets": pts}
    return out


def score_with(k: dict, system: str, s: dict):
    sc = s["cos"]["ecapa"] if system == "ecapa" else fuse(s["cos"], k["norm"])
    a, b = k[system]["threshold"]
    c, e = k[system]["margin"]
    thr = lambda dm: a + b * np.log(np.clip(dm, 3.0, 20.0))  # noqa: E731
    floor = k[system].get("marginFloor", 0.0)
    mfn = lambda dm: np.maximum(floor, c + e * np.log(np.clip(dm, 3.0, 20.0)))  # noqa: E731
    return sc, thr, mfn


def main() -> None:
    cv = [load_set(n) for n in sd.FIT_SETS]
    result = {"targetError": TARGET_ERROR, "crossValidation": {}, "shipped": None}
    # 2-fold by SVQ person: fit on CV + one fold, report on the other
    for fold in (0, 1):
        k = fit_constants(merge(cv + [load_set(n, 1 - fold) for n in sd.TEST_SETS]))
        res = {"constants": k}
        for system in ("ecapa", "fused"):
            print(f"\n===== fold {fold}, {system}: threshold {k[system]['threshold']}, margin {k[system]['margin']}")
            for n in sd.TEST_SETS:
                s = load_set(n, fold)
                res[f"{system}/{n}"] = evaluate(s, *score_with(k, system, s), f"held out: {n} fold {fold}")
        result["crossValidation"][str(fold)] = res
    result["shipped"] = fit_constants(merge(cv + [load_set(n) for n in sd.TEST_SETS]))
    print("\nshipped constants (CV + all SVQ):", json.dumps(result["shipped"], indent=1))
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print("wrote", OUT_JSON)


def main_cv_only() -> None:
    fit_sets = [load_set(n) for n in sd.FIT_SETS]
    s_fit = merge(fit_sets)
    imp = s_fit["y"] == 0
    norm = {m: (float(s_fit["cos"][m][imp].mean()), float(s_fit["cos"][m][imp].std())) for m in MODELS}
    result = {"norm": norm, "targetError": TARGET_ERROR, "systems": {}}
    for system in ("ecapa", "fused"):
        score_of = (lambda s: s["cos"]["ecapa"]) if system == "ecapa" else (lambda s: fuse(s["cos"], norm))
        sc = score_of(s_fit)
        a, b, pts = sd.fit_threshold(sc, s_fit["dmin"], s_fit["y"])
        thr = lambda dm, a=a, b=b: a + b * np.log(np.clip(dm, 3.0, 20.0))  # noqa: E731
        c, e = fit_margin(sc, s_fit["dmin"], s_fit["y"], thr)
        mfn = lambda dm, c=c, e=e: np.maximum(0.0, c + e * np.log(np.clip(dm, 3.0, 20.0)))  # noqa: E731
        print(f"\n===== {system}: threshold {a:.4f} + {b:.4f} ln d  (buckets {pts});  unsure margin {c:.4f} + {e:.4f} ln d")
        sysres = {"threshold": [a, b], "margin": [c, e], "fit": evaluate(s_fit, sc, thr, mfn, "fit (Common Voice hi + en)")}
        for n in sd.TEST_SETS:
            s = load_set(n)
            sysres[n] = evaluate(s, score_of(s), thr, mfn, f"held out: {n}")
        result["systems"][system] = sysres
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
