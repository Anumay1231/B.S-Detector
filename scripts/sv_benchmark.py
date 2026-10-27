"""
Speaker-verification benchmark on Hindi and Indian-English speech, including
the "same room, same microphone" case that broke the web app.

    python scripts/sv_benchmark.py embed --model ecapa      # cache embeddings
    python scripts/sv_benchmark.py eval  --model ecapa      # score every backend

Why this exists: the deployed threshold (0.2528) was calibrated on VoxCeleb1,
where an impostor pair is two celebrities on two different YouTube channels.
In the app the impostor is a friend on the same laptop mic in the same room, and
usually the same language and accent. Nothing in VoxCeleb measures that.

Data (all public, speaker-labelled, downloaded through the Hugging Face cache):
  - Common Voice 17 Hindi (fixie-ai/common_voice_17_0, CC0): train split for the
    adaptation backend, test split for evaluation. Speaker = client_id.
  - Common Voice 17 English, accents tagged "India and South Asia" kept separate.
  - Google SVQ hi_in / en_in (google/svq, CC-BY-4.0): phone recordings in clean,
    background-speech, media-noise and traffic conditions.

"Same room" condition: both clips of a trial go through the same simulated room
(a synthetic reverberant impulse response, one microphone colouration filter and
one recorded-style noise bed). Genuine and impostor trials get the same treatment,
so a model that embeds the room instead of the voice shows up as impostor scores
rising, which is exactly the failure seen at home.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "src"))

import speaker  # noqa: E402  (speech_only, the app's exact preprocessing)

SR = 16_000
OUT = ROOT / "outputs" / "sv_bench"
HF = Path.home() / ".cache" / "huggingface" / "hub"
ROOMS = 3
MAX_SECONDS = 8.0


# ---------------------------------------------------------------- data


def _decode(blob: bytes) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, SR).numpy()
    return audio[: int(MAX_SECONDS * SR)]


def _files(pattern: str) -> list[str]:
    return sorted(glob.glob(str(HF / pattern)))


def _cv_files(lang: str, split: str) -> list[str]:
    # Hindi shards sit in hi/<split>/chunk-*/, English ones in en/<split>-*.parquet
    base = "datasets--fixie-ai--common_voice_17_0/snapshots/*"
    return _files(f"{base}/{lang}/{split}/*/*.parquet") + _files(f"{base}/{lang}/{split}-*.parquet")


def cv_speakers(lang: str, split: str) -> set[str]:
    import pyarrow.parquet as pq

    return {c for f in _cv_files(lang, split) for c in pq.read_table(f, columns=["client_id"]).column(0).to_pylist()}


def half(cid: str) -> int:
    """Stable speaker-level 50/50 split."""
    import hashlib

    return hashlib.md5(cid.encode()).digest()[0] & 1


def load_cv(lang: str, splits, per_speaker: int, max_speakers: int, exclude: set[str] = frozenset(), part: int | None = None, seed: int = 0):
    """Common Voice clips grouped by speaker (client_id). `exclude` / `part` keep train and test speaker-disjoint."""
    import pyarrow.parquet as pq

    rows = []
    for split in [splits] if isinstance(splits, str) else splits:
        for f in _cv_files(lang, split):
            t = pq.read_table(f, columns=["client_id", "gender", "path", "audio"])
            for cid, g, p, a in zip(*(t.column(c).to_pylist() for c in ["client_id", "gender", "path", "audio"])):
                if cid in exclude or (part is not None and half(cid) != part):
                    continue
                rows.append((cid, (g or "").split("_")[0] or "unknown", a["bytes"]))
    by = defaultdict(list)
    for cid, g, b in rows:
        by[cid].append((g, b))
    rng = np.random.default_rng(seed)
    spk = [s for s, v in by.items() if len(v) >= 2]
    rng.shuffle(spk)
    out = []
    for s in spk[:max_speakers]:
        items = by[s]
        for i in rng.permutation(len(items))[:per_speaker]:
            g, b = items[i]
            out.append({"speaker": f"cv-{lang}-{s[:12]}", "gender": g, "env": "cv", "audio": b})
    return out


def load_svq(locale: str, per_speaker_env: int, seed: int = 0):
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    out = []
    for f in _files(f"datasets--google--svq/snapshots/*/2.0.0/utts_{locale}_*.parquet"):
        t = pq.read_table(f, columns=["speaker_id", "speaker_gender", "environment", "waveform"])
        by = defaultdict(list)
        for s, g, e, w in zip(*(t.column(c).to_pylist() for c in ["speaker_id", "speaker_gender", "environment", "waveform"])):
            by[(s, g, e)].append(w["bytes"])
        for (s, g, e), blobs in by.items():
            for i in rng.permutation(len(blobs))[:per_speaker_env]:
                out.append({"speaker": f"svq-{locale}-{s}", "gender": g, "env": e, "audio": blobs[i]})
    return out


# ---------------------------------------------------------------- rooms


def make_room(seed: int):
    """One simulated room + mic: a reverberant impulse response, a mic EQ and a noise bed."""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(1000 + seed)
    rt60 = rng.uniform(0.35, 0.8)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / 0.25  # strong direct path, as with a laptop mic at arm's length
    rir /= np.sqrt(np.sum(rir**2))
    lo, hi = rng.uniform(80, 250), rng.uniform(3500, 7000)
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    # a room tone: low-passed hum + fan-like broadband noise
    base = rng.standard_normal(SR * 60)
    bn, an = butter(1, rng.uniform(300, 1500) / (SR / 2))
    bed = lfilter(bn, an, base) + 0.3 * base + 0.5 * np.sin(2 * np.pi * rng.choice([50, 100]) * np.arange(SR * 60) / SR)
    bed /= np.sqrt(np.mean(bed**2))
    return {"rir": rir, "eq": (b, a), "bed": bed, "snr": rng.uniform(12, 20), "rt60": rt60}


def in_room(audio: np.ndarray, room, rng) -> np.ndarray:
    from scipy.signal import fftconvolve, lfilter

    wet = fftconvolve(audio, room["rir"])[: len(audio)]
    wet = lfilter(*room["eq"], wet)
    pad = np.zeros(int(0.6 * SR))
    wet = np.concatenate([pad, wet, pad])
    start = rng.integers(0, len(room["bed"]) - len(wet))
    noise = room["bed"][start : start + len(wet)]
    p_sig = np.mean(wet[len(pad) : -len(pad)] ** 2) + 1e-12
    wet = wet + noise * np.sqrt(p_sig / 10 ** (room["snr"] / 10))
    return (wet / (np.abs(wet).max() + 1e-9) * 0.7).astype(np.float32)


# ---------------------------------------------------------------- models


class Ecapa:
    name = "ecapa"

    def __init__(self):
        from speaker_verification.encoder import SpeakerEncoder

        self.enc = SpeakerEncoder(device="cpu")

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torch

        return self.enc.encode(torch.from_numpy(wav).unsqueeze(0)).cpu().numpy()


MODELS = {"ecapa": Ecapa}


def get_model(name: str):
    if name in MODELS:
        return MODELS[name]()
    import sv_models  # alternatives live next to this script

    return sv_models.MODELS[name]()


# ---------------------------------------------------------------- embed


def corpora():
    # Hindi CV's own "train" split has only 3 speakers and nearly every other speaker
    # is in "test", so Hindi is split 50/50 by speaker over the whole validated set.
    return {
        "cv_hi_train": lambda: load_cv("hi", "validated", per_speaker=8, max_speakers=400, part=0),
        "cv_en_train": lambda: load_cv("en", "train", per_speaker=4, max_speakers=600, exclude=cv_speakers("en", "test")),
        "cv_hi_test": lambda: load_cv("hi", "validated", per_speaker=6, max_speakers=200, part=1, seed=1),
        "cv_en_test": lambda: load_cv("en", "test", per_speaker=6, max_speakers=150, seed=1),
        "svq_hi": lambda: load_svq("hi_in", per_speaker_env=12),
        "svq_en": lambda: load_svq("en_in", per_speaker_env=12),
    }


#: Only the evaluation sets are also rendered in the simulated rooms.
EVAL_SETS = ("cv_hi_test", "cv_en_test", "svq_hi", "svq_en")


def embed(model_name: str, only: list[str] | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = get_model(model_name)
    rooms = [make_room(r) for r in range(ROOMS)]
    for name, loader in corpora().items():
        if only and name not in only:
            continue
        path = OUT / f"{model_name}__{name}.npz"
        if path.exists():
            print(f"[skip] {path.name}")
            continue
        t0 = time.time()
        items = loader()
        conds = ["clean"] + ([f"room{r}" for r in range(ROOMS)] if name in EVAL_SETS else [])
        embs = {c: [] for c in conds}
        keep = []
        rng = np.random.default_rng(7)
        for i, it in enumerate(items):
            try:
                audio = _decode(it["audio"])
            except Exception as exc:  # a broken file in the corpus
                print("  skip", exc)
                continue
            vecs = {}
            for c in conds:
                wav = audio if c == "clean" else in_room(audio, rooms[int(c[4:])], rng)
                wav = speaker.speech_only(wav)
                if len(wav) < SR:
                    break
                vecs[c] = model(wav)
            if len(vecs) != len(conds):
                continue
            for c in conds:
                embs[c].append(vecs[c])
            keep.append(it)
            if i % 100 == 0:
                print(f"  {name}: {i}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(
            path,
            speaker=np.array([k["speaker"] for k in keep]),
            gender=np.array([k["gender"] for k in keep]),
            env=np.array([k["env"] for k in keep]),
            **{f"emb_{c}": np.stack(v) for c, v in embs.items()},
        )
        print(f"[done] {path.name}: {len(keep)} clips, {len(set(k['speaker'] for k in keep))} speakers, {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- scoring


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate([gen, imp])
    labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    order = np.argsort(-scores)
    labels, scores = labels[order], scores[order]
    tp = np.cumsum(labels) / len(gen)
    fp = np.cumsum(1 - labels) / len(imp)
    frr, far = 1 - tp, fp
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2), float(scores[i])


def trials(spk: np.ndarray, gender: np.ndarray, env: np.ndarray, rng, max_each: int = 6000, same_gender: bool = True):
    """Genuine: same speaker. Impostor: different speaker, same gender (harder, and the usual failure)."""
    n = len(spk)
    i, j = np.triu_indices(n, 1)
    same = spk[i] == spk[j]
    imp = ~same & ((gender[i] == gender[j]) if same_gender else True)
    gi, ii = np.flatnonzero(same), np.flatnonzero(imp)
    gi = rng.choice(gi, min(len(gi), max_each), replace=False)
    ii = rng.choice(ii, min(len(ii), max_each), replace=False)
    return (i[gi], j[gi]), (i[ii], j[ii])


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Backend:
    """Embedding -> score. Fitted only on the train corpora (speaker-disjoint from test)."""

    def __init__(self, kind: str, train: dict | None = None, lda_dim: int = 150, cohort_k: int = 300):
        self.kind = kind
        self.mean = None
        self.W = None
        self.cohort = None
        self.k = cohort_k
        if train is None or kind == "cosine":
            return
        X, y = train["emb"], train["speaker"]
        if "center" in kind or "lda" in kind:
            self.mean = X.mean(axis=0)
        if "lda" in kind:
            Xc = l2n(X - self.mean)
            self.W = lda(Xc, y, lda_dim)
        if "asnorm" in kind:
            # one averaged embedding per cohort speaker
            spk = np.unique(y)
            self.cohort = self.project(np.stack([X[y == s].mean(axis=0) for s in spk]))

    def project(self, X: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            X = X - self.mean
        X = l2n(X)
        if self.W is not None:
            X = l2n(X @ self.W)
        return X

    def score(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a, b = self.project(A), self.project(B)
        s = np.sum(a * b, axis=-1)
        if self.cohort is None:
            return s
        ca = np.sort(a @ self.cohort.T, axis=1)[:, -self.k :]
        cb = np.sort(b @ self.cohort.T, axis=1)[:, -self.k :]
        return 0.5 * ((s - ca.mean(1)) / ca.std(1) + (s - cb.mean(1)) / cb.std(1))


def lda(X: np.ndarray, y: np.ndarray, dim: int) -> np.ndarray:
    classes = np.unique(y)
    mu = X.mean(axis=0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for c in classes:
        Xc = X[y == c]
        m = Xc.mean(axis=0)
        Sw += (Xc - m).T @ (Xc - m)
        Sb += len(Xc) * np.outer(m - mu, m - mu)
    Sw /= len(X)
    Sb /= len(X)
    Sw += 1e-3 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    vals, vecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-vals.real)[:dim]
    return vecs[:, order].real


def load(model: str, name: str) -> dict:
    d = np.load(OUT / f"{model}__{name}.npz")
    return {k: d[k] for k in d.files}


def evaluate(model: str, deployed: float, backends: list[str], save: bool = True) -> dict:
    tr = [load(model, n) for n in ("cv_hi_train", "cv_en_train")]
    train = {"emb": np.concatenate([t["emb_clean"] for t in tr]), "speaker": np.concatenate([t["speaker"] for t in tr])}
    fitted = {b: Backend(b, train) for b in backends}
    rng = np.random.default_rng(0)
    report: dict = {"model": model, "deployedThreshold": deployed, "sets": {}}
    for name in EVAL_SETS:
        d = load(model, name)
        g, imp = trials(d["speaker"], d["gender"], d["env"], rng)
        conds = [k[4:] for k in d if k.startswith("emb_")]
        rep = {"speakers": int(len(set(d["speaker"]))), "genuine": int(len(g[0])), "impostor": int(len(imp[0])), "conditions": {}}
        for c in conds:
            E = d[f"emb_{c}"]
            rep["conditions"][c] = {}
            for b, be in fitted.items():
                sg, si = be.score(E[g[0]], E[g[1]]), be.score(E[imp[0]], E[imp[1]])
                e, thr = eer(sg, si)
                row = {"eer": e, "eerThreshold": thr, "impostorMean": float(si.mean()), "impostorP99": float(np.percentile(si, 99)), "genuineMean": float(sg.mean())}
                if b == "cosine":
                    row["farAtDeployed"] = float(np.mean(si >= deployed))
                    row["frrAtDeployed"] = float(np.mean(sg < deployed))
                rep["conditions"][c][b] = row
        report["sets"][name] = rep
    if save:
        (OUT / f"report_{model}.json").write_text(json.dumps(report, indent=2))
    return report


def print_report(r: dict) -> None:
    print(f"\n=== {r['model']} (deployed threshold {r['deployedThreshold']:.4f}) ===")
    for name, rep in r["sets"].items():
        print(f"\n{name}: {rep['speakers']} speakers, {rep['genuine']} genuine / {rep['impostor']} same-gender impostor trials")
        for c, rows in rep["conditions"].items():
            for b, row in rows.items():
                extra = f"  FAR@dep {row['farAtDeployed']*100:5.1f}%  FRR@dep {row['frrAtDeployed']*100:5.1f}%" if "farAtDeployed" in row else ""
                print(f"  {c:7s} {b:18s} EER {row['eer']*100:5.2f}%  thr {row['eerThreshold']:+.3f}  imp mean {row['impostorMean']:+.3f} p99 {row['impostorP99']:+.3f}  gen mean {row['genuineMean']:+.3f}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "eval"])
    ap.add_argument("--model", default="ecapa")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--deployed", type=float, default=speaker.CALIBRATED_THRESHOLD)
    ap.add_argument("--backends", nargs="*", default=["cosine", "center", "center+asnorm", "lda", "lda+asnorm"])
    a = ap.parse_args()
    if a.cmd == "embed":
        embed(a.model, a.only)
    else:
        print_report(evaluate(a.model, a.deployed, a.backends))


if __name__ == "__main__":
    main()
