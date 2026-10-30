"""
Phase 12: retrain the SLS head so it stops flagging real phone recordings.

Runs in a CUDA environment (see docs/deepfake-improvement-plan.md, phase 12):

    python scripts/phase12_data.py          # SVQ, Common Voice, IndicSynth
    python scripts/phase12_fakes.py         # Edge-TTS and XTTS-v2 fakes
    python scripts/train_phase12.py [--epochs 10] [--out models/sls_head_phase12.pth]

- Starts from models/sls_head_silence_invariant.pth; the XLS-R backbone stays frozen.
- Real: SVQ hi_in/en_in (all four environments) and Common Voice hi/en.
  Fake: IndicSynth XTTS-v2/FreeVC24, Edge-TTS stock voices, XTTS-v2 clones.
- Both classes get the same augmentation, drawn fresh every epoch: the window
  starts at 0 s, mid-utterance, or after 0-1 s of added quiet; then a phone line
  (8 kHz, mu-law), MP3 or Opus at low bitrates, background noise, random gain.
- Splits are by person and by TTS voice (phase12_common.py); demo speakers are in
  none of them. The checkpoint is chosen on val, never on test.

Writes the best head to --out and the training log to outputs/phase12_train.json.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from phase12_common import DATA, ROOT  # noqa: E402
from phase12_real import GATED, ai4bharat, common_voice, fleurs, indicsynth, indictts, svq  # noqa: E402

SR = 16_000
WINDOW = 64_600
MAX_SECONDS = 8  # longer clips are cut, to keep everything in RAM (16 GB)
CAP = {"train": 40, "val": 12, "test": 12}  # clips per speaker (Edge and XTTS fakes: no cap)
SOURCE_CAP = {"indictts-hi": 500}  # two studio speakers: take more of each
TOTAL = {"train": 1600, "val": 200, "test": 400}  # clips per source and split, taken round-robin over people
CONDITIONS = ("start", "phone", "mp3", "opus", "mid", "quiet_lead", "noise", "room", "replay")


# ------------------------------------------------------------------ audio


def decode(data: bytes | Path) -> np.ndarray | None:
    try:
        x, sr = sf.read(io.BytesIO(data) if isinstance(data, bytes) else data, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return None
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    x = x[: MAX_SECONDS * SR]
    if len(x) < SR or np.abs(x).max() < 1e-4:
        return None
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def tile(x: np.ndarray) -> np.ndarray:
    """What the server does with a clip shorter than the window."""
    if len(x) >= WINDOW:
        return x[:WINDOW]
    return np.tile(x, WINDOW // len(x) + 1)[:WINDOW]


def lead_trim(x: np.ndarray) -> np.ndarray:
    """Drop leading silence (frames under 1% of the peak frame energy)."""
    f = 400
    n = len(x) // f
    if n < 2:
        return x
    e = (x[: n * f].reshape(n, f) ** 2).mean(axis=1)
    on = np.argmax(e >= 0.01 * e.max())
    return x[on * f :]


def phone(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = rng.uniform(250, 350), rng.uniform(3200, 3500)
    y = resample_poly(x, 1, 2)
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=8000, output="sos"), y)
    mu = 255.0
    y = np.clip(y / max(1e-6, np.abs(y).max()) * np.abs(x).max(), -1, 1)
    q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) + 1) / 2 * 255)
    y = np.sign(q / 255 * 2 - 1) * ((1 + mu) ** np.abs(q / 255 * 2 - 1) - 1) / mu
    return resample_poly(y, 2, 1).astype(np.float32)[: len(x)]


def codec(x: np.ndarray, fmt: str, rng: np.random.Generator) -> np.ndarray:
    kind = {"format": "MP3"} if fmt == "mp3" else {"format": "OGG", "subtype": "OPUS"}
    buf = io.BytesIO()
    try:
        sf.write(buf, x, SR, compression_level=float(rng.uniform(0.6, 0.95)), **kind)
    except sf.LibsndfileError:  # some levels are refused near the top of the range
        buf = io.BytesIO()
        sf.write(buf, x, SR, **kind)
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float32")
    y = y if y.ndim == 1 else y.mean(axis=1)
    out = np.zeros_like(x)
    out[: min(len(x), len(y))] = y[: len(x)]
    return out


def room(x: np.ndarray, rng: np.random.Generator, replay: bool = False) -> np.ndarray:
    """A random room and mic (or, with replay, a small loudspeaker played into a room):
    reverberant impulse response, band-limiting EQ and a noise bed. Every call draws a
    new room, so none matches the fixed rooms the demo same-room clips were made with."""
    from scipy.signal import fftconvolve

    rt60 = rng.uniform(0.15, 0.8)
    t = np.arange(int(rt60 * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    rir[0] = rng.uniform(2, 8)  # direct path: close mic (high) to across the room (low)
    rir /= np.sqrt(np.sum(rir**2))
    y = fftconvolve(x, rir)[: len(x)]
    lo, hi = (rng.uniform(200, 450), rng.uniform(5000, 7500)) if replay else (rng.uniform(60, 250), rng.uniform(3500, 7900))
    y = sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), y)
    p = float((y**2).mean()) + 1e-12
    y = y + noise(len(y), rng, float(np.sqrt(p / 10 ** (rng.uniform(10, 30) / 10))))
    return (y / (np.abs(y).max() + 1e-9) * np.abs(x).max()).astype(np.float32)


def noise(n: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """Room-tone-ish noise: white noise with a gentle low-pass."""
    w = rng.standard_normal(n).astype(np.float32)
    return (sosfilt(butter(2, rng.uniform(1500, 6000), fs=SR, output="sos"), w) * level).astype(np.float32)


def augment(clip: np.ndarray, rng: np.random.Generator, condition: str | None = None) -> np.ndarray:
    """One 4.04 s training window. `condition` fixes the recipe (for val/test)."""
    x = clip.astype(np.float32) / 32767
    cond = condition or rng.choice(["start", "mid", "quiet_lead"], p=[0.45, 0.35, 0.2])
    crop = cond if cond in ("start", "mid", "quiet_lead") else "start"
    if crop == "mid" and len(x) > WINDOW + SR // 2:
        s = int(rng.integers(SR // 2, len(x) - WINDOW))
        x = x[s : s + WINDOW]
    elif crop == "quiet_lead":
        x = lead_trim(x)
        lead = int(rng.uniform(0.05, 1.0) * SR)
        x = np.concatenate([noise(lead, rng, float(10 ** rng.uniform(-4.3, -2.5))), x])
    x = tile(x)
    channel = cond if cond in ("phone", "mp3", "opus", "noise") else None
    if condition is None:
        r = rng.random()
        if r < 0.25:
            x = room(x, rng)
        elif r < 0.33:
            x = room(x, rng, replay=True)
        channel = rng.choice([None, "phone", "mp3", "opus"], p=[0.35, 0.25, 0.2, 0.2])
        if rng.random() < 0.2:
            x = x + noise(len(x), rng, float(np.sqrt((x**2).mean()) * 10 ** (-rng.uniform(10, 30) / 20)))
        x = x * float(10 ** rng.uniform(-0.8, 0.15)) / max(1e-4, float(np.abs(x).max())) * 0.8
    elif cond in ("room", "replay"):
        x = room(x, rng, replay=cond == "replay")
    if channel == "phone":
        x = phone(x, rng)
    elif channel in ("mp3", "opus"):
        x = codec(x, channel, rng)
    elif channel == "noise":
        x = x + noise(len(x), rng, float(np.sqrt((x**2).mean()) * 10 ** (-15 / 20)))
    return np.clip(x, -1, 1).astype(np.float32)


def safe_augment(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """augment(), but one odd clip never stops an hour-long run: fall back to the plain window."""
    try:
        return augment(clip, rng)
    except Exception as exc:  # noqa: BLE001
        print(f"  augment failed ({type(exc).__name__}: {exc}); using the plain window", flush=True)
        return tile(clip.astype(np.float32) / 32767)


# ------------------------------------------------------------------ data


def collect() -> list[dict]:
    """Every clip as {label (1 real / 0 fake), source, split, speaker, audio int16}."""
    rng = random.Random(0)
    items: list[dict] = []

    def add(rows: list[dict], label: int, cap: bool = True) -> None:
        by = defaultdict(list)
        for r in rows:
            by[(r["split"], r["speaker"])].append(r)
        kept = Counter()
        groups = []
        for (split, _), rs in by.items():
            rng.shuffle(rs)
            groups.append((split, rs[: SOURCE_CAP.get(rs[0]["source"], CAP[split]) if cap else None]))
        rng.shuffle(groups)
        # round-robin: everyone's first clip, then everyone's second, ... until the total is reached
        for k in range(max((len(rs) for _, rs in groups), default=0)):
            for split, rs in groups:
                if k >= len(rs) or kept[(rs[k]["source"], split)] >= TOTAL[split]:
                    continue
                r = rs[k]
                a = decode(r["audio"] if "audio" in r else DATA / r["file"])
                if a is None:
                    continue
                items.append({"label": label, "source": r["source"], "split": split, "speaker": r["speaker"],
                              "lang": r.get("lang", ""), "audio": a})
                kept[(r["source"], split)] += 1
        print("  " + ", ".join(f"{s}/{sp}: {n}" for (s, sp), n in sorted(kept.items())), flush=True)

    t0 = time.time()
    print("real: SVQ", flush=True)
    add(svq(), 1)
    for lang in ("hi", "en"):
        print(f"real: Common Voice {lang}", flush=True)
        add(common_voice(lang), 1)
    print("real: FLEURS Hindi, IndicTTS Hindi (studio)", flush=True)
    add(fleurs(), 1)
    add(indictts(), 1)
    # gated AI4Bharat sets, when they have been downloaded by hand
    for name, lang in (("svarah", "en"), ("lahaja", "hi"), ("kathbath", "hi")):
        if any((GATED / name).glob("*.parquet")):
            print(f"real: {name}", flush=True)
            add(ai4bharat(name, lang), 1)
    print("fake: IndicSynth", flush=True)
    add(indicsynth(), 0)
    fakes = json.loads((DATA / "fakes" / "fakes.json").read_text(encoding="utf-8"))
    for f in fakes:
        f["source"] = "edge" if f["kind"] == "edge" else f"xtts-{f['lang']}"
        f["speaker"] = f.get("voice") or f.get("clonedSpeaker")
    print("fake: Edge-TTS and XTTS-v2", flush=True)
    add(fakes, 0, cap=False)
    print(f"collected {len(items)} clips in {time.time() - t0:.0f} s", flush=True)
    return items


def balanced_epoch(items: list[dict], n_per_class: int, rng: np.random.Generator) -> list[dict]:
    """n clips per class, spread evenly over the sources within each class."""
    out = []
    for label in (0, 1):
        by = defaultdict(list)
        for it in items:
            if it["label"] == label:
                by[it["source"]].append(it)
        per = n_per_class // len(by)
        for rs in by.values():
            out += [rs[i] for i in rng.choice(len(rs), per, replace=len(rs) < per)]
    rng.shuffle(out)
    return out


def fixed_set(items: list[dict], seed: int, per_source: int) -> list[tuple[dict, str, np.ndarray]]:
    """A fixed, seeded evaluation set: each clip under one condition, cycling through all."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for it in items:
        by[it["source"]].append(it)
    out = []
    for src, rs in sorted(by.items()):
        pick = [rs[i] for i in rng.choice(len(rs), min(per_source, len(rs)), replace=False)]
        for k, it in enumerate(pick):
            cond = CONDITIONS[k % len(CONDITIONS)]
            out.append((it, cond, augment(it["audio"], rng, cond)))
    return out


# ------------------------------------------------------------------ model


def load_models(device: str, head_path: Path):
    import inference as engine
    from transformers import Wav2Vec2Model

    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).half().eval()
    for p in backbone.parameters():
        p.requires_grad = False
    head = engine.SLSHead().to(device)
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    head.load_state_dict({k.removeprefix("module."): v for k, v in state.items() if not k.startswith("ssl_model.")})
    return backbone, head


@torch.no_grad()
def hidden(backbone, x: torch.Tensor) -> list[torch.Tensor]:
    return [h.float() for h in backbone(x.half(), output_hidden_states=True).hidden_states[1:]]


@torch.no_grad()
def score(backbone, head, waves: list[np.ndarray], device: str, bs: int = 16) -> np.ndarray:
    """P(spoof) per window, as the server computes it (index 0 is spoof)."""
    head.eval()
    out = []
    for i in range(0, len(waves), bs):
        x = torch.from_numpy(np.stack(waves[i : i + bs])).to(device)
        out.append(torch.exp(head(hidden(backbone, x))[0])[:, 0].cpu().numpy())
    return np.concatenate(out)


def report(evalset, p_spoof: np.ndarray) -> dict:
    """Balanced accuracy at 0.5, overall and per condition and per source."""
    rows = [(it["label"], cond, it["source"], ps) for (it, cond, _), ps in zip(evalset, p_spoof)]

    def ba(sel):
        real = [ps < 0.5 for lab, _, _, ps in sel if lab == 1]
        fake = [ps >= 0.5 for lab, _, _, ps in sel if lab == 0]
        parts = [np.mean(v) for v in (real, fake) if v]
        return float(np.mean(parts)) if parts else float("nan")

    by_cond = {c: ba([r for r in rows if r[1] == c]) for c in CONDITIONS}
    by_src = {s: float(np.mean([(ps < 0.5) == (lab == 1) for lab, _, src, ps in rows if src == s])) for s in sorted({r[2] for r in rows})}
    return {"balancedAccuracy": ba(rows), "worstCondition": min(by_cond.values()), "byCondition": by_cond, "bySource": by_src}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=2400, help="clips per class per epoch")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--init", type=Path, default=ROOT / "models" / "sls_head_silence_invariant.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sls_head_phase12.pth")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    a = ap.parse_args()
    if a.smoke:
        CAP.update(train=1, val=1, test=1)
        SOURCE_CAP.clear()
        a.epochs, a.per_class = 1, 64
        a.out = a.out.with_name("sls_head_phase12_smoke.pth")
    device = "cuda"
    torch.manual_seed(12)
    rng = np.random.default_rng(12)

    items = collect()
    train = [it for it in items if it["split"] == "train"]
    val = fixed_set([it for it in items if it["split"] == "val"], seed=1, per_source=12 if a.smoke else 140)
    print(f"train clips {len(train)}; val windows {len(val)}", flush=True)

    backbone, head = load_models(device, a.init)
    log = {"args": {k: str(v) for k, v in vars(a).items()}, "epochs": []}
    base = report(val, score(backbone, head, [w for _, _, w in val], device))
    print(f"epoch 0 (init) val BA {base['balancedAccuracy']:.4f} worst {base['worstCondition']:.4f} {base['byCondition']}", flush=True)
    log["epochs"].append({"epoch": 0, "val": base})
    best = (base["balancedAccuracy"] + base["worstCondition"]) / 2
    best_epoch = 0
    torch.save(head.state_dict(), a.out)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        head.train()
        batch = balanced_epoch(train, a.per_class, rng)
        losses = []
        for i in range(0, len(batch), a.batch):
            b = batch[i : i + a.batch]
            x = torch.from_numpy(np.stack([safe_augment(it["audio"], rng) for it in b])).to(device)
            y = torch.tensor([0 if it["label"] == 0 else 1 for it in b], device=device)
            logp, _ = head(hidden(backbone, x))
            loss = F.nll_loss(logp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = report(val, score(backbone, head, [w for _, _, w in val], device))
        sel = (v["balancedAccuracy"] + v["worstCondition"]) / 2
        print(f"epoch {epoch} loss {np.mean(losses):.4f} val BA {v['balancedAccuracy']:.4f} worst {v['worstCondition']:.4f} "
              f"{ {k: round(x, 3) for k, x in v['byCondition'].items()} } [{time.time() - t0:.0f} s]", flush=True)
        log["epochs"].append({"epoch": epoch, "loss": float(np.mean(losses)), "val": v})
        if sel > best:
            best, best_epoch = sel, epoch
            torch.save(head.state_dict(), a.out)
            print(f"  saved (selection score {sel:.4f})", flush=True)
    log["bestEpoch"] = best_epoch
    out = ROOT / "outputs" / "phase12_train.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log, indent=1), encoding="utf-8", newline="")
    print(f"best epoch {best_epoch}; head saved to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
