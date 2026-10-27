#!/usr/bin/env python3
"""scripts/eval_ecapa_comparison.py
-----------------------------------
Evaluates baseline ECAPA vs Indian-adapted ECAPA on held-out Indian speaker datasets
(CV17 Hindi Test and Google SVQ) across clean and simulated same-room conditions.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_ecapa")

SR = 16_000
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def make_room(seed: int) -> dict:
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(1000 + seed)
    rt60 = rng.uniform(0.35, 0.8)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / 0.25
    rir /= np.sqrt(np.sum(rir ** 2))
    lo, hi = rng.uniform(80, 250), rng.uniform(3500, 7000)
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    base = rng.standard_normal(SR * 60)
    bn, an = butter(1, rng.uniform(300, 1500) / (SR / 2))
    bed = lfilter(bn, an, base) + 0.3 * base + 0.5 * np.sin(2 * np.pi * rng.choice([50, 100]) * np.arange(SR * 60) / SR)
    bed /= np.sqrt(np.mean(bed ** 2))
    return {"rir": rir, "eq": (b, a), "bed": bed, "snr": rng.uniform(12, 20), "rt60": rt60}


def apply_room(audio: np.ndarray, room: dict, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve, lfilter
    wet = fftconvolve(audio, room["rir"])[: len(audio)]
    wet = lfilter(*room["eq"], wet)
    pad = np.zeros(int(0.3 * SR))
    wet = np.concatenate([pad, wet, pad])
    start = rng.integers(0, max(1, len(room["bed"]) - len(wet)))
    noise = room["bed"][start : start + len(wet)]
    p_sig = np.mean(wet[len(pad) : -len(pad)] ** 2) + 1e-12
    wet = wet + noise * np.sqrt(p_sig / 10 ** (room["snr"] / 10))
    peak = np.abs(wet).max() + 1e-9
    return (wet / peak * 0.7).astype(np.float32)


def decode_audio_bytes(blob: bytes) -> np.ndarray:
    import torchaudio
    a, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a.astype(np.float32)


def get_speaker_split(cid: str) -> int:
    import hashlib
    return hashlib.md5(cid.encode()).digest()[0] & 1


def load_held_out_trials(max_speakers: int = 80, per_spk: int = 4) -> List[Tuple[np.ndarray, np.ndarray, int, str, str]]:
    """Generates trials on strictly held-out Indian speakers (part=1, unseen in training).
    Returns tuples of: (wav1, wav2, is_genuine (1/0), condition ('clean'/'same_room'), dataset_name).
    """
    import pyarrow.parquet as pq

    trials = []
    rng = np.random.default_rng(2026)
    room = make_room(777)

    # 1. CV 17 Hindi Test (part=1)
    by_speaker_cv = defaultdict(list)
    cv_files = sorted(HF_CACHE.glob("datasets--fixie-ai--common_voice_17_0/snapshots/*/hi/validated/*/*.parquet"))
    for f in cv_files:
        t = pq.read_table(f, columns=["client_id", "audio"])
        for cid, a in zip(t.column("client_id").to_pylist(), t.column("audio").to_pylist()):
            if get_speaker_split(cid) == 1:
                by_speaker_cv[f"cv-hi-{cid[:12]}"].append(a["bytes"])

    cv_spks = [s for s, v in by_speaker_cv.items() if len(v) >= 2][:max_speakers]
    logger.info(f"Loaded {len(cv_spks)} held-out CV Hindi speakers.")

    # 2. SVQ Indian English (part=1)
    by_speaker_svq = defaultdict(list)
    svq_files = sorted(HF_CACHE.glob("datasets--google--svq/snapshots/*/*/*en_in*.parquet"))
    for f in svq_files:
        t = pq.read_table(f, columns=["speaker_id", "waveform"])
        for spk, w in zip(t.column("speaker_id").to_pylist(), t.column("waveform").to_pylist()):
            if get_speaker_split(spk) == 1:
                by_speaker_svq[f"svq-en-{spk}"].append(w["bytes"])

    svq_spks = [s for s, v in by_speaker_svq.items() if len(v) >= 2][:max_speakers]
    logger.info(f"Loaded {len(svq_spks)} held-out SVQ Indian English speakers.")

    for dname, spks, by_spk in [("CV-Hindi-Test", cv_spks, by_speaker_cv), ("SVQ-Indian-English", svq_spks, by_speaker_svq)]:
        # Build genuine pairs
        for s in spks:
            b1, b2 = by_spk[s][0], by_spk[s][1]
            w1, w2 = decode_audio_bytes(b1), decode_audio_bytes(b2)
            trials.append((w1, w2, 1, "clean", dname))

            # same room
            w1_r = apply_room(w1, room, rng)
            w2_r = apply_room(w2, room, rng)
            trials.append((w1_r, w2_r, 1, "same_room", dname))

        # Build impostor pairs
        for i in range(len(spks) - 1):
            s1, s2 = spks[i], spks[i + 1]
            w1, w2 = decode_audio_bytes(by_spk[s1][0]), decode_audio_bytes(by_spk[s2][0])
            trials.append((w1, w2, 0, "clean", dname))

            # same room
            w1_r = apply_room(w1, room, rng)
            w2_r = apply_room(w2, room, rng)
            trials.append((w1_r, w2_r, 0, "same_room", dname))

    return trials


def compute_eer(gen: np.ndarray, imp: np.ndarray) -> Tuple[float, float]:
    scores = np.concatenate([gen, imp])
    labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    order = np.argsort(-scores)
    labels, scores = labels[order], scores[order]
    tp = np.cumsum(labels) / max(1, len(gen))
    fp = np.cumsum(1 - labels) / max(1, len(imp))
    frr, far = 1 - tp, fp
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2), float(scores[i])


def evaluate_model(classifier, trials: list, device: str) -> dict:
    compute_features = classifier.mods.compute_features
    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model
    embedding_model.eval()

    def embed_wav(w: np.ndarray) -> np.ndarray:
        crop_len = int(3.0 * SR)
        if len(w) >= crop_len:
            w = w[:crop_len]
        else:
            w = np.tile(w, (crop_len // max(1, len(w))) + 1)[:crop_len]
        t = torch.from_numpy(w).unsqueeze(0).to(device).float()
        lens = torch.ones(1, device=device)
        with torch.no_grad():
            f = compute_features(t)
            f = mean_var_norm(f, lens)
            e = embedding_model(f, lens).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    results = defaultdict(lambda: {"gen": [], "imp": []})
    for w1, w2, label, cond, dname in trials:
        e1 = embed_wav(w1)
        e2 = embed_wav(w2)
        cos = float(np.dot(e1, e2))
        key = f"{dname}__{cond}"
        if label == 1:
            results[key]["gen"].append(cos)
        else:
            results[key]["imp"].append(cos)

    metrics = {}
    for key, data in results.items():
        gen = np.array(data["gen"])
        imp = np.array(data["imp"])
        eer_val, eer_th = compute_eer(gen, imp)
        # Threshold at 0.25 (typical deployed)
        far_025 = float(np.mean(imp >= 0.25))
        frr_025 = float(np.mean(gen < 0.25))
        acc_025 = float((np.mean(gen >= 0.25) + np.mean(imp < 0.25)) / 2.0)
        metrics[key] = {
            "eer": eer_val,
            "eer_th": eer_th,
            "gen_mean": float(np.mean(gen)),
            "imp_mean": float(np.mean(imp)),
            "far_025": far_025,
            "frr_025": frr_025,
            "acc_025": acc_025,
        }
    return metrics


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running evaluation on {device}...")

    trials = load_held_out_trials(max_speakers=70)
    logger.info(f"Total evaluation pairs: {len(trials)}")

    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    logger.info("--- Evaluating Pretrained Baseline (VoxCeleb ECAPA) ---")
    base_results = evaluate_model(classifier, trials, device)

    adapted_path = Path("models/ecapa_indian_adapted.pth")
    if not adapted_path.exists():
        logger.error(f"Adapted checkpoint {adapted_path} not found!")
        return

    logger.info(f"--- Loading Adapted Weights from {adapted_path} ---")
    classifier.mods.embedding_model.load_state_dict(torch.load(adapted_path, map_location=device))
    adapted_results = evaluate_model(classifier, trials, device)

    print("\n" + "=" * 90)
    print("                     ECAPA-TDNN INDIAN SPEAKER VERIFICATION BENCHMARK")
    print("=" * 90)
    print(f"{'Condition':<32} | {'Metric':<18} | {'Baseline (VoxCeleb)':<20} | {'Adapted (Indian)':<20}")
    print("-" * 90)

    for k in sorted(base_results.keys()):
        b = base_results[k]
        a = adapted_results[k]
        print(f"{k:<32} | {'EER':<18} | {b['eer']*100:6.2f}%              | {a['eer']*100:6.2f}%")
        print(f"{'':<32} | {'Accuracy @ 0.25':<18} | {b['acc_025']*100:6.2f}%              | {a['acc_025']*100:6.2f}%")
        print(f"{'':<32} | {'Gen / Imp Cos Mean':<18} | {b['gen_mean']:+.2f} / {b['imp_mean']:+.2f}         | {a['gen_mean']:+.2f} / {a['imp_mean']:+.2f}")
        print("-" * 90)


if __name__ == "__main__":
    main()
