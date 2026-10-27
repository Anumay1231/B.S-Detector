#!/usr/bin/env python3
"""scripts/train_ecapa_indian.py
--------------------------------
Fine-tunes ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) on Indian speakers
(Common Voice 17 Hindi + Google SVQ Indian English / Hindi) with simulated
room acoustics to boost same-room speaker verification accuracy.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ecapa_finetune")

SR = 16_000
CROP_SECONDS = 3.0
CROP_SAMPLES = int(CROP_SECONDS * SR)  # 48,000 samples
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# ---------------------------------------------------------------------------
# Room Simulation (matches scripts/sv_benchmark.py exactly)
# ---------------------------------------------------------------------------

def make_room(seed: int) -> dict:
    from scipy.signal import butter
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
    from scipy.signal import lfilter
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


# ---------------------------------------------------------------------------
# Data Loading & Partitioning
# ---------------------------------------------------------------------------

def ensure_datasets_downloaded():
    logger.info("Ensuring Common Voice 17 Hindi and Google SVQ are available...")
    token = os.environ.get("HF_TOKEN") or None
    if not token and (Path.home() / ".cache" / "huggingface" / "token").exists():
        token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()

    # 1. Download CV 17 Hindi validated parquet shards
    cv_pattern = "datasets--fixie-ai--common_voice_17_0/snapshots/*"
    if not list(HF_CACHE.glob(f"{cv_pattern}/hi/validated/*/*.parquet")):
        logger.info("Downloading fixie-ai/common_voice_17_0 (Hindi validated split)...")
        snapshot_download(
            repo_id="fixie-ai/common_voice_17_0",
            repo_type="dataset",
            allow_patterns=["hi/validated/*"],
            token=token,
        )

    # 2. Download Google SVQ Indian English and Hindi shards
    svq_pattern = "datasets--google--svq/snapshots/*"
    if not list(HF_CACHE.glob(f"{svq_pattern}/*/*hi_in*.parquet")):
        logger.info("Downloading google/svq (hi_in & en_in splits)...")
        snapshot_download(
            repo_id="google/svq",
            repo_type="dataset",
            allow_patterns=["*hi_in*", "*en_in*"],
            token=token,
        )
    logger.info("Datasets ready.")


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


def collect_training_samples(max_speakers: int = 500, per_speaker: int = 30) -> Tuple[List[dict], Dict[str, int]]:
    import pyarrow.parquet as pq

    ensure_datasets_downloaded()
    logger.info("Collecting speaker-disjoint Indian training data...")
    by_speaker = defaultdict(list)

    # 1. Common Voice 17 Hindi (train half: half(cid) == 0)
    cv_files = sorted(HF_CACHE.glob("datasets--fixie-ai--common_voice_17_0/snapshots/*/hi/validated/*/*.parquet"))
    for f in cv_files:
        t = pq.read_table(f, columns=["client_id", "audio"])
        for cid, a in zip(t.column("client_id").to_pylist(), t.column("audio").to_pylist()):
            if get_speaker_split(cid) == 0:  # Train split
                by_speaker[f"cv-hi-{cid[:12]}"].append(a["bytes"])

    # 2. Google SVQ (hi_in & en_in) (train half: half(cid) == 0)
    svq_files = sorted(HF_CACHE.glob("datasets--google--svq/snapshots/*/*/*hi_in*.parquet")) + \
                sorted(HF_CACHE.glob("datasets--google--svq/snapshots/*/*/*en_in*.parquet"))
    for f in svq_files:
        t = pq.read_table(f, columns=["speaker_id", "waveform"])
        for spk, w in zip(t.column("speaker_id").to_pylist(), t.column("waveform").to_pylist()):
            if get_speaker_split(spk) == 0:
                by_speaker[f"svq-{spk}"].append(w["bytes"])

    # Filter speakers with at least 3 utterances
    qualified = [s for s, v in by_speaker.items() if len(v) >= 3]
    rng = np.random.default_rng(42)
    rng.shuffle(qualified)
    selected_speakers = qualified[:max_speakers]
    speaker_to_id = {spk: idx for idx, spk in enumerate(selected_speakers)}

    samples = []
    for spk in selected_speakers:
        blobs = by_speaker[spk]
        rng.shuffle(blobs)
        for b in blobs[:per_speaker]:
            samples.append({"speaker": spk, "label": speaker_to_id[spk], "bytes": b})

    logger.info(f"Collected {len(samples)} utterances across {len(selected_speakers)} Indian speakers.")
    return samples, speaker_to_id


def collect_validation_pairs(num_speakers: int = 50, per_spk: int = 4) -> List[Tuple[np.ndarray, np.ndarray, int, str]]:
    """Builds held-out validation pairs from part=1 (strictly unseen during training)."""
    import pyarrow.parquet as pq

    by_speaker = defaultdict(list)
    cv_files = sorted(HF_CACHE.glob("datasets--fixie-ai--common_voice_17_0/snapshots/*/hi/validated/*/*.parquet"))
    for f in cv_files:
        t = pq.read_table(f, columns=["client_id", "audio"])
        for cid, a in zip(t.column("client_id").to_pylist(), t.column("audio").to_pylist()):
            if get_speaker_split(cid) == 1:  # Held-out split
                by_speaker[f"cv-hi-{cid[:12]}"].append(a["bytes"])

    svq_files = sorted(HF_CACHE.glob("datasets--google--svq/snapshots/*/*/*hi_in*.parquet")) + \
                sorted(HF_CACHE.glob("datasets--google--svq/snapshots/*/*/*en_in*.parquet"))
    for f in svq_files:
        t = pq.read_table(f, columns=["speaker_id", "waveform"])
        for spk, w in zip(t.column("speaker_id").to_pylist(), t.column("waveform").to_pylist()):
            if get_speaker_split(spk) == 1:
                by_speaker[f"svq-{spk}"].append(w["bytes"])

    qualified = [s for s, v in by_speaker.items() if len(v) >= per_spk]
    rng = np.random.default_rng(999)
    rng.shuffle(qualified)
    selected = qualified[:num_speakers]

    cached_audio = {}
    for s in selected:
        blobs = by_speaker[s]
        rng.shuffle(blobs)
        cached_audio[s] = [decode_audio_bytes(b) for b in blobs[:per_spk]]

    room = make_room(42)
    pairs = []
    # Genuine pairs (clean & same-room)
    for s in selected:
        wavs = cached_audio[s]
        if len(wavs) >= 2:
            pairs.append((wavs[0], wavs[1], 1, "clean"))
            # same room
            w0_room = apply_room(wavs[0], room, rng)
            w1_room = apply_room(wavs[1], room, rng)
            pairs.append((w0_room, w1_room, 1, "room"))

    # Impostor pairs (clean & same-room)
    for i in range(len(selected) - 1):
        s1, s2 = selected[i], selected[i + 1]
        w1, w2 = cached_audio[s1][0], cached_audio[s2][0]
        pairs.append((w1, w2, 0, "clean"))
        # same room (the classic hard failure mode)
        w1_room = apply_room(w1, room, rng)
        w2_room = apply_room(w2, room, rng)
        pairs.append((w1_room, w2_room, 0, "room"))

    return pairs


# ---------------------------------------------------------------------------
# Dataset & Batch Augmentation
# ---------------------------------------------------------------------------

class IndianSpeakerDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples
        self.rooms = [make_room(i) for i in range(5)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = decode_audio_bytes(item["bytes"])

        rng = np.random.default_rng()
        # 50% chance of simulated acoustic room augmentation
        if rng.random() < 0.50:
            room = rng.choice(self.rooms)
            audio = apply_room(audio, room, rng)

        # 3.0-second random crop / tile
        if len(audio) >= CROP_SAMPLES:
            start = rng.integers(0, len(audio) - CROP_SAMPLES + 1)
            crop = audio[start : start + CROP_SAMPLES]
        elif len(audio) > 1600:
            repeats = (CROP_SAMPLES // len(audio)) + 1
            crop = np.tile(audio, repeats)[:CROP_SAMPLES]
        else:
            crop = np.zeros(CROP_SAMPLES, dtype=np.float32)

        return torch.from_numpy(crop), item["label"]


# ---------------------------------------------------------------------------
# Additive Angular Margin Loss (AAM-Softmax)
# ---------------------------------------------------------------------------

class AAMSoftmax(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return F.cross_entropy(output, labels)


def evaluate_pairs(embedding_model, compute_features, mean_var_norm, pairs, device) -> dict:
    embedding_model.eval()
    clean_gen, clean_imp = [], []
    room_gen, room_imp = [], []

    with torch.no_grad():
        for w1, w2, label, cond in pairs:
            def prep(w):
                if len(w) >= CROP_SAMPLES:
                    w = w[:CROP_SAMPLES]
                else:
                    w = np.tile(w, (CROP_SAMPLES // max(1, len(w))) + 1)[:CROP_SAMPLES]
                t = torch.from_numpy(w).unsqueeze(0).to(device).float()
                lens = torch.ones(1, device=device)
                f = compute_features(t)
                f = mean_var_norm(f, lens)
                e = embedding_model(f, lens).squeeze().cpu().numpy()
                return e / (np.linalg.norm(e) + 1e-9)

            e1 = prep(w1)
            e2 = prep(w2)
            cos = float(np.dot(e1, e2))

            if cond == "clean":
                if label == 1:
                    clean_gen.append(cos)
                else:
                    clean_imp.append(cos)
            else:
                if label == 1:
                    room_gen.append(cos)
                else:
                    room_imp.append(cos)

    def calc_eer(gen, imp):
        scores = np.concatenate([gen, imp])
        labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
        order = np.argsort(-scores)
        labels, scores = labels[order], scores[order]
        tp = np.cumsum(labels) / max(1, len(gen))
        fp = np.cumsum(1 - labels) / max(1, len(imp))
        frr, far = 1 - tp, fp
        i = np.argmin(np.abs(frr - far))
        return float((frr[i] + far[i]) / 2), float(scores[i])

    clean_eer, clean_th = calc_eer(np.array(clean_gen), np.array(clean_imp))
    room_eer, room_th = calc_eer(np.array(room_gen), np.array(room_imp))
    embedding_model.train()
    return {
        "clean_eer": clean_eer,
        "clean_th": clean_th,
        "room_eer": room_eer,
        "room_th": room_th,
        "clean_gen_mean": float(np.mean(clean_gen)),
        "clean_imp_mean": float(np.mean(clean_imp)),
        "room_gen_mean": float(np.mean(room_gen)),
        "room_imp_mean": float(np.mean(room_imp)),
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(epochs: int = 15, batch_size: int = 64, lr: float = 1e-4, per_speaker: int = 30, out_dir: str = "models"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} (PyTorch {torch.__version__})")

    samples, speaker_to_id = collect_training_samples(max_speakers=500, per_speaker=per_speaker)
    dataset = IndianSpeakerDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    val_pairs = collect_validation_pairs(num_speakers=50, per_spk=4)
    logger.info(f"Collected {len(val_pairs)} held-out validation pairs (part=1, unseen speakers).")

    # Load pretrained ECAPA-TDNN from SpeechBrain
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    
    compute_features = classifier.mods.compute_features
    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model

    # Target loss: AAM-Softmax on 192-dim embeddings
    aam_loss = AAMSoftmax(in_features=192, out_features=len(speaker_to_id), s=30.0, m=0.20).to(device)

    # Optimizer with cosine decay
    optimizer = torch.optim.AdamW(
        list(embedding_model.parameters()) + list(aam_loss.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_model_path = out_path / "ecapa_indian_adapted.pth"

    logger.info("Evaluating baseline pretrained ECAPA on held-out validation pairs...")
    base_metrics = evaluate_pairs(embedding_model, compute_features, mean_var_norm, val_pairs, device)
    logger.info(
        f"Baseline -> Clean EER: {base_metrics['clean_eer']*100:.2f}% | "
        f"Same-Room EER: {base_metrics['room_eer']*100:.2f}% "
        f"(Room Gen Mean: {base_metrics['room_gen_mean']:.3f}, Imp Mean: {base_metrics['room_imp_mean']:.3f})"
    )

    logger.info(f"Starting ECAPA fine-tuning for {epochs} epochs...")
    embedding_model.train()
    best_room_eer = base_metrics["room_eer"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        batches = 0

        for wavs, labels in dataloader:
            wavs = wavs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            wav_lens = torch.ones(wavs.shape[0], device=device)
            feats = compute_features(wavs)
            feats = mean_var_norm(feats, wav_lens)
            embs = embedding_model(feats, wav_lens).squeeze(1)  # (B, 192)

            loss = aam_loss(embs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=3.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, batches)
        elapsed = time.time() - t0

        val_metrics = evaluate_pairs(embedding_model, compute_features, mean_var_norm, val_pairs, device)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - lr: {scheduler.get_last_lr()[0]:.2e} - "
            f"Time: {elapsed:.1f}s | Val Clean EER: {val_metrics['clean_eer']*100:.2f}%, "
            f"Room EER: {val_metrics['room_eer']*100:.2f}% (Imp Mean: {val_metrics['room_imp_mean']:.3f})"
        )

        if val_metrics["room_eer"] <= best_room_eer or epoch == epochs:
            best_room_eer = val_metrics["room_eer"]
            torch.save(embedding_model.state_dict(), saved_model_path)
            logger.info(f"  --> Saved new best checkpoint to {saved_model_path}")

    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--per-speaker", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, per_speaker=args.per_speaker, out_dir=args.out_dir)
