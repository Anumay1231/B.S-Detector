"""scripts/run_indicsynth_finetune.py: End-to-end IndicSynth augmentation, fine-tuning, and evaluation.

Streams IndicSynth Hindi (male/female, FreeVC/XTTS-v2/VITS), pairs with bona fide Hindi,
extracts XLS-R 300M features, fine-tunes from the N=500 SLS checkpoint, and evaluates.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from scipy.signal import resample
from transformers import Wav2Vec2Model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("indicsynth_pipeline")

TARGET_SR = 16000
INPUT_LENGTH = 64600  # 4.04s at 16kHz
HIDDEN_DIM = 1024
NUM_LAYERS = 24


# ---------------------------------------------------------------------------
# Model definitions (exact match to SLS architecture)
# ---------------------------------------------------------------------------

class SLSHead(nn.Module):
    """SLS classifier head matching models/sls_head_n500_seed2_best.pth."""

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(HIDDEN_DIM, 1)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        weights = []
        for layer_out in hidden_states_list:
            pooled = layer_out.mean(dim=1)  # (B, 1024)
            w = torch.sigmoid(self.fc0(pooled))  # (B, 1)
            weights.append(w)

        fused = torch.zeros_like(hidden_states_list[0])
        for w, layer_out in zip(weights, hidden_states_list):
            fused = fused + w.unsqueeze(-1) * layer_out

        x = fused.unsqueeze(1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=3)
        x = x.flatten(start_dim=1)
        x = self.selu(self.fc1(x))
        x = self.fc3(x)
        return self.logsoftmax(x)


def load_n500_weights(sls_head: SLSHead, checkpoint_path: Path) -> SLSHead:
    """Load weights from local N=500 checkpoint."""
    logger.info(f"Loading checkpoint weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("ssl_model.", "")
        if clean_k in sls_head.state_dict():
            cleaned[clean_k] = v
    sls_head.load_state_dict(cleaned, strict=False)
    logger.info(f"Loaded {len(cleaned)} keys into SLS head.")
    return sls_head


# ---------------------------------------------------------------------------
# Audio Preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(raw_audio: np.ndarray, orig_sr: int) -> torch.Tensor:
    """Resamples to 16kHz mono and pads/truncates to 64,600 samples."""
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    if np.issubdtype(raw_audio.dtype, np.integer):
        max_val = float(np.iinfo(raw_audio.dtype).max)
        raw_audio = raw_audio.astype(np.float32) / max_val
    else:
        raw_audio = raw_audio.astype(np.float32)

    if orig_sr != TARGET_SR:
        num_target_samples = int(round(len(raw_audio) * float(TARGET_SR) / float(orig_sr)))
        raw_audio = resample(raw_audio, num_target_samples).astype(np.float32)

    # Pad or truncate
    if len(raw_audio) < INPUT_LENGTH:
        padded = np.zeros(INPUT_LENGTH, dtype=np.float32)
        padded[:len(raw_audio)] = raw_audio
        raw_audio = padded
    else:
        raw_audio = raw_audio[:INPUT_LENGTH]

    return torch.from_numpy(raw_audio)


# ---------------------------------------------------------------------------
# Data Streaming & Stratification
# ---------------------------------------------------------------------------

def collect_indicsynth_spoofs(target_per_cell: int = 100) -> List[dict]:
    """Collects spoof samples from IndicSynth across Gender x Generator."""
    logger.info("Streaming IndicSynth (Hindi)...")
    ds = load_dataset("vdivyasharma/IndicSynth", name="Hindi", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {
        ("male", "freevc24"): 0,
        ("male", "xtts_v2"): 0,
        ("male", "vits"): 0,
        ("female", "freevc24"): 0,
        ("female", "xtts_v2"): 0,
        ("female", "vits"): 0,
    }
    samples = []

    for item in ds:
        gender = str(item.get("Gender", "")).strip().lower()
        model_str = str(item.get("Generative Model", "")).strip().lower()

        if "freevc" in model_str:
            model = "freevc24"
        elif "xtts" in model_str:
            model = "xtts_v2"
        elif "vits" in model_str:
            model = "vits"
        else:
            continue

        if (gender, model) not in counts or counts[(gender, model)] >= target_per_cell:
            continue

        raw_bytes = item["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(raw_bytes))
        wave = preprocess_audio(data, sr)

        samples.append({
            "id": f"indicsynth_{gender}_{model}_{counts[(gender, model)]:04d}",
            "label": 0,  # spoof
            "gender": gender,
            "generator": model,
            "speaker_id": str(item.get("Speaker ID", "")),
            "waveform": wave,
        })
        counts[(gender, model)] += 1

        if all(c >= target_per_cell for c in counts.values()):
            break

    logger.info(f"Collected {len(samples)} IndicSynth spoof samples: {counts}")
    return samples


def collect_fleurs_bonafide(target_per_gender: int = 300) -> List[dict]:
    """Collects genuine Hindi samples from google/fleurs (male and female)."""
    logger.info("Streaming google/fleurs (hi_in) for bona fide speech...")
    ds = load_dataset("google/fleurs", "hi_in", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {"male": 0, "female": 0}
    samples = []

    for item in ds:
        # gender 0 = male, 1 = female in FLEURS
        raw_g = item.get("gender")
        gender = "female" if raw_g == 1 else "male"

        if counts[gender] >= target_per_gender:
            continue

        raw_bytes = item["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(raw_bytes))
        wave = preprocess_audio(data, sr)

        samples.append({
            "id": f"fleurs_{gender}_{counts[gender]:04d}",
            "label": 1,  # bonafide
            "gender": gender,
            "generator": "bonafide_human",
            "speaker_id": str(item.get("speaker_id", "")),
            "waveform": wave,
        })
        counts[gender] += 1

        if counts["male"] >= target_per_gender and counts["female"] >= target_per_gender:
            break

    logger.info(f"Collected {len(samples)} FLEURS bona fide samples: {counts}")
    return samples


# ---------------------------------------------------------------------------
# Feature Extraction with Frozen XLS-R 300M
# ---------------------------------------------------------------------------

def extract_features(
    samples: List[dict], backbone: Wav2Vec2Model, device: str = "cuda", batch_size: int = 16
) -> None:
    """Passes waveforms through XLS-R backbone and stores 24-layer hidden states."""
    logger.info(f"Extracting XLS-R hidden states for {len(samples)} clips on {device}...")
    backbone.eval()

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        waves = torch.stack([s["waveform"] for s in batch]).to(device)

        with torch.no_grad():
            outputs = backbone(waves, output_hidden_states=True)
            # outputs.hidden_states is a tuple of 25 tensors: [conv_out, layer1, ..., layer24]
            # Take layers 1..24, each (B, T, 1024)
            layers = [h.cpu().half() for h in outputs.hidden_states[1:]]

        for b_idx, s in enumerate(batch):
            # Store list of 24 tensors (1, T, 1024)
            s["hidden_states"] = [layers[l][b_idx : b_idx + 1] for l in range(NUM_LAYERS)]
            # Free raw waveform to save RAM
            del s["waveform"]

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(samples):
            logger.info(f"Feature extraction progress: {min(i + batch_size, len(samples))}/{len(samples)}")


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    """Computes EER, Accuracy, and optimal threshold."""
    from sklearn.metrics import roc_curve

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    best_thresh = float(thresholds[idx])

    preds = (scores_arr >= best_thresh).astype(int)
    acc = float((preds == labels_arr).mean())

    return {"eer": eer * 100.0, "accuracy": acc * 100.0, "threshold": best_thresh}


def evaluate(sls_head: SLSHead, samples: List[dict], device: str = "cuda") -> Dict[str, any]:
    """Evaluates SLSHead, reporting overall, per-gender, and per-generator metrics."""
    sls_head.eval()
    all_labels = []
    all_scores = []

    gender_groups = {"male": ([], []), "female": ([], [])}
    gen_groups = {}

    with torch.no_grad():
        for s in samples:
            hidden = [h.float().to(device) for h in s["hidden_states"]]
            log_prob = sls_head(hidden)  # (1, 2)
            bonafide_score = log_prob[0, 1].item()  # higher = more bonafide

            lbl = s["label"]
            g = s["gender"]
            gen = s["generator"]

            all_labels.append(lbl)
            all_scores.append(bonafide_score)

            if g in gender_groups:
                gender_groups[g][0].append(lbl)
                gender_groups[g][1].append(bonafide_score)

            if gen not in gen_groups:
                gen_groups[gen] = ([], [])
            gen_groups[gen][0].append(lbl)
            gen_groups[gen][1].append(bonafide_score)

    overall = compute_metrics(all_labels, all_scores)

    # Per gender metrics
    per_gender = {}
    for g, (g_lbls, g_scores) in gender_groups.items():
        if len(set(g_lbls)) > 1:
            per_gender[g] = compute_metrics(g_lbls, g_scores)

    # Generator-specific spoof detection rates at overall threshold
    thresh = overall["threshold"]
    gen_accuracies = {}
    for gen, (gen_lbls, gen_scs) in gen_groups.items():
        if gen == "bonafide_human":
            acc = float((np.array(gen_scs) >= thresh).mean()) * 100.0
        else:
            acc = float((np.array(gen_scs) < thresh).mean()) * 100.0
        gen_accuracies[gen] = acc

    return {
        "overall": overall,
        "per_gender": per_gender,
        "generator_accuracy": gen_accuracies,
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting IndicSynth Augmentation Pipeline on {device}...")

    # 1. Collect Data
    spoofs = collect_indicsynth_spoofs(target_per_cell=100)  # 600 total (300 M, 300 F)
    bonafides = collect_fleurs_bonafide(target_per_gender=300)  # 600 total (300 M, 300 F)
    all_samples = spoofs + bonafides
    logger.info(f"Total dataset collected: {len(all_samples)} clips (600 spoof, 600 bona fide).")

    # 2. Partition into Train (50%) and Held-out Test (50%)
    np.random.seed(42)
    np.random.shuffle(spoofs)
    np.random.shuffle(bonafides)

    train_samples = spoofs[:300] + bonafides[:300]
    test_samples = spoofs[300:] + bonafides[300:]
    np.random.shuffle(train_samples)
    np.random.shuffle(test_samples)

    logger.info(f"Split: {len(train_samples)} Train clips, {len(test_samples)} Test clips.")

    # 3. Extract XLS-R Backbone Features
    logger.info("Loading XLS-R 300M backbone...")
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    extract_features(all_samples, backbone, device=device, batch_size=16)
    del backbone
    torch.cuda.empty_cache()

    # 4. Load N=500 SLS Head Checkpoint
    checkpoint_path = Path("models/sls_head_n500_seed2_best.pth")
    if not checkpoint_path.exists():
        checkpoint_path = Path("/home/yashgarg/B.S-Detector/models/sls_head_n500_seed2_best.pth")

    sls_head = SLSHead().to(device)
    load_n500_weights(sls_head, checkpoint_path)

    # 5. Zero-Shot Baseline Evaluation on Test Split
    logger.info("=== ZERO-SHOT EVALUATION (Pre-adaptation N=500 Checkpoint) ===")
    baseline_metrics = evaluate(sls_head, test_samples, device=device)
    logger.info(f"Baseline Overall: {baseline_metrics['overall']}")
    logger.info(f"Baseline Per Gender: {baseline_metrics['per_gender']}")
    logger.info(f"Baseline Per Generator: {baseline_metrics['generator_accuracy']}")

    # 6. Fine-Tuning SLS Head on Augmented Data
    logger.info("=== FINE-TUNING SLS HEAD ON AUGMENTED DATA ===")
    sls_head.train()
    optimizer = torch.optim.Adam(sls_head.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.NLLLoss(weight=torch.tensor([0.5, 0.5], device=device))

    epochs = 10
    batch_size = 8
    indices = np.arange(len(train_samples))

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0

        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            batch_items = [train_samples[j] for j in b_idx]

            batched_hidden = []
            for l in range(NUM_LAYERS):
                layer_tensors = [it["hidden_states"][l] for it in batch_items]
                batched_hidden.append(torch.cat(layer_tensors, dim=0).float().to(device))

            labels = torch.tensor([it["label"] for it in batch_items], dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_probs = sls_head(batched_hidden)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(indices) // batch_size)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # 7. Post-Adaptation Evaluation
    logger.info("=== POST-ADAPTATION EVALUATION (Fine-tuned Checkpoint) ===")
    adapted_metrics = evaluate(sls_head, test_samples, device=device)
    logger.info(f"Adapted Overall: {adapted_metrics['overall']}")
    logger.info(f"Adapted Per Gender: {adapted_metrics['per_gender']}")
    logger.info(f"Adapted Per Generator: {adapted_metrics['generator_accuracy']}")

    # 8. Save Model and Results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    save_model_path = models_dir / "sls_head_n500_indicsynth_best.pth"
    torch.save(sls_head.state_dict(), save_model_path)
    logger.info(f"Saved fine-tuned model checkpoint to {save_model_path}")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_checkpoint": "sls_head_n500_seed2_best.pth",
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "baseline": baseline_metrics,
        "adapted": adapted_metrics,
    }
    with open(output_dir / "indicsynth_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to outputs/indicsynth_results.json")

    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    # Exit cleanly without background worker thread deadlock
    os._exit(0)


if __name__ == "__main__":
    main()
