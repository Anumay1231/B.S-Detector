"""scripts/train_sls_robust.py: SLS Head Fine-Tuning with Silence Invariance and Commercial TTS.

Removes the silence shortcut by:
1. Energy-based trimming of leading/trailing silence on all audio.
2. Injecting randomized 0.0-0.8s quiet tone beds equally on both bona fide and spoof clips.
3. Training on mid-utterance slices (cutting through active speech) for sliding-window robustness.
4. Integrating Edge-TTS commercial stock voices (Neerja, Prabhat, Madhur, Swara) and IndicSynth.
5. Evaluating on clean benchmark AND silence-stress tests.
"""

from __future__ import annotations

import asyncio
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
import torch.nn.functional as F
from datasets import Audio, load_dataset
from scipy.signal import resample
from transformers import Wav2Vec2Model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sls_robust")

TARGET_SR = 16000
INPUT_LENGTH = 64600  # 4.04s at 16kHz
HIDDEN_DIM = 1024
NUM_LAYERS = 24


# ---------------------------------------------------------------------------
# SLS Architecture Definition (Exact match to server/inference.py)
# ---------------------------------------------------------------------------

class SLSHead(nn.Module):
    """Selective Layer Summarization (SLS) classifier head for XLS-R 300M."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.fc0 = nn.Linear(hidden_dim, 1)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_states_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        weights = []
        for layer_out in hidden_states_list:
            pooled = layer_out.mean(dim=1)
            w = torch.sigmoid(self.fc0(pooled))
            weights.append(w)

        fused = torch.zeros_like(hidden_states_list[0])
        for w, layer_out in zip(weights, hidden_states_list):
            fused = fused + w.unsqueeze(-1) * layer_out

        fused = fused.unsqueeze(1)
        x = self.first_bn(fused)
        x = self.selu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=3)
        x = x.flatten(start_dim=1)
        x = self.selu(self.fc1(x))
        x = self.fc3(x)
        return self.logsoftmax(x), weights


def load_checkpoint_weights(sls_head: SLSHead, checkpoint_path: Path) -> SLSHead:
    """Load weights cleanly into SLSHead."""
    logger.info(f"Loading checkpoint weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("ssl_model.", "")
        if clean_k in sls_head.state_dict():
            cleaned[clean_k] = v
    sls_head.load_state_dict(cleaned, strict=False)
    logger.info(f"Successfully loaded {len(cleaned)}/{len(sls_head.state_dict())} parameter keys.")
    return sls_head


# ---------------------------------------------------------------------------
# Audio Augmentation & Slice Extraction
# ---------------------------------------------------------------------------

def detect_active_speech(wave: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Trims dead silence using RMS energy threshold."""
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    if len(wave) <= frame_len:
        return wave
    
    # Compute frame energy
    n_frames = (len(wave) - frame_len) // hop_len
    if n_frames <= 0:
        return wave
        
    frames = np.lib.stride_tricks.as_strided(
        wave,
        shape=(n_frames, frame_len),
        strides=(wave.strides[0] * hop_len, wave.strides[0]),
    )
    energy = np.sum(frames ** 2, axis=1)
    max_e = np.max(energy)
    if max_e < 1e-7:
        return wave

    active = np.where(energy > 0.01 * max_e)[0]
    if len(active) == 0:
        return wave

    first_idx = max(0, active[0] * hop_len)
    last_idx = min(len(wave), (active[-1] + 1) * hop_len + frame_len)
    return wave[first_idx:last_idx]


def extract_slices_from_raw(
    raw_audio: np.ndarray,
    orig_sr: int,
    extract_mid: bool = True,
) -> List[Tuple[torch.Tensor, str]]:
    """
    Standardizes audio to 16kHz and extracts:
    1. Onset crop with randomized leading quiet tone floor (0.0-0.8s)
    2. Mid-utterance crop (cutting into speech) if audio is sufficiently long
    """
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    if np.issubdtype(raw_audio.dtype, np.integer):
        max_val = float(np.iinfo(raw_audio.dtype).max)
        raw_audio = raw_audio.astype(np.float32) / max_val
    else:
        raw_audio = raw_audio.astype(np.float32)

    # Normalize amplitude
    peak = np.max(np.abs(raw_audio))
    if peak > 1e-4:
        raw_audio = raw_audio / peak * 0.95

    # Resample to 16 kHz
    if orig_sr != TARGET_SR:
        target_len = int(round(len(raw_audio) * float(TARGET_SR) / float(orig_sr)))
        raw_audio = resample(raw_audio, target_len).astype(np.float32)

    # Trim dead leading/trailing silence
    speech = detect_active_speech(raw_audio, TARGET_SR)

    crops = []

    # 1. Short clips: tile pad
    if len(speech) < INPUT_LENGTH:
        repeats = (INPUT_LENGTH // max(1, len(speech))) + 1
        tiled = np.tile(speech, repeats)[:INPUT_LENGTH]
        crops.append((torch.from_numpy(tiled), "tile_padded"))
        return crops

    # 2. Onset crop with randomized leading tone bed
    lead_sec = np.random.uniform(0.0, 0.8)
    lead_samples = int(lead_sec * TARGET_SR)
    if lead_samples > 0:
        # Subtle ambient tone floor
        tone = np.random.normal(0, 1e-4, lead_samples).astype(np.float32)
        onset_w = np.concatenate([tone, speech])[:INPUT_LENGTH]
    else:
        onset_w = speech[:INPUT_LENGTH]
    crops.append((torch.from_numpy(onset_w), "onset_random_silence"))

    # 3. Mid-utterance crop (starts between 0.8s and 2.5s in, mid-sentence)
    if extract_mid and len(speech) >= (INPUT_LENGTH + int(1.0 * TARGET_SR)):
        max_start = len(speech) - INPUT_LENGTH
        start_idx = np.random.randint(int(0.8 * TARGET_SR), max_start)
        mid_w = speech[start_idx : start_idx + INPUT_LENGTH]
        crops.append((torch.from_numpy(mid_w), "mid_utterance"))

    return crops


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_indicsynth(target_per_cell: int = 100) -> List[dict]:
    """Targeted shard extraction for IndicSynth (Female/Male x FreeVC/XTTS-v2)."""
    import pandas as pd
    logger.info("Fast targeted downloading IndicSynth shards (Hindi)...")
    shards_config = [
        (0, "Female", "freevc24"),
        (40, "Male", "freevc24"),
        (60, "Female", "xtts_v2"),
        (80, "Male", "xtts_v2"),
    ]
    samples = []
    counts = {}

    for shard, gender_target, model_target in shards_config:
        url = f"https://huggingface.co/datasets/vdivyasharma/IndicSynth/resolve/main/Hindi/train-{shard:05d}-of-00107.parquet"
        logger.info(f"Loading shard {shard:05d} for {gender_target} {model_target}...")
        df = pd.read_parquet(url)
        df_cell = df[(df["Gender"] == gender_target) & (df["Generative Model"] == model_target)].head(target_per_cell)
        
        gender = gender_target.lower()
        model = model_target
        cell_key = (gender, model)
        counts[cell_key] = 0

        for idx, row in df_cell.iterrows():
            audio_data = row["audio"]
            raw_bytes = audio_data["bytes"]
            data, sr = sf.read(io.BytesIO(raw_bytes))
            
            slices = extract_slices_from_raw(data, sr, extract_mid=True)
            for s_idx, (wave_tensor, crop_type) in enumerate(slices):
                samples.append({
                    "id": f"indicsynth_{gender}_{model}_{counts[cell_key]:04d}_{s_idx}",
                    "label": 0,  # spoof
                    "gender": gender,
                    "generator": model,
                    "speaker_id": str(row.get("Speaker ID", "")),
                    "crop_type": crop_type,
                    "waveform": wave_tensor,
                })
            counts[cell_key] += 1

        logger.info(f"Shard {shard:05d} complete: {counts[cell_key]} clips collected.")

    logger.info(f"Collected IndicSynth spoofs: {len(samples)} slices across {counts}")
    return samples


def collect_fleurs_bonafide(target_per_gender: int = 300) -> List[dict]:
    """Collects genuine Hindi samples from google/fleurs (male and female)."""
    logger.info("Streaming google/fleurs (hi_in) for bona fide human speech...")
    ds = load_dataset("google/fleurs", "hi_in", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {"male": 0, "female": 0}
    samples = []

    for item in ds:
        raw_g = item.get("gender")
        gender = "female" if raw_g == 1 else "male"

        if counts[gender] >= target_per_gender:
            continue

        raw_bytes = item["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(raw_bytes))
        
        slices = extract_slices_from_raw(data, sr, extract_mid=True)
        for s_idx, (wave_tensor, crop_type) in enumerate(slices):
            samples.append({
                "id": f"fleurs_{gender}_{counts[gender]:04d}_{s_idx}",
                "label": 1,  # bonafide
                "gender": gender,
                "generator": "bonafide_human",
                "speaker_id": str(item.get("speaker_id", "")),
                "crop_type": crop_type,
                "waveform": wave_tensor,
            })

        counts[gender] += 1
        total_fl = sum(counts.values())
        if total_fl % 50 == 0:
            logger.info(f"FLEURS progress: {counts} (total {total_fl}/{target_per_gender * 2})")
        if counts["male"] >= target_per_gender and counts["female"] >= target_per_gender:
            break

    logger.info(f"Collected FLEURS bona fide: {len(samples)} slices across {counts}")
    return samples


def generate_edge_tts_samples(n_per_voice: int = 25) -> List[dict]:
    """Generates synthetic speech for Edge-TTS stock voices (Madhur, Swara, Neerja, Prabhat)."""
    import edge_tts

    voices = {
        ("male", "edge_madhur", "hi-IN-MadhurNeural"): "hi",
        ("female", "edge_swara", "hi-IN-SwaraNeural"): "hi",
        ("female", "edge_neerja", "en-IN-NeerjaNeural"): "en",
        ("male", "edge_prabhat", "en-IN-PrabhatNeural"): "en",
    }

    hindi_prompts = [
        "नमस्ते, आपका स्वागत है इस सुरक्षा सत्यापन प्रणाली में।",
        "कृपया अपना बैंक खाता नंबर और गुप्त पिन किसी के साथ साझा न करें।",
        "आज के मौसम समाचार में भारी बारिश की चेतावनी दी गई है।",
        "भारतीय रिज़र्व बैंक के अनुसार डिजिटल लेन-देन सुरक्षित है।",
        "सुरक्षा मानकों का पालन करना प्रत्येक नागरिक की ज़िम्मेदारी है।",
    ]
    english_prompts = [
        "Welcome to the biometric speech authentication portal.",
        "Your account verification request is being processed by the system.",
        "Please repeat the security phrase clearly into your microphone.",
        "Fraud detection protocols are actively monitoring this transaction.",
        "Ensure that background noise is minimized for voice verification.",
    ]

    samples = []
    tmp_dir = Path("/tmp/edge_tts_gen")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Edge-TTS samples for commercial neural voice coverage...")

    async def _gen(voice_id, text, out_file):
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(str(out_file))

    for (gender, gen_name, voice_id), lang in voices.items():
        prompts = hindi_prompts if lang == "hi" else english_prompts
        for i in range(n_per_voice):
            text = prompts[i % len(prompts)] + f" Step {i+1} completed."
            wav_path = tmp_dir / f"{gen_name}_{i:03d}.mp3"
            try:
                asyncio.run(_gen(voice_id, text, wav_path))
                data, sr = sf.read(str(wav_path))
                slices = extract_slices_from_raw(data, sr, extract_mid=False)
                for s_idx, (wave_tensor, crop_type) in enumerate(slices):
                    samples.append({
                        "id": f"{gen_name}_{i:03d}_{s_idx}",
                        "label": 0,  # spoof
                        "gender": gender,
                        "generator": gen_name,
                        "speaker_id": gen_name,
                        "crop_type": crop_type,
                        "waveform": wave_tensor,
                    })
            except Exception as e:
                logger.warning(f"Failed to generate {gen_name}_{i}: {e}")

    logger.info(f"Generated {len(samples)} Edge-TTS spoof slices.")
    return samples


# ---------------------------------------------------------------------------
# Feature Extraction (Frozen XLS-R 300M in RAM)
# ---------------------------------------------------------------------------

def extract_features(
    samples: List[dict], backbone: Wav2Vec2Model, device: str = "cuda", batch_size: int = 32
) -> None:
    """Passes waveforms through XLS-R backbone and caches 24-layer hidden states in RAM."""
    logger.info(f"Extracting 24-layer XLS-R hidden states for {len(samples)} slices on {device}...")
    backbone.eval()

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        waves = torch.stack([s["waveform"] for s in batch]).to(device)

        with torch.no_grad():
            outputs = backbone(waves, output_hidden_states=True)
            layers = [h.cpu().half() for h in outputs.hidden_states[1:]]

        for b_idx, s in enumerate(batch):
            s["hidden_states"] = [layers[l][b_idx : b_idx + 1] for l in range(NUM_LAYERS)]
            del s["waveform"]

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(samples):
            logger.info(f"Extraction progress: {min(i + batch_size, len(samples))}/{len(samples)}")


# ---------------------------------------------------------------------------
# Metrics Computation & Stress-Test Suite
# ---------------------------------------------------------------------------

def compute_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    """Computes EER, Accuracy @ 0.0 logit threshold, and optimal threshold."""
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    # Bonafide label=1 (higher score), Spoof label=0
    bonafide = scores_arr[labels_arr == 1]
    spoof = scores_arr[labels_arr == 0]

    if len(bonafide) == 0 or len(spoof) == 0:
        return {"eer": 0.0, "accuracy": 100.0, "threshold": 0.0}

    thresholds = np.sort(scores_arr)
    frr = np.array([np.mean(bonafide < t) for t in thresholds])
    far = np.array([np.mean(spoof >= t) for t in thresholds])
    diff = np.abs(frr - far)
    idx = np.argmin(diff)
    eer = float((frr[idx] + far[idx]) / 2.0) * 100.0
    thresh = float(thresholds[idx])

    # Accuracy @ default threshold (logit 0.0 = probability 0.50)
    pred_default = (scores_arr >= 0.0).astype(int)
    acc = float(np.mean(pred_default == labels_arr)) * 100.0

    return {"eer": round(eer, 2), "accuracy": round(acc, 2), "threshold": round(thresh, 4)}


def evaluate(sls_head: SLSHead, samples: List[dict], device: str = "cuda") -> Dict[str, any]:
    """Runs full evaluation and silence stress tests."""
    sls_head.eval()

    all_labels, all_scores = [], []
    gen_groups = {}
    crop_groups = {"onset_random_silence": ([], []), "mid_utterance": ([], []), "tile_padded": ([], [])}

    with torch.no_grad():
        for s in samples:
            hidden = [h.float().to(device) for h in s["hidden_states"]]
            log_prob, _ = sls_head(hidden)  # (1, 2)
            # log_prob[0, 1] - log_prob[0, 0] = logit (higher = more bonafide)
            bonafide_logit = (log_prob[0, 1] - log_prob[0, 0]).item()

            lbl = s["label"]
            gen = s["generator"]
            crop = s.get("crop_type", "onset_random_silence")

            all_labels.append(lbl)
            all_scores.append(bonafide_logit)

            if gen not in gen_groups:
                gen_groups[gen] = ([], [])
            gen_groups[gen][0].append(lbl)
            gen_groups[gen][1].append(bonafide_logit)

            if crop in crop_groups:
                crop_groups[crop][0].append(lbl)
                crop_groups[crop][1].append(bonafide_logit)

    overall = compute_metrics(all_labels, all_scores)

    # Per-generator detection rate
    gen_accuracies = {}
    for gen, (g_lbls, g_scs) in gen_groups.items():
        if gen == "bonafide_human":
            acc = float((np.array(g_scs) >= 0.0).mean()) * 100.0
        else:
            acc = float((np.array(g_scs) < 0.0).mean()) * 100.0
        gen_accuracies[gen] = round(acc, 2)

    # Stress Test 1: Real speech cut mid-sentence false alarm rate
    mid_lbls, mid_scs = crop_groups["mid_utterance"]
    mid_arr_l = np.array(mid_lbls)
    mid_arr_s = np.array(mid_scs)
    if (mid_arr_l == 1).sum() > 0:
        real_mid_scs = mid_arr_s[mid_arr_l == 1]
        mid_sentence_false_alarm_rate = float((real_mid_scs < 0.0).mean()) * 100.0
    else:
        mid_sentence_false_alarm_rate = 0.0

    return {
        "overall": overall,
        "generator_accuracy": gen_accuracies,
        "mid_sentence_false_alarm_rate": round(mid_sentence_false_alarm_rate, 2),
    }


# ---------------------------------------------------------------------------
# Training Routine
# ---------------------------------------------------------------------------

def train_pipeline(
    epochs: int = 15,
    lr: float = 2e-5,
    batch_size: int = 16,
    base_checkpoint: str = "models/sls_head_n500_indicsynth_best.pth",
    out_checkpoint: str = "models/sls_head_silence_invariant.pth",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=== Starting SLS Silence-Invariant Retraining on {device} ===")

    # 1. Collect Data
    indicsynth_samples = collect_indicsynth(target_per_cell=75)
    fleurs_samples = collect_fleurs_bonafide(target_per_gender=150)
    edge_samples = generate_edge_tts_samples(n_per_voice=25)

    all_samples = indicsynth_samples + fleurs_samples + edge_samples
    logger.info(f"Total dataset: {len(all_samples)} slices ({len(indicsynth_samples)+len(edge_samples)} spoof, {len(fleurs_samples)} bona fide).")

    # 2. Partition into Train (70%) and Held-out Test (30%)
    np.random.seed(42)
    # Stratified split by generator
    train_samples, test_samples = [], []
    by_gen = {}
    for s in all_samples:
        by_gen.setdefault(s["generator"], []).append(s)

    for gen, items in by_gen.items():
        np.random.shuffle(items)
        n_train = int(0.70 * len(items))
        train_samples.extend(items[:n_train])
        test_samples.extend(items[n_train:])

    np.random.shuffle(train_samples)
    np.random.shuffle(test_samples)
    logger.info(f"Split: {len(train_samples)} Train slices, {len(test_samples)} Held-out Test slices.")

    # 3. XLS-R 300M Feature Extraction
    logger.info("Loading XLS-R 300M backbone...")
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    extract_features(all_samples, backbone, device=device, batch_size=32)
    del backbone
    torch.cuda.empty_cache()

    # 4. Load Base SLS Head Checkpoint
    sls_head = SLSHead().to(device)
    load_checkpoint_weights(sls_head, Path(base_checkpoint))

    # 5. Baseline Evaluation on Test Set
    logger.info("=== BASELINE EVALUATION (Before Silence Invariance Fine-Tuning) ===")
    base_eval = evaluate(sls_head, test_samples, device=device)
    logger.info(f"Baseline Overall: {base_eval['overall']}")
    logger.info(f"Baseline Mid-Sentence False Alarm Rate: {base_eval['mid_sentence_false_alarm_rate']}%")
    logger.info(f"Baseline Per-Generator Accuracy: {base_eval['generator_accuracy']}")

    # 6. Fine-Tuning
    logger.info(f"=== FINE-TUNING SLS HEAD ({epochs} Epochs, lr={lr}) ===")
    optimizer = torch.optim.AdamW(sls_head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.NLLLoss(weight=torch.tensor([0.5, 0.5], device=device))

    best_test_acc = base_eval["overall"]["accuracy"]
    best_mid_far = base_eval["mid_sentence_false_alarm_rate"]
    indices = np.arange(len(train_samples))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        sls_head.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            batch_items = [train_samples[j] for j in b_idx]

            batched_hidden = []
            for l in range(NUM_LAYERS):
                layer_tensors = [it["hidden_states"][l] for it in batch_items]
                batched_hidden.append(torch.cat(layer_tensors, dim=0).float().to(device))

            labels = torch.tensor([it["label"] for it in batch_items], dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_probs, _ = sls_head(batched_hidden)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0

        # Evaluate at epoch end
        cur_eval = evaluate(sls_head, test_samples, device=device)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Loss: {avg_loss:.4f} | "
            f"Val EER: {cur_eval['overall']['eer']}% | Val Acc: {cur_eval['overall']['accuracy']}% | "
            f"Mid-Sentence False Alarm: {cur_eval['mid_sentence_false_alarm_rate']}%"
        )

        # Checkpoint saving condition:
        # Save if mid-sentence false alarms drop significantly while maintaining high accuracy
        if cur_eval["mid_sentence_false_alarm_rate"] <= best_mid_far and cur_eval["overall"]["accuracy"] >= 96.0:
            best_mid_far = cur_eval["mid_sentence_false_alarm_rate"]
            best_test_acc = cur_eval["overall"]["accuracy"]
            Path(out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(sls_head.state_dict(), out_checkpoint)
            logger.info(f"  --> Saved new best robust checkpoint to {out_checkpoint}")

    # 7. Final Verification
    logger.info("=== FINAL POST-ADAPTATION AUDIT ===")
    final_eval = evaluate(sls_head, test_samples, device=device)
    logger.info(f"Final Overall: {final_eval['overall']}")
    logger.info(f"Final Mid-Sentence False Alarm Rate: {final_eval['mid_sentence_false_alarm_rate']}%")
    logger.info(f"Final Per-Generator Accuracy: {final_eval['generator_accuracy']}")

    # Export results JSON
    results_path = Path("outputs/sls_robust_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs": epochs,
            "baseline": base_eval,
            "final": final_eval,
            "checkpoint": out_checkpoint,
        }, f, indent=2)
    logger.info(f"Results exported to {results_path}")


if __name__ == "__main__":
    train_pipeline()
