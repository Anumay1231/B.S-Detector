"""scripts/train_sls_channel_robust.py: SLS Head Fine-Tuning with Over-the-Air Channel & Phone Replay Robustness.

Enhances deepfake detection robustness against acoustic transmission, phone speaker playback,
and room reverberation by:
1. Acoustic channel simulation: Phone transducer bandpass (250-7000Hz), mid-resonance,
   soft-clipping saturation, room impulse response (RIR) convolution, and mic noise floor.
2. Dual-channel training: 50% direct digital audio and 50% simulated over-the-air phone replay.
3. Silence & mid-utterance invariance: Preserves randomized leading tone beds and mid-speech crops.
4. Multilingual & commercial voice coverage: IndicSynth (FreeVC, XTTS), Edge-TTS, and FLEURS Hindi.
5. Warm-starts from models/sls_head_silence_invariant.pth.
6. Evaluates both clean digital performance AND over-the-air replay detection rate.
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
from scipy.signal import butter, fftconvolve, lfilter, resample
from transformers import Wav2Vec2Model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sls_channel_robust")

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
# Over-the-Air & Acoustic Channel Simulation
# ---------------------------------------------------------------------------

def make_room_impulse(seed: int, sr: int = TARGET_SR) -> dict:
    """Generates synthetic room impulse response and ambient room tone bed."""
    rng = np.random.default_rng(seed)
    rt60 = rng.uniform(0.20, 0.55)  # typical living room / office
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / 0.30
    rir /= np.sqrt(np.sum(rir ** 2))

    # Room low/high pass color
    lo, hi = rng.uniform(100, 250), rng.uniform(4000, 7000)
    b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")

    # Ambient room tone bed
    base = rng.standard_normal(sr * 30)
    bn, an = butter(1, rng.uniform(200, 1200) / (sr / 2))
    bed = lfilter(bn, an, base) + 0.2 * base
    bed /= (np.sqrt(np.mean(bed ** 2)) + 1e-9)

    return {"rir": rir, "eq": (b, a), "bed": bed, "snr": rng.uniform(18, 30), "rt60": rt60}


def apply_phone_speaker(audio: np.ndarray, rng: np.random.Generator, sr: int = TARGET_SR) -> np.ndarray:
    """
    Simulates physical playback from a smartphone speaker:
    - Bandpass roll-off (cutting low bass < 250Hz and high-freq vocoder shimmer > 6500Hz)
    - Resonance bump around 2-3.5 kHz (small phone enclosure resonance)
    - Soft-clipping saturation (speaker diaphragm non-linearity)
    """
    lo = rng.uniform(240, 320)
    hi = rng.uniform(6200, 7200)
    b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    filtered = lfilter(b, a, audio)

    # Phone chassis resonance peak
    res_freq = rng.uniform(2000, 3200)
    b_res, a_res = butter(1, [max(100, res_freq - 250) / (sr / 2), min(sr / 2 - 100, res_freq + 250) / (sr / 2)], btype="band")
    mid_boost = lfilter(b_res, a_res, filtered) * rng.uniform(0.3, 0.6)
    out = filtered + mid_boost

    # Mild diaphragm soft-clipping (tanh)
    gain = rng.uniform(1.2, 1.8)
    out = np.tanh(out * gain)
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * 0.9).astype(np.float32)


def apply_acoustic_channel(audio: np.ndarray, seed: int, sr: int = TARGET_SR) -> np.ndarray:
    """Combines phone speaker playback + room impulse response + mic noise floor."""
    rng = np.random.default_rng(seed)
    # 1. Phone speaker filter
    phone_wave = apply_phone_speaker(audio, rng, sr)

    # 2. Room impulse response convolution
    room = make_room_impulse(seed, sr)
    wet = fftconvolve(phone_wave, room["rir"])[: len(phone_wave)]
    wet = lfilter(*room["eq"], wet)

    # 3. Add ambient mic noise floor
    if len(wet) < len(room["bed"]):
        start = rng.integers(0, len(room["bed"]) - len(wet))
        noise = room["bed"][start : start + len(wet)]
    else:
        repeats = (len(wet) // len(room["bed"])) + 1
        noise = np.tile(room["bed"], repeats)[: len(wet)]

    p_sig = np.mean(wet ** 2) + 1e-12
    p_noise = 10 ** (room["snr"] / 10)
    replayed = wet + noise * np.sqrt(p_sig / p_noise)

    peak = np.max(np.abs(replayed)) + 1e-9
    return (replayed / peak * 0.85).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio Slicing & Preprocessing
# ---------------------------------------------------------------------------

def detect_active_speech(wave: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Trims dead silence using RMS energy threshold."""
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    if len(wave) <= frame_len:
        return wave

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
    seed_idx: int = 0,
) -> List[Tuple[torch.Tensor, str, str]]:
    """
    Standardizes audio to 16kHz and extracts slices across both:
    1. Direct clean digital channel (with randomized quiet room tone beds)
    2. Over-the-air phone speaker replay channel (phone EQ + RIR + mic noise)
    Returns list of (waveform_tensor, crop_type, channel_type).
    """
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    if np.issubdtype(raw_audio.dtype, np.integer):
        max_val = float(np.iinfo(raw_audio.dtype).max)
        raw_audio = raw_audio.astype(np.float32) / max_val
    else:
        raw_audio = raw_audio.astype(np.float32)

    peak = np.max(np.abs(raw_audio))
    if peak > 1e-4:
        raw_audio = raw_audio / peak * 0.95

    if orig_sr != TARGET_SR:
        target_len = int(round(len(raw_audio) * float(TARGET_SR) / float(orig_sr)))
        raw_audio = resample(raw_audio, target_len).astype(np.float32)

    speech_clean = detect_active_speech(raw_audio, TARGET_SR)
    if len(speech_clean) < int(0.5 * TARGET_SR):
        return []

    # Generate phone replay version of speech
    speech_replay = apply_acoustic_channel(speech_clean, seed=seed_idx, sr=TARGET_SR)

    crops = []

    for speech_ver, ch_name in [(speech_clean, "clean_digital"), (speech_replay, "acoustic_replay")]:
        # Short audio: tile pad
        if len(speech_ver) < INPUT_LENGTH:
            repeats = (INPUT_LENGTH // max(1, len(speech_ver))) + 1
            tiled = np.tile(speech_ver, repeats)[:INPUT_LENGTH]
            crops.append((torch.from_numpy(tiled), "tile_padded", ch_name))
            continue

        # Onset crop with subtle room-tone lead-in
        lead_sec = np.random.uniform(0.0, 0.6)
        lead_samples = int(lead_sec * TARGET_SR)
        if lead_samples > 0:
            tone = np.random.normal(0, 1e-4, lead_samples).astype(np.float32)
            onset_w = np.concatenate([tone, speech_ver])[:INPUT_LENGTH]
        else:
            onset_w = speech_ver[:INPUT_LENGTH]
        crops.append((torch.from_numpy(onset_w), "onset_random_silence", ch_name))

        # Mid-utterance crop (cutting into speech mid-sentence)
        if len(speech_ver) >= (INPUT_LENGTH + int(0.8 * TARGET_SR)):
            max_start = len(speech_ver) - INPUT_LENGTH
            start_idx = np.random.randint(int(0.5 * TARGET_SR), max_start)
            mid_w = speech_ver[start_idx : start_idx + INPUT_LENGTH]
            crops.append((torch.from_numpy(mid_w), "mid_utterance", ch_name))

    return crops


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_indicsynth(target_per_cell: int = 30) -> List[dict]:
    """Targeted shard extraction for IndicSynth (Female/Male x FreeVC/XTTS-v2)."""
    import pandas as pd
    logger.info("Downloading IndicSynth shards (Hindi)...")
    shards_config = [
        (0, "Female", "freevc24"),
        (40, "Male", "freevc24"),
        (60, "Female", "xtts_v2"),
        (80, "Male", "xtts_v2"),
    ]
    samples = []
    seed_ctr = 1000

    for shard, gender_target, model_target in shards_config:
        url = f"https://huggingface.co/datasets/vdivyasharma/IndicSynth/resolve/main/Hindi/train-{shard:05d}-of-00107.parquet"
        logger.info(f"Loading shard {shard:05d} for {gender_target} {model_target}...")
        df = pd.read_parquet(url)
        df_cell = df[(df["Gender"] == gender_target) & (df["Generative Model"] == model_target)].head(target_per_cell)

        gender = gender_target.lower()
        model = model_target

        for idx, row in df_cell.iterrows():
            audio_data = row["audio"]
            data, sr = sf.read(io.BytesIO(audio_data["bytes"]))

            slices = extract_slices_from_raw(data, sr, seed_idx=seed_ctr)
            seed_ctr += 1
            for s_idx, (wave_tensor, crop_type, ch_name) in enumerate(slices):
                samples.append({
                    "id": f"indicsynth_{gender}_{model}_{seed_ctr}_{s_idx}",
                    "label": 0,  # spoof
                    "gender": gender,
                    "generator": model,
                    "crop_type": crop_type,
                    "channel": ch_name,
                    "waveform": wave_tensor,
                })

    logger.info(f"Collected IndicSynth spoofs: {len(samples)} slices.")
    return samples


def collect_fleurs_bonafide(target_per_gender: int = 65) -> List[dict]:
    """Collects genuine Hindi samples from google/fleurs (male and female)."""
    logger.info("Streaming google/fleurs (hi_in) for bona fide human speech...")
    ds = load_dataset("google/fleurs", "hi_in", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    counts = {"male": 0, "female": 0}
    samples = []
    seed_ctr = 5000

    for item in ds:
        raw_g = item.get("gender")
        gender = "female" if raw_g == 1 else "male"

        if counts[gender] >= target_per_gender:
            continue

        data, sr = sf.read(io.BytesIO(item["audio"]["bytes"]))

        slices = extract_slices_from_raw(data, sr, seed_idx=seed_ctr)
        seed_ctr += 1
        for s_idx, (wave_tensor, crop_type, ch_name) in enumerate(slices):
            samples.append({
                "id": f"fleurs_{gender}_{counts[gender]:04d}_{s_idx}",
                "label": 1,  # bonafide
                "gender": gender,
                "generator": "bonafide_human",
                "crop_type": crop_type,
                "channel": ch_name,
                "waveform": wave_tensor,
            })

        counts[gender] += 1
        if counts["male"] >= target_per_gender and counts["female"] >= target_per_gender:
            break

    logger.info(f"Collected FLEURS bona fide: {len(samples)} slices across {counts}.")
    return samples


def generate_edge_tts_samples(n_per_voice: int = 15) -> List[dict]:
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
    seed_ctr = 9000

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
                slices = extract_slices_from_raw(data, sr, seed_idx=seed_ctr)
                seed_ctr += 1
                for s_idx, (wave_tensor, crop_type, ch_name) in enumerate(slices):
                    samples.append({
                        "id": f"{gen_name}_{i:03d}_{s_idx}",
                        "label": 0,  # spoof
                        "gender": gender,
                        "generator": gen_name,
                        "crop_type": crop_type,
                        "channel": ch_name,
                        "waveform": wave_tensor,
                    })
            except Exception as e:
                logger.warning(f"Failed to generate {gen_name}_{i}: {e}")

    logger.info(f"Generated {len(samples)} Edge-TTS spoof slices.")
    return samples


# ---------------------------------------------------------------------------
# Feature Extraction (XLS-R 300M 24-Layer Caching)
# ---------------------------------------------------------------------------

def extract_features(samples: List[dict], backbone: Wav2Vec2Model, device: str = "cuda", batch_size: int = 32):
    """Batches 4.04s waveform tensors through frozen XLS-R and attaches 24 layer hidden states."""
    logger.info(f"Extracting XLS-R hidden states for {len(samples)} slices (batch size {batch_size})...")
    t0 = time.time()
    n_batches = (len(samples) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch_slice = samples[b * batch_size : (b + 1) * batch_size]
        stacked = torch.stack([s["waveform"] for s in batch_slice]).float().to(device)

        with torch.no_grad():
            outputs = backbone(stacked, output_hidden_states=True)
            # 24 layers (excluding embedding layer)
            hidden_states = outputs.hidden_states[1:]

        for i, s in enumerate(batch_slice):
            s["hidden_states"] = [hidden_states[l][i : i + 1].half().cpu() for l in range(NUM_LAYERS)]
            del s["waveform"]

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            logger.info(f"Extracted {min((b + 1) * batch_size, len(samples))}/{len(samples)} slices ({(b+1)/n_batches*100:.1f}%)")

    torch.cuda.empty_cache()
    logger.info(f"Feature extraction completed in {time.time() - t0:.1f}s.")


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

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

    pred_default = (scores_arr >= 0.0).astype(int)
    acc = float(np.mean(pred_default == labels_arr)) * 100.0

    return {"eer": round(eer, 2), "accuracy": round(acc, 2), "threshold": round(thresh, 4)}


def evaluate(sls_head: SLSHead, samples: List[dict], criterion: nn.Module, device: str = "cuda") -> Dict[str, any]:
    """Runs evaluation computing loss, EER, and breakdown across clean vs acoustic replay."""
    sls_head.eval()

    all_labels, all_scores = [], []
    gen_groups = {}
    ch_groups = {"clean_digital": ([], []), "acoustic_replay": ([], [])}
    total_val_loss = 0.0

    with torch.no_grad():
        for s in samples:
            hidden = [h.float().to(device) for h in s["hidden_states"]]
            log_prob, _ = sls_head(hidden)  # (1, 2)
            lbl = s["label"]
            target = torch.tensor([lbl], dtype=torch.long, device=device)
            loss = criterion(log_prob, target)
            total_val_loss += loss.item()

            bonafide_logit = (log_prob[0, 1] - log_prob[0, 0]).item()
            gen = s["generator"]
            ch = s.get("channel", "clean_digital")

            all_labels.append(lbl)
            all_scores.append(bonafide_logit)

            if gen not in gen_groups:
                gen_groups[gen] = ([], [])
            gen_groups[gen][0].append(lbl)
            gen_groups[gen][1].append(bonafide_logit)

            if ch in ch_groups:
                ch_groups[ch][0].append(lbl)
                ch_groups[ch][1].append(bonafide_logit)

    overall = compute_metrics(all_labels, all_scores)
    val_loss = total_val_loss / max(1, len(samples))

    # Channel accuracies
    clean_acc = compute_metrics(ch_groups["clean_digital"][0], ch_groups["clean_digital"][1])["accuracy"] if ch_groups["clean_digital"][0] else 100.0
    replay_acc = compute_metrics(ch_groups["acoustic_replay"][0], ch_groups["acoustic_replay"][1])["accuracy"] if ch_groups["acoustic_replay"][0] else 100.0

    # Per-generator detection rate
    gen_accuracies = {}
    for gen, (g_lbls, g_scs) in gen_groups.items():
        if gen == "bonafide_human":
            acc = float((np.array(g_scs) >= 0.0).mean()) * 100.0
        else:
            acc = float((np.array(g_scs) < 0.0).mean()) * 100.0
        gen_accuracies[gen] = round(acc, 2)

    return {
        "val_loss": round(val_loss, 4),
        "overall": overall,
        "clean_accuracy": clean_acc,
        "replay_accuracy": replay_acc,
        "generator_accuracy": gen_accuracies,
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    epochs: int = 15,
    lr: float = 2e-5,
    batch_size: int = 16,
    base_checkpoint: str = "models/sls_head_silence_invariant.pth",
    out_checkpoint: str = "models/sls_head_channel_robust.pth",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=== Starting Channel-Robust SLS Training on {device} ===")

    # 1. Collect Data with Dual-Channel Simulation
    indicsynth_samples = collect_indicsynth(target_per_cell=75)
    fleurs_samples = collect_fleurs_bonafide(target_per_gender=150)
    edge_samples = generate_edge_tts_samples(n_per_voice=25)

    all_samples = indicsynth_samples + fleurs_samples + edge_samples
    logger.info(f"Total dataset: {len(all_samples)} slices ({len(indicsynth_samples)+len(edge_samples)} spoof, {len(fleurs_samples)} bona fide).")

    # 2. Partition into Train (70%) and Held-out Test (30%)
    np.random.seed(42)
    by_gen = {}
    for s in all_samples:
        by_gen.setdefault(s["generator"], []).append(s)

    train_samples, test_samples = [], []
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

    # 4. Load Base SLS Head Checkpoint (Warm-Start)
    sls_head = SLSHead().to(device)
    load_checkpoint_weights(sls_head, Path(base_checkpoint))

    # 5. Baseline Evaluation on Test Set
    criterion = nn.NLLLoss(weight=torch.tensor([0.5, 0.5], device=device))
    logger.info("=== BASELINE EVALUATION (Before Channel Robustness Fine-Tuning) ===")
    base_eval = evaluate(sls_head, test_samples, criterion, device=device)
    logger.info(f"Baseline Val Loss: {base_eval['val_loss']} | Val EER: {base_eval['overall']['eer']}% | Overall Acc: {base_eval['overall']['accuracy']}%")
    logger.info(f"Baseline Clean Acc: {base_eval['clean_accuracy']}% | Replay Acc: {base_eval['replay_accuracy']}%")
    logger.info(f"Baseline Generator Acc: {base_eval['generator_accuracy']}")

    # 6. Fine-Tuning
    logger.info(f"=== FINE-TUNING SLS HEAD ({epochs} Epochs, lr={lr}) ===")
    optimizer = torch.optim.AdamW(sls_head.parameters(), lr=lr, weight_decay=1e-4)

    best_replay_acc = base_eval["replay_accuracy"]
    best_overall_acc = base_eval["overall"]["accuracy"]
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

        avg_train_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0

        cur_eval = evaluate(sls_head, test_samples, criterion, device=device)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {cur_eval['val_loss']:.4f} | Val EER: {cur_eval['overall']['eer']}% | "
            f"Overall Acc: {cur_eval['overall']['accuracy']}% | Clean Acc: {cur_eval['clean_accuracy']}% | "
            f"Replay Acc: {cur_eval['replay_accuracy']}%"
        )

        # Checkpointing: Save if acoustic replay accuracy improves while clean accuracy stays high (>= 96%)
        if cur_eval["replay_accuracy"] >= best_replay_acc and cur_eval["clean_accuracy"] >= 96.0:
            best_replay_acc = cur_eval["replay_accuracy"]
            best_overall_acc = cur_eval["overall"]["accuracy"]
            Path(out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(sls_head.state_dict(), out_checkpoint)
            logger.info(f"  --> Saved new best channel-robust checkpoint to {out_checkpoint}")

    # 7. Final Verification
    logger.info("=== FINAL POST-ADAPTATION AUDIT ===")
    final_eval = evaluate(sls_head, test_samples, criterion, device=device)
    logger.info(f"Final Overall: {final_eval['overall']}")
    logger.info(f"Final Clean Acc: {final_eval['clean_accuracy']}% | Replay Acc: {final_eval['replay_accuracy']}%")
    logger.info(f"Final Generator Acc: {final_eval['generator_accuracy']}")

    # Export results JSON
    results_path = Path("outputs/sls_channel_robust_results.json")
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
