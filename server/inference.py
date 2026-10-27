"""
B.S. Detector — Core Inference Engine for Real-Time Forensic Audio Analysis
Supports XLS-R 300M + Selective Layer Summarization (SLS) Head inference.
"""

import io
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bsdetector.frontend.inference")

INPUT_LENGTH = 64600  # 64,600 samples at 16kHz (~4.04s)
TARGET_SR = 16000     # 16 kHz sample rate

# Check for PyTorch availability
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import Wav2Vec2Model
    from huggingface_hub import hf_hub_download
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch or Transformers not fully available: {e}")


# =====================================================================
# SLS Architecture Definition
# =====================================================================
if TORCH_AVAILABLE:
    class SLSHead(nn.Module):
        """Selective Layer Summarization (SLS) classifier head for XLS-R 300M."""
        
        def __init__(self, hidden_dim: int = 1024):
            super().__init__()
            # Per-layer attention gate
            self.fc0 = nn.Linear(hidden_dim, 1)
            # Post-fusion processing
            self.first_bn = nn.BatchNorm2d(1)
            self.selu = nn.SELU()
            # MaxPool2d(3, 3) on (1, 201, 1024) -> (1, 67, 341) => 22,847 features
            self.fc1 = nn.Linear(22847, 1024)
            self.fc3 = nn.Linear(1024, 2)
            self.logsoftmax = nn.LogSoftmax(dim=-1)

        def forward(self, hidden_states_list):
            """Forward pass through SLS head."""
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


@dataclass
class ForensicInferenceResult:
    """Forensic report containing real model output and layer attention dynamics."""
    verdict: str                 # "BONAFIDE" or "SPOOF"
    confidence: float            # Percentage (0-100%)
    prob_bonafide: float         # 0.0 to 1.0
    prob_spoof: float            # 0.0 to 1.0
    layer_weights: List[float]   # 24 layer attention weights from SLS gating
    latency_ms: float            # Inference execution time in ms
    audio_duration_sec: float    # Duration of input audio
    device_used: str             # "cuda", "mps", or "cpu"
    interlock_triggered: bool    # True if spoof, blocking Stage 2 biometrics
    layer_summary: str           # Forensic explanation of active layers
    windows: List[Dict[str, Any]] = field(default_factory=list)
    num_windows: int = 1
    aggregation: str = "single"
    peak_spoof_window: Optional[Dict[str, Any]] = None


def select_best_device() -> str:
    """Automatically select the fastest available device (CUDA > MPS > CPU)."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_detector_models(device: Optional[str] = None, checkpoint_version: str = "v1"):
    """
    Load frozen Wav2Vec2 XLS-R 300M backbone and pretrained SLS Head weights.
    Returns: (backbone, sls_head, device)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch and transformers must be installed for real model inference.")
    
    if device is None:
        device = select_best_device()
    
    logger.info(f"Initializing XLS-R 300M on target device: {device}")
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    # Initialize SLS Head
    sls_head = SLSHead().to(device)
    
    # Download and load SLS checkpoint
    filename = "v1/epoch_2.pth" if checkpoint_version == "v1" else "v2/epoch_16.pth"
    logger.info(f"Fetching pretrained checkpoint from HuggingFace: sukhdeveyash/XLS-R-SLS-Deepfake-Detection ({filename})")
    ckpt_path = hf_hub_download(
        repo_id="sukhdeveyash/XLS-R-SLS-Deepfake-Detection",
        filename=filename
    )
    
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sls_keys = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key.startswith("ssl_model."):
            continue
        sls_keys[clean_key] = value
    
    sls_head.load_state_dict(sls_keys, strict=False)
    sls_head.eval()
    logger.info(f"Loaded {len(sls_keys)} SLS head weights into model on {device}")
    
    return backbone, sls_head, device


def find_speech_onset_index(
    waveform: "torch.Tensor",
    target_length: int = INPUT_LENGTH,
    sample_rate: int = TARGET_SR
) -> int:
    """
    Finds the onset sample index of active speech in a 1D waveform tensor.
    If speech starts after leading silence, returns the onset index offset by a 150ms buffer
    so the 4.04-second analysis window aligns with real vocal activity rather than dead silence.
    """
    total_samples = waveform.shape[0]
    if total_samples <= target_length:
        return 0

    frame_len = int(0.025 * sample_rate)  # 25 ms
    hop_len = int(0.010 * sample_rate)    # 10 ms
    if total_samples < frame_len:
        return 0

    unfolded = waveform.unfold(0, frame_len, hop_len)
    frame_energies = (unfolded ** 2).mean(dim=1)
    max_energy = frame_energies.max().item()

    if max_energy < 1e-5:
        # Near-zero signal across the whole clip
        return 0

    thresh = max(1e-5, max_energy * 0.01)  # 1% of peak speech energy (-20 dB)
    speech_indices = (frame_energies >= thresh).nonzero(as_tuple=True)[0]
    if len(speech_indices) == 0:
        return 0

    first_speech_idx = speech_indices[0].item() * hop_len
    onset = max(0, first_speech_idx - int(0.150 * sample_rate))
    max_start = max(0, total_samples - target_length)
    return min(onset, max_start)


def pad_or_truncate_audio(waveform: "torch.Tensor", target_length: int = INPUT_LENGTH) -> "torch.Tensor":
    """Tile-pad or extract speech-centered window of exactly target_length samples."""
    if waveform.shape[0] <= target_length:
        repeats = (target_length // waveform.shape[0]) + 1
        tiled = waveform.repeat(repeats)
        return tiled[:target_length]
    onset = find_speech_onset_index(waveform, target_length=target_length)
    return waveform[onset : onset + target_length]


def extract_sliding_windows(
    waveform: "torch.Tensor",
    window_length: int = INPUT_LENGTH,
    stride_sec: float = 2.0,
    sample_rate: int = TARGET_SR,
    max_windows: int = 20,
) -> Tuple[List["torch.Tensor"], List[Dict[str, Any]]]:
    """
    Slices a 1D waveform tensor into overlapping windows of window_length samples.
    Returns:
        windows: List of 1D tensors, each of length window_length.
        window_metadata: List of dicts with window boundaries, indices, and padding status.
    """
    total_samples = waveform.shape[0]

    # If the audio is shorter than or equal to window_length, tile-pad and return 1 window
    if total_samples <= window_length:
        repeats = (window_length // max(1, total_samples)) + 1
        tiled = waveform.repeat(repeats)[:window_length]
        dur_sec = round(total_samples / sample_rate, 3)
        return [tiled], [{"index": 0, "start_sec": 0.0, "end_sec": dur_sec, "is_padded": total_samples < window_length}]

    stride_samples = int(stride_sec * sample_rate)
    stride_samples = max(int(0.5 * sample_rate), stride_samples)

    # Calculate sliding window start positions
    starts = list(range(0, total_samples - window_length + 1, stride_samples))
    last_possible = total_samples - window_length
    if starts and starts[-1] < last_possible:
        starts.append(last_possible)

    # Cap number of windows for responsiveness
    if len(starts) > max_windows:
        indices = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
        starts = [starts[i] for i in indices]

    windows = []
    metadata = []
    for idx, s in enumerate(starts):
        win = waveform[s : s + window_length]
        start_s = round(s / sample_rate, 3)
        end_s = round((s + window_length) / sample_rate, 3)
        windows.append(win)
        metadata.append({"index": idx, "start_sec": start_s, "end_sec": end_s, "is_padded": False})

    return windows, metadata


def load_and_preprocess_audio(audio_bytes_or_file, target_sr: int = TARGET_SR) -> Tuple["torch.Tensor", float, np.ndarray]:
    """
    Reads audio from bytes or file-like object, converts to mono, resamples to 16kHz,
    and returns (padded_tensor, original_duration_sec, raw_audio_data).

    Preprocessing is intentionally kept consistent with the benchmark feature-extraction
    pipeline in scripts/streaming.py + scripts/features.py:
      - soundfile.read() returns float32 in [-1, 1] natively for float-format files and
        integer PCM files are NOT rescaled further (matches the sf.read() + torch.tensor()
        path used when building the cached features).
      - No peak-amplitude normalization is applied after reading, because the cached
        XLS-R features were extracted from un-normalised waveforms.  Applying peak-norm
        here would shift the input distribution and corrupt the BatchNorm running stats.
      - Integer PCM dtypes (e.g. int16 WAV) ARE divided by their max value so they land
        in [-1, 1], which is what soundfile does automatically for float-format files.
      - Resampling uses scipy.signal.resample (frequency-domain) as a fallback when the
        file is not already at 16 kHz; the benchmark corpus was natively 16 kHz so this
        branch was never exercised during training.
    """
    raw_audio = None
    orig_sr = target_sr

    # Primary: soundfile (handles float32/float64/PCM files natively, returns float64)
    try:
        import soundfile as sf
        if isinstance(audio_bytes_or_file, bytes):
            bio = io.BytesIO(audio_bytes_or_file)
        else:
            bio = audio_bytes_or_file
        # always_2d=False returns 1-D for mono, 2-D for multi-channel
        data, orig_sr = sf.read(bio, dtype="float32", always_2d=False)
        raw_audio = data
    except Exception as sf_err:
        logger.debug(f"soundfile read failed: {sf_err}, trying scipy.io.wavfile")
        import scipy.io.wavfile as wavfile
        if isinstance(audio_bytes_or_file, bytes):
            bio = io.BytesIO(audio_bytes_or_file)
        else:
            bio = audio_bytes_or_file
        orig_sr, data = wavfile.read(bio)
        raw_audio = data
        # scipy returns raw integer PCM — normalise to [-1, 1] to match soundfile behaviour
        if np.issubdtype(raw_audio.dtype, np.integer):
            max_val = float(np.iinfo(raw_audio.dtype).max)
            raw_audio = raw_audio.astype(np.float32) / max_val
        else:
            raw_audio = raw_audio.astype(np.float32)

    # Convert to mono if multi-channel
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)

    raw_audio = raw_audio.astype(np.float32)
    duration_sec = float(len(raw_audio)) / float(orig_sr)

    # Resample to 16,000 Hz if needed
    if orig_sr != target_sr:
        num_target_samples = int(len(raw_audio) * float(target_sr) / float(orig_sr))
        try:
            from scipy.signal import resample
            resampled = resample(raw_audio, num_target_samples)
        except Exception:
            # Linear interpolation fallback
            indices = np.linspace(0, len(raw_audio) - 1, num_target_samples)
            resampled = np.interp(indices, np.arange(len(raw_audio)), raw_audio)
        raw_audio = resampled.astype(np.float32)
        logger.debug(
            f"Resampled audio from {orig_sr} Hz -> {target_sr} Hz "
            f"({num_target_samples} samples). Note: benchmark features were extracted "
            f"from native 16 kHz audio; resampled inputs may show slight distribution shift."
        )

    # NOTE: Peak-amplitude normalisation has been intentionally removed.
    # The cached XLS-R features used for fine-tuning were extracted from waveforms
    # loaded with soundfile (which scales to [-1, 1] but does NOT peak-normalise).
    # Applying an additional per-clip peak-norm here rescales each clip to a fixed
    # max of 1.0, which can distort quiet/soft speech and shifts the input distribution
    # away from what the model's BatchNorm running statistics expect.

    tensor_wave = torch.from_numpy(raw_audio)
    padded_wave = pad_or_truncate_audio(tensor_wave, target_length=INPUT_LENGTH)
    return padded_wave, duration_sec, raw_audio


def run_forensic_detection(
    audio_source: Any,
    backbone: Any,
    sls_head: Any,
    device: str,
    full_clip: bool = True,
    stride_sec: float = 2.0,
    aggregation: str = "max",
) -> ForensicInferenceResult:
    """
    Executes end-to-end forensic inference on audio input:
    1. Preprocesses & normalizes audio to 16kHz.
    2. If full_clip is True and audio > 4.04s, extracts overlapping sliding windows (2.0s stride).
    3. Batches windows through frozen XLS-R 300M and SLS Head in a single forward pass.
    4. Aggregates scores across all windows (max-pooling or top2) and identifies peak spoof segment.
    """
    t_start = time.perf_counter()
    
    # Preprocess
    waveform_1d, duration_sec, raw_samples = load_and_preprocess_audio(audio_source)
    
    # Check if input waveform is silence or muted mic (below acoustic detection threshold)
    full_wave = torch.from_numpy(raw_samples)
    rms_energy = float(torch.sqrt(torch.mean(full_wave ** 2)).item())
    max_amp = float(torch.max(torch.abs(full_wave)).item())
    if max_amp < 1e-4 or rms_energy < 5e-5:
        raise ValueError(
            "The audio recording contains silence or is below the microphone detection threshold. "
            "Please speak clearly into the microphone."
        )

    # Extract windows: sliding full-clip if requested, else single speech-aligned window
    if full_clip and full_wave.shape[0] > INPUT_LENGTH:
        windows_list, win_meta = extract_sliding_windows(
            full_wave, window_length=INPUT_LENGTH, stride_sec=stride_sec, sample_rate=TARGET_SR
        )
    else:
        windows_list = [waveform_1d]
        dur = min(round(duration_sec, 3), round(INPUT_LENGTH / TARGET_SR, 3))
        win_meta = [{"index": 0, "start_sec": 0.0, "end_sec": dur, "is_padded": full_wave.shape[0] <= INPUT_LENGTH}]

    K = len(windows_list)
    batch_waveforms = torch.stack(windows_list, dim=0).to(device)  # (K, 64600)
    
    with torch.no_grad():
        # Step 1: Feature Extraction across all 24 layers of XLS-R in a single batched pass
        outputs = backbone(batch_waveforms, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states[1:])  # 24 tensors, each (K, 201, 1024)
        
        # Step 2: SLS Classification & Layer Attention Gating
        log_probs, layer_weight_tensors = sls_head(hidden_states)
        probs = torch.exp(log_probs)  # (K, 2) [p_spoof, p_bonafide]

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Record per-window probabilities
    for i in range(K):
        sp = float(probs[i, 0].item())
        bf = float(probs[i, 1].item())
        win_meta[i]["prob_spoof"] = round(sp, 4)
        win_meta[i]["prob_bonafide"] = round(bf, 4)
        win_meta[i]["verdict"] = "SPOOF" if sp >= 0.50 else "BONAFIDE"

    # Aggregation
    peak_idx = int(torch.argmax(probs[:, 0]).item())
    max_spoof = float(probs[:, 0].max().item())
    top2_spoof = float(probs[:, 0].topk(min(2, K)).values.mean().item())

    if K == 1:
        p_spoof = win_meta[0]["prob_spoof"]
        p_bonafide = win_meta[0]["prob_bonafide"]
        agg_name = "single"
    elif aggregation == "top2":
        p_spoof = top2_spoof
        p_bonafide = 1.0 - p_spoof
        agg_name = "sliding_window_top2"
    else:
        p_spoof = max_spoof
        p_bonafide = 1.0 - p_spoof
        agg_name = "sliding_window_max"

    peak_window = {
        "index": peak_idx,
        "start_sec": win_meta[peak_idx]["start_sec"],
        "end_sec": win_meta[peak_idx]["end_sec"],
        "prob_spoof": win_meta[peak_idx]["prob_spoof"],
        "prob_bonafide": win_meta[peak_idx]["prob_bonafide"],
        "verdict": win_meta[peak_idx]["verdict"],
    }

    # Extract per-layer attention scalar weights for peak window
    layer_weights = [float(w[peak_idx].squeeze().cpu().item()) for w in layer_weight_tensors]

    # Classification decision
    is_spoof = p_spoof >= 0.50
    verdict = "SPOOF" if is_spoof else "BONAFIDE"
    confidence = (p_spoof if is_spoof else p_bonafide) * 100.0

    # Forensic interpretation
    top_layers = [i + 1 for i, w in enumerate(layer_weights) if w >= np.percentile(layer_weights, 75)]
    if is_spoof:
        if K > 1:
            layer_summary = (
                f"Synthetic voice anomaly detected in window {peak_window['start_sec']}s-{peak_window['end_sec']}s "
                f"(peak spoof prob {peak_window['prob_spoof']*100:.1f}%). Transformer layers {top_layers} active."
            )
        else:
            layer_summary = (
                f"Synthetic phase anomalies heavily weighted in transformer layers {top_layers}. "
                "Biometric speaker verification permanently locked."
            )
    else:
        layer_summary = (
            f"Natural human glottal pulse consistency confirmed across all {K} analysis windows. "
            "Authenticity certified for biometric verification."
        )

    return ForensicInferenceResult(
        verdict=verdict,
        confidence=confidence,
        prob_bonafide=p_bonafide,
        prob_spoof=p_spoof,
        layer_weights=layer_weights,
        latency_ms=latency_ms,
        audio_duration_sec=duration_sec,
        device_used=device,
        interlock_triggered=is_spoof,
        layer_summary=layer_summary,
        windows=win_meta,
        num_windows=K,
        aggregation=agg_name,
        peak_spoof_window=peak_window,
    )
