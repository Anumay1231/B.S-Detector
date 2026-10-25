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


def pad_or_truncate_audio(waveform: "torch.Tensor", target_length: int = INPUT_LENGTH) -> "torch.Tensor":
    """Tile-pad or truncate audio tensor to exactly 64,600 samples."""
    if waveform.shape[0] >= target_length:
        return waveform[:target_length]
    repeats = (target_length // waveform.shape[0]) + 1
    tiled = waveform.repeat(repeats)
    return tiled[:target_length]


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
    device: str
) -> ForensicInferenceResult:
    """
    Executes end-to-end forensic inference on audio input:
    1. Preprocesses & normalizes audio to 16kHz (64,600 samples).
    2. Passes waveform through frozen XLS-R 300M to extract all 24 transformer hidden states.
    3. Feeds states through SLS head with per-layer attention gating.
    4. Computes real class probabilities, layer attention scores, and latency.
    """
    t_start = time.perf_counter()
    
    # Preprocess
    waveform_1d, duration_sec, raw_samples = load_and_preprocess_audio(audio_source)
    waveform_2d = waveform_1d.unsqueeze(0).to(device)  # (1, 64600)
    
    with torch.no_grad():
        # Step 1: Feature Extraction across all 24 layers of XLS-R
        outputs = backbone(waveform_2d, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states[1:])  # 24 tensors, each (1, 201, 1024)
        
        # Step 2: SLS Classification & Layer Attention Gating
        log_probs, layer_weight_tensors = sls_head(hidden_states)
        probs = torch.exp(log_probs).squeeze(0)  # [p_spoof, p_bonafide]
        
        p_spoof = float(probs[0].item())
        p_bonafide = float(probs[1].item())
        
        # Extract per-layer attention scalar weights (24 values)
        layer_weights = [float(w.squeeze().cpu().item()) for w in layer_weight_tensors]

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    
    # Classification decision
    is_spoof = p_spoof >= 0.50
    verdict = "SPOOF" if is_spoof else "BONAFIDE"
    confidence = (p_spoof if is_spoof else p_bonafide) * 100.0
    
    # Forensic interpretation based on layer weights
    top_layers = [i + 1 for i, w in enumerate(layer_weights) if w >= np.percentile(layer_weights, 75)]
    if is_spoof:
        layer_summary = (
            f"Synthetic phase anomalies heavily weighted in transformer layers {top_layers}. "
            "Biometric speaker verification permanently locked."
        )
    else:
        layer_summary = (
            f"Natural human glottal pulse consistency confirmed across layers {top_layers}. "
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
        layer_summary=layer_summary
    )
