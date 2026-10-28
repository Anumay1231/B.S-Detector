"""Local inference script for Audio Deepfake Detection using XLS-R 300M + SLS.

Run inference locally on any audio clip (.wav, .flac, .mp3, .pt) using CPU or Apple Silicon MPS.

Usage:
    python scripts/infer_local.py --audio path/to/sample.wav
    python scripts/infer_local.py --audio path/to/sample.wav --model models/sls_head_channel_robust.pth --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import soundfile as sf

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model import (
    INPUT_LENGTH,
    SLSHead,
    extract_hidden_states,
    load_backbone,
    pad_or_truncate,
)


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio from .wav, .flac, .mp3, or .pt tensor, converting to mono 16kHz."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "waveform" in data:
            wf = data["waveform"]
        else:
            wf = data
    else:
        wf_np, sr = sf.read(str(path))
        if len(wf_np.shape) > 1:
            wf_np = wf_np.mean(axis=1)  # Stereo to mono
        if sr != target_sr:
            # Resample using torchaudio or linear interpolation
            import torchaudio.transforms as T
            wf_tensor = torch.tensor(wf_np, dtype=torch.float32).unsqueeze(0)
            resampler = T.Resample(sr, target_sr)
            wf = resampler(wf_tensor).squeeze(0)
        else:
            wf = torch.tensor(wf_np, dtype=torch.float32)

    return pad_or_truncate(wf, INPUT_LENGTH)


def predict(
    audio_path: str | Path,
    model_path: str | Path,
    device: str = "cpu",
    backbone=None,
    sls_head=None,
) -> dict:
    """Predict whether an audio clip is Bonafide (Real) or Spoof (Fake)."""
    device = torch.device(device)

    # 1. Load backbone if not provided
    if backbone is None:
        backbone = load_backbone(device=device)

    # 2. Load SLS Head
    if sls_head is None:
        sls_head = SLSHead()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        sls_head.load_state_dict(state_dict)
        sls_head.to(device)
        sls_head.eval()

    # 3. Load and preprocess audio
    waveform_1d = load_audio(audio_path)
    if waveform_1d.abs().max() < 1e-4:
        raise ValueError(
            f"Audio file '{audio_path}' contains silence or is below the acoustic detection threshold. "
            "Please provide a recording with clear human speech."
        )
    waveform = waveform_1d.unsqueeze(0).to(device)  # (1, 64600)

    # 4. Extract 24-layer representations
    with torch.no_grad():
        hidden_states = extract_hidden_states(backbone, waveform, device=device)
        log_probs = sls_head(hidden_states)  # (1, 2): [spoof, bonafide]
        probs = torch.exp(log_probs).squeeze(0)  # [p_spoof, p_bonafide]

    pred_idx = probs.argmax().item()
    labels = ["Spoof (Deepfake)", "Bona fide (Genuine Speech)"]
    label = labels[pred_idx]
    confidence = probs[pred_idx].item() * 100.0

    return {
        "file": str(audio_path),
        "prediction": label,
        "is_bonafide": pred_idx == 1,
        "confidence": confidence,
        "spoof_prob": probs[0].item(),
        "bonafide_prob": probs[1].item(),
        "log_probs": log_probs.squeeze(0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run local audio deepfake inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    default_head = (
        _PROJECT_ROOT / "models" / "sls_head_channel_robust.pth"
        if (_PROJECT_ROOT / "models" / "sls_head_channel_robust.pth").exists()
        else _PROJECT_ROOT / "models" / "sls_head_n500_indicsynth_best.pth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_head),
        help="Path to fine-tuned SLS head checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on ('cpu', 'mps', 'cuda')",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Running inference on: {args.audio} (device: {args.device})")

    res = predict(args.audio, args.model, device=args.device)

    print("\n" + "=" * 55)
    print("AUDIO DEEPFAKE DETECTION RESULT")
    print("=" * 55)
    print(f"File:         {res['file']}")
    print(f"Prediction:   {res['prediction']}")
    print(f"Confidence:   {res['confidence']:.2f}%")
    print(f"Spoof Prob:   {res['spoof_prob']:.4f}")
    print(f"Bonafide Prob:{res['bonafide_prob']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
