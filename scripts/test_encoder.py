#!/usr/bin/env python
"""
scripts/test_encoder.py
------------------------

Manual integration-test CLI for the Phase 4 ECAPA-TDNN speaker encoder
(src/speaker_verification/encoder.py). This is a diagnostic tool, not part
of the pytest suite (see tests/test_encoder.py for automated tests).

Single-file usage:
    python scripts/test_encoder.py path/to/audio.wav

Prints device info, original vs. processed audio metadata, and embedding
shape/dtype/device/norm/inference time. The full embedding vector is not
printed by default.

Two-file usage:
    python scripts/test_encoder.py reference.wav test.wav

Extracts an embedding for each file independently and reports their
shapes and basic statistics side by side. This does NOT perform speaker
verification: no similarity score and no MATCH/NON_MATCH/UNCERTAIN
decision is computed here — that belongs to a later phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running this script directly without installing the package
# (no pyproject.toml / setup.py has been added yet).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import AudioError, get_audio_info, preprocess_audio  # noqa: E402
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder, timed_encode  # noqa: E402


def _report_single(encoder: SpeakerEncoder, path: str) -> bool:
    """Process one file and print its full diagnostic report.
    Returns True on success, False on failure.
    """
    print(f"Input file: {path}")
    print(f"Device: {encoder.describe_device()}")

    try:
        info = get_audio_info(path)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False

    print(f"Original sample rate: {info.sample_rate} Hz")

    try:
        waveform, sample_rate = preprocess_audio(path)
    except AudioError as exc:
        print(f"ERROR during preprocessing: {exc}", file=sys.stderr)
        return False

    print(f"Processed sample rate: {sample_rate} Hz")
    print(f"Processed channels: {waveform.shape[0]}")
    print(f"Waveform shape: {tuple(waveform.shape)}")

    try:
        embedding, elapsed = timed_encode(encoder, waveform)
    except (EncoderError, Exception) as exc:  # noqa: BLE001
        print(f"ERROR during encoding: {exc}", file=sys.stderr)
        return False

    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding device: {embedding.device}")
    print(f"Embedding norm: {torch.linalg.norm(embedding).item():.6f}")
    print(f"Inference time: {elapsed * 1000:.2f} ms")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(encoder.device_info.gpu_index or 0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    return True


def _report_two_file(encoder: SpeakerEncoder, path_a: str, path_b: str) -> bool:
    """Extract embeddings for two files independently and report their
    shapes/statistics. Does NOT compute similarity or a MATCH/NON_MATCH
    decision — that is out of scope for this phase.
    """
    embeddings = {}
    for label, path in (("reference", path_a), ("test", path_b)):
        print(f"--- {label}: {path} ---")
        try:
            waveform, sample_rate = preprocess_audio(path)
        except AudioError as exc:
            print(f"ERROR during preprocessing '{path}': {exc}", file=sys.stderr)
            return False

        try:
            embedding, elapsed = timed_encode(encoder, waveform)
        except (EncoderError, Exception) as exc:  # noqa: BLE001
            print(f"ERROR during encoding '{path}': {exc}", file=sys.stderr)
            return False

        embeddings[label] = embedding
        print(f"  Processed sample rate: {sample_rate} Hz")
        print(f"  Waveform shape: {tuple(waveform.shape)}")
        print(f"  Embedding shape: {tuple(embedding.shape)}")
        print(f"  Embedding dtype: {embedding.dtype}")
        print(f"  Embedding mean: {embedding.mean().item():.6f}")
        print(f"  Embedding std: {embedding.std().item():.6f}")
        print(f"  Embedding norm: {torch.linalg.norm(embedding).item():.6f}")
        print(f"  Inference time: {elapsed * 1000:.2f} ms")
        print()

    print(
        "Note: this is a two-file EMBEDDING EXTRACTION report only. No "
        "cosine similarity, threshold, or MATCH/NON_MATCH/UNCERTAIN "
        "decision is computed here — that belongs to a later phase."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ECAPA-TDNN speaker embedding(s) from one or "
        "two audio files and report diagnostics. Does not perform "
        "speaker verification."
    )
    parser.add_argument("audio_path", help="Path to a WAV or FLAC file.")
    parser.add_argument(
        "second_audio_path",
        nargs="?",
        default=None,
        help="Optional second file (e.g. for a reference/test pair).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override (e.g. 'cpu', 'cuda'). Default: "
        "auto-select CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    try:
        encoder = SpeakerEncoder(device=args.device)
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1

    if args.second_audio_path is None:
        ok = _report_single(encoder, args.audio_path)
    else:
        ok = _report_two_file(encoder, args.audio_path, args.second_audio_path)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
