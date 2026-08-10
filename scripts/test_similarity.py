#!/usr/bin/env python
"""
scripts/test_similarity.py
----------------------------

Real ECAPA-TDNN integration test for cosine similarity. This is a
diagnostic tool, not part of the pytest suite (see tests/test_similarity.py
for deterministic unit tests using hand-constructed vectors).

Usage:
    python scripts/test_similarity.py reference.wav test.wav

Pipeline:

    reference.wav -> audio preprocessing -> SpeakerEncoder -> embedding A
    test.wav      -> audio preprocessing -> SpeakerEncoder -> embedding B
    embedding A, embedding B -> cosine_similarity() -> single score

This script reports the raw cosine similarity score ONLY. It does NOT
compute a threshold, and it does NOT produce a MATCH / NON_MATCH /
UNCERTAIN verdict — a raw score has no established meaning about whether
two recordings are from the same speaker until it is interpreted against
a threshold calibrated on representative genuine/impostor trial data (a
later phase). Do not interpret the printed score as a verification
result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running this script directly without installing the package
# (no pyproject.toml / setup.py has been added yet).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import AudioError, preprocess_audio  # noqa: E402
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder  # noqa: E402
from speaker_verification.similarity import SimilarityError, cosine_similarity  # noqa: E402


def _encode(encoder: SpeakerEncoder, path: str) -> torch.Tensor:
    waveform, _sample_rate = preprocess_audio(path)
    return encoder.encode(waveform)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ECAPA-TDNN embeddings for two audio files and "
        "report their cosine similarity. Does NOT produce a MATCH/NON_MATCH "
        "verdict — that requires a calibrated threshold (a later phase)."
    )
    parser.add_argument("reference_path", help="Path to the reference WAV/FLAC file.")
    parser.add_argument("test_path", help="Path to the test WAV/FLAC file.")
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override (e.g. 'cpu', 'cuda'). Default: "
        "auto-select CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    print(f"Reference file: {args.reference_path}")
    print(f"Test file: {args.test_path}")

    try:
        encoder = SpeakerEncoder(device=args.device)
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1

    print(f"Device: {encoder.describe_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(encoder.device_info.gpu_index or 0)}")

    try:
        embedding_a = _encode(encoder, args.reference_path)
        embedding_b = _encode(encoder, args.test_path)
    except AudioError as exc:
        print(f"ERROR during audio preprocessing: {exc}", file=sys.stderr)
        return 1
    except EncoderError as exc:
        print(f"ERROR during embedding extraction: {exc}", file=sys.stderr)
        return 1

    print(f"Reference embedding shape: {tuple(embedding_a.shape)}")
    print(f"Test embedding shape: {tuple(embedding_b.shape)}")
    print(f"Reference embedding device: {embedding_a.device}")
    print(f"Test embedding device: {embedding_b.device}")

    # Time the similarity operation separately from ECAPA inference.
    if embedding_a.device.type == "cuda":
        torch.cuda.synchronize(embedding_a.device)
    start = time.perf_counter()
    try:
        score = cosine_similarity(embedding_a, embedding_b)
    except SimilarityError as exc:
        print(f"ERROR computing similarity: {exc}", file=sys.stderr)
        return 1
    if embedding_a.device.type == "cuda":
        torch.cuda.synchronize(embedding_a.device)
    elapsed = time.perf_counter() - start

    print(f"Cosine similarity: {score:.6f}")
    print(f"Inference time (similarity only): {elapsed * 1000:.4f} ms")
    print()
    print(
        "Note: this is a raw similarity score, not a verification result. "
        "No threshold, calibration, or MATCH/NON_MATCH/UNCERTAIN decision "
        "is applied here — that requires threshold calibration on "
        "representative genuine/impostor trial data (a later phase)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
