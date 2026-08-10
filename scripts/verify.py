#!/usr/bin/env python
"""
scripts/verify.py
------------------

Run the full speaker-verification pipeline on a reference/test audio pair
and print a MATCH / NON_MATCH decision, using a threshold YOU supply.

Usage:
    python scripts/verify.py reference.wav test.wav --threshold 0.6123 --calibrated
    python scripts/verify.py reference.wav test.wav --threshold 0.6      # uncalibrated example

There is NO default threshold. You must pass --threshold explicitly. If
that threshold came from real calibration (speaker_verification.calibration
.calibrate(...).eer_threshold on representative genuine/impostor trial
data — see scripts/calibrate.py), also pass --calibrated so the result is
correctly labeled. Without --calibrated, the result is clearly marked
UNCALIBRATED — this script will not let an ad-hoc threshold be silently
mistaken for a validated one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import AudioError  # noqa: E402
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder  # noqa: E402
from speaker_verification.similarity import SimilarityError  # noqa: E402
from speaker_verification.verifier import InvalidThresholdError, SpeakerVerifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reference/test audio through preprocessing -> "
        "ECAPA encoder -> cosine similarity -> threshold decision, and "
        "print MATCH/NON_MATCH plus the raw score. Requires an explicit "
        "--threshold; there is no default/magic value."
    )
    parser.add_argument("reference_path", help="Path to the reference WAV/FLAC file.")
    parser.add_argument("test_path", help="Path to the test WAV/FLAC file.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold in [-1.0, 1.0]. score >= threshold -> MATCH. "
        "No default is provided — see docs/calibration.md for how to "
        "obtain a real, calibrated value.",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Assert that --threshold came from real calibration "
        "(calibrate() on representative genuine/impostor trial data). "
        "Omit this flag for ad-hoc/example thresholds.",
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
        verifier = SpeakerVerifier(
            encoder=encoder, threshold=args.threshold, is_calibrated=args.calibrated
        )
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1
    except InvalidThresholdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Reference file: {args.reference_path}")
    print(f"Test file: {args.test_path}")
    print(f"Device: {encoder.describe_device()}")
    print(f"Threshold: {args.threshold:.6f} ({'calibrated' if args.calibrated else 'UNCALIBRATED'})")
    print()

    try:
        result = verifier.verify_files(args.reference_path, args.test_path)
    except AudioError as exc:
        print(f"ERROR during audio preprocessing: {exc}", file=sys.stderr)
        return 1
    except (EncoderError, SimilarityError) as exc:
        print(f"ERROR during verification: {exc}", file=sys.stderr)
        return 1

    print(f"Cosine similarity: {result.score:.6f}")
    print(f"Decision: {result.outcome.value}")
    print()

    if not result.is_calibrated:
        print(
            "*** This threshold is UNCALIBRATED. The decision above has "
            "not been validated against real genuine/impostor trial data "
            "and should not be treated as reliable. See "
            "docs/calibration.md. ***"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
