#!/usr/bin/env python
"""
scripts/score_personal_recording.py
--------------------------------------

Phase 7L: score the developer's own two personal recordings using the
EXISTING pipeline (audio.py -> encoder.py -> similarity.py), clearly
labeled as OUT-OF-DOMAIN / PROJECT-SPECIFIC VALIDATION.

This is explicitly NOT calibration data:
    - It is never mixed into the VoxCeleb genuine/impostor score pools
      used by scripts/run_voxceleb_calibration.py.
    - It never changes the VoxCeleb-derived threshold automatically.
    - The two personal recordings themselves stay out of Git (see
      .gitignore: 'tests/recording test/').

Usage:
    python scripts/score_personal_recording.py
    python scripts/score_personal_recording.py path/to/a.wav path/to/b.wav

With no arguments, this script looks for the two files this project's
developer has already recorded locally:
    tests/recording test/Recording.wav
    tests/recording test/Recording (2).wav
and explains how to supply different files if those aren't present.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import AudioError  # noqa: E402
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder  # noqa: E402
from speaker_verification.similarity import SimilarityError, cosine_similarity  # noqa: E402

DEFAULT_PATH_A = "tests/recording test/Recording.wav"
DEFAULT_PATH_B = "tests/recording test/Recording (2).wav"


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 2:
        path_a, path_b = args
    elif len(args) == 0:
        path_a, path_b = DEFAULT_PATH_A, DEFAULT_PATH_B
    else:
        print("Usage: python scripts/score_personal_recording.py [file_a file_b]", file=sys.stderr)
        return 2

    if not Path(path_a).is_file() or not Path(path_b).is_file():
        print(f"Could not find both recordings:")
        print(f"    {path_a} (exists: {Path(path_a).is_file()})")
        print(f"    {path_b} (exists: {Path(path_b).is_file()})")
        print()
        print(
            "These are personal recordings kept outside Git (see "
            "tests/recording test/ in .gitignore); they only exist on "
            "the developer's own machine. Pass two files explicitly if "
            "you want to score a different pair:"
        )
        print("    python scripts/score_personal_recording.py a.wav b.wav")
        return 0  # Clean exit -- this is expected/documented, not an error.

    try:
        encoder = SpeakerEncoder()
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1

    try:
        embedding_a = encoder.encode_file(path_a)
        embedding_b = encoder.encode_file(path_b)
        score = cosine_similarity(embedding_a, embedding_b)
    except (AudioError, EncoderError, SimilarityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Personal same-speaker validation (OUT-OF-DOMAIN / PROJECT-SPECIFIC) ===")
    print(f"Reference: {path_a}")
    print(f"Test:      {path_b}")
    print(f"Device: {encoder.describe_device()}")
    print(f"Cosine similarity: {score:.6f}")
    print()
    print(
        "This score is NOT part of VoxCeleb calibration and does NOT "
        "change the VoxCeleb-derived threshold (see "
        "scripts/run_voxceleb_calibration.py / docs/calibration.md). It "
        "is a single, uncalibrated, out-of-domain data point -- useful "
        "as a sanity check that the pipeline runs on real project audio, "
        "not as evidence of system accuracy in any domain."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
