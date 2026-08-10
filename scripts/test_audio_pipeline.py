#!/usr/bin/env python
"""
scripts/test_audio_pipeline.py
-------------------------------

Manual smoke-test CLI for the Phase 3 audio preprocessing pipeline
(src/speaker_verification/audio.py). This is a diagnostic tool, not part
of the pytest suite (see tests/test_audio.py for automated tests).

Usage:
    python scripts/test_audio_pipeline.py path/to/audio.wav

Prints original (pre-preprocessing) and processed (post-preprocessing)
audio metadata side by side, and explicitly flags stereo/multi-channel ->
mono conversion when it occurs. Does not modify the input file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package
# (no pyproject.toml / setup.py has been added yet).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import (  # noqa: E402
    AudioError,
    get_audio_info,
    preprocess_audio,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the audio preprocessing pipeline on a single file "
        "and report before/after metadata."
    )
    parser.add_argument("audio_path", help="Path to a WAV or FLAC file.")
    args = parser.parse_args()

    print(f"File: {args.audio_path}")

    try:
        info = get_audio_info(args.audio_path)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Original sample rate: {info.sample_rate} Hz")
    print(f"Original channels: {info.num_channels}")
    print(f"Original duration: {info.duration_seconds:.4f} s")

    try:
        waveform, sample_rate = preprocess_audio(args.audio_path)
    except AudioError as exc:
        print(f"ERROR during preprocessing: {exc}", file=sys.stderr)
        return 1

    processed_channels = waveform.shape[0]
    processed_duration = waveform.shape[1] / sample_rate

    print(f"Processed sample rate: {sample_rate} Hz")
    print(f"Processed channels: {processed_channels}")
    print(f"Processed duration: {processed_duration:.4f} s")
    print(f"Processed waveform shape: {tuple(waveform.shape)}")

    if info.num_channels > 1 and processed_channels == 1:
        print(
            f"Note: input had {info.num_channels} channels and was "
            f"converted to mono."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
