#!/usr/bin/env python
"""
scripts/calibrate.py
---------------------

Run threshold calibration against a labeled trial file and report FAR,
FRR, EER, and the resulting candidate threshold.

Usage:
    python scripts/calibrate.py [--trials data/trials.csv]

Expected trial file format (CSV with a header row):

    reference,test,label
    data/reference/speakerA_1.wav,data/reference/speakerA_2.wav,genuine
    data/reference/speakerA_1.wav,data/test/speakerB_1.wav,impostor
    ...

Where:
    - "reference" and "test" are paths to WAV/FLAC files (relative paths
      are resolved relative to the current working directory).
    - "label" is either "genuine" (both files are the same speaker) or
      "impostor" (different speakers). Case-insensitive.

If the trial file does not exist, this script explains exactly how to
create one and exits cleanly (exit code 0) — it does NOT fabricate trial
data or pretend calibration ran.

IMPORTANT: calibration requires REAL genuine AND impostor trials to be
meaningful. A handful of trials (as opposed to dozens+) will produce a
numerically valid but statistically unreliable result — this script
prints that caveat explicitly whenever it applies (see
speaker_verification.calibration.MIN_RELIABLE_TRIALS_PER_CLASS).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.audio import AudioError  # noqa: E402
from speaker_verification.calibration import (  # noqa: E402
    CalibrationError,
    calibrate,
)
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder  # noqa: E402
from speaker_verification.similarity import SimilarityError, cosine_similarity  # noqa: E402

DEFAULT_TRIALS_PATH = "data/trials.csv"
VALID_LABELS = {
    "genuine": "genuine",
    "same": "genuine",
    "match": "genuine",
    "impostor": "impostor",
    "different": "impostor",
    "non_match": "impostor",
    "nonmatch": "impostor",
}


def _explain_missing_trials_file(path: str) -> None:
    print(f"No trial file found at: {path}")
    print()
    print("Calibration requires labeled genuine AND impostor trial pairs.")
    print("To create one, make a CSV file with this header and rows:")
    print()
    print("    reference,test,label")
    print("    path/to/speakerA_1.wav,path/to/speakerA_2.wav,genuine")
    print("    path/to/speakerA_1.wav,path/to/speakerB_1.wav,impostor")
    print()
    print("Where:")
    print("  - 'reference' and 'test' are paths to WAV/FLAC audio files.")
    print("  - 'label' is 'genuine' (same speaker) or 'impostor' (different")
    print("    speaker), case-insensitive.")
    print()
    print(
        "This project currently has only two genuine recordings from one "
        "speaker and no impostor recordings, so a real calibration file "
        "cannot yet be produced — see docs/calibration.md for the exact "
        "data requirement. Run this script again once trial data exists."
    )


def _read_trials(path: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"reference", "test", "label"} - set(
            h.strip().lower() for h in reader.fieldnames
        ):
            raise ValueError(
                f"'{path}' must have a header row with exactly the "
                f"columns: reference,test,label (got: {reader.fieldnames})"
            )
        for line_num, row in enumerate(reader, start=2):
            label_raw = (row.get("label") or "").strip().lower()
            if label_raw not in VALID_LABELS:
                raise ValueError(
                    f"'{path}' line {line_num}: invalid label "
                    f"'{row.get('label')}' — must be 'genuine' or "
                    f"'impostor' (case-insensitive)."
                )
            reference = (row.get("reference") or "").strip()
            test = (row.get("test") or "").strip()
            if not reference or not test:
                raise ValueError(
                    f"'{path}' line {line_num}: 'reference' and 'test' "
                    f"must both be non-empty file paths."
                )
            rows.append((reference, test, VALID_LABELS[label_raw]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run threshold calibration from a labeled trial CSV "
        "(genuine/impostor pairs) and report FAR, FRR, EER, and the "
        "resulting candidate threshold."
    )
    parser.add_argument(
        "--trials",
        default=DEFAULT_TRIALS_PATH,
        help=f"Path to the trial CSV file (default: {DEFAULT_TRIALS_PATH}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override (e.g. 'cpu', 'cuda'). Default: "
        "auto-select CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    if not Path(args.trials).is_file():
        _explain_missing_trials_file(args.trials)
        return 0  # Clean exit — this is expected/documented, not an error.

    try:
        rows = _read_trials(args.trials)
    except (ValueError, OSError) as exc:
        print(f"ERROR reading '{args.trials}': {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"'{args.trials}' contains no trial rows. Nothing to calibrate.")
        return 0

    try:
        encoder = SpeakerEncoder(device=args.device)
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} trial row(s) from '{args.trials}'.")
    print(f"Device: {encoder.describe_device()}")
    print()

    embedding_cache: Dict[str, torch.Tensor] = {}

    def get_embedding(path: str) -> torch.Tensor:
        if path not in embedding_cache:
            embedding_cache[path] = encoder.encode_file(path)
        return embedding_cache[path]

    genuine_scores: List[float] = []
    impostor_scores: List[float] = []
    skipped = 0

    for reference, test, label in rows:
        try:
            embedding_a = get_embedding(reference)
            embedding_b = get_embedding(test)
            score = cosine_similarity(embedding_a, embedding_b)
        except (AudioError, EncoderError, SimilarityError) as exc:
            print(
                f"WARNING: skipping trial ({reference}, {test}, {label}): {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if label == "genuine":
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

    print(
        f"Scored {len(genuine_scores)} genuine and {len(impostor_scores)} "
        f"impostor trial(s)"
        + (f" ({skipped} skipped due to errors)." if skipped else ".")
    )
    print()

    try:
        result = calibrate(genuine_scores, impostor_scores)
    except CalibrationError as exc:
        print(f"Calibration UNAVAILABLE: {exc}", file=sys.stderr)
        return 1

    print(f"Genuine trials:  {result.num_genuine}")
    print(
        f"  score mean/std/min/max: "
        f"{result.genuine_stats.mean:.6f} / {result.genuine_stats.std:.6f} / "
        f"{result.genuine_stats.min:.6f} / {result.genuine_stats.max:.6f}"
    )
    print(f"Impostor trials: {result.num_impostor}")
    print(
        f"  score mean/std/min/max: "
        f"{result.impostor_stats.mean:.6f} / {result.impostor_stats.std:.6f} / "
        f"{result.impostor_stats.min:.6f} / {result.impostor_stats.max:.6f}"
    )
    print()
    print(f"ROC-AUC: {result.roc_auc:.6f}")
    print(f"EER: {result.eer:.6f}")
    print(f"EER threshold: {result.eer_threshold:.6f}")
    print(f"EER method: {result.eer_method}")
    print()

    if not result.is_statistically_reliable:
        print("*** WARNING: this calibration is NOT statistically reliable. ***")
        for warning in result.warnings:
            print(f"  - {warning}")
        print(
            "Do not use this threshold as a production decision boundary. "
            "See docs/calibration.md for the data requirement."
        )
    else:
        print(
            "Sample size meets the documented minimum "
            "(speaker_verification.calibration.MIN_RELIABLE_TRIALS_PER_CLASS) "
            "for a statistically reliable EER estimate. This does not by "
            "itself guarantee the trial data is representative of your "
            "real deployment population — review it before trusting the "
            "threshold in production."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
