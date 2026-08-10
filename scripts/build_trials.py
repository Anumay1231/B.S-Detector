#!/usr/bin/env python
"""
scripts/build_trials.py
-------------------------

Phase 7A/7B/7F/7G: turn an official dataset trial-list file into a
standardized, sampled, calibration/evaluation-split trials CSV that the
rest of the Phase 7 pipeline (extract_embeddings.py, score_trials.py,
run_voxceleb_calibration.py) consumes.

This script does NOT compute any embeddings or similarity scores -- it
only parses trial metadata, checks which files exist, samples a
deterministic subset, and assigns each kept trial to a calibration or
evaluation split. See src/speaker_verification/datasets/ for the
underlying logic.

Usage:
    python scripts/build_trials.py \\
        --dataset voxceleb \\
        --trials /path/to/voxceleb1_test_list.txt \\
        --audio-root /path/to/voxceleb1/wav \\
        --output data/voxceleb/trials/subset.csv \\
        --max-genuine 1000 \\
        --max-impostor 1000 \\
        --seed 42 \\
        --calibration-fraction 0.5 \\
        --split-seed 42

If --trials does not exist, this script explains exactly what needs to
be obtained from the official VoxCeleb site and exits cleanly (exit
code 0) rather than erroring or fabricating trial data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.datasets.trials import (  # noqa: E402
    Trial,
    sample_trials,
    speaker_overlap,
    split_trials_by_speaker,
    summarize_trials,
    validate_trials,
)
from speaker_verification.datasets.voxceleb import (  # noqa: E402
    VOXCELEB_SOURCE_URL,
    parse_trial_list,
)

CSV_FIELDNAMES = [
    "trial_id",
    "reference_id",
    "test_id",
    "reference_path",
    "test_path",
    "label",
    "reference_speaker",
    "test_speaker",
    "split",
]

DATASET_PARSERS = {
    "voxceleb": parse_trial_list,
}


def _explain_missing_trial_list(path: str) -> None:
    print(f"No trial-list file found at: {path}")
    print()
    print(
        "This script does not download or generate a trial-list file -- "
        "it must be obtained from the official VoxCeleb site:"
    )
    print(f"    {VOXCELEB_SOURCE_URL}")
    print()
    print("You need TWO things from that site (or wherever you legally")
    print("obtained VoxCeleb access):")
    print("  1. A verification trial-list file (VoxCeleb1, VoxCeleb1-E, or")
    print("     VoxCeleb1-H, 'cleaned' or not) -- a text file with lines")
    print("     like:")
    print("         1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav")
    print("         0 id10270/x6uYqmx31kE/00001.wav id10300/ize_eiCFEg0/00003.wav")
    print("     (1 = genuine/same-speaker, 0 = impostor/different-speaker)")
    print("  2. The corresponding audio files on disk, laid out as")
    print("     <audio_root>/<speaker_id>/<video_id>/<utterance>.wav")
    print()
    print("Pass them as:")
    print("    python scripts/build_trials.py --trials <trial_list_file> "
          "--audio-root <audio_dir> ...")
    print()
    print(
        "VoxCeleb access requires completing VGG's own registration/"
        "access process. This project does not scrape, bypass, or "
        "mirror that dataset."
    )


def build_subset(
    all_trials: List[Trial],
    max_genuine,
    max_impostor,
    seed: int,
    calibration_fraction: float,
    split_seed: int,
):
    """Pure (no I/O) pipeline: validate -> sample -> split. Returns a
    dict with everything the CLI needs to report and write out. Kept
    separate from main() so it is directly unit-testable with synthetic
    Trial objects (Phase 7K)."""
    valid, missing = validate_trials(all_trials)
    subset = sample_trials(valid, max_genuine=max_genuine, max_impostor=max_impostor, seed=seed)
    calibration, evaluation, dropped = split_trials_by_speaker(
        subset, calibration_fraction=calibration_fraction, seed=split_seed
    )
    overlap = speaker_overlap(calibration, evaluation)

    return {
        "valid": valid,
        "missing": missing,
        "subset": subset,
        "calibration": calibration,
        "evaluation": evaluation,
        "dropped": dropped,
        "speaker_overlap": overlap,
    }


def write_trials_csv(output_path: str, calibration: List[Trial], evaluation: List[Trial]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for split_name, trials in (("calibration", calibration), ("evaluation", evaluation)):
            for t in trials:
                writer.writerow(
                    {
                        "trial_id": t.trial_id,
                        "reference_id": t.reference_id,
                        "test_id": t.test_id,
                        "reference_path": t.reference_path,
                        "test_path": t.test_path,
                        "label": t.label,
                        "reference_speaker": t.reference_speaker,
                        "test_speaker": t.test_speaker,
                        "split": split_name,
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a dataset trial-list file, validate file "
        "existence, sample a deterministic subset, split it into "
        "speaker-disjoint calibration/evaluation sets, and write a "
        "standardized trials CSV."
    )
    parser.add_argument("--dataset", default="voxceleb", choices=sorted(DATASET_PARSERS))
    parser.add_argument("--trials", required=True, help="Path to the official trial-list file.")
    parser.add_argument("--audio-root", required=True, help="Local directory containing the dataset audio.")
    parser.add_argument("--output", default="data/voxceleb/trials/subset.csv")
    parser.add_argument("--max-genuine", type=int, default=None)
    parser.add_argument("--max-impostor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic subset sampling.")
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="Fraction of unique speakers assigned to the calibration side of the split.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42, help="Seed for the calibration/evaluation speaker split."
    )
    args = parser.parse_args()

    if not Path(args.trials).is_file():
        _explain_missing_trial_list(args.trials)
        return 0  # Clean exit -- this is expected/documented, not an error.

    parse_fn = DATASET_PARSERS[args.dataset]
    all_trials = parse_fn(args.trials, args.audio_root)
    print(f"Parsed {len(all_trials)} trial(s) from '{args.trials}'.")

    result = build_subset(
        all_trials,
        max_genuine=args.max_genuine,
        max_impostor=args.max_impostor,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        split_seed=args.split_seed,
    )

    print(f"Valid (audio files found): {len(result['valid'])}")
    print(f"Missing (audio files NOT found, excluded): {len(result['missing'])}")
    if result["missing"]:
        preview = result["missing"][:5]
        for t in preview:
            print(f"    missing: {t.reference_path} or {t.test_path} (trial {t.trial_id})")
        if len(result["missing"]) > len(preview):
            print(f"    ... and {len(result['missing']) - len(preview)} more.")

    subset_summary = summarize_trials(result["subset"])
    print()
    print(f"Sampled subset: {subset_summary.num_trials} trials "
          f"({subset_summary.num_genuine} genuine, {subset_summary.num_impostor} impostor), "
          f"{subset_summary.num_speakers} unique speakers, "
          f"{subset_summary.num_utterances} unique utterances "
          f"(seed={args.seed}).")

    cal_summary = summarize_trials(result["calibration"])
    eval_summary = summarize_trials(result["evaluation"])
    print()
    print(f"Calibration split: {cal_summary.num_trials} trials "
          f"({cal_summary.num_genuine} genuine, {cal_summary.num_impostor} impostor), "
          f"{cal_summary.num_speakers} speakers (split-seed={args.split_seed}, "
          f"calibration-fraction={args.calibration_fraction}).")
    print(f"Evaluation split:  {eval_summary.num_trials} trials "
          f"({eval_summary.num_genuine} genuine, {eval_summary.num_impostor} impostor), "
          f"{eval_summary.num_speakers} speakers.")
    print(f"Dropped (speaker split across both sides): {len(result['dropped'])} trial(s).")
    print(f"Speaker overlap between calibration and evaluation: "
          f"{len(result['speaker_overlap'])} speaker(s) "
          f"{'(SPEAKER-DISJOINT -- good)' if not result['speaker_overlap'] else '*** LEAKAGE DETECTED ***'}.")

    if not result["calibration"] or not result["evaluation"]:
        print()
        print(
            "WARNING: calibration or evaluation split is empty after "
            "sampling/splitting. Increase --max-genuine/--max-impostor "
            "or check that the trial list has enough distinct speakers."
        )

    write_trials_csv(args.output, result["calibration"], result["evaluation"])
    print()
    print(f"Wrote {len(result['calibration']) + len(result['evaluation'])} trial(s) to '{args.output}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
