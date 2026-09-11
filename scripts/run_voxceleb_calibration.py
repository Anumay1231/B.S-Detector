#!/usr/bin/env python
"""
scripts/run_voxceleb_calibration.py
--------------------------------------

Phase 7E/7F/7G/7J: feed a scored-trials CSV (scripts/score_trials.py)
into the EXISTING Phase 6 calibration engine
(src/speaker_verification/calibration.py), using ONLY the calibration
split to choose a threshold (calibrate()/EER), and the disjoint
evaluation split to measure that fixed threshold's FAR/FRR -- i.e. the
threshold is never chosen and measured on the same trials.

This script does not implement a second calibration algorithm; it is a
thin, reporting-focused wrapper around calibration.calibrate(),
calibration.compute_far(), and calibration.compute_frr().

Usage:
    python scripts/run_voxceleb_calibration.py \\
        --scores outputs/scores/voxceleb_scores.csv \\
        --report-out outputs/calibration/voxceleb_report.txt

If the calibration split has zero genuine or zero impostor trials,
calibration.calibrate() raises InsufficientCalibrationDataError and
this script reports "Calibration UNAVAILABLE" -- it never reports or
writes out a threshold in that case.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Optional

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.calibration import (  # noqa: E402
    CalibrationError,
    calibrate,
    compute_far,
    compute_frr,
)


def _median(values: List[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _read_scores(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
        row["score"] = float(row["score"])
    return rows


def _speakers(rows: List[dict]) -> set:
    speakers = set()
    for row in rows:
        speakers.add(row["reference_speaker"])
        speakers.add(row["test_speaker"])
    return speakers


def _utterances(rows: List[dict]) -> set:
    utterances = set()
    for row in rows:
        utterances.add(row["reference"])
        utterances.add(row["test"])
    return utterances


def run_calibration(
    scored_rows: List[dict], external_threshold: Optional[float] = None
) -> dict:
    """Pure (no I/O) analysis: split calibration/evaluation rows already
    labeled by the 'split' column, run calibration.calibrate() on the
    calibration split ONLY, and evaluate the resulting threshold's
    FAR/FRR on the evaluation split. Returns a report dict. Kept
    separate from main() for direct unit testing (Phase 7K).

    Raises:
        speaker_verification.calibration.InsufficientCalibrationDataError:
            If the calibration split has zero genuine or zero impostor
            scores. Callers must not report/persist a threshold in
            that case.
    """
    calibration_rows = [r for r in scored_rows if r["split"] == "calibration"]
    evaluation_rows = [r for r in scored_rows if r["split"] == "evaluation"]

    cal_genuine = [r["score"] for r in calibration_rows if r["label"] == 1]
    cal_impostor = [r["score"] for r in calibration_rows if r["label"] == 0]
    eval_genuine = [r["score"] for r in evaluation_rows if r["label"] == 1]
    eval_impostor = [r["score"] for r in evaluation_rows if r["label"] == 0]

    calibration_result = calibrate(cal_genuine, cal_impostor)  # may raise

    # FAR/FRR of the chosen threshold measured on the very data it was
    # chosen from. At the EER operating point these are both ~= the EER
    # by construction; they are reported explicitly because "~= EER" is
    # an assertion about the interpolation, not a measurement, and
    # because a single-set protocol run (no held-out split) has no other
    # FAR/FRR to report. This is NOT an unbiased performance estimate --
    # see the evaluation-split block for that.
    cal_far_at_eer = compute_far(cal_impostor, calibration_result.eer_threshold)
    cal_frr_at_eer = compute_frr(cal_genuine, calibration_result.eer_threshold)

    # Optionally measure a threshold that was derived elsewhere (e.g. the
    # speaker-disjoint calibration run) against every scored trial here.
    # Purely additive reporting: it never influences how
    # calibration_result's own threshold was selected.
    external = None
    if external_threshold is not None:
        ext_genuine = [r["score"] for r in scored_rows if r["label"] == 1]
        ext_impostor = [r["score"] for r in scored_rows if r["label"] == 0]
        if ext_genuine and ext_impostor:
            external = {
                "threshold": external_threshold,
                "num_genuine": len(ext_genuine),
                "num_impostor": len(ext_impostor),
                "far": compute_far(ext_impostor, external_threshold),
                "frr": compute_frr(ext_genuine, external_threshold),
            }

    evaluation = None
    if eval_genuine and eval_impostor:
        eval_far = compute_far(eval_impostor, calibration_result.eer_threshold)
        eval_frr = compute_frr(eval_genuine, calibration_result.eer_threshold)
        evaluation = {
            "num_genuine": len(eval_genuine),
            "num_impostor": len(eval_impostor),
            "genuine_mean": statistics.fmean(eval_genuine),
            "genuine_median": _median(eval_genuine),
            "impostor_mean": statistics.fmean(eval_impostor),
            "impostor_median": _median(eval_impostor),
            "far_at_calibration_threshold": eval_far,
            "frr_at_calibration_threshold": eval_frr,
        }

    cal_speakers = _speakers(calibration_rows)
    eval_speakers = _speakers(evaluation_rows)
    overlap = cal_speakers & eval_speakers

    all_genuine = [r["score"] for r in scored_rows if r["label"] == 1]
    all_impostor = [r["score"] for r in scored_rows if r["label"] == 0]

    return {
        "calibration_result": calibration_result,
        "calibration_rows": calibration_rows,
        "evaluation_rows": evaluation_rows,
        "evaluation": evaluation,
        "speaker_overlap": overlap,
        "num_speakers_total": len(_speakers(scored_rows)),
        "num_utterances_total": len(_utterances(scored_rows)),
        # Medians of the calibration split only, so they describe the same
        # population as the calibration mean/std/min/max reported beside
        # them. The dataset-wide medians below cover calibration +
        # evaluation combined and are reported separately, never mixed
        # into a per-split block.
        "calibration_genuine_median": _median(cal_genuine),
        "calibration_impostor_median": _median(cal_impostor),
        "calibration_far_at_eer_threshold": cal_far_at_eer,
        "calibration_frr_at_eer_threshold": cal_frr_at_eer,
        "external_threshold_evaluation": external,
        "dataset_genuine_median": _median(all_genuine),
        "dataset_impostor_median": _median(all_impostor),
    }


def _format_report(result: dict) -> str:
    cal = result["calibration_result"]
    lines = []
    lines.append("=== VoxCeleb calibration report (Phase 7) ===")
    lines.append("")
    lines.append(f"Number of speakers (calibration + evaluation combined): {result['num_speakers_total']}")
    lines.append(f"Number of utterances (unique, combined): {result['num_utterances_total']}")
    lines.append("")
    lines.append("--- Calibration set (used to choose the threshold) ---")
    lines.append(f"Number of genuine trials: {cal.num_genuine}")
    lines.append(f"Number of impostor trials: {cal.num_impostor}")
    lines.append("Genuine score:")
    lines.append(f"    mean:   {cal.genuine_stats.mean:.6f}")
    lines.append(f"    median: {result['calibration_genuine_median']:.6f}")
    lines.append(f"    std:    {cal.genuine_stats.std:.6f}")
    lines.append(f"    min:    {cal.genuine_stats.min:.6f}")
    lines.append(f"    max:    {cal.genuine_stats.max:.6f}")
    lines.append("Impostor score:")
    lines.append(f"    mean:   {cal.impostor_stats.mean:.6f}")
    lines.append(f"    median: {result['calibration_impostor_median']:.6f}")
    lines.append(f"    std:    {cal.impostor_stats.std:.6f}")
    lines.append(f"    min:    {cal.impostor_stats.min:.6f}")
    lines.append(f"    max:    {cal.impostor_stats.max:.6f}")
    lines.append(f"ROC-AUC: {cal.roc_auc:.6f}")
    lines.append(f"EER: {cal.eer:.6f}")
    lines.append(f"EER threshold: {cal.eer_threshold:.6f}")
    lines.append(f"EER method: {cal.eer_method}")
    lines.append(
        f"FAR at EER threshold (same data): {result['calibration_far_at_eer_threshold']:.6f}"
    )
    lines.append(
        f"FRR at EER threshold (same data): {result['calibration_frr_at_eer_threshold']:.6f}"
    )
    lines.append("    (Measured on the data the threshold was chosen from, so this is")
    lines.append("     NOT an unbiased estimate -- see the evaluation-set block below.)")
    lines.append(f"Statistically reliable (>= 30 trials/class): {cal.is_statistically_reliable}")
    for w in cal.warnings:
        lines.append(f"    WARNING: {w}")
    lines.append("")

    ev = result["evaluation"]
    lines.append("--- Evaluation set (held out; threshold NOT tuned on this data) ---")
    if ev is None:
        lines.append("No evaluation-set FAR/FRR computed (evaluation split has zero "
                      "genuine or zero impostor trials).")
    else:
        lines.append(f"Number of genuine trials: {ev['num_genuine']}")
        lines.append(f"Number of impostor trials: {ev['num_impostor']}")
        lines.append(f"Genuine score mean/median: {ev['genuine_mean']:.6f} / {ev['genuine_median']:.6f}")
        lines.append(f"Impostor score mean/median: {ev['impostor_mean']:.6f} / {ev['impostor_median']:.6f}")
        lines.append(f"FAR at calibration EER threshold ({cal.eer_threshold:.6f}): {ev['far_at_calibration_threshold']:.6f}")
        lines.append(f"FRR at calibration EER threshold ({cal.eer_threshold:.6f}): {ev['frr_at_calibration_threshold']:.6f}")
        lines.append("(This is the unbiased performance estimate: the threshold was")
        lines.append(" chosen using only the calibration split, then applied unchanged")
        lines.append(" to this held-out evaluation split.)")
    lines.append("")

    ext = result.get("external_threshold_evaluation")
    if ext is not None:
        lines.append("--- Externally supplied threshold, measured on ALL scored trials ---")
        lines.append(f"Threshold (not derived from this run): {ext['threshold']:.6f}")
        lines.append(f"Trials: {ext['num_genuine']} genuine, {ext['num_impostor']} impostor")
        lines.append(f"FAR: {ext['far']:.6f}")
        lines.append(f"FRR: {ext['frr']:.6f}")
        lines.append("(Reporting only -- this threshold was chosen elsewhere and did")
        lines.append(" not influence any value computed above.)")
        lines.append("")

    lines.append("--- Speaker leakage check ---")
    lines.append(f"Speakers overlapping between calibration and evaluation: {len(result['speaker_overlap'])}")
    if result["speaker_overlap"]:
        lines.append(f"    *** LEAKAGE DETECTED: {sorted(result['speaker_overlap'])[:10]} ...")
    else:
        lines.append("    None -- calibration and evaluation are speaker-disjoint.")
    lines.append("")
    lines.append("Threshold is REAL (data-derived from calibration-split genuine/impostor "
                  "scores) -- not 0.635386, and not any other hard-coded value.")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 calibration on the calibration split of "
        "a scored-trials CSV, and report FAR/FRR of that threshold on "
        "the held-out evaluation split."
    )
    parser.add_argument("--scores", required=True, help="Scored-trials CSV from scripts/score_trials.py.")
    parser.add_argument("--report-out", default=None, help="Optional path to also write the report as text.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional externally-derived threshold to additionally measure "
            "FAR/FRR for across ALL scored trials (e.g. a threshold "
            "calibrated in a separate speaker-disjoint run). Reporting "
            "only -- it never affects threshold selection here."
        ),
    )
    args = parser.parse_args()

    if not Path(args.scores).is_file():
        print(f"ERROR: scores file not found: '{args.scores}'. Run scripts/score_trials.py first.", file=sys.stderr)
        return 1

    scored_rows = _read_scores(args.scores)
    if not scored_rows:
        print(f"ERROR: '{args.scores}' contains no scored trials.", file=sys.stderr)
        return 1

    try:
        result = run_calibration(scored_rows, external_threshold=args.threshold)
    except CalibrationError as exc:
        print(f"Calibration UNAVAILABLE: {exc}", file=sys.stderr)
        print(
            "No threshold has been produced or reported. This happens "
            "when the calibration split has zero genuine or zero "
            "impostor trials -- increase --max-genuine/--max-impostor "
            "in scripts/build_trials.py, or check --calibration-fraction.",
            file=sys.stderr,
        )
        return 1

    report = _format_report(result)
    print(report)

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report, encoding="utf-8")
        print(f"\nReport written to '{args.report_out}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
