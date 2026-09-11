"""
test_run_voxceleb_calibration.py
------------------------------------

Unit tests for scripts/run_voxceleb_calibration.py's run_calibration():
integrates the EXISTING Phase 6 calibration.calibrate() with a
calibration/evaluation split and a speaker-leakage check, all on
synthetic scored-trial rows (no real VoxCeleb data or model needed).
"""

from __future__ import annotations

import pytest

from run_voxceleb_calibration import run_calibration
from speaker_verification.calibration import InsufficientCalibrationDataError


def _row(trial_id, score, label, split, ref_speaker, test_speaker):
    return {
        "trial_id": trial_id,
        "reference": f"{ref_speaker}/v/1.wav",
        "test": f"{test_speaker}/v/2.wav",
        "label": label,
        "score": score,
        "split": split,
        "reference_speaker": ref_speaker,
        "test_speaker": test_speaker,
    }


def _well_separated_rows():
    rows = []
    # Calibration split: clearly separated genuine/impostor scores.
    for i in range(10):
        rows.append(_row(f"cg{i}", 0.8 + i * 0.01, 1, "calibration", f"cid{i}", f"cid{i}"))
    for i in range(10):
        rows.append(_row(f"ci{i}", 0.1 + i * 0.01, 0, "calibration", f"cidA{i}", f"cidB{i}"))
    # Evaluation split: disjoint speaker ids, also separated.
    for i in range(10):
        rows.append(_row(f"eg{i}", 0.75 + i * 0.01, 1, "evaluation", f"eid{i}", f"eid{i}"))
    for i in range(10):
        rows.append(_row(f"ei{i}", 0.15 + i * 0.01, 0, "evaluation", f"eidA{i}", f"eidB{i}"))
    return rows


# --------------------------------------------------------------------------
# Calibration integration (Phase 7K #9)
# --------------------------------------------------------------------------


def test_far_frr_at_eer_threshold_are_reported_and_near_the_eer() -> None:
    """At the EER operating point FAR and FRR should both sit close to
    the reported EER -- that is what makes it the *equal* error rate."""
    result = run_calibration(_well_separated_rows())
    cal = result["calibration_result"]

    far = result["calibration_far_at_eer_threshold"]
    frr = result["calibration_frr_at_eer_threshold"]

    assert 0.0 <= far <= 1.0
    assert 0.0 <= frr <= 1.0
    assert abs(far - cal.eer) <= 0.1
    assert abs(frr - cal.eer) <= 0.1


def test_external_threshold_is_absent_unless_requested() -> None:
    result = run_calibration(_well_separated_rows())
    assert result["external_threshold_evaluation"] is None


def test_external_threshold_measured_over_all_trials_without_changing_calibration() -> None:
    """An externally supplied threshold must be reported over every
    scored trial and must not perturb the calibration result."""
    rows = _well_separated_rows()
    baseline = run_calibration(rows)
    result = run_calibration(rows, external_threshold=0.5)

    ext = result["external_threshold_evaluation"]
    assert ext is not None
    assert ext["threshold"] == 0.5
    # All trials, both splits -- not just the calibration split.
    assert ext["num_genuine"] == len([r for r in rows if r["label"] == 1])
    assert ext["num_impostor"] == len([r for r in rows if r["label"] == 0])

    # Threshold selection is untouched by the reporting-only argument.
    assert result["calibration_result"].eer_threshold == baseline["calibration_result"].eer_threshold
    assert result["calibration_result"].eer == baseline["calibration_result"].eer


def test_single_split_protocol_keeps_every_trial_and_reports_no_evaluation() -> None:
    """Full-protocol mode: when every speaker is on the calibration
    side, no trial is dropped and there is no held-out split to report."""
    rows = _well_separated_rows()
    single = [dict(r, split="calibration") for r in rows]

    result = run_calibration(single)

    assert result["calibration_result"].num_genuine == len([r for r in rows if r["label"] == 1])
    assert result["calibration_result"].num_impostor == len([r for r in rows if r["label"] == 0])
    assert result["evaluation"] is None
    assert result["speaker_overlap"] == set()


def test_calibration_medians_describe_the_calibration_split_only() -> None:
    """The calibration block's median must come from the calibration
    split alone, so it describes the same population as the mean/std
    reported beside it -- not calibration + evaluation combined."""
    import statistics

    rows = _well_separated_rows()
    result = run_calibration(rows)

    cal_genuine = [r["score"] for r in rows if r["split"] == "calibration" and r["label"] == 1]
    cal_impostor = [r["score"] for r in rows if r["split"] == "calibration" and r["label"] == 0]
    all_genuine = [r["score"] for r in rows if r["label"] == 1]
    all_impostor = [r["score"] for r in rows if r["label"] == 0]

    assert result["calibration_genuine_median"] == statistics.median(cal_genuine)
    assert result["calibration_impostor_median"] == statistics.median(cal_impostor)

    # The fixture is built so the two populations genuinely differ; if
    # they ever coincide this test would silently stop checking anything.
    assert statistics.median(cal_genuine) != statistics.median(all_genuine)
    assert statistics.median(cal_impostor) != statistics.median(all_impostor)


def test_report_does_not_put_dataset_wide_median_in_calibration_block() -> None:
    from run_voxceleb_calibration import _format_report

    result = run_calibration(_well_separated_rows())
    report = _format_report(result)

    calibration_block = report.split("--- Evaluation set")[0]
    assert "dataset-wide" not in calibration_block


def test_run_calibration_produces_eer_and_threshold() -> None:
    result = run_calibration(_well_separated_rows())
    cal = result["calibration_result"]
    assert cal.num_genuine == 10
    assert cal.num_impostor == 10
    assert 0.0 <= cal.eer <= 1.0
    assert cal.eer_threshold is not None


def test_run_calibration_evaluates_on_held_out_split() -> None:
    result = run_calibration(_well_separated_rows())
    assert result["evaluation"] is not None
    assert result["evaluation"]["num_genuine"] == 10
    assert result["evaluation"]["num_impostor"] == 10
    assert 0.0 <= result["evaluation"]["far_at_calibration_threshold"] <= 1.0
    assert 0.0 <= result["evaluation"]["frr_at_calibration_threshold"] <= 1.0


def test_run_calibration_insufficient_calibration_data_raises() -> None:
    # Calibration split has genuine trials but zero impostor trials --
    # mirrors this project's real single-speaker situation.
    rows = [_row("g1", 0.6, 1, "calibration", "id1", "id1")]
    with pytest.raises(InsufficientCalibrationDataError):
        run_calibration(rows)


def test_run_calibration_empty_evaluation_split_reports_none() -> None:
    rows = []
    for i in range(5):
        rows.append(_row(f"cg{i}", 0.8, 1, "calibration", f"cid{i}", f"cid{i}"))
    for i in range(5):
        rows.append(_row(f"ci{i}", 0.2, 0, "calibration", f"cidA{i}", f"cidB{i}"))
    result = run_calibration(rows)
    assert result["evaluation"] is None


# --------------------------------------------------------------------------
# Speaker leakage detection (Phase 7K #11)
# --------------------------------------------------------------------------


def test_run_calibration_detects_speaker_leakage() -> None:
    rows = _well_separated_rows()
    # Deliberately reuse a calibration speaker in the evaluation split.
    rows.append(_row("leak1", 0.9, 1, "evaluation", "cid0", "cid0"))
    result = run_calibration(rows)
    assert "cid0" in result["speaker_overlap"]


def test_run_calibration_no_leakage_when_disjoint() -> None:
    result = run_calibration(_well_separated_rows())
    assert result["speaker_overlap"] == set()


# --------------------------------------------------------------------------
# Dataset-wide counts
# --------------------------------------------------------------------------


def test_run_calibration_reports_total_speakers_and_utterances() -> None:
    result = run_calibration(_well_separated_rows())
    # 10 genuine (1 speaker each) + 10 impostor (2 speakers each) per
    # split, 2 splits -> (10 + 20) * 2 = 60 unique speaker ids.
    assert result["num_speakers_total"] == 60
    assert result["num_utterances_total"] > 0
