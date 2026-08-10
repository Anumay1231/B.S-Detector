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
