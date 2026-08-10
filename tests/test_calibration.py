"""
test_calibration.py
--------------------

Deterministic unit tests for src/speaker_verification/calibration.py.

IMPORTANT: all scores used in this file are SYNTHETIC — hand-constructed
numbers chosen to exercise specific mathematical behavior (perfect
separation, full overlap, boundary conditions, etc.). They are NOT real
speaker-verification results and must never be read as evidence about
this project's real-world accuracy. For the one real (uncalibrated)
score obtained from actual ECAPA embeddings, see docs/calibration.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from speaker_verification.calibration import (
    CalibrationResult,
    InsufficientCalibrationDataError,
    InvalidScoreError,
    MIN_RELIABLE_TRIALS_PER_CLASS,
    calibrate,
    compute_eer,
    compute_far,
    compute_frr,
    compute_score_statistics,
)


# --------------------------------------------------------------------------
# Perfect genuine/impostor separation (synthetic)
# --------------------------------------------------------------------------


def test_perfect_separation_gives_zero_eer_and_full_auc() -> None:
    # Synthetic: all genuine scores strictly above all impostor scores.
    genuine = [0.80, 0.85, 0.90, 0.95]
    impostor = [0.10, 0.15, 0.20, 0.30]

    result = calibrate(genuine, impostor)

    assert result.eer == pytest.approx(0.0, abs=1e-9)
    assert result.roc_auc == pytest.approx(1.0, abs=1e-9)
    # The EER threshold must fall strictly between the two clusters.
    assert max(impostor) < result.eer_threshold < min(genuine)


# --------------------------------------------------------------------------
# Fully overlapping / identical distributions (synthetic, worst case)
# --------------------------------------------------------------------------


def test_identical_distributions_give_eer_near_half() -> None:
    # Synthetic: genuine and impostor scores are drawn from the exact
    # same set of values -> best possible threshold is no better than
    # chance (EER ~= 0.5, AUC ~= 0.5).
    genuine = [0.1, 0.3, 0.5, 0.7, 0.9]
    impostor = [0.1, 0.3, 0.5, 0.7, 0.9]

    result = calibrate(genuine, impostor)

    assert result.eer == pytest.approx(0.5, abs=1e-6)
    assert result.roc_auc == pytest.approx(0.5, abs=1e-6)


def test_all_scores_identical_value() -> None:
    # Synthetic degenerate case: every score (genuine and impostor) is
    # exactly the same single value.
    genuine = [0.5] * 10
    impostor = [0.5] * 10

    result = calibrate(genuine, impostor)

    assert result.eer == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------
# Overlapping-but-separable distributions (synthetic)
# --------------------------------------------------------------------------


def test_overlapping_distributions_give_intermediate_eer() -> None:
    # Synthetic: mostly separated, with one impostor score (0.60) inside
    # the genuine cluster's range, hand-picked so FAR == FRR == 0.25
    # exactly at threshold 0.575 (see calculation in the module
    # docstring's EER method) - not a coincidence, deliberately chosen
    # to make this test's expected value exact rather than approximate.
    genuine = [0.55, 0.65, 0.75, 0.85]
    impostor = [0.25, 0.35, 0.45, 0.60]

    result = calibrate(genuine, impostor)

    assert result.eer == pytest.approx(0.25, abs=1e-9)
    assert result.eer_threshold == pytest.approx(0.575, abs=1e-9)
    assert 0.5 < result.roc_auc < 1.0


# --------------------------------------------------------------------------
# FAR / FRR at explicit thresholds (synthetic, hand-checked)
# --------------------------------------------------------------------------


def test_far_frr_at_explicit_threshold() -> None:
    genuine = [0.9, 0.8, 0.3]  # one genuine trial below the threshold
    impostor = [0.2, 0.1, 0.85]  # one impostor trial above the threshold
    threshold = 0.5

    far = compute_far(impostor, threshold)
    frr = compute_frr(genuine, threshold)

    # FAR: impostor scores >= 0.5 -> just 0.85 -> 1/3
    assert far == pytest.approx(1 / 3, abs=1e-9)
    # FRR: genuine scores < 0.5 -> just 0.3 -> 1/3
    assert frr == pytest.approx(1 / 3, abs=1e-9)


def test_far_is_one_when_threshold_below_all_impostor_scores() -> None:
    impostor = [0.2, 0.3, 0.4]
    assert compute_far(impostor, -1.0) == pytest.approx(1.0)


def test_far_is_zero_when_threshold_above_all_impostor_scores() -> None:
    impostor = [0.2, 0.3, 0.4]
    assert compute_far(impostor, 1.0) == pytest.approx(0.0)


def test_frr_is_zero_when_threshold_below_all_genuine_scores() -> None:
    genuine = [0.6, 0.7, 0.8]
    assert compute_frr(genuine, -1.0) == pytest.approx(0.0)


def test_frr_is_one_when_threshold_above_all_genuine_scores() -> None:
    genuine = [0.6, 0.7, 0.8]
    assert compute_frr(genuine, 1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Score statistics
# --------------------------------------------------------------------------


def test_score_statistics_basic() -> None:
    stats = compute_score_statistics([0.2, 0.4, 0.6])
    assert stats.count == 3
    assert stats.mean == pytest.approx(0.4)
    assert stats.min == pytest.approx(0.2)
    assert stats.max == pytest.approx(0.6)


def test_score_statistics_single_value_has_zero_std() -> None:
    stats = compute_score_statistics([0.5])
    assert stats.count == 1
    assert stats.std == pytest.approx(0.0)


def test_score_statistics_accepts_tensor() -> None:
    stats = compute_score_statistics(torch.tensor([0.1, 0.2, 0.3]))
    assert stats.count == 3


# --------------------------------------------------------------------------
# Empty / one-sided calibration data -> explicit "unavailable" error
# --------------------------------------------------------------------------


def test_empty_genuine_and_impostor_raises() -> None:
    with pytest.raises(InsufficientCalibrationDataError):
        calibrate([], [])


def test_only_genuine_scores_raises() -> None:
    # Mirrors this project's real current situation: one genuine trial,
    # zero impostor trials -> calibration must be explicitly unavailable.
    with pytest.raises(InsufficientCalibrationDataError):
        calibrate([0.635386], [])


def test_only_impostor_scores_raises() -> None:
    with pytest.raises(InsufficientCalibrationDataError):
        calibrate([], [0.2, 0.3])


# --------------------------------------------------------------------------
# Invalid scores
# --------------------------------------------------------------------------


def test_nan_score_raises() -> None:
    with pytest.raises(InvalidScoreError):
        calibrate([0.5, float("nan")], [0.2, 0.3])


def test_inf_score_raises() -> None:
    with pytest.raises(InvalidScoreError):
        calibrate([0.5, float("inf")], [0.2, 0.3])
    with pytest.raises(InvalidScoreError):
        calibrate([0.5, 0.6], [0.2, float("-inf")])


def test_out_of_range_score_raises() -> None:
    with pytest.raises(InvalidScoreError):
        calibrate([0.5, 1.5], [0.2, 0.3])  # 1.5 is not a valid cosine similarity


def test_non_numeric_score_raises() -> None:
    with pytest.raises(InvalidScoreError):
        calibrate([0.5, "not a number"], [0.2, 0.3])  # type: ignore[list-item]


# --------------------------------------------------------------------------
# Statistical reliability flag / warnings (small-sample honesty)
# --------------------------------------------------------------------------


def test_small_sample_is_flagged_unreliable() -> None:
    result = calibrate([0.8, 0.9], [0.1, 0.2])
    assert result.num_genuine < MIN_RELIABLE_TRIALS_PER_CLASS
    assert result.is_statistically_reliable is False
    assert len(result.warnings) > 0


def test_large_sample_is_flagged_reliable() -> None:
    # Synthetic: enough trials in each class to clear the documented
    # minimum. Still not "real" data, just enough of it.
    genuine = [0.8 + 0.001 * i for i in range(MIN_RELIABLE_TRIALS_PER_CLASS)]
    impostor = [0.1 + 0.001 * i for i in range(MIN_RELIABLE_TRIALS_PER_CLASS)]
    result = calibrate(genuine, impostor)
    assert result.is_statistically_reliable is True
    assert result.warnings == ()


# --------------------------------------------------------------------------
# compute_eer() return shape / method documentation
# --------------------------------------------------------------------------


def test_compute_eer_returns_method_description() -> None:
    eer, threshold, method = compute_eer([0.8, 0.9], [0.1, 0.2])
    assert isinstance(eer, float)
    assert isinstance(threshold, float)
    assert isinstance(method, str) and len(method) > 0


def test_compute_eer_requires_both_classes() -> None:
    with pytest.raises(InsufficientCalibrationDataError):
        compute_eer([0.5], [])


# --------------------------------------------------------------------------
# CalibrationResult structural sanity
# --------------------------------------------------------------------------


def test_calibration_result_operating_points_cover_full_range() -> None:
    result = calibrate([0.6, 0.7, 0.8], [0.2, 0.3, 0.4])
    assert isinstance(result, CalibrationResult)
    fars = [p.far for p in result.operating_points]
    frrs = [p.frr for p in result.operating_points]
    # The FAR/FRR curve must span from (FAR=1, FRR=0) to (FAR=0, FRR=1).
    assert fars[0] == pytest.approx(1.0)
    assert frrs[0] == pytest.approx(0.0)
    assert fars[-1] == pytest.approx(0.0)
    assert frrs[-1] == pytest.approx(1.0)


def test_calibration_result_thresholds_are_sorted_ascending() -> None:
    result = calibrate([0.6, 0.7, 0.8], [0.2, 0.3, 0.4])
    thresholds = [p.threshold for p in result.operating_points]
    assert thresholds == sorted(thresholds)


def test_far_is_non_increasing_with_threshold() -> None:
    result = calibrate([0.4, 0.6, 0.8], [0.1, 0.3, 0.5])
    fars = [p.far for p in result.operating_points]
    assert all(fars[i] >= fars[i + 1] - 1e-12 for i in range(len(fars) - 1))


def test_frr_is_non_decreasing_with_threshold() -> None:
    result = calibrate([0.4, 0.6, 0.8], [0.1, 0.3, 0.5])
    frrs = [p.frr for p in result.operating_points]
    assert all(frrs[i] <= frrs[i + 1] + 1e-12 for i in range(len(frrs) - 1))
