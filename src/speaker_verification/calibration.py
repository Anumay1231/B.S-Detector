"""
calibration.py
---------------

Threshold calibration and evaluation for the ECAPA-TDNN cosine-similarity
baseline, from LABELED trial data.

Scope (unchanged from the Phase 1 scoping correction — see CLAUDE.md /
docs/architecture.md):
    - genuine/impostor trial score handling
    - score distribution statistics
    - False Acceptance Rate (FAR) and False Rejection Rate (FRR) as a
      function of threshold
    - an Equal Error Rate (EER) operating point
    - ROC curve points (and ROC-AUC) as a byproduct of the FAR/FRR curve
    - threshold selection, with the selection criterion explicitly
      documented (see select_threshold())

Fuzzy logic, MATCH/NON_MATCH decisions, and ML classification are
explicitly OUT OF SCOPE here. This module only produces numbers
(statistics, curves, a candidate threshold) from labeled score data — the
decision itself is made by verifier.py, which is handed a threshold as
configuration rather than deriving one itself.

Threshold direction (see verifier.py): higher cosine similarity means
more likely same speaker, so a candidate/selected threshold ``t`` is
always meant to be used as ``score >= t -> MATCH``.

IMPORTANT — calibration requires real, labeled data:
    A meaningful threshold cannot be produced from zero genuine trials or
    zero impostor trials — calibrate() raises
    InsufficientCalibrationDataError in that case rather than returning a
    fabricated or default number. Even with a nonzero-but-small number of
    trials, the result is numerically valid but NOT statistically
    reliable; calibrate() flags this explicitly via
    CalibrationResult.is_statistically_reliable and
    CalibrationResult.warnings rather than silently presenting a
    tiny-sample result as trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Union

import torch

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Below this many trials in EITHER class, results are numerically valid
#: but flagged as not statistically reliable. This is a commonly cited
#: rule-of-thumb minimum for a remotely stable EER estimate, not a
#: rigorous statistical guarantee, and is documented as such.
MIN_RELIABLE_TRIALS_PER_CLASS: int = 30

#: Small offset used to place the two outermost threshold candidates
#: strictly below the minimum and strictly above the maximum observed
#: score, so the FAR/FRR curve's endpoints (FAR=1,FRR=0 and FAR=0,FRR=1)
#: are always represented among the candidates.
_BOUNDARY_EPSILON: float = 1e-6

ScoreInput = Union[Sequence[float], torch.Tensor]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class CalibrationError(Exception):
    """Base class for all errors raised by this module."""


class InsufficientCalibrationDataError(CalibrationError):
    """Raised when calibration is attempted with zero genuine trials
    and/or zero impostor trials. A threshold cannot be meaningfully
    derived from one-sided or empty data — this is raised rather than
    silently returning a default/fabricated threshold.
    """


class InvalidScoreError(CalibrationError):
    """Raised when a score is not a finite real number (NaN/Inf), or is
    outside the mathematically valid cosine-similarity range."""


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreStatistics:
    """Basic descriptive statistics for one class of trial scores."""

    count: int
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class ThresholdOperatingPoint:
    """FAR/FRR at one candidate threshold."""

    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class CalibrationResult:
    """Full output of calibrate(): everything needed to select and
    justify a decision threshold from labeled trial data.

    Attributes:
        num_genuine: Number of genuine (same-speaker) trial scores.
        num_impostor: Number of impostor (different-speaker) trial scores.
        genuine_stats: Descriptive statistics of the genuine scores.
        impostor_stats: Descriptive statistics of the impostor scores.
        operating_points: FAR/FRR evaluated at every candidate threshold,
            sorted by threshold ascending. This is also the ROC curve
            data (FAR is the false-positive rate; 1 - FRR is the
            true-positive rate).
        roc_auc: Area under the ROC curve (FAR on x-axis, 1-FRR on
            y-axis), via the trapezoidal rule over `operating_points`.
            1.0 = perfect separation, 0.5 = no better than chance.
        eer: Equal Error Rate — the FAR (~= FRR) at the point where the
            FAR and FRR curves cross (see select_threshold() /
            compute_eer() for the interpolation method).
        eer_threshold: The threshold at the EER operating point.
        eer_method: Human-readable description of how eer/eer_threshold
            were derived (documents the interpolation method used).
        is_statistically_reliable: False if either class has fewer than
            MIN_RELIABLE_TRIALS_PER_CLASS trials. This does NOT mean the
            numbers are wrong — the arithmetic is exact for the data
            given — it means the sample is too small to trust as a
            general-purpose threshold.
        warnings: Human-readable caveats about this specific calibration
            run (e.g. small sample size). Always check this before
            treating a CalibrationResult as production-ready.
    """

    num_genuine: int
    num_impostor: int
    genuine_stats: ScoreStatistics
    impostor_stats: ScoreStatistics
    operating_points: Tuple[ThresholdOperatingPoint, ...]
    roc_auc: float
    eer: float
    eer_threshold: float
    eer_method: str
    is_statistically_reliable: bool
    warnings: Tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Validation / conversion
# --------------------------------------------------------------------------


def _to_float_list(scores: ScoreInput, name: str) -> List[float]:
    if scores is None:
        raise InvalidScoreError(f"{name} must not be None.")

    if isinstance(scores, torch.Tensor):
        if scores.ndim != 1:
            raise InvalidScoreError(
                f"{name} must be a 1-D sequence of scores, got a tensor "
                f"of shape {tuple(scores.shape)}."
            )
        values = scores.detach().to("cpu").tolist()
    else:
        try:
            values = [float(s) for s in scores]
        except (TypeError, ValueError) as exc:
            raise InvalidScoreError(
                f"{name} must be a sequence of numbers or a 1-D "
                f"torch.Tensor, got {type(scores)!r} containing at least "
                f"one non-numeric element: {exc}"
            ) from exc

    for value in values:
        if not math.isfinite(value):
            raise InvalidScoreError(
                f"{name} contains a non-finite value ({value}); all "
                f"scores must be finite real numbers."
            )
        if value < -1.0 - 1e-4 or value > 1.0 + 1e-4:
            raise InvalidScoreError(
                f"{name} contains a value ({value}) outside the "
                f"mathematically valid cosine-similarity range "
                f"[-1.0, 1.0]. This usually indicates the input is not "
                f"actually a cosine similarity score."
            )

    return values


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def compute_score_statistics(scores: ScoreInput, name: str = "scores") -> ScoreStatistics:
    """Compute basic descriptive statistics for a set of trial scores.

    Raises:
        InvalidScoreError: If scores is empty, contains non-finite
            values, or contains values outside [-1, 1].
    """
    values = _to_float_list(scores, name)
    if not values:
        raise InvalidScoreError(f"{name} is empty; cannot compute statistics.")

    tensor = torch.tensor(values, dtype=torch.float64)
    # unbiased=False (population std) is well-defined even for count == 1,
    # unlike the default unbiased estimator which divides by (n - 1).
    std = tensor.std(unbiased=False).item() if len(values) > 1 else 0.0

    return ScoreStatistics(
        count=len(values),
        mean=tensor.mean().item(),
        std=std,
        min=tensor.min().item(),
        max=tensor.max().item(),
    )


# --------------------------------------------------------------------------
# FAR / FRR
# --------------------------------------------------------------------------


def compute_far(impostor_scores: ScoreInput, threshold: float) -> float:
    """False Acceptance Rate at `threshold`: the fraction of impostor
    (different-speaker) trials that would be incorrectly accepted as a
    MATCH.

    Decision rule (see verifier.py): score >= threshold -> MATCH.
    So FAR(t) = P(impostor_score >= t).
    """
    values = _to_float_list(impostor_scores, "impostor_scores")
    if not values:
        raise InvalidScoreError("impostor_scores is empty; cannot compute FAR.")
    accepted = sum(1 for v in values if v >= threshold)
    return accepted / len(values)


def compute_frr(genuine_scores: ScoreInput, threshold: float) -> float:
    """False Rejection Rate at `threshold`: the fraction of genuine
    (same-speaker) trials that would be incorrectly rejected as a
    NON_MATCH.

    Decision rule (see verifier.py): score >= threshold -> MATCH.
    So FRR(t) = P(genuine_score < t).
    """
    values = _to_float_list(genuine_scores, "genuine_scores")
    if not values:
        raise InvalidScoreError("genuine_scores is empty; cannot compute FRR.")
    rejected = sum(1 for v in values if v < threshold)
    return rejected / len(values)


def _threshold_candidates(genuine: List[float], impostor: List[float]) -> List[float]:
    """Candidate thresholds to evaluate: the midpoints between every pair
    of consecutive unique observed scores (genuine + impostor combined),
    plus one sentinel just below the minimum and one just above the
    maximum so the curve's endpoints (FAR=1,FRR=0 and FAR=0,FRR=1) are
    always included.

    Using midpoints (rather than the observed scores themselves) avoids
    ambiguity from ties and gives a clean, strictly-ordered step function
    for FAR(t) and FRR(t).
    """
    unique_sorted = sorted(set(genuine) | set(impostor))
    candidates = [unique_sorted[0] - _BOUNDARY_EPSILON]
    candidates.extend(
        (unique_sorted[i] + unique_sorted[i + 1]) / 2.0
        for i in range(len(unique_sorted) - 1)
    )
    candidates.append(unique_sorted[-1] + _BOUNDARY_EPSILON)
    return candidates


def compute_far_frr_curve(
    genuine_scores: ScoreInput, impostor_scores: ScoreInput
) -> Tuple[ThresholdOperatingPoint, ...]:
    """FAR and FRR evaluated at every candidate threshold (see
    _threshold_candidates), sorted by threshold ascending.

    Raises:
        InsufficientCalibrationDataError: If either input is empty.
    """
    genuine = _to_float_list(genuine_scores, "genuine_scores")
    impostor = _to_float_list(impostor_scores, "impostor_scores")

    if not genuine or not impostor:
        raise InsufficientCalibrationDataError(
            f"Cannot compute a FAR/FRR curve with {len(genuine)} genuine "
            f"and {len(impostor)} impostor trial(s); both must be "
            f"non-empty. Calibration requires labeled genuine AND "
            f"impostor trials."
        )

    points = []
    for t in _threshold_candidates(genuine, impostor):
        far = compute_far(impostor, t)
        frr = compute_frr(genuine, t)
        points.append(ThresholdOperatingPoint(threshold=t, far=far, frr=frr))
    return tuple(points)


# --------------------------------------------------------------------------
# EER
# --------------------------------------------------------------------------


def compute_eer(
    genuine_scores: ScoreInput, impostor_scores: ScoreInput
) -> Tuple[float, float, str]:
    """Compute the Equal Error Rate operating point.

    Method: evaluate FAR(t) and FRR(t) at every candidate threshold
    (see _threshold_candidates — midpoints between consecutive unique
    observed scores). As the threshold increases, FAR is non-increasing
    and FRR is non-decreasing, so (FAR - FRR) is non-increasing. We scan
    for the first point where (FAR - FRR) crosses from >= 0 to <= 0:

        - If some candidate threshold has FAR == FRR exactly, that is
          the EER point directly.
        - Otherwise, we linearly interpolate between the last candidate
          where FAR >= FRR and the next candidate where FAR < FRR, to
          estimate the threshold where the two curves would cross, and
          report the interpolated FAR/FRR value there as the EER. This
          is a standard approach for EER estimation from a finite set of
          scored trials (the true FAR/FRR curves are step functions;
          interpolation estimates the crossing point between steps).
        - If FAR and FRR never cross (e.g. perfect separation with a
          flat gap where both are 0), the candidate with the smallest
          |FAR - FRR| is used, preferring the smaller EER value if tied.

    Returns:
        (eer, eer_threshold, method_description)

    Raises:
        InsufficientCalibrationDataError: If either input is empty.
    """
    curve = compute_far_frr_curve(genuine_scores, impostor_scores)

    diffs = [p.far - p.frr for p in curve]

    # Exact match on some candidate.
    for point, diff in zip(curve, diffs):
        if diff == 0.0:
            return (
                point.far,
                point.threshold,
                "Exact match: FAR == FRR at an evaluated candidate "
                "threshold (midpoint between consecutive unique observed "
                "scores).",
            )

    # Look for a sign change of (FAR - FRR) between consecutive candidates.
    for i in range(len(curve) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 > 0.0 and d1 < 0.0:
            p0, p1 = curve[i], curve[i + 1]
            # Linear interpolation of the threshold where (FAR - FRR) == 0.
            span = d0 - d1
            fraction = d0 / span if span != 0 else 0.5
            eer_threshold = p0.threshold + fraction * (p1.threshold - p0.threshold)
            eer_far = p0.far + fraction * (p1.far - p0.far)
            eer_frr = p0.frr + fraction * (p1.frr - p0.frr)
            eer = (eer_far + eer_frr) / 2.0
            return (
                eer,
                eer_threshold,
                "Linear interpolation between the two candidate "
                "thresholds bracketing the point where (FAR - FRR) "
                "changes sign.",
            )

    # No sign change found (e.g. perfect separation with a flat gap, or a
    # degenerate case) — fall back to the candidate minimizing |FAR - FRR|.
    best_index = min(range(len(curve)), key=lambda i: abs(diffs[i]))
    best_point = curve[best_index]
    eer = (best_point.far + best_point.frr) / 2.0
    return (
        eer,
        best_point.threshold,
        "No sign change in (FAR - FRR) across candidates (e.g. perfect "
        "separation); used the candidate threshold minimizing "
        "|FAR - FRR|.",
    )


def compute_roc_auc(operating_points: Sequence[ThresholdOperatingPoint]) -> float:
    """Area under the ROC curve (FAR on the x-axis, 1 - FRR / true
    positive rate on the y-axis), via the trapezoidal rule.

    1.0 = perfect separation between genuine and impostor scores.
    0.5 = no better than chance.

    IMPORTANT: `operating_points` must be in threshold-ascending order,
    exactly as returned by compute_far_frr_curve() / CalibrationResult.
    Since FAR(t) is provably non-increasing and FRR(t) non-decreasing as
    the threshold t increases, simply reversing that order yields FAR
    ascending with a correctly paired (monotonic) TPR at every point —
    including at tied FAR values, where the specific pairing matters for
    the trapezoidal area. (Re-sorting by FAR alone, ignoring this
    ordering, silently breaks the correct pairing at tied FAR values and
    understates the AUC — do not do that.)
    """
    points = list(reversed(operating_points))
    auc = 0.0
    for i in range(len(points) - 1):
        x0, x1 = points[i].far, points[i + 1].far
        y0, y1 = 1.0 - points[i].frr, 1.0 - points[i + 1].frr
        auc += abs(x1 - x0) * (y0 + y1) / 2.0
    return auc


# --------------------------------------------------------------------------
# Top-level calibration entry point
# --------------------------------------------------------------------------


def calibrate(
    genuine_scores: ScoreInput, impostor_scores: ScoreInput
) -> CalibrationResult:
    """Run full calibration: statistics, FAR/FRR curve, ROC-AUC, and the
    EER operating point, from labeled genuine and impostor trial scores.

    This is the criterion used for threshold selection in this module:
    **the Equal Error Rate (EER) threshold** — the point where a
    false-acceptance error and a false-rejection error are equally
    likely. This is a standard, well-documented default operating point
    for biometric/speaker verification systems; other criteria (e.g.
    minimizing FAR at a fixed FRR for a security-sensitive application)
    could be selected from `result.operating_points` instead, but EER is
    what `CalibrationResult.eer_threshold` reports.

    Args:
        genuine_scores: Cosine similarity scores from same-speaker
            (genuine) trials.
        impostor_scores: Cosine similarity scores from different-speaker
            (impostor) trials.

    Returns:
        A CalibrationResult. Check `result.is_statistically_reliable` and
        `result.warnings` before treating the threshold as trustworthy.

    Raises:
        InsufficientCalibrationDataError: If either genuine_scores or
            impostor_scores is empty. Calibration is explicitly reported
            as unavailable in this case rather than falling back to a
            default/fabricated threshold.
        InvalidScoreError: If any score is non-finite or outside the
            valid cosine-similarity range.
    """
    genuine = _to_float_list(genuine_scores, "genuine_scores")
    impostor = _to_float_list(impostor_scores, "impostor_scores")

    if not genuine or not impostor:
        raise InsufficientCalibrationDataError(
            f"Calibration requires at least one genuine AND at least one "
            f"impostor trial. Got {len(genuine)} genuine and "
            f"{len(impostor)} impostor trial(s). Calibration is "
            f"UNAVAILABLE — no threshold has been produced. Provide "
            f"labeled trial data for both classes (see "
            f"scripts/calibrate.py and data/trials.csv)."
        )

    genuine_stats = compute_score_statistics(genuine, "genuine_scores")
    impostor_stats = compute_score_statistics(impostor, "impostor_scores")
    operating_points = compute_far_frr_curve(genuine, impostor)
    roc_auc = compute_roc_auc(operating_points)
    eer, eer_threshold, eer_method = compute_eer(genuine, impostor)

    warnings: List[str] = []
    is_reliable = True
    if genuine_stats.count < MIN_RELIABLE_TRIALS_PER_CLASS:
        is_reliable = False
        warnings.append(
            f"Only {genuine_stats.count} genuine trial(s) — fewer than "
            f"the recommended minimum of {MIN_RELIABLE_TRIALS_PER_CLASS}. "
            f"The computed threshold/EER are numerically correct for "
            f"this data but are NOT statistically reliable and should "
            f"not be treated as a production threshold."
        )
    if impostor_stats.count < MIN_RELIABLE_TRIALS_PER_CLASS:
        is_reliable = False
        warnings.append(
            f"Only {impostor_stats.count} impostor trial(s) — fewer "
            f"than the recommended minimum of "
            f"{MIN_RELIABLE_TRIALS_PER_CLASS}. The computed "
            f"threshold/EER are numerically correct for this data but "
            f"are NOT statistically reliable and should not be treated "
            f"as a production threshold."
        )

    return CalibrationResult(
        num_genuine=genuine_stats.count,
        num_impostor=impostor_stats.count,
        genuine_stats=genuine_stats,
        impostor_stats=impostor_stats,
        operating_points=operating_points,
        roc_auc=roc_auc,
        eer=eer,
        eer_threshold=eer_threshold,
        eer_method=eer_method,
        is_statistically_reliable=is_reliable,
        warnings=tuple(warnings),
    )
