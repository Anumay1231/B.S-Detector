"""
verifier.py
-----------

High-level orchestration: takes two embeddings (or two audio files),
computes cosine similarity via similarity.py, and applies a
CALLER-SUPPLIED threshold to produce a MATCH / NON_MATCH decision.

    Audio
      |
    audio.preprocess_audio()
      |
    ECAPA encoder (encoder.SpeakerEncoder)
      |
    192-D embedding
      |
    cosine similarity (similarity.cosine_similarity)
      |
    threshold (supplied by the caller — see below)
      |
    MATCH / NON_MATCH

Threshold direction: higher cosine similarity means more likely
same-speaker, so the decision rule is always:

    score >= threshold  ->  MATCH
    score <  threshold  ->  NON_MATCH

IMPORTANT — this module never invents a threshold:
    SpeakerVerifier requires an explicit `threshold` argument with no
    default value. There is no hard-coded "magic number" anywhere in
    this file. The threshold should normally come from
    calibration.calibrate(...).eer_threshold on real, representative
    genuine/impostor trial data (see calibration.py and
    docs/calibration.md). If you construct a SpeakerVerifier with a
    threshold that did NOT come from real calibration (e.g. for a quick
    manual test), you must pass `is_calibrated=False` (the default) so
    that fact is preserved and surfaced in every VerificationResult
    rather than silently presented as a trustworthy decision.

This module does not implement fuzzy logic, ML classification, or
deepfake detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from speaker_verification.encoder import SpeakerEncoder
from speaker_verification.similarity import cosine_similarity


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class VerifierError(Exception):
    """Base class for all errors raised by this module."""


class InvalidThresholdError(VerifierError):
    """Raised when a threshold is missing, non-finite, or outside the
    valid cosine-similarity range."""


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


class VerificationOutcome(str, Enum):
    """The two possible outcomes. Deliberately just two values — no
    UNCERTAIN/third state is introduced here; that would require its own
    documented criterion, which is out of scope for this phase.
    """

    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"


@dataclass(frozen=True)
class VerificationResult:
    """The full, transparent result of one verification decision.

    The raw score is ALWAYS included and is never hidden behind the
    outcome label — callers can always see exactly what score produced
    a given decision, and whether the threshold used was an actual
    calibrated value.

    Attributes:
        outcome: MATCH or NON_MATCH.
        score: The raw cosine similarity score (float, ~[-1.0, 1.0]).
        threshold: The threshold that was applied to produce `outcome`.
        is_calibrated: Whether `threshold` came from real calibration
            (calibration.calibrate() on labeled genuine/impostor trial
            data) as opposed to a manually supplied / example value. This
            is metadata supplied by the caller at SpeakerVerifier
            construction time — it is not independently verified here.
        reference_path: Optional source path, if verify_files() was used.
        test_path: Optional source path, if verify_files() was used.
    """

    outcome: VerificationOutcome
    score: float
    threshold: float
    is_calibrated: bool
    reference_path: Optional[str] = None
    test_path: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        calibration_note = (
            "calibrated threshold"
            if self.is_calibrated
            else "UNCALIBRATED threshold — not a validated decision boundary"
        )
        return (
            f"{self.outcome.value} (score={self.score:.6f}, "
            f"threshold={self.threshold:.6f}, {calibration_note})"
        )


# --------------------------------------------------------------------------
# Threshold validation
# --------------------------------------------------------------------------


def _validate_threshold(threshold: float) -> float:
    if threshold is None:
        raise InvalidThresholdError(
            "threshold must not be None. SpeakerVerifier requires an "
            "explicit threshold — there is no default/magic value. Use "
            "calibration.calibrate(...).eer_threshold on real "
            "genuine/impostor trial data, or pass an explicit value and "
            "set is_calibrated=False if it is not from real calibration."
        )
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InvalidThresholdError(
            f"threshold must be a real number, got {threshold!r}."
        ) from exc
    if not math.isfinite(value):
        raise InvalidThresholdError(f"threshold must be finite, got {value}.")
    if value < -1.0 - 1e-4 or value > 1.0 + 1e-4:
        raise InvalidThresholdError(
            f"threshold ({value}) is outside the valid cosine-similarity "
            f"range [-1.0, 1.0]. This usually indicates a mistake (e.g. "
            f"passing a percentage or an unrelated number)."
        )
    return value


# --------------------------------------------------------------------------
# SpeakerVerifier
# --------------------------------------------------------------------------


class SpeakerVerifier:
    """Applies a supplied decision threshold to cosine-similarity scores
    between pairs of speaker embeddings.

    Example:
        >>> from speaker_verification.calibration import calibrate  # doctest: +SKIP
        >>> result = calibrate(genuine_scores, impostor_scores)  # doctest: +SKIP
        >>> verifier = SpeakerVerifier(  # doctest: +SKIP
        ...     encoder=my_encoder,
        ...     threshold=result.eer_threshold,
        ...     is_calibrated=True,
        ... )
        >>> outcome = verifier.verify_files("reference.wav", "test.wav")  # doctest: +SKIP
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        threshold: float,
        is_calibrated: bool = False,
    ) -> None:
        """
        Args:
            encoder: An already-constructed SpeakerEncoder (see
                encoder.py). Not constructed here, so it can be shared
                across multiple SpeakerVerifier instances / calls without
                reloading the model.
            threshold: The decision threshold. REQUIRED — there is no
                default. score >= threshold -> MATCH.
            is_calibrated: Set True only if `threshold` came from real
                calibration (calibration.calibrate() on representative
                genuine/impostor trial data). Defaults to False so that
                ad-hoc/example thresholds are never silently presented as
                validated. This is recorded on every VerificationResult.

        Raises:
            InvalidThresholdError: If threshold is missing, non-finite,
                or outside [-1.0, 1.0].
        """
        self.encoder = encoder
        self.threshold = _validate_threshold(threshold)
        self.is_calibrated = is_calibrated

    def _decide(self, score: float) -> VerificationOutcome:
        return (
            VerificationOutcome.MATCH
            if score >= self.threshold
            else VerificationOutcome.NON_MATCH
        )

    def verify_embeddings(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> VerificationResult:
        """Compare two already-extracted embeddings.

        Raises:
            speaker_verification.similarity.SimilarityError: For any
                invalid embedding input (see similarity.py).
        """
        score = cosine_similarity(embedding_a, embedding_b)
        return VerificationResult(
            outcome=self._decide(score),
            score=score,
            threshold=self.threshold,
            is_calibrated=self.is_calibrated,
        )

    def verify_files(self, reference_path: str, test_path: str) -> VerificationResult:
        """Run the full pipeline on two audio files: preprocess -> encode
        -> cosine similarity -> threshold decision.

        Raises:
            speaker_verification.audio.AudioError: For any audio
                loading/preprocessing failure.
            speaker_verification.encoder.EncoderError: For any embedding
                extraction failure.
            speaker_verification.similarity.SimilarityError: For any
                invalid embedding comparison.
        """
        embedding_a = self.encoder.encode_file(reference_path)
        embedding_b = self.encoder.encode_file(test_path)
        score = cosine_similarity(embedding_a, embedding_b)
        return VerificationResult(
            outcome=self._decide(score),
            score=score,
            threshold=self.threshold,
            is_calibrated=self.is_calibrated,
            reference_path=reference_path,
            test_path=test_path,
        )
