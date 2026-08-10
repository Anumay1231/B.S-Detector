"""
test_verifier.py
-----------------

Unit tests for src/speaker_verification/verifier.py.

These tests exercise SpeakerVerifier's threshold/decision logic directly
(via verify_embeddings(), using hand-constructed tensors) and its
orchestration logic (via verify_files(), using a lightweight stand-in
encoder rather than the real pretrained model — no network access
required). They do NOT test the real ECAPA-TDNN model; that is covered
separately by tests/test_encoder.py and the real-data integration script
scripts/test_similarity.py / scripts/verify.py.
"""

from __future__ import annotations

import pytest
import torch

from speaker_verification.similarity import InvalidEmbeddingError
from speaker_verification.verifier import (
    InvalidThresholdError,
    SpeakerVerifier,
    VerificationOutcome,
    VerificationResult,
)


class _StubEncoder:
    """A minimal stand-in for SpeakerEncoder used only to test
    SpeakerVerifier's orchestration logic (verify_files) without
    downloading or running the real pretrained model. Maps fixed file
    paths to fixed, deterministic embeddings.
    """

    def __init__(self, embeddings_by_path: dict) -> None:
        self._embeddings_by_path = embeddings_by_path

    def encode_file(self, path: str) -> torch.Tensor:
        return self._embeddings_by_path[path]


# --------------------------------------------------------------------------
# Threshold validation (never a hard-coded magic default)
# --------------------------------------------------------------------------


def test_threshold_is_a_required_argument() -> None:
    import inspect

    sig = inspect.signature(SpeakerVerifier.__init__)
    assert sig.parameters["threshold"].default is inspect.Parameter.empty, (
        "SpeakerVerifier's threshold must have no default value — it "
        "must always be supplied explicitly by the caller."
    )


def test_none_threshold_raises() -> None:
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold=None)


def test_nan_threshold_raises() -> None:
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold=float("nan"))


def test_inf_threshold_raises() -> None:
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold=float("inf"))


def test_out_of_range_threshold_raises() -> None:
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold=1.5)
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold=-1.5)


def test_non_numeric_threshold_raises() -> None:
    with pytest.raises(InvalidThresholdError):
        SpeakerVerifier(encoder=_StubEncoder({}), threshold="high")  # type: ignore[arg-type]


def test_valid_threshold_at_extremes_accepted() -> None:
    SpeakerVerifier(encoder=_StubEncoder({}), threshold=-1.0)
    SpeakerVerifier(encoder=_StubEncoder({}), threshold=1.0)
    SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.0)


# --------------------------------------------------------------------------
# MATCH when score >= threshold
# --------------------------------------------------------------------------


def test_match_when_score_above_threshold() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.5)
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])  # identical -> score = 1.0
    result = verifier.verify_embeddings(a, b)
    assert result.outcome == VerificationOutcome.MATCH
    assert result.score == pytest.approx(1.0)


def test_match_when_score_exactly_equals_threshold() -> None:
    # Boundary condition: score == threshold -> MATCH (score >= threshold).
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=1.0)
    a = torch.tensor([0.3, -0.5, 0.8])
    b = a.clone()  # identical vectors -> score = 1.0 exactly
    result = verifier.verify_embeddings(a, b)
    assert result.score == pytest.approx(1.0, abs=1e-6)
    assert result.outcome == VerificationOutcome.MATCH


# --------------------------------------------------------------------------
# NON_MATCH when score < threshold
# --------------------------------------------------------------------------


def test_non_match_when_score_below_threshold() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.5)
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])  # orthogonal -> score = 0.0
    result = verifier.verify_embeddings(a, b)
    assert result.outcome == VerificationOutcome.NON_MATCH
    assert result.score == pytest.approx(0.0, abs=1e-6)


def test_non_match_just_below_threshold() -> None:
    # A score just below the threshold must NON_MATCH, not MATCH —
    # confirms strict "<" on the rejection side. a and b are unit
    # vectors with a known angle giving cosine similarity == 0.99
    # exactly; threshold is set just above that.
    import math

    a = torch.tensor([1.0, 0.0])
    angle = math.acos(0.99)
    b = torch.tensor([math.cos(angle), math.sin(angle)])

    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.995)
    result = verifier.verify_embeddings(a, b)
    assert result.score == pytest.approx(0.99, abs=1e-4)
    assert result.score < 0.995
    assert result.outcome == VerificationOutcome.NON_MATCH


# --------------------------------------------------------------------------
# Raw score is always exposed, regardless of outcome
# --------------------------------------------------------------------------


def test_raw_score_always_present_on_match() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.1)
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 0.0])
    result = verifier.verify_embeddings(a, b)
    assert isinstance(result.score, float)


def test_raw_score_always_present_on_non_match() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.9)
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    result = verifier.verify_embeddings(a, b)
    assert isinstance(result.score, float)
    assert result.score == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# is_calibrated flag: never silently implied
# --------------------------------------------------------------------------


def test_default_is_calibrated_is_false() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.5)
    assert verifier.is_calibrated is False
    a = torch.tensor([1.0, 0.0])
    result = verifier.verify_embeddings(a, a.clone())
    assert result.is_calibrated is False
    assert "UNCALIBRATED" in str(result)


def test_explicit_is_calibrated_true_propagates() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.5, is_calibrated=True)
    a = torch.tensor([1.0, 0.0])
    result = verifier.verify_embeddings(a, a.clone())
    assert result.is_calibrated is True
    assert "UNCALIBRATED" not in str(result)


# --------------------------------------------------------------------------
# Invalid embeddings delegate to similarity.py's validation
# --------------------------------------------------------------------------


def test_verify_embeddings_rejects_invalid_input() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.5)
    with pytest.raises(InvalidEmbeddingError):
        verifier.verify_embeddings(torch.zeros(4), torch.tensor([1.0, 0.0, 0.0, 0.0]))


# --------------------------------------------------------------------------
# verify_files(): orchestration (preprocess -> encode -> similarity ->
# decision), using a stub encoder rather than the real model
# --------------------------------------------------------------------------


def test_verify_files_orchestration_match() -> None:
    embedding = torch.tensor([0.6, 0.8, 0.0])
    stub = _StubEncoder(
        {
            "reference.wav": embedding,
            "test.wav": embedding.clone(),
        }
    )
    verifier = SpeakerVerifier(encoder=stub, threshold=0.5)
    result = verifier.verify_files("reference.wav", "test.wav")

    assert isinstance(result, VerificationResult)
    assert result.outcome == VerificationOutcome.MATCH
    assert result.score == pytest.approx(1.0, abs=1e-6)
    assert result.reference_path == "reference.wav"
    assert result.test_path == "test.wav"


def test_verify_files_orchestration_non_match() -> None:
    stub = _StubEncoder(
        {
            "reference.wav": torch.tensor([1.0, 0.0, 0.0]),
            "different.wav": torch.tensor([0.0, 1.0, 0.0]),
        }
    )
    verifier = SpeakerVerifier(encoder=stub, threshold=0.5)
    result = verifier.verify_files("reference.wav", "different.wav")

    assert result.outcome == VerificationOutcome.NON_MATCH
    assert result.score == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# __str__ formatting sanity
# --------------------------------------------------------------------------


def test_str_contains_score_and_threshold() -> None:
    verifier = SpeakerVerifier(encoder=_StubEncoder({}), threshold=0.42)
    a = torch.tensor([1.0, 0.0])
    result = verifier.verify_embeddings(a, a.clone())
    text = str(result)
    assert "MATCH" in text
    assert "0.42" in text
