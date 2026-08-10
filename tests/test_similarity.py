"""
test_similarity.py
-------------------

Deterministic unit tests for src/speaker_verification/similarity.py.

These tests use hand-constructed vectors (not real speaker embeddings) so
that the expected cosine similarity is known exactly ahead of time. They
verify the *mathematics and input validation* of cosine_similarity() in
isolation from the ECAPA-TDNN model. For a real-embeddings integration
test, see scripts/test_similarity.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from speaker_verification.similarity import (
    EmbeddingMismatchError,
    InvalidEmbeddingError,
    cosine_similarity,
)

# Floating-point tolerance for approximate equality checks.
ATOL = 1e-5


# --------------------------------------------------------------------------
# 1. Identical embeddings -> ~1.0
# --------------------------------------------------------------------------


def test_identical_embeddings_score_near_one() -> None:
    a = torch.tensor([0.3, -0.5, 0.8, 0.1])
    b = a.clone()
    score = cosine_similarity(a, b)
    assert score == pytest.approx(1.0, abs=ATOL)


# --------------------------------------------------------------------------
# 2. Same-direction, different magnitude -> ~1.0 (scale-insensitive)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.001, 0.5, 2.0, 100.0, 1e6])
def test_scaled_same_direction_score_near_one(scale: float) -> None:
    a = torch.tensor([0.3, -0.5, 0.8, 0.1])
    b = a * scale
    score = cosine_similarity(a, b)
    assert score == pytest.approx(1.0, abs=ATOL)


# --------------------------------------------------------------------------
# 3. Orthogonal embeddings -> ~0.0
# --------------------------------------------------------------------------


def test_orthogonal_embeddings_score_near_zero() -> None:
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    score = cosine_similarity(a, b)
    assert score == pytest.approx(0.0, abs=ATOL)


# --------------------------------------------------------------------------
# 4. Opposite-direction embeddings -> ~-1.0
# --------------------------------------------------------------------------


def test_opposite_embeddings_score_near_negative_one() -> None:
    a = torch.tensor([0.3, -0.5, 0.8, 0.1])
    b = -a
    score = cosine_similarity(a, b)
    assert score == pytest.approx(-1.0, abs=ATOL)


# --------------------------------------------------------------------------
# 5. Different magnitudes, known angle
# --------------------------------------------------------------------------


def test_known_angle_different_magnitudes() -> None:
    # a = (1, 0), b = (0, 5): 90 degrees apart regardless of b's magnitude.
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 5.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=ATOL)

    # a = (1, 1), b = (3, 3) (scaled, same direction) -> 1.0
    a = torch.tensor([1.0, 1.0])
    b = torch.tensor([3.0, 3.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=ATOL)

    # a = (1, 0), b = (-2, 0) (scaled, opposite direction) -> -1.0
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-2.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=ATOL)


def test_score_is_within_valid_range_for_random_vectors() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        a = torch.randn(192)
        b = torch.randn(192)
        score = cosine_similarity(a, b)
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6


# --------------------------------------------------------------------------
# 6. Mismatched dimensions
# --------------------------------------------------------------------------


def test_mismatched_embedding_dimensions_raises() -> None:
    a = torch.zeros(192)
    a[0] = 1.0
    b = torch.zeros(128)
    b[0] = 1.0
    with pytest.raises(EmbeddingMismatchError):
        cosine_similarity(a, b)


def test_mismatched_ndim_raises() -> None:
    a = torch.ones(4)
    b = torch.ones(2, 4)
    with pytest.raises(EmbeddingMismatchError):
        cosine_similarity(a, b)


def test_mismatched_batch_size_raises() -> None:
    a = torch.ones(2, 4)
    b = torch.ones(3, 4)
    with pytest.raises(EmbeddingMismatchError):
        cosine_similarity(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this environment.")
def test_mismatched_device_raises() -> None:
    a = torch.ones(4)  # cpu
    b = torch.ones(4).to("cuda")
    with pytest.raises(EmbeddingMismatchError):
        cosine_similarity(a, b)


# --------------------------------------------------------------------------
# 7. NaN input
# --------------------------------------------------------------------------


def test_nan_input_raises() -> None:
    a = torch.tensor([1.0, float("nan"), 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a, b)
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(b, a)


# --------------------------------------------------------------------------
# 8. Inf input
# --------------------------------------------------------------------------


def test_inf_input_raises() -> None:
    a = torch.tensor([1.0, float("inf"), 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a, b)

    a_neg_inf = torch.tensor([1.0, float("-inf"), 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a_neg_inf, b)


# --------------------------------------------------------------------------
# 9. Zero vector
# --------------------------------------------------------------------------


def test_zero_vector_raises() -> None:
    zero = torch.zeros(4)
    nonzero = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(zero, nonzero)
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(nonzero, zero)
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(zero, zero)


def test_near_zero_vector_raises() -> None:
    # Effectively zero due to floating-point underflow, not exactly 0.0.
    near_zero = torch.full((4,), 1e-10)
    nonzero = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(near_zero, nonzero)


# --------------------------------------------------------------------------
# 10. Empty tensor
# --------------------------------------------------------------------------


def test_empty_tensor_raises() -> None:
    empty = torch.empty(0)
    nonzero = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(empty, nonzero)


# --------------------------------------------------------------------------
# Additional input-validation coverage
# --------------------------------------------------------------------------


def test_none_embedding_raises() -> None:
    a = torch.tensor([1.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(None, a)  # type: ignore[arg-type]
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a, None)  # type: ignore[arg-type]


def test_non_tensor_input_raises() -> None:
    a = torch.tensor([1.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity([1.0, 0.0], a)  # type: ignore[arg-type]


def test_non_floating_dtype_raises() -> None:
    a = torch.tensor([1, 0, 0], dtype=torch.int64)
    b = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a, b)


def test_wrong_ndim_raises() -> None:
    a = torch.ones(1, 1, 4)
    b = torch.ones(4)
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(a, b)


def test_inputs_are_not_mutated() -> None:
    a = torch.tensor([0.3, -0.5, 0.8, 0.1])
    b = torch.tensor([0.1, 0.2, -0.3, 0.9])
    a_before = a.clone()
    b_before = b.clone()
    cosine_similarity(a, b)
    assert torch.equal(a, a_before)
    assert torch.equal(b, b_before)


# --------------------------------------------------------------------------
# 11. CPU tensors
# --------------------------------------------------------------------------


def test_cpu_tensors() -> None:
    a = torch.tensor([1.0, 0.0, 0.0], device="cpu")
    b = torch.tensor([1.0, 0.0, 0.0], device="cpu")
    score = cosine_similarity(a, b)
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=ATOL)


# --------------------------------------------------------------------------
# 12. CUDA tensors, if available
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this environment.")
def test_cuda_tensors() -> None:
    a = torch.tensor([0.3, -0.5, 0.8, 0.1], device="cuda")
    b = a.clone()
    score = cosine_similarity(a, b)
    assert isinstance(score, float)
    assert score == pytest.approx(1.0, abs=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this environment.")
def test_cuda_result_matches_cpu_result() -> None:
    torch.manual_seed(1)
    a_cpu = torch.randn(192)
    b_cpu = torch.randn(192)
    cpu_score = cosine_similarity(a_cpu, b_cpu)

    a_cuda = a_cpu.to("cuda")
    b_cuda = b_cpu.to("cuda")
    cuda_score = cosine_similarity(a_cuda, b_cuda)

    assert cuda_score == pytest.approx(cpu_score, abs=1e-4)


# --------------------------------------------------------------------------
# Batched input (explicitly invited by the spec: "handle reasonable batch
# dimensions if useful")
# --------------------------------------------------------------------------


def test_batched_input_returns_per_pair_scores() -> None:
    a = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    b = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0]),  # identical -> 1.0
            torch.tensor([0.0, 1.0, 0.0]),  # orthogonal -> 0.0
            torch.tensor([-1.0, 0.0, 0.0]),  # opposite -> -1.0
        ]
    )
    scores = cosine_similarity(a, b)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (3,)
    assert scores[0].item() == pytest.approx(1.0, abs=ATOL)
    assert scores[1].item() == pytest.approx(0.0, abs=ATOL)
    assert scores[2].item() == pytest.approx(-1.0, abs=ATOL)
