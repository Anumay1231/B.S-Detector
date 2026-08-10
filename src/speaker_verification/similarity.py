"""
similarity.py
--------------

Cosine similarity between two speaker embeddings.

This module answers exactly one question: "how similar are these two
speaker embeddings?" — it returns a raw similarity score and nothing
else.

    embedding_a
    embedding_b
           |
    cosine similarity
           |
    single scalar score

Cosine similarity measures the cosine of the angle between two vectors:

    similarity = dot(A, B) / (||A|| * ||B||)

Its range is approximately ``[-1.0, 1.0]``: ``1.0`` means the two vectors
point in exactly the same direction (regardless of magnitude), ``0.0``
means they are orthogonal (unrelated directions), and ``-1.0`` means they
point in exactly opposite directions. Because cosine similarity only
depends on direction, not magnitude, it is insensitive to embedding
scale.

This module deliberately does NOT implement any of the following — they
belong to later phases:
    - a similarity threshold
    - a MATCH / NON_MATCH / UNCERTAIN decision
    - FAR, FRR, EER, ROC/ROC-AUC
    - calibration or fuzzy logic

A cosine similarity score produced here has **no established meaning**
about whether two recordings are from the same speaker until it is
interpreted against a threshold calibrated on representative genuine and
impostor trial data — a later phase. Do not treat a raw score from this
module as a verification result.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Below this L2 norm, an embedding is treated as a zero vector: cosine
#: similarity is mathematically undefined (division by zero), so we raise
#: a clear error instead of returning torch's epsilon-clamped fallback
#: value, which would look like a legitimate (but meaningless) score.
ZERO_NORM_EPSILON: float = 1e-8


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SimilarityError(Exception):
    """Base class for all errors raised by this module."""


class InvalidEmbeddingError(SimilarityError, ValueError):
    """Raised when an embedding input is not usable: wrong type, wrong
    number of dimensions, empty, non-finite (NaN/Inf), or effectively a
    zero vector (undefined direction)."""


class EmbeddingMismatchError(SimilarityError, ValueError):
    """Raised when two embeddings are not comparable: mismatched
    dimensionality, mismatched batch size, or residing on different
    devices."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_embedding(embedding: torch.Tensor, name: str) -> None:
    """Validate a single embedding tensor in isolation (type, shape,
    finiteness, non-zero norm). Does not compare it against another
    embedding — see cosine_similarity() for pairwise checks.
    """
    if embedding is None:
        raise InvalidEmbeddingError(f"{name} must not be None.")

    if not isinstance(embedding, torch.Tensor):
        raise InvalidEmbeddingError(
            f"{name} must be a torch.Tensor, got {type(embedding)!r}."
        )

    if not torch.is_floating_point(embedding):
        raise InvalidEmbeddingError(
            f"{name} must be a floating-point tensor (got dtype "
            f"{embedding.dtype})."
        )

    if embedding.ndim not in (1, 2):
        raise InvalidEmbeddingError(
            f"{name} must be 1-D (embedding_dim,) for a single embedding "
            f"or 2-D (batch, embedding_dim) for a batch, got "
            f"{embedding.ndim}-D shape {tuple(embedding.shape)}."
        )

    if embedding.numel() == 0 or embedding.shape[-1] == 0:
        raise InvalidEmbeddingError(f"{name} is empty (zero elements).")

    if not torch.isfinite(embedding).all():
        raise InvalidEmbeddingError(
            f"{name} contains non-finite values (NaN or Inf)."
        )

    norm = torch.linalg.vector_norm(embedding, ord=2, dim=-1)
    if torch.any(norm < ZERO_NORM_EPSILON):
        raise InvalidEmbeddingError(
            f"{name} has a zero (or near-zero) norm; cosine similarity is "
            f"undefined for a zero vector (division by zero). This "
            f"usually indicates a bug upstream (e.g. an all-zero or "
            f"empty-audio embedding) rather than a valid speaker "
            f"embedding."
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def cosine_similarity(
    embedding_a: torch.Tensor, embedding_b: torch.Tensor
) -> Union[float, torch.Tensor]:
    """Compute the cosine similarity between two speaker embeddings.

    This is a pure comparison function: it does not threshold, calibrate,
    or interpret the result in any way. It only answers "how similar are
    these two embeddings?".

    Args:
        embedding_a: A single embedding of shape ``(embedding_dim,)``, or
            a batch of shape ``(batch, embedding_dim)``. As produced by
            ``speaker_verification.encoder.SpeakerEncoder.encode()`` for
            the single-embedding case.
        embedding_b: Same shape convention as ``embedding_a``. If both
            inputs are batched, batch sizes must match (pairwise
            comparison, i.e. ``embedding_a[i]`` vs ``embedding_b[i]``).

    Returns:
        If both inputs are 1-D: a Python ``float`` in the range
        approximately ``[-1.0, 1.0]``.
        If both inputs are 2-D (batched): a 1-D ``torch.Tensor`` of shape
        ``(batch,)`` with one score per pair, on the same device as the
        inputs.

    Raises:
        InvalidEmbeddingError: If either embedding is None, not a
            floating-point ``torch.Tensor``, has an unsupported number of
            dimensions, is empty, contains NaN/Inf, or has a zero (or
            near-zero) norm.
        EmbeddingMismatchError: If the two embeddings have mismatched
            embedding dimensionality, mismatched batch size, or reside on
            different devices.

    Notes:
        - Neither input tensor is modified.
        - No unnecessary device transfers are performed: the computation
          runs on whatever device the (matching) input tensors are
          already on. Only the final scalar result (1-D case) is copied
          to host memory as a Python float, which is unavoidable for a
          scalar return value.
        - Uses ``torch.nn.functional.cosine_similarity`` (PyTorch's
          established, numerically-stable implementation) rather than a
          manual re-implementation of dot(A, B) / (||A|| ||B||).
    """
    _validate_embedding(embedding_a, "embedding_a")
    _validate_embedding(embedding_b, "embedding_b")

    if embedding_a.ndim != embedding_b.ndim:
        raise EmbeddingMismatchError(
            f"embedding_a is {embedding_a.ndim}-D "
            f"{tuple(embedding_a.shape)} but embedding_b is "
            f"{embedding_b.ndim}-D {tuple(embedding_b.shape)}. Both must "
            f"either be single embeddings (1-D) or batches (2-D)."
        )

    if embedding_a.shape[-1] != embedding_b.shape[-1]:
        raise EmbeddingMismatchError(
            f"Embedding dimension mismatch: embedding_a has dimension "
            f"{embedding_a.shape[-1]}, embedding_b has dimension "
            f"{embedding_b.shape[-1]}. Both embeddings must come from the "
            f"same encoder/model."
        )

    if embedding_a.ndim == 2 and embedding_a.shape[0] != embedding_b.shape[0]:
        raise EmbeddingMismatchError(
            f"Batch size mismatch: embedding_a has batch size "
            f"{embedding_a.shape[0]}, embedding_b has batch size "
            f"{embedding_b.shape[0]}."
        )

    if embedding_a.device != embedding_b.device:
        raise EmbeddingMismatchError(
            f"embedding_a is on device {embedding_a.device} but "
            f"embedding_b is on device {embedding_b.device}. Move both "
            f"embeddings to the same device before calling "
            f"cosine_similarity() — this function does not perform "
            f"implicit/unnecessary device transfers."
        )

    if embedding_a.ndim == 1:
        # F.cosine_similarity reduces over `dim`, so add a temporary batch
        # dimension of size 1, compute, then extract the single score.
        score = F.cosine_similarity(
            embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), dim=-1
        )
        return float(score.item())

    # Batched case: one score per row, shape (batch,).
    return F.cosine_similarity(embedding_a, embedding_b, dim=-1)
