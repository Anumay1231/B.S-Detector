"""
test_score_trials.py
-----------------------

Unit tests for scripts/score_trials.py's score_trials(): computing
cosine-similarity scores for trial rows from cached embeddings, using
the EXISTING similarity.cosine_similarity() (not a reimplementation).
"""

from __future__ import annotations

import math

import torch

from score_trials import score_trials
from speaker_verification.embedding_cache import EmbeddingCache


def _row(trial_id, ref_id, test_id, label, split="calibration",
         ref_speaker="id1", test_speaker="id1"):
    return {
        "trial_id": trial_id,
        "reference_id": ref_id,
        "test_id": test_id,
        "label": str(label),
        "split": split,
        "reference_speaker": ref_speaker,
        "test_speaker": test_speaker,
    }


def test_score_trials_computes_cosine_similarity() -> None:
    cache = EmbeddingCache("/unused")
    cache.put("a.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)
    cache.put("b.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)  # identical -> score 1.0

    rows = [_row("t1", "a.wav", "b.wav", label=1)]
    scored, skipped = score_trials(rows, cache)

    assert len(scored) == 1
    assert skipped == []
    assert scored[0]["trial_id"] == "t1"
    assert math.isclose(scored[0]["score"], 1.0, abs_tol=1e-6)
    assert scored[0]["label"] == "1"
    assert scored[0]["split"] == "calibration"


def test_score_trials_orthogonal_embeddings_score_near_zero() -> None:
    cache = EmbeddingCache("/unused")
    cache.put("a.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)
    cache.put("b.wav", torch.tensor([0.0, 1.0]), model_source="m", sample_rate=16000)

    rows = [_row("t1", "a.wav", "b.wav", label=0)]
    scored, skipped = score_trials(rows, cache)

    assert math.isclose(scored[0]["score"], 0.0, abs_tol=1e-6)


def test_score_trials_skips_missing_embeddings() -> None:
    cache = EmbeddingCache("/unused")
    cache.put("a.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)
    # "b.wav" was never cached (e.g. extraction failed for it).

    rows = [_row("t1", "a.wav", "b.wav", label=1)]
    scored, skipped = score_trials(rows, cache)

    assert scored == []
    assert len(skipped) == 1
    assert "b.wav" in skipped[0]["reason"]


def test_score_trials_preserves_speaker_columns_for_leakage_checks() -> None:
    cache = EmbeddingCache("/unused")
    cache.put("a.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)
    cache.put("b.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)

    rows = [_row("t1", "a.wav", "b.wav", label=1, ref_speaker="id42", test_speaker="id42")]
    scored, _ = score_trials(rows, cache)

    assert scored[0]["reference_speaker"] == "id42"
    assert scored[0]["test_speaker"] == "id42"


def test_score_trials_does_not_apply_a_threshold() -> None:
    # score_trials() must return raw scores only -- no MATCH/NON_MATCH
    # field, no threshold parameter accepted.
    cache = EmbeddingCache("/unused")
    cache.put("a.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)
    cache.put("b.wav", torch.tensor([1.0, 0.0]), model_source="m", sample_rate=16000)

    rows = [_row("t1", "a.wav", "b.wav", label=1)]
    scored, _ = score_trials(rows, cache)

    assert "outcome" not in scored[0]
    assert "threshold" not in scored[0]
