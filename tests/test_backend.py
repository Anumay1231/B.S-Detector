"""Domain-adapted scoring backend for ECAPA embeddings."""
import numpy as np

from speaker_verification.backend import ScoreBackend

rng = np.random.default_rng(0)


def test_plain_backend_is_cosine():
    b = ScoreBackend(mean=None, lda=None, cohort=None, top_k=10, threshold=0.5)
    a, c = rng.standard_normal(192), rng.standard_normal(192)
    cos = a @ c / np.linalg.norm(a) / np.linalg.norm(c)
    assert np.isclose(b.score(a, c), cos)
    assert np.isclose(b.score(a, a), 1.0)


def test_mean_removal_separates_speakers_sharing_a_channel_offset():
    offset = 5 * rng.standard_normal(192)  # what a shared mic/language adds to every embedding
    s1, s2 = rng.standard_normal(192), rng.standard_normal(192)
    raw = ScoreBackend(None, None, None, 10, 0.5)
    centred = ScoreBackend(offset, None, None, 10, 0.5)
    assert raw.score(s1 + offset, s2 + offset) > 0.8
    assert abs(centred.score(s1 + offset, s2 + offset)) < 0.3


def test_round_trip_and_asnorm(tmp_path):
    cohort = rng.standard_normal((50, 192))
    cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
    b = ScoreBackend(rng.standard_normal(192) * 0.01, None, cohort, 20, 3.0, "test")
    b.save(tmp_path / "b.npz")
    b2 = ScoreBackend.load(tmp_path / "b.npz")
    a, c = rng.standard_normal(192), rng.standard_normal(192)
    assert np.isclose(b.score(a, c), b2.score(a, c))
    assert b2.description == "test" and b2.threshold == 3.0
    # AS-norm: a pair identical to itself scores far above the cohort
    assert b2.score(a, a) > 3
