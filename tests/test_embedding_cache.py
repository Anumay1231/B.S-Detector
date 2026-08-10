"""
test_embedding_cache.py
--------------------------

Unit tests for src/speaker_verification/embedding_cache.py: put/get,
save/load round-trip, and reuse (the mechanism scripts/extract_embeddings.py
relies on to avoid recomputing embeddings for utterances already cached,
including duplicate utterances referenced by multiple trials).

All embeddings here are small synthetic tensors -- no real model is
required.
"""

from __future__ import annotations

import torch

from speaker_verification.embedding_cache import EmbeddingCache


def test_put_and_get_round_trip() -> None:
    cache = EmbeddingCache("/unused")
    embedding = torch.tensor([0.1, 0.2, 0.3])
    cache.put("id1/v/1.wav", embedding, model_source="test-model", sample_rate=16000)

    assert "id1/v/1.wav" in cache
    retrieved = cache.get("id1/v/1.wav")
    assert torch.equal(retrieved, embedding)


def test_get_missing_returns_none() -> None:
    cache = EmbeddingCache("/unused")
    assert cache.get("nope") is None
    assert "nope" not in cache


def test_record_metadata_populated() -> None:
    cache = EmbeddingCache("/unused")
    cache.put("id1/v/1.wav", torch.zeros(4), model_source="my-model", sample_rate=16000)
    record = cache.record("id1/v/1.wav")
    assert record is not None
    assert record.utterance_id == "id1/v/1.wav"
    assert record.embedding_dim == 4
    assert record.model_source == "my-model"
    assert record.sample_rate == 16000
    assert record.extracted_at  # non-empty timestamp string


def test_save_then_load_round_trip(tmp_path) -> None:
    cache_dir = str(tmp_path / "cache")
    cache = EmbeddingCache(cache_dir)
    cache.put("a", torch.tensor([1.0, 2.0]), model_source="m", sample_rate=16000)
    cache.put("b", torch.tensor([3.0, 4.0]), model_source="m", sample_rate=16000)
    cache.save()

    reloaded = EmbeddingCache(cache_dir)
    reloaded.load()

    assert len(reloaded) == 2
    assert torch.equal(reloaded.get("a"), torch.tensor([1.0, 2.0]))
    assert torch.equal(reloaded.get("b"), torch.tensor([3.0, 4.0]))
    assert reloaded.record("a").embedding_dim == 2


def test_load_on_empty_directory_starts_empty(tmp_path) -> None:
    cache = EmbeddingCache(str(tmp_path / "does_not_exist_yet"))
    cache.load()  # must not raise
    assert len(cache) == 0


def test_cache_reuse_avoids_recomputation(tmp_path) -> None:
    # Simulates scripts/extract_embeddings.py's skip-if-cached logic:
    # after loading a saved cache, an already-present utterance is
    # never re-embedded.
    cache_dir = str(tmp_path / "cache")
    cache = EmbeddingCache(cache_dir)
    cache.put("id1/v/1.wav", torch.tensor([9.0]), model_source="m", sample_rate=16000)
    cache.save()

    reloaded = EmbeddingCache(cache_dir)
    reloaded.load()

    calls = []

    def fake_encode(utterance_id):
        calls.append(utterance_id)
        return torch.tensor([0.0])

    for utterance_id in ["id1/v/1.wav", "id2/v/1.wav"]:
        if utterance_id not in reloaded:
            reloaded.put(utterance_id, fake_encode(utterance_id), model_source="m", sample_rate=16000)

    # Only the NOT-already-cached utterance triggered "encoding".
    assert calls == ["id2/v/1.wav"]
    assert len(reloaded) == 2


def test_duplicate_utterance_put_overwrites_not_duplicates() -> None:
    # Two trials referencing the same utterance_id must collapse to one
    # cache entry, not two.
    cache = EmbeddingCache("/unused")
    cache.put("shared.wav", torch.tensor([1.0]), model_source="m", sample_rate=16000)
    cache.put("shared.wav", torch.tensor([2.0]), model_source="m", sample_rate=16000)
    assert len(cache) == 1
    assert torch.equal(cache.get("shared.wav"), torch.tensor([2.0]))


def test_save_is_atomic_leaves_no_tmp_files(tmp_path) -> None:
    cache_dir = str(tmp_path / "cache")
    cache = EmbeddingCache(cache_dir)
    cache.put("a", torch.tensor([1.0]), model_source="m", sample_rate=16000)
    cache.save()

    leftover_tmp = list(tmp_path.glob("cache/*.tmp"))
    assert leftover_tmp == []
