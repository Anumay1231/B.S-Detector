"""
test_extract_embeddings.py
-----------------------------

Unit tests for scripts/extract_embeddings.py's pure/testable functions:
collect_unique_utterances() (duplicate-utterance dedup) and
extract_embeddings() (cache-aware batch extraction, using a stub
encoder rather than the real pretrained model).
"""

from __future__ import annotations

import csv

import torch

from extract_embeddings import collect_unique_utterances, extract_embeddings
from speaker_verification.embedding_cache import EmbeddingCache


class _StubEncoder:
    """Minimal stand-in for SpeakerEncoder.encode_file(); tracks calls
    so tests can assert an already-cached utterance is never
    re-encoded."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on or set()

    def encode_file(self, path: str) -> torch.Tensor:
        self.calls.append(path)
        if path in self.fail_on:
            from speaker_verification.encoder import EncoderError

            raise EncoderError(f"synthetic failure for {path}")
        # Deterministic, path-derived fake embedding.
        return torch.tensor([float(len(path)), 1.0, 2.0])


def _write_trials_csv(path, rows):
    fieldnames = [
        "trial_id", "reference_id", "test_id", "reference_path", "test_path",
        "label", "reference_speaker", "test_speaker", "split",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
# collect_unique_utterances: duplicate-utterance handling (Phase 7K #7)
# --------------------------------------------------------------------------


def test_collect_unique_utterances_dedupes(tmp_path) -> None:
    csv_path = tmp_path / "trials.csv"
    _write_trials_csv(csv_path, [
        {"trial_id": "t1", "reference_id": "id1/v/1.wav", "test_id": "id1/v/2.wav",
         "reference_path": "/a/id1/v/1.wav", "test_path": "/a/id1/v/2.wav",
         "label": 1, "reference_speaker": "id1", "test_speaker": "id1", "split": "calibration"},
        # t2 reuses id1/v/1.wav as its "test" utterance -- should collapse
        # to a single cache entry, not be processed twice.
        {"trial_id": "t2", "reference_id": "id2/v/1.wav", "test_id": "id1/v/1.wav",
         "reference_path": "/a/id2/v/1.wav", "test_path": "/a/id1/v/1.wav",
         "label": 0, "reference_speaker": "id2", "test_speaker": "id1", "split": "calibration"},
    ])

    unique = collect_unique_utterances(str(csv_path))
    assert set(unique.keys()) == {"id1/v/1.wav", "id1/v/2.wav", "id2/v/1.wav"}
    assert unique["id1/v/1.wav"] == "/a/id1/v/1.wav"


# --------------------------------------------------------------------------
# extract_embeddings: cache-aware extraction (Phase 7K #5, #6, #7)
# --------------------------------------------------------------------------


def test_extract_embeddings_processes_new_and_skips_cached(tmp_path) -> None:
    cache = EmbeddingCache(str(tmp_path / "cache"))
    cache.put("already_cached.wav", torch.tensor([9.0]), model_source="m", sample_rate=16000)

    encoder = _StubEncoder()
    unique = {"already_cached.wav": "/a/already_cached.wav", "new.wav": "/a/new.wav"}

    stats = extract_embeddings(
        unique, encoder, cache, model_source="m", sample_rate=16000, save_every=0,
    )

    assert stats["already_cached"] == 1
    assert stats["attempted"] == 1
    assert stats["num_processed"] == 1
    assert encoder.calls == ["/a/new.wav"]
    assert "new.wav" in cache


def test_extract_embeddings_handles_failures_separately(tmp_path) -> None:
    cache = EmbeddingCache(str(tmp_path / "cache"))
    encoder = _StubEncoder(fail_on={"/a/bad.wav"})
    unique = {"good.wav": "/a/good.wav", "bad.wav": "/a/bad.wav"}

    stats = extract_embeddings(
        unique, encoder, cache, model_source="m", sample_rate=16000, save_every=0,
    )

    assert stats["num_processed"] == 1
    assert stats["num_failed"] == 1
    assert "good.wav" in cache
    assert "bad.wav" not in cache
    assert stats["failures"][0][0] == "bad.wav"


def test_extract_embeddings_respects_limit(tmp_path) -> None:
    cache = EmbeddingCache(str(tmp_path / "cache"))
    encoder = _StubEncoder()
    unique = {f"u{i}.wav": f"/a/u{i}.wav" for i in range(5)}

    stats = extract_embeddings(
        unique, encoder, cache, model_source="m", sample_rate=16000, save_every=0, limit=2,
    )

    assert stats["attempted"] == 2
    assert stats["num_processed"] == 2


def test_extract_embeddings_saves_cache_to_disk(tmp_path) -> None:
    cache_dir = str(tmp_path / "cache")
    cache = EmbeddingCache(cache_dir)
    encoder = _StubEncoder()
    unique = {"u1.wav": "/a/u1.wav"}

    extract_embeddings(unique, encoder, cache, model_source="m", sample_rate=16000, save_every=0)

    reloaded = EmbeddingCache(cache_dir)
    reloaded.load()
    assert "u1.wav" in reloaded
