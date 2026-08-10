"""
embedding_cache.py
-------------------

An on-disk cache of speaker embeddings, keyed by dataset-relative
utterance id, so ``scripts/extract_embeddings.py`` never re-runs the
(comparatively expensive) ECAPA-TDNN forward pass for an utterance it
has already embedded -- including when the SAME utterance is referenced
by multiple trials (Phase 7K "duplicate utterance handling").

Format choice: this project's initial target scale (Phase 7H) is on the
order of a few thousand unique utterances (~1,000 genuine + ~1,000
impostor trials touches at most a few thousand distinct files, usually
fewer since utterances are commonly reused across trials). At that
scale, one consolidated ``torch.save()`` file (a dict of
``{utterance_id: FloatTensor}``) plus a JSON-Lines manifest (one
metadata record per line -- human-readable, diffable, and easy to
reason about) is simpler and faster than writing one file per
utterance, and avoids filesystem overhead from potentially thousands of
tiny files. This does NOT scale indefinitely: a full VoxCeleb2 run
(1M+ utterances) would need a sharded format (e.g. per-speaker shards,
or a columnar format like Parquet) to avoid one very large file and
slow full-cache rewrites. That is explicitly out of scope for Phase 7
(see Phase 7H: start small, scale later) and is noted here rather than
built speculatively ahead of need.

Per utterance, the cache stores:
    - utterance_id (the cache key -- NOT the audio file path, so the
      cache is portable across different audio_root locations for the
      same dataset)
    - the embedding tensor itself
    - embedding_dim
    - model_source (e.g. "speechbrain/spkrec-ecapa-voxceleb")
    - sample_rate (the rate the encoder/audio pipeline uses)
    - extracted_at (ISO-8601 UTC timestamp)

It deliberately does NOT store the source audio.

Writes are atomic (temp file + os.replace) so a crash mid-save cannot
corrupt a previously-good cache on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import torch

#: Filename for the consolidated embedding tensor dict within a cache
#: directory.
EMBEDDINGS_FILENAME: str = "embeddings.pt"

#: Filename for the JSON-Lines metadata manifest within a cache
#: directory.
MANIFEST_FILENAME: str = "manifest.jsonl"


class EmbeddingCacheError(Exception):
    """Base class for all errors raised by this module."""


@dataclass(frozen=True)
class EmbeddingCacheRecord:
    """Metadata for one cached embedding (everything except the tensor
    itself, which lives in the embeddings.pt dict)."""

    utterance_id: str
    embedding_dim: int
    model_source: str
    sample_rate: int
    extracted_at: str


class EmbeddingCache:
    """Loads/saves a directory-backed embedding cache.

    Example:
        >>> cache = EmbeddingCache("outputs/embeddings/voxceleb")  # doctest: +SKIP
        >>> cache.load()  # doctest: +SKIP
        >>> if "id10270/x/00001.wav" not in cache:  # doctest: +SKIP
        ...     embedding = encoder.encode_file(path)  # doctest: +SKIP
        ...     cache.put("id10270/x/00001.wav", embedding, model_source="...", sample_rate=16000)  # doctest: +SKIP
        >>> cache.save()  # doctest: +SKIP
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._records: Dict[str, EmbeddingCacheRecord] = {}

    @property
    def _embeddings_path(self) -> str:
        return os.path.join(self.cache_dir, EMBEDDINGS_FILENAME)

    @property
    def _manifest_path(self) -> str:
        return os.path.join(self.cache_dir, MANIFEST_FILENAME)

    # ----------------------------------------------------------------
    # Load / save
    # ----------------------------------------------------------------

    def load(self) -> None:
        """Load an existing cache from ``cache_dir``, if present. Safe
        to call on an empty/nonexistent ``cache_dir`` -- starts empty
        rather than raising, so a first run needs no special-casing."""
        if os.path.isfile(self._embeddings_path):
            # weights_only=True restricts unpickling to a safe allowlist
            # (plain tensors/dicts, which is exactly what this cache
            # stores) -- deliberately avoids executing arbitrary code
            # from a cache file, even though this project only ever
            # writes its own cache files.
            self._embeddings = torch.load(
                self._embeddings_path, map_location="cpu", weights_only=True
            )
        else:
            self._embeddings = {}

        self._records = {}
        if os.path.isfile(self._manifest_path):
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._records[data["utterance_id"]] = EmbeddingCacheRecord(**data)

    def save(self) -> None:
        """Persist the current in-memory cache to ``cache_dir``,
        atomically (temp file + ``os.replace``) so a crash mid-write
        cannot corrupt a previously-good cache on disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self._atomic_write_torch(self._embeddings_path, self._embeddings)
        self._atomic_write_manifest(self._manifest_path, self._records)

    def _atomic_write_torch(self, path: str, obj: object) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _atomic_write_manifest(
        self, path: str, records: Dict[str, EmbeddingCacheRecord]
    ) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for utterance_id in sorted(records):
                    f.write(json.dumps(asdict(records[utterance_id])) + "\n")
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ----------------------------------------------------------------
    # Access
    # ----------------------------------------------------------------

    def __contains__(self, utterance_id: str) -> bool:
        return utterance_id in self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)

    def get(self, utterance_id: str) -> Optional[torch.Tensor]:
        """Return the cached embedding for ``utterance_id``, or None if
        not cached."""
        return self._embeddings.get(utterance_id)

    def put(
        self,
        utterance_id: str,
        embedding: torch.Tensor,
        model_source: str,
        sample_rate: int,
    ) -> None:
        """Add (or overwrite) one utterance's embedding and metadata in
        memory. Call ``save()`` afterward to persist to disk -- this
        method itself does not touch the filesystem, so callers can
        batch many ``put()`` calls before one ``save()`` (see
        scripts/extract_embeddings.py's periodic-checkpoint behavior).
        """
        embedding_cpu = embedding.detach().to("cpu")
        self._embeddings[utterance_id] = embedding_cpu
        self._records[utterance_id] = EmbeddingCacheRecord(
            utterance_id=utterance_id,
            embedding_dim=int(embedding_cpu.shape[-1]),
            model_source=model_source,
            sample_rate=sample_rate,
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

    def record(self, utterance_id: str) -> Optional[EmbeddingCacheRecord]:
        """Return the metadata record for ``utterance_id``, or None if
        not cached."""
        return self._records.get(utterance_id)

    def utterance_ids(self) -> list:
        """All utterance ids currently in the cache."""
        return list(self._embeddings.keys())
