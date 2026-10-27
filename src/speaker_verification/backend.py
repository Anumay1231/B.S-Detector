"""
backend.py
----------
Domain-adapted scoring for ECAPA-TDNN embeddings (Phase 9).

The VoxCeleb-trained encoder and its VoxCeleb-calibrated cosine threshold do
not transfer to Indian speakers: on Hindi and Indian-English test speakers,
same-gender impostors score around 0.2 on average and over a third of them
cross the 0.2528 VoxCeleb EER threshold (see docs/speaker-verification-hindi.md
and scripts/sv_benchmark.py). This module adds a small backend, trained on
speaker-labelled Hindi + English Common Voice embeddings, between the encoder
and the threshold:

    embedding
        |  subtract the in-domain mean, length-normalise
        |  LDA projection learnt on in-domain speakers, length-normalise
        |  cosine
        |  adaptive score normalisation (AS-norm) against an in-domain cohort
    score  (in standard deviations above the cohort, not a cosine)

The encoder itself is unchanged. All parameters are loaded from one .npz file
produced by scripts/train_sv_backend.py; nothing is learnt at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


def _np(x: ArrayLike) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


@dataclass(frozen=True)
class ScoreBackend:
    """Mean-centring + LDA + AS-norm. Any stage left as None is skipped."""

    mean: Optional[np.ndarray]
    lda: Optional[np.ndarray]
    cohort: Optional[np.ndarray]
    top_k: int
    threshold: float
    description: str = ""

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScoreBackend":
        d = np.load(path, allow_pickle=False)
        get = lambda k: d[k] if k in d.files and d[k].size else None  # noqa: E731
        return cls(
            mean=get("mean"),
            lda=get("lda"),
            cohort=get("cohort"),
            top_k=int(d["top_k"]),
            threshold=float(d["threshold"]),
            description=str(d["description"]) if "description" in d.files else "",
        )

    def save(self, path: Union[str, Path]) -> None:
        empty = np.zeros(0)
        np.savez(
            path,
            mean=self.mean if self.mean is not None else empty,
            lda=self.lda if self.lda is not None else empty,
            cohort=self.cohort if self.cohort is not None else empty,
            top_k=self.top_k,
            threshold=self.threshold,
            description=self.description,
        )

    def project(self, emb: ArrayLike) -> np.ndarray:
        x = _np(emb)
        if self.mean is not None:
            x = x - self.mean
        x = _l2n(x)
        if self.lda is not None:
            x = _l2n(x @ self.lda)
        return x

    def score(self, a: ArrayLike, b: ArrayLike) -> float:
        """Score one pair of raw encoder embeddings; higher means more alike."""
        pa, pb = self.project(a), self.project(b)
        s = float(pa @ pb)
        if self.cohort is None:
            return s
        k = min(self.top_k, len(self.cohort))
        ca = np.sort(self.cohort @ pa)[-k:]
        cb = np.sort(self.cohort @ pb)[-k:]
        return 0.5 * ((s - ca.mean()) / (ca.std() + 1e-9) + (s - cb.mean()) / (cb.std() + 1e-9))
