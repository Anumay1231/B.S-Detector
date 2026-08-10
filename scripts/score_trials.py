#!/usr/bin/env python
"""
scripts/score_trials.py
-------------------------

Phase 7D: compute a cosine-similarity score for every trial in a trials
CSV, using cached embeddings (scripts/extract_embeddings.py) and the
EXISTING similarity.cosine_similarity() -- no second cosine-similarity
implementation is introduced here.

Usage:
    python scripts/score_trials.py \\
        --trials data/voxceleb/trials/subset.csv \\
        --cache-dir outputs/embeddings/voxceleb \\
        --output outputs/scores/voxceleb_scores.csv

Output CSV columns:
    trial_id,reference,test,label,score,split,reference_speaker,test_speaker

(reference/test hold the utterance ids; label is 1=genuine/0=impostor,
matching Phase 7D's example format, with split/speaker columns carried
through for downstream calibration/evaluation and speaker-leakage
reporting.)

NO threshold is applied here -- this script only produces raw scores.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from speaker_verification.embedding_cache import EmbeddingCache  # noqa: E402
from speaker_verification.similarity import SimilarityError, cosine_similarity  # noqa: E402

SCORE_CSV_FIELDNAMES = [
    "trial_id",
    "reference",
    "test",
    "label",
    "score",
    "split",
    "reference_speaker",
    "test_speaker",
]


def score_trials(trials_rows: List[dict], cache: EmbeddingCache) -> Tuple[List[dict], List[dict]]:
    """Score every trial row (as read from a build_trials.py CSV) using
    cached embeddings and the existing cosine_similarity(). Returns
    (scored_rows, skipped_rows). A trial is skipped (not scored) if
    either embedding is missing from the cache or the embeddings are
    not comparable (similarity.py's own validation) -- skips are
    reported separately rather than silently dropped.
    """
    scored: List[dict] = []
    skipped: List[dict] = []

    for row in trials_rows:
        reference_id = row["reference_id"]
        test_id = row["test_id"]
        embedding_a = cache.get(reference_id)
        embedding_b = cache.get(test_id)

        if embedding_a is None or embedding_b is None:
            reason = []
            if embedding_a is None:
                reason.append(f"missing cached embedding for reference '{reference_id}'")
            if embedding_b is None:
                reason.append(f"missing cached embedding for test '{test_id}'")
            skipped.append({**row, "reason": "; ".join(reason)})
            continue

        try:
            score = cosine_similarity(embedding_a, embedding_b)
        except SimilarityError as exc:
            skipped.append({**row, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        scored.append(
            {
                "trial_id": row["trial_id"],
                "reference": reference_id,
                "test": test_id,
                "label": row["label"],
                "score": score,
                "split": row["split"],
                "reference_speaker": row["reference_speaker"],
                "test_speaker": row["test_speaker"],
            }
        )

    return scored, skipped


def _read_trials_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_scores_csv(path: str, rows: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every trial in a trials CSV using cached "
        "embeddings and the existing cosine_similarity() function. "
        "Does not apply a threshold."
    )
    parser.add_argument("--trials", required=True, help="Trials CSV produced by scripts/build_trials.py.")
    parser.add_argument("--cache-dir", default="outputs/embeddings/voxceleb")
    parser.add_argument("--output", default="outputs/scores/voxceleb_scores.csv")
    args = parser.parse_args()

    if not Path(args.trials).is_file():
        print(f"ERROR: trials file not found: '{args.trials}'. Run scripts/build_trials.py first.", file=sys.stderr)
        return 1

    cache = EmbeddingCache(args.cache_dir)
    cache.load()
    if len(cache) == 0:
        print(
            f"ERROR: embedding cache at '{args.cache_dir}' is empty. Run "
            f"scripts/extract_embeddings.py first.",
            file=sys.stderr,
        )
        return 1

    trials_rows = _read_trials_csv(args.trials)
    print(f"Loaded {len(trials_rows)} trial(s) from '{args.trials}'.")
    print(f"Loaded {len(cache)} cached embedding(s) from '{args.cache_dir}'.")

    scored, skipped = score_trials(trials_rows, cache)

    print(f"Scored: {len(scored)}")
    print(f"Skipped (missing/invalid embeddings): {len(skipped)}")
    if skipped:
        for row in skipped[:10]:
            print(f"    trial {row['trial_id']}: {row['reason']}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more.")

    if not scored:
        print("ERROR: no trials were scored; nothing to write.", file=sys.stderr)
        return 1

    _write_scores_csv(args.output, scored)
    print(f"Wrote {len(scored)} score(s) to '{args.output}'.")
    print("No threshold was applied -- see scripts/run_voxceleb_calibration.py.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
