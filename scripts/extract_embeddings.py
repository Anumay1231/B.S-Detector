#!/usr/bin/env python
"""
scripts/extract_embeddings.py
-------------------------------

Phase 7C: offline, cached, batch embedding extraction for a trials CSV
produced by scripts/build_trials.py.

Loads the ECAPA-TDNN encoder ONCE per process (never per-file), reuses
any embedding already present in the on-disk cache
(src/speaker_verification/embedding_cache.py), and reports throughput
and (on CUDA) peak GPU memory.

Usage:
    python scripts/extract_embeddings.py \\
        --trials data/voxceleb/trials/subset.csv \\
        --cache-dir outputs/embeddings/voxceleb \\
        --save-every 200

This script does NOT compute similarity scores or a threshold -- see
scripts/score_trials.py and scripts/run_voxceleb_calibration.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from speaker_verification.audio import AudioError  # noqa: E402
from speaker_verification.embedding_cache import EmbeddingCache  # noqa: E402
from speaker_verification.encoder import EncoderError, ModelLoadError, SpeakerEncoder  # noqa: E402


def collect_unique_utterances(trials_csv_path: str) -> Dict[str, str]:
    """Read a trials CSV (as produced by scripts/build_trials.py) and
    return a dict mapping each unique utterance_id to its resolved
    local path, deduplicating utterances that are referenced by more
    than one trial (Phase 7K "duplicate utterance handling") -- each
    such utterance is embedded only once.
    """
    unique: Dict[str, str] = {}
    with open(trials_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique[row["reference_id"]] = row["reference_path"]
            unique[row["test_id"]] = row["test_path"]
    return unique


def extract_embeddings(
    unique_utterances: Dict[str, str],
    encoder,
    cache: EmbeddingCache,
    model_source: str,
    sample_rate: int,
    progress_every: int = 50,
    save_every: int = 200,
    limit: "int | None" = None,
) -> dict:
    """Encode every utterance in ``unique_utterances`` not already
    present in ``cache``, saving the cache periodically. Returns a
    stats dict. ``encoder`` only needs an ``encode_file(path) ->
    torch.Tensor`` method (see encoder.SpeakerEncoder), so tests can
    substitute a lightweight stand-in without the real pretrained
    model.
    """
    to_process = [
        (utterance_id, path)
        for utterance_id, path in unique_utterances.items()
        if utterance_id not in cache
    ]
    already_cached = len(unique_utterances) - len(to_process)

    if limit is not None:
        to_process = to_process[:limit]

    failures: List[Tuple[str, str, str]] = []
    num_processed = 0
    total_encode_seconds = 0.0
    since_last_save = 0

    for i, (utterance_id, path) in enumerate(to_process, start=1):
        start = time.perf_counter()
        try:
            embedding = encoder.encode_file(path)
        except (AudioError, EncoderError) as exc:
            failures.append((utterance_id, path, f"{type(exc).__name__}: {exc}"))
            continue
        elapsed = time.perf_counter() - start

        cache.put(utterance_id, embedding, model_source=model_source, sample_rate=sample_rate)
        num_processed += 1
        total_encode_seconds += elapsed
        since_last_save += 1

        if progress_every and i % progress_every == 0:
            print(f"  ... {i}/{len(to_process)} processed "
                  f"({num_processed} ok, {len(failures)} failed)")

        if save_every and since_last_save >= save_every:
            cache.save()
            since_last_save = 0

    cache.save()

    return {
        "already_cached": already_cached,
        "attempted": len(to_process),
        "num_processed": num_processed,
        "num_failed": len(failures),
        "failures": failures,
        "total_encode_seconds": total_encode_seconds,
        "avg_encode_seconds": (total_encode_seconds / num_processed) if num_processed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and cache ECAPA-TDNN embeddings for every "
        "unique utterance referenced by a trials CSV (see "
        "scripts/build_trials.py)."
    )
    parser.add_argument("--trials", required=True, help="Trials CSV produced by scripts/build_trials.py.")
    parser.add_argument("--cache-dir", default="outputs/embeddings/voxceleb")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', 'cuda'. Default: auto-select.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint the cache to disk every N newly-processed files.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many NEW utterances (for quick tests/dry runs).")
    args = parser.parse_args()

    if not Path(args.trials).is_file():
        print(f"ERROR: trials file not found: '{args.trials}'. Run scripts/build_trials.py first.", file=sys.stderr)
        return 1

    unique_utterances = collect_unique_utterances(args.trials)
    print(f"{len(unique_utterances)} unique utterance(s) referenced by '{args.trials}'.")

    cache = EmbeddingCache(args.cache_dir)
    cache.load()
    print(f"Loaded existing cache at '{args.cache_dir}': {len(cache)} embedding(s) already present.")

    init_start = time.perf_counter()
    try:
        encoder = SpeakerEncoder(device=args.device)
    except ModelLoadError as exc:
        print(f"ERROR: could not load the pretrained model: {exc}", file=sys.stderr)
        return 1
    init_seconds = time.perf_counter() - init_start
    print(f"Model initialized in {init_seconds:.3f}s on {encoder.describe_device()}.")

    if encoder.device_info.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(encoder.device_info.device)

    stats = extract_embeddings(
        unique_utterances,
        encoder,
        cache,
        model_source=encoder.model_source,
        sample_rate=16_000,
        progress_every=args.progress_every,
        save_every=args.save_every,
        limit=args.limit,
    )

    print()
    print("=== Embedding extraction report ===")
    print(f"Device: {encoder.describe_device()}")
    print(f"Model initialization time: {init_seconds:.3f}s")
    print(f"Already cached (skipped): {stats['already_cached']}")
    print(f"Attempted (new): {stats['attempted']}")
    print(f"Succeeded: {stats['num_processed']}")
    print(f"Failed: {stats['num_failed']}")
    if stats["num_processed"]:
        throughput = stats["num_processed"] / stats["total_encode_seconds"] if stats["total_encode_seconds"] > 0 else float("inf")
        print(f"Total embedding time: {stats['total_encode_seconds']:.3f}s")
        print(f"Average embedding time: {stats['avg_encode_seconds'] * 1000:.2f} ms/file")
        print(f"Throughput: {throughput:.2f} files/sec")
    if encoder.device_info.device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(encoder.device_info.device) / (1024 ** 2)
        print(f"Peak GPU memory allocated: {peak_mb:.1f} MiB")
    print(f"Cache now contains {len(cache)} embedding(s) at '{args.cache_dir}'.")

    if stats["failures"]:
        print()
        print(f"{len(stats['failures'])} failure(s):")
        for utterance_id, path, error in stats["failures"][:20]:
            print(f"    {utterance_id} ({path}): {error}")
        if len(stats["failures"]) > 20:
            print(f"    ... and {len(stats['failures']) - 20} more.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
