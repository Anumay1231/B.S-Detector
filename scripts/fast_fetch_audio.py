#!/usr/bin/env python3
"""
Fast Direct-Fetch Audio Downloader for SEA-Spoof.

Instead of sequentially streaming through 439,000 rows across all languages,
this script:
1. Calculates which exact Parquet shards contain our target row IDs.
2. Reads ONLY the target rows directly from each shard using fsspec/pyarrow.
3. Decodes the audio in-memory and saves 16kHz .pt waveforms to data/deepfake_cache/audio/.

Downloads only the needed ~800 clips directly in <60 seconds on cloud networks.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.splits import (
    create_fewshot_pools,
    create_test_split,
    get_all_needed_ids,
    load_hindi_metadata,
)
from scripts.streaming import CACHE_DIR

logger = logging.getLogger("fast_fetch")

DATASET_NAME = "Jack-ppkdczgx/SEA-Spoof"
ROWS_PER_SHARD = 30000  # SEA-Spoof shards typically contain ~25,000-30,000 rows


def fetch_target_audio_direct(
    target_row_ids: Set[str],
    cache_dir: Path = CACHE_DIR,
    dataset_name: str = DATASET_NAME,
) -> Dict[str, Path]:
    """Download audio for only the target row_ids directly from HuggingFace parquet files."""
    import fsspec
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check already cached
    already_cached = {}
    still_needed = set()
    for rid in target_row_ids:
        p = cache_dir / f"{rid}.pt"
        if p.exists():
            already_cached[rid] = p
        else:
            still_needed.add(rid)

    logger.info(f"Target clips: {len(target_row_ids)} total | Already cached: {len(already_cached)} | Needed: {len(still_needed)}")
    if not still_needed:
        return already_cached

    # Connect to HuggingFace filesystem
    fs = fsspec.filesystem("hf")
    shard_pattern = f"datasets/{dataset_name}/data/train"
    all_files = fs.ls(shard_pattern, detail=False)
    shard_files = sorted([f for f in all_files if f.endswith(".parquet")])
    logger.info(f"Found {len(shard_files)} shards on HuggingFace CDN.")

    # Silence noisy HTTP request loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    start_time = time.time()
    saved_count = 0

    # Dynamically determine which shards contain the still_needed row_ids.
    # SEA-Spoof has ~439,000 rows partitioned across 30 shards (~14,640 rows per shard).
    target_shard_indices = set()
    for rid in still_needed:
        try:
            int_id = int(str(rid).split("_")[-1])
            est_shard = int_id // 14640
            for s in [est_shard - 1, est_shard, est_shard + 1]:
                if 0 <= s < len(shard_files):
                    target_shard_indices.add(s)
        except Exception:
            pass

    if not target_shard_indices:
        target_shard_indices = set(range(len(shard_files)))

    relevant_shards = [
        f for i, f in enumerate(shard_files) if i in target_shard_indices
    ]
    logger.info(
        f"Scanning {len(relevant_shards)} relevant shards (out of {len(shard_files)}) "
        f"for {len(still_needed)} needed clips: {[f.split('/')[-1] for f in relevant_shards]}"
    )

    total_needed_start = len(still_needed)
    # Progress bar 1: Shards
    shard_pbar = tqdm(relevant_shards, desc="Scanning Shards", unit="shard", dynamic_ncols=True)
    # Progress bar 2: Audio clips cached
    clip_pbar = tqdm(total=total_needed_start, desc="Caching Clips", unit="clip", dynamic_ncols=True)

    for shard_path in shard_pbar:
        if not still_needed:
            break

        shard_name = shard_path.split("/")[-1]
        shard_pbar.set_postfix({"shard": shard_name, "needed": len(still_needed)})

        try:
            with fs.open(shard_path, "rb") as f:
                pf = pq.ParquetFile(f)
                num_rgs = pf.metadata.num_row_groups

                for rg in range(num_rgs):
                    if not still_needed:
                        break

                    # 1. Read row_id column ONLY (fast byte range, zero audio transfer)
                    id_table = pf.read_row_group(rg, columns=["row_id"])
                    row_ids_in_rg = id_table.column("row_id").to_pylist()

                    # Find matches in this row group
                    matched_indices = [
                        idx for idx, rid in enumerate(row_ids_in_rg) if rid in still_needed
                    ]

                    if not matched_indices:
                        continue

                    # 2. Read audio column for ONLY the matched rows in this row group
                    audio_table = pf.read_row_group(rg, columns=["audio"])
                    audio_data_list = audio_table.column("audio").to_pylist()

                    for idx in matched_indices:
                        rid = row_ids_in_rg[idx]
                        audio_entry = audio_data_list[idx]

                        # Extract waveform
                        waveform = None
                        sr = 16000

                        if isinstance(audio_entry, dict):
                            if "bytes" in audio_entry and audio_entry["bytes"] is not None:
                                raw_bytes = audio_entry["bytes"]
                                wav_arr, sr = sf.read(io.BytesIO(raw_bytes))
                                waveform = torch.tensor(wav_arr, dtype=torch.float32)
                            elif "array" in audio_entry and audio_entry["array"] is not None:
                                waveform = torch.tensor(audio_entry["array"], dtype=torch.float32)
                                sr = audio_entry.get("sampling_rate", 16000)

                        if waveform is None:
                            continue

                        # Save .pt waveform
                        out_path = cache_dir / f"{rid}.pt"
                        torch.save({"waveform": waveform, "sampling_rate": sr}, out_path)
                        already_cached[rid] = out_path
                        still_needed.remove(rid)
                        saved_count += 1
                        clip_pbar.update(1)

        except Exception as e:
            logger.warning(f"Error processing {shard_name}: {e}")
            continue

    shard_pbar.close()
    clip_pbar.close()

    elapsed = time.time() - start_time
    logger.info(f"\nDirect fetch complete in {elapsed:.1f}s! Total clips cached: {len(already_cached)}/{len(target_row_ids)}")
    return already_cached


def main():
    parser = argparse.ArgumentParser(description="Fast Direct-Fetch Audio Downloader")
    parser.add_argument("--parquet-path", type=str, default="data/sea_spoof_en_hi_metadata.parquet")
    parser.add_argument("--audio-cache-dir", type=str, default="data/deepfake_cache/audio")
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--fewshot-sizes", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--num-seeds", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parquet_path = _PROJECT_ROOT / args.parquet_path
    audio_cache_dir = _PROJECT_ROOT / args.audio_cache_dir

    logger.info("=" * 70)
    logger.info("FAST DIRECT-FETCH AUDIO DOWNLOADER")
    logger.info("=" * 70)

    df_hindi = load_hindi_metadata(parquet_path)
    test_ids, _ = create_test_split(df_hindi, test_per_class=args.test_per_class, seed=42)
    fewshot_pools = create_fewshot_pools(
        df_hindi,
        test_ids=test_ids,
        fewshot_sizes=args.fewshot_sizes,
        num_seeds=args.num_seeds,
    )
    all_needed_ids = get_all_needed_ids(test_ids, fewshot_pools)

    # Fetch only the missing clips
    fetch_target_audio_direct(all_needed_ids, cache_dir=audio_cache_dir)


if __name__ == "__main__":
    main()
