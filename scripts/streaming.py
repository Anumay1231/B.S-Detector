"""Stream audio from HuggingFace SEA-Spoof dataset and cache locally.

Iterates the HF streaming dataset once, matching row_ids against a target
set, and saves raw audio waveforms for matched rows to disk.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATASET_NAME = "Jack-ppkdczgx/SEA-Spoof"
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "deepfake_cache" / "audio"


def stream_and_cache_audio(
    target_ids: set,
    cache_dir: Path = CACHE_DIR,
    verify_first_n: int = 5,
) -> dict:
    """Stream HF dataset and cache audio for target row_ids.

    First attempts fast targeted shard extraction using fsspec and pyarrow.
    Falls back to sequential streaming if needed.

    Args:
        target_ids: Set of row_id strings to capture.
        cache_dir: Directory to save cached audio files.
        verify_first_n: Print the first N streamed examples for ID verification.

    Returns:
        Dict mapping row_id -> Path to cached .pt file for all found IDs.
    """
    import io
    from collections import defaultdict
    import fsspec
    import pyarrow.parquet as pq
    import soundfile as sf
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check which IDs are already cached
    already_cached = {}
    still_needed = set()
    for rid in target_ids:
        cached_path = cache_dir / f"{rid}.pt"
        if cached_path.exists():
            already_cached[rid] = cached_path
        else:
            still_needed.add(rid)

    if not still_needed:
        logger.info(
            f"All {len(target_ids)} audio clips already cached in {cache_dir}"
        )
        return already_cached

    logger.info(
        f"Need to fetch {len(still_needed)} clips "
        f"({len(already_cached)} already cached)"
    )

    found = dict(already_cached)

    # Attempt fast targeted shard extraction
    # SEA-Spoof row_id pattern: sea_bench_clean_v2_train_0000XXXXX
    # 15,000 rows per shard, 75 row groups of 200 rows per shard
    shard_rg_map = defaultdict(lambda: defaultdict(set))
    unmatched_pattern = set()

    for rid in still_needed:
        if rid.startswith("sea_bench_clean_v2_train_"):
            try:
                row_idx = int(rid.split("_")[-1])
                shard_idx = row_idx // 15000
                rg_idx = (row_idx % 15000) // 200
                shard_rg_map[shard_idx][rg_idx].add(rid)
            except ValueError:
                unmatched_pattern.add(rid)
        else:
            unmatched_pattern.add(rid)

    if shard_rg_map:
        total_shards = len(shard_rg_map)
        total_rgs = sum(len(rgs) for rgs in shard_rg_map.values())
        logger.info(
            f"Targeted shard extraction: {len(still_needed) - len(unmatched_pattern)} clips "
            f"span {total_shards} shards and {total_rgs} row groups."
        )

        try:
            fs = fsspec.filesystem("hf")
            for shard_idx, rgs_dict in sorted(shard_rg_map.items()):
                shard_path = f"datasets/{DATASET_NAME}/data/train/train-{shard_idx:05d}.parquet"
                needed_rgs = sorted(rgs_dict.keys())
                target_in_shard = set().union(*rgs_dict.values())
                logger.info(
                    f"  Opening shard {shard_idx:05d} ({len(needed_rgs)} row groups, "
                    f"{len(target_in_shard)} target clips)..."
                )

                with fs.open(shard_path, "rb") as f:
                    pf = pq.ParquetFile(f)
                    
                    # Batch row group reads (5 at a time for optimal I/O)
                    chunk_size = 5
                    for i in range(0, len(needed_rgs), chunk_size):
                        chunk = needed_rgs[i : i + chunk_size]
                        tbl = pf.read_row_groups(chunk, columns=["row_id", "audio"])
                        df_chunk = tbl.to_pandas()
                        for _, row in df_chunk.iterrows():
                            rid = row["row_id"]
                            if rid in still_needed:
                                audio_dict = row["audio"]
                                if isinstance(audio_dict, dict) and audio_dict.get("bytes"):
                                    wf, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
                                    wf_tensor = torch.tensor(wf, dtype=torch.float32)
                                    cache_path = cache_dir / f"{rid}.pt"
                                    torch.save({"waveform": wf_tensor, "sampling_rate": sr}, cache_path)
                                    found[rid] = cache_path
                                    still_needed.discard(rid)

                logger.info(
                    f"  Shard {shard_idx:05d} complete. Progress: "
                    f"{len(found)}/{len(target_ids)} cached."
                )

        except Exception as e:
            logger.warning(f"Targeted shard extraction encountered error: {e}. Falling back to sequential stream.")

    # Fallback to streaming dataset if any IDs remain
    if still_needed:
        from datasets import load_dataset
        import datasets

        logger.info(f"Fallback streaming for remaining {len(still_needed)} clips from {DATASET_NAME}...")
        try:
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            ds = ds.cast_column("audio", datasets.Audio(decode=False))
        except Exception as e:
            logger.error(f"Failed to load streaming dataset: {e}")
            raise

        streamed_count = 0
        id_column = None
        for example in ds:
            streamed_count += 1
            if id_column is None:
                if "row_id" in example:
                    id_column = "row_id"
                elif "utterance_id" in example:
                    id_column = "utterance_id"
                else:
                    id_column = "row_id"

            example_id = example.get(id_column)
            if example_id in still_needed:
                audio_data = example.get("audio", {})
                if isinstance(audio_data, dict) and audio_data.get("bytes"):
                    wf, sr = sf.read(io.BytesIO(audio_data["bytes"]))
                    wf_tensor = torch.tensor(wf, dtype=torch.float32)
                    cache_path = cache_dir / f"{example_id}.pt"
                    torch.save({"waveform": wf_tensor, "sampling_rate": sr}, cache_path)
                    found[example_id] = cache_path
                    still_needed.discard(example_id)

            if not still_needed:
                break

    logger.info(f"Audio caching complete: {len(found)}/{len(target_ids)} clips ready.")
    return found


def load_cached_audio(row_id: str, cache_dir: Path = CACHE_DIR) -> torch.Tensor:
    """Load a cached audio waveform by row_id.

    Args:
        row_id: The row_id of the cached clip.
        cache_dir: Cache directory.

    Returns:
        1D float32 tensor of audio samples.
    """
    cache_path = cache_dir / f"{row_id}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached audio not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    return data["waveform"]
