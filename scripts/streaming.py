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
CACHE_DIR = Path("/Users/yashgarg/Documents/bsdetector/B.S-Detector/data/deepfake_cache/audio")


def stream_and_cache_audio(
    target_ids: set,
    cache_dir: Path = CACHE_DIR,
    verify_first_n: int = 5,
) -> dict:
    """Stream HF dataset and cache audio for target row_ids.

    Performs a single pass over the streaming dataset, checking each
    example's row_id against the target set. When matched, saves the
    raw waveform to disk as a .pt file.

    Args:
        target_ids: Set of row_id strings to capture.
        cache_dir: Directory to save cached audio files.
        verify_first_n: Print the first N streamed examples for ID verification.

    Returns:
        Dict mapping row_id -> Path to cached .pt file for all found IDs.
    """
    from datasets import load_dataset

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
        f"Need to stream {len(still_needed)} clips "
        f"({len(already_cached)} already cached)"
    )

    logger.info(f"Loading streaming dataset: {DATASET_NAME}")
    try:
        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        logger.error(
            f"Failed to load streaming dataset. "
            f"Ensure you have access (huggingface-cli login) and the dataset "
            f"'{DATASET_NAME}' is accessible. Error: {e}"
        )
        raise

    found = dict(already_cached)
    streamed_count = 0
    matched_count = len(already_cached)
    start_time = time.time()
    last_log_time = start_time

    # Determine the ID column name from the first example
    id_column = None

    for example in ds:
        streamed_count += 1

        # On first example, determine which column matches our IDs
        if id_column is None:
            if "row_id" in example and example["row_id"] in target_ids:
                id_column = "row_id"
            elif "utterance_id" in example and example["utterance_id"] in target_ids:
                id_column = "utterance_id"
            else:
                # Check both columns
                if "row_id" in example:
                    id_column = "row_id"
                elif "utterance_id" in example:
                    id_column = "utterance_id"
                else:
                    logger.error(
                        f"Dataset has neither 'row_id' nor 'utterance_id'. "
                        f"Available columns: {list(example.keys())}"
                    )
                    raise KeyError("No matching ID column found")
            logger.info(f"Using ID column: '{id_column}'")

        # Verify first N examples
        if streamed_count <= verify_first_n:
            example_id = example.get(id_column, "N/A")
            logger.info(
                f"  Stream example #{streamed_count}: "
                f"{id_column}='{example_id}', "
                f"in_target={example_id in target_ids}"
            )

        # Check if this example's ID is in our target set
        example_id = example.get(id_column)
        if example_id is None:
            continue

        if example_id in still_needed:
            # Extract and save audio
            audio_data = example.get("audio", {})
            if isinstance(audio_data, dict):
                waveform = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 16000)
            else:
                logger.warning(
                    f"Unexpected audio format for {example_id}: {type(audio_data)}"
                )
                continue

            if waveform is None:
                logger.warning(f"No audio array for {example_id}, skipping")
                continue

            # Convert to tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            else:
                waveform = waveform.float()

            # Resample if not 16kHz (SEA-Spoof should already be 16kHz)
            if sr != 16000:
                logger.warning(
                    f"Audio {example_id} has sr={sr}, expected 16000. "
                    f"Consider resampling."
                )

            # Save to cache
            cache_path = cache_dir / f"{example_id}.pt"
            torch.save({"waveform": waveform, "sampling_rate": sr}, cache_path)
            found[example_id] = cache_path
            still_needed.discard(example_id)
            matched_count += 1

        # Progress logging every 30 seconds
        now = time.time()
        if now - last_log_time >= 30:
            elapsed = now - start_time
            rate = streamed_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Streamed {streamed_count:,} examples "
                f"({rate:.0f}/sec), "
                f"matched {matched_count}/{len(target_ids)}, "
                f"remaining {len(still_needed)}"
            )
            last_log_time = now

        # Early exit if all needed IDs found
        if not still_needed:
            logger.info(
                f"All {len(target_ids)} target IDs found after "
                f"streaming {streamed_count:,} examples"
            )
            break

    elapsed = time.time() - start_time
    logger.info(
        f"Streaming complete: {streamed_count:,} examples in {elapsed:.0f}s, "
        f"matched {matched_count}/{len(target_ids)}"
    )

    if still_needed:
        logger.warning(
            f"{len(still_needed)} target IDs NOT found in stream: "
            f"{list(still_needed)[:10]}..."
        )

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
