"""Extract and cache XLS-R hidden states for audio clips.

Runs the frozen XLS-R 300M backbone on cached audio waveforms and saves
the full 24-layer hidden states to disk as .pt files.  Re-running skips
clips that are already cached.
"""

import logging
import time
from pathlib import Path
from typing import List

import torch

from .model import (
    INPUT_LENGTH,
    extract_hidden_states,
    load_backbone,
    pad_or_truncate,
)
from .streaming import load_cached_audio

logger = logging.getLogger(__name__)

FEATURE_CACHE_DIR = Path(
    "/Users/yashgarg/Documents/bsdetector/B.S-Detector/data/deepfake_cache/features"
)


def extract_and_cache_features(
    row_ids: list,
    backbone=None,
    device: str = "cpu",
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
    audio_cache_dir: Path = None,
) -> None:
    """Extract XLS-R hidden states for a list of clips and cache to disk.

    For each row_id, loads the cached audio, pads/truncates to INPUT_LENGTH,
    runs the frozen backbone, and saves the 24-layer hidden states as a
    stacked float16 tensor.

    Args:
        row_ids: List of row_id strings whose audio is already cached.
        backbone: Pre-loaded Wav2Vec2Model. If None, loads it fresh.
        device: Device for backbone inference.
        feature_cache_dir: Where to save feature .pt files.
        audio_cache_dir: Where cached audio lives (uses default if None).
    """
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which clips still need extraction
    needed = []
    for rid in row_ids:
        feat_path = feature_cache_dir / f"{rid}.pt"
        if not feat_path.exists():
            needed.append(rid)

    if not needed:
        logger.info(
            f"All {len(row_ids)} feature files already cached in "
            f"{feature_cache_dir}"
        )
        return

    logger.info(
        f"Extracting features for {len(needed)} clips "
        f"({len(row_ids) - len(needed)} already cached)"
    )

    # Load backbone if not provided
    if backbone is None:
        backbone = load_backbone(device=device)

    start_time = time.time()
    batch_size = 2

    for start_idx in range(0, len(needed), batch_size):
        batch_rids = needed[start_idx : start_idx + batch_size]
        waveforms = []
        for rid in batch_rids:
            if audio_cache_dir is not None:
                waveform = load_cached_audio(rid, cache_dir=audio_cache_dir)
            else:
                waveform = load_cached_audio(rid)
            waveform = pad_or_truncate(waveform, INPUT_LENGTH)
            waveforms.append(waveform)

        waveform_batch = torch.stack(waveforms, dim=0)
        hidden_states = extract_hidden_states(backbone, waveform_batch, device=device)

        for b_idx, rid in enumerate(batch_rids):
            stacked = torch.stack(
                [hs[b_idx].detach().cpu().half() for hs in hidden_states]
            )  # (24, T, 1024) on CPU
            feat_path = feature_cache_dir / f"{rid}.pt"
            torch.save(stacked, feat_path)

        del hidden_states, waveform_batch
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        curr_count = min(start_idx + batch_size, len(needed))
        if curr_count % 10 == 0 or curr_count == len(needed):
            elapsed = time.time() - start_time
            rate = curr_count / elapsed if elapsed > 0 else 0
            eta = (len(needed) - curr_count) / rate if rate > 0 else 0
            logger.info(
                f"  Extracted {curr_count}/{len(needed)} "
                f"({rate:.1f} clips/sec, ETA {eta:.0f}s)"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Feature extraction complete: {len(needed)} clips in {elapsed:.0f}s"
    )


def load_cached_features(
    row_id: str,
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
) -> List[torch.Tensor]:
    """Load cached hidden states for a single clip.

    Args:
        row_id: The row_id of the clip.
        feature_cache_dir: Directory containing cached feature files.

    Returns:
        List of 24 tensors, each (T, 1024) in float32.
    """
    feat_path = feature_cache_dir / f"{row_id}.pt"
    if not feat_path.exists():
        raise FileNotFoundError(f"Cached features not found: {feat_path}")

    # Load stacked tensor (24, T, 1024) in float16 and convert to float32
    stacked = torch.load(feat_path, map_location="cpu", weights_only=True).float()

    # Split back into list of 24 tensors
    return [stacked[i] for i in range(stacked.shape[0])]


def load_features_batch(
    row_ids: list,
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
) -> List[List[torch.Tensor]]:
    """Load cached features for multiple clips.

    Args:
        row_ids: List of row_id strings.
        feature_cache_dir: Directory containing cached feature files.

    Returns:
        List (one per clip) of lists (24 layers) of tensors (T, 1024).
    """
    return [load_cached_features(rid, feature_cache_dir) for rid in row_ids]


def collate_features(
    features_list: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """Collate a batch of per-clip features into batched layer tensors.

    Args:
        features_list: List of B items, each a list of 24 (T, D) tensors.

    Returns:
        List of 24 tensors, each (B, T, D).
    """
    num_layers = len(features_list[0])
    batched = []
    for layer_idx in range(num_layers):
        layer_tensors = [clip[layer_idx] for clip in features_list]
        batched.append(torch.stack(layer_tensors, dim=0))
    return batched
