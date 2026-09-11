"""Orchestrator script for audio deepfake detection experiments.

Runs the full pipeline:
  1. Load/generate Hindi metadata
  2. Create test and few-shot splits
  3. Stream and cache audio from HuggingFace
  4. Extract and cache XLS-R features
  5. Zero-shot evaluation
  6. Few-shot experiments (N=10, N=50 across 3 seeds)
  7. Print comparison table

Usage:
  python -m scripts.run_deepfake_detection [options]
  python scripts/run_deepfake_detection.py [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for both invocation styles
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from the scripts package
# ---------------------------------------------------------------------------
from scripts.splits import (
    create_fewshot_pools,
    create_test_split,
    get_all_needed_ids,
    get_labels_for_ids,
    load_hindi_metadata,
)
from scripts.streaming import stream_and_cache_audio
from scripts.features import extract_and_cache_features, FEATURE_CACHE_DIR
from scripts.experiment import evaluate_zeroshot, run_fewshot_experiment
from scripts.metrics import print_comparison_table


# ---------------------------------------------------------------------------
# Metadata generation (if parquet doesn't exist locally)
# ---------------------------------------------------------------------------
METADATA_COLUMNS = [
    "row_id", "utterance_id", "text", "language", "label", "spoof_type",
    "category", "split", "text_source", "mapping_source", "text_granularity",
    "is_text_exact", "source_model", "source_dataset", "speaker_or_voice",
    "sampling_rate", "audio_was_resampled",
]


def generate_metadata_parquet(output_path: Path, dataset_name: str = "Jack-ppkdczgx/SEA-Spoof") -> Path:
    """Stream the full dataset (metadata only, no audio) and save as parquet.

    This replaces the pre-existing parquet that was generated on a different
    machine. Uses select_columns to avoid downloading audio bytes.

    Args:
        output_path: Where to save the parquet file.
        dataset_name: HuggingFace dataset identifier.

    Returns:
        Path to the saved parquet file.
    """
    import pandas as pd
    from datasets import load_dataset

    logger = logging.getLogger(__name__)
    logger.info(f"Generating metadata parquet from {dataset_name} (streaming, metadata only)...")

    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.select_columns(METADATA_COLUMNS)

    rows = []
    for i, sample in enumerate(ds):
        rows.append(sample)
        if (i + 1) % 10000 == 0:
            logger.info(f"  Streamed {i + 1:,} metadata rows...")

    logger.info(f"  Total rows streamed: {len(rows):,}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Metadata parquet saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run audio deepfake detection experiments (zero-shot + few-shot)"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Path to sea_spoof_en_hi_metadata.parquet. If not found, generates it.",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=str,
        default=None,
        help="Directory for cached audio .pt files.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default=None,
        help="Directory for cached feature .pt files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for few-shot fine-tuning.",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip audio streaming (assume already cached).",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction (assume already cached).",
    )
    args = parser.parse_args()

    # ---- Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("deepfake_detection")

    # ---- Resolve paths ----
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.parquet_path) if args.parquet_path else data_dir / "sea_spoof_en_hi_metadata.parquet"
    audio_cache_dir = Path(args.audio_cache_dir) if args.audio_cache_dir else data_dir / "deepfake_cache" / "audio"
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else data_dir / "deepfake_cache" / "features"

    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Audio Deepfake Detection — XLS-R + SLS Pipeline")
    logger.info("=" * 70)
    logger.info(f"Parquet:          {parquet_path}")
    logger.info(f"Audio cache:      {audio_cache_dir}")
    logger.info(f"Feature cache:    {feature_cache_dir}")
    logger.info(f"Device:           {args.device}")
    logger.info(f"Epochs:           {args.epochs}")

    # ---- Step 1: Load metadata ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Load Hindi metadata")
    logger.info("=" * 70)

    if not parquet_path.exists():
        logger.info(f"Parquet not found at {parquet_path}, generating from HuggingFace...")
        generate_metadata_parquet(parquet_path)

    df_hindi = load_hindi_metadata(parquet_path)

    # ---- Step 2: Create splits ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Create test and few-shot splits")
    logger.info("=" * 70)

    test_ids, test_df = create_test_split(df_hindi)
    fewshot_pools = create_fewshot_pools(df_hindi, test_ids)
    all_needed_ids = get_all_needed_ids(test_ids, fewshot_pools)
    labels_dict = get_labels_for_ids(df_hindi, all_needed_ids)

    logger.info(f"Test set: {len(test_ids)} samples")
    logger.info(f"Total unique IDs needed: {len(all_needed_ids)}")
    logger.info(f"Labels extracted: {len(labels_dict)}")

    # ---- Step 3: Stream and cache audio ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Stream and cache audio")
    logger.info("=" * 70)

    if args.skip_streaming:
        logger.info("Skipping audio streaming (--skip-streaming)")
    else:
        t0 = time.time()
        found = stream_and_cache_audio(all_needed_ids, cache_dir=audio_cache_dir)
        logger.info(f"Audio streaming completed in {time.time() - t0:.0f}s, found {len(found)} clips")

        missing = all_needed_ids - set(found.keys())
        if missing:
            logger.warning(f"{len(missing)} IDs not found in stream!")
            # Remove missing IDs from experiment sets
            test_ids -= missing
            for key in list(fewshot_pools.keys()):
                fewshot_pools[key]["row_ids"] -= missing

    # ---- Step 4: Extract and cache features ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Extract and cache XLS-R features")
    logger.info("=" * 70)

    if args.skip_features:
        logger.info("Skipping feature extraction (--skip-features)")
    else:
        t0 = time.time()
        extract_and_cache_features(
            row_ids=list(all_needed_ids),
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            audio_cache_dir=audio_cache_dir,
        )
        logger.info(f"Feature extraction completed in {time.time() - t0:.0f}s")

    # ---- Step 5: Zero-shot evaluation ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Zero-shot evaluation")
    logger.info("=" * 70)

    zeroshot_metrics = evaluate_zeroshot(
        test_row_ids=test_ids,
        labels_dict=labels_dict,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
    )

    # ---- Step 6: Few-shot experiments ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Few-shot experiments")
    logger.info("=" * 70)

    fewshot_10_metrics = run_fewshot_experiment(
        n=10,
        fewshot_pools=fewshot_pools,
        test_row_ids=test_ids,
        labels_dict=labels_dict,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
        epochs=args.epochs,
    )

    fewshot_50_metrics = run_fewshot_experiment(
        n=50,
        fewshot_pools=fewshot_pools,
        test_row_ids=test_ids,
        labels_dict=labels_dict,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
        epochs=args.epochs,
    )

    # ---- Step 7: Comparison table ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Final comparison table")
    logger.info("=" * 70)

    results = {
        "Zero-shot": [zeroshot_metrics],
        "Few-shot N=10": fewshot_10_metrics,
        "Few-shot N=50": fewshot_50_metrics,
    }
    print("\n")
    print_comparison_table(results)

    logger.info("\n" + "=" * 70)
    logger.info("All experiments complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
