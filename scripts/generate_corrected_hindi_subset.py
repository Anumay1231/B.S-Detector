"""Generate a corrected Hindi subset from SEA-Spoof with proper random sampling.

Fixes the shard/label confound in the original data extraction by:
  1. Streaming ALL metadata from the full SEA-Spoof dataset (no audio yet)
  2. Collecting every Hindi row
  3. Sampling a balanced subset (N_PER_CLASS spoof + N_PER_CLASS bonafide)
     stratified by source_model so TTS diversity is proportionally represented
  4. Saving the corrected metadata parquet
  5. Printing diversity diagnostics for verification

Usage:
    python scripts/generate_corrected_hindi_subset.py [--n-per-class 400] [--seed 42]

After running this script:
  1. Delete old caches:  data/deepfake_cache/audio/  and  data/deepfake_cache/features/
  2. Re-run the full pipeline:
     .\.venv\Scripts\python.exe scripts\run_deepfake_detection.py --device cuda --test-per-class 250
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "Jack-ppkdczgx/SEA-Spoof"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "sea_spoof_en_hi_metadata_v2.parquet"

METADATA_COLUMNS = [
    "row_id", "utterance_id", "text", "language", "label", "spoof_type",
    "category", "split", "text_source", "mapping_source", "text_granularity",
    "is_text_exact", "source_model", "source_dataset", "speaker_or_voice",
    "sampling_rate", "audio_was_resampled",
]


def collect_hindi_metadata(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    """Read ALL Hindi metadata from SEA-Spoof parquet shards (no audio download).

    Uses fsspec's HuggingFace filesystem to read only metadata columns
    from each parquet shard, completely skipping the massive audio column.
    This is orders of magnitude faster than streaming the full dataset.

    Returns:
        DataFrame of all Hindi rows with metadata columns only.
    """
    import fsspec
    import pyarrow.parquet as pq

    logger = logging.getLogger(__name__)
    logger.info(f"Collecting Hindi metadata from {dataset_name} (metadata-only, no audio)...")

    fs = fsspec.filesystem("hf")

    # List all train parquet shards
    shard_pattern = f"datasets/{dataset_name}/data/train"
    try:
        all_files = fs.ls(shard_pattern, detail=False)
    except Exception as e:
        logger.error(f"Failed to list shards: {e}")
        raise

    shard_files = sorted([f for f in all_files if f.endswith(".parquet")])
    logger.info(f"Found {len(shard_files)} train shards")

    all_hindi_dfs = []
    total_rows = 0

    for i, shard_path in enumerate(shard_files):
        logger.info(f"  Reading shard {i+1}/{len(shard_files)}: {shard_path.split('/')[-1]}...")
        start = time.time()

        try:
            with fs.open(shard_path, "rb") as f:
                pf = pq.ParquetFile(f)
                # Read only metadata columns (skip 'audio' which is huge)
                available_cols = pf.schema_arrow.names
                cols_to_read = [c for c in METADATA_COLUMNS if c in available_cols]

                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=cols_to_read)
                    df_rg = table.to_pandas()
                    total_rows += len(df_rg)

                    # Filter for Hindi
                    if "language" in df_rg.columns:
                        df_hindi_rg = df_rg[df_rg["language"] == "hi"]
                        if len(df_hindi_rg) > 0:
                            all_hindi_dfs.append(df_hindi_rg)
                            logger.info(
                                f"    RG {rg_idx}: {len(df_rg)} rows, "
                                f"{len(df_hindi_rg)} Hindi"
                            )

            elapsed = time.time() - start
            logger.info(f"    Shard done in {elapsed:.1f}s")

        except Exception as e:
            logger.warning(f"    Failed to read shard {shard_path}: {e}")
            continue

    if not all_hindi_dfs:
        raise ValueError("No Hindi rows found in any shard!")

    df = pd.concat(all_hindi_dfs, ignore_index=True)
    logger.info(
        f"Collection complete: {total_rows:,} total rows scanned, "
        f"{len(df)} Hindi rows collected from {len(all_hindi_dfs)} row groups"
    )
    return df


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample n_per_class from each label, stratified by source_model.

    For spoof clips: proportional representation of each TTS source_model.
    For bonafide clips: simple random sample (typically one source_model).

    Args:
        df: Full Hindi metadata DataFrame.
        n_per_class: Number of samples per class (spoof, bonafide).
        seed: Random seed for reproducibility.

    Returns:
        Balanced DataFrame with n_per_class * 2 rows.
    """
    logger = logging.getLogger(__name__)
    rng = np.random.RandomState(seed)

    sampled_parts = []

    for label in ["spoof", "bonafide"]:
        label_df = df[df["label"] == label].copy()
        if len(label_df) < n_per_class:
            raise ValueError(
                f"Not enough {label} samples: have {len(label_df)}, need {n_per_class}"
            )

        # Count source_model diversity
        model_counts = label_df["source_model"].value_counts()
        logger.info(f"  {label}: {len(label_df)} total, {len(model_counts)} source_models")
        for model, count in model_counts.items():
            logger.info(f"    {model}: {count}")

        if len(model_counts) > 1:
            # Stratified: proportional allocation across source_models
            model_names = model_counts.index.tolist()
            total_in_label = len(label_df)

            # Proportional allocation (at least 1 per model)
            allocations = {}
            remaining = n_per_class
            for model in model_names:
                alloc = max(1, int(round(n_per_class * model_counts[model] / total_in_label)))
                alloc = min(alloc, model_counts[model])  # can't sample more than available
                allocations[model] = alloc
                remaining -= alloc

            # Distribute any remainder to the largest groups
            while remaining > 0:
                for model in model_names:
                    if remaining <= 0:
                        break
                    if allocations[model] < model_counts[model]:
                        allocations[model] += 1
                        remaining -= 1

            # Handle over-allocation
            while remaining < 0:
                for model in reversed(model_names):
                    if remaining >= 0:
                        break
                    if allocations[model] > 1:
                        allocations[model] -= 1
                        remaining += 1

            logger.info(f"  Stratified allocations for {label}:")
            for model, alloc in allocations.items():
                logger.info(f"    {model}: {alloc}/{model_counts[model]}")

            for model, alloc in allocations.items():
                model_df = label_df[label_df["source_model"] == model]
                sampled = model_df.sample(n=alloc, random_state=rng)
                sampled_parts.append(sampled)
        else:
            # Single source_model: simple random sample
            sampled = label_df.sample(n=n_per_class, random_state=rng)
            sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, ignore_index=True)
    logger.info(f"  Final sample: {len(result)} rows")
    return result


def print_diagnostics(df: pd.DataFrame, title: str = "CORRECTED SUBSET DIAGNOSTICS"):
    """Print diversity diagnostics for verification."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")
    print(f"\nLabel distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  {label}: {count}")

    for label in ["spoof", "bonafide"]:
        subset = df[df["label"] == label]
        print(f"\n--- {label.upper()} ({len(subset)} clips) ---")

        print(f"  source_model:")
        for model, count in subset["source_model"].value_counts().items():
            pct = 100 * count / len(subset)
            print(f"    {model or '(empty)'}: {count} ({pct:.1f}%)")

        print(f"  source_dataset:")
        for ds, count in subset["source_dataset"].value_counts().items():
            pct = 100 * count / len(subset)
            print(f"    {ds or '(empty)'}: {count} ({pct:.1f}%)")

        print(f"  sampling_rate:")
        for sr, count in subset["sampling_rate"].value_counts().items():
            print(f"    {sr}: {count}")

        print(f"  audio_was_resampled:")
        for val, count in subset["audio_was_resampled"].value_counts().items():
            print(f"    {val}: {count}")

    # Check for row_id range spread
    if "row_id" in df.columns:
        for label in ["spoof", "bonafide"]:
            ids = sorted(df[df["label"] == label]["row_id"].tolist())
            print(f"\n  {label} row_id range: [{ids[0]} ... {ids[-1]}]")
            # Check if they're contiguous (bad) or spread out (good)
            try:
                nums = [int(rid.split("_")[-1]) for rid in ids]
                spread = max(nums) - min(nums)
                density = len(nums) / (spread + 1) if spread > 0 else 1.0
                print(f"    Numeric spread: {spread}, density: {density:.4f}")
                if density > 0.9:
                    print("    ⚠️  WARNING: Row IDs are nearly contiguous — possible shard clustering!")
                else:
                    print("    ✓  Row IDs are spread out — good diversity signal")
            except (ValueError, IndexError):
                pass

    print("\n" + "=" * 70)


def main():
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Generate corrected Hindi subset with proper random sampling"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=400,
        help="Number of samples per class (default: 400 -> 800 total)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Only audit the full Hindi pool, don't create subset"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("corrected_sampling")

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    # Step 1: Collect all Hindi metadata
    logger.info("=" * 70)
    logger.info("STEP 1: Collect all Hindi metadata from SEA-Spoof")
    logger.info("=" * 70)

    df_hindi_full = collect_hindi_metadata()

    print_diagnostics(df_hindi_full, "FULL HINDI POOL DIAGNOSTICS")

    if args.audit_only:
        logger.info("Audit-only mode. Exiting.")
        return

    # Step 2: Stratified sampling
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Stratified random sampling")
    logger.info("=" * 70)

    df_sampled = stratified_sample(df_hindi_full, args.n_per_class, args.seed)

    # Step 3: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sampled.to_parquet(output_path, index=False)
    logger.info(f"Saved corrected metadata to {output_path}")

    # Step 4: Diagnostics
    print_diagnostics(df_sampled, "CORRECTED SUBSET DIAGNOSTICS")

    # Step 5: Compare with old
    old_path = output_path.parent / "sea_spoof_en_hi_metadata.parquet"
    if old_path.exists():
        df_old = pd.read_parquet(old_path)
        overlap = set(df_old["row_id"]) & set(df_sampled["row_id"])
        print(f"\nOverlap with old v1 subset: {len(overlap)}/{len(df_old)} row_ids")
        print("(Some overlap is expected by chance; the key is diversity, not disjointness)")

    logger.info("\n" + "=" * 70)
    logger.info("DONE. Next steps:")
    logger.info("  1. Delete old caches:")
    logger.info("     rmdir /s /q data\\deepfake_cache\\audio")
    logger.info("     rmdir /s /q data\\deepfake_cache\\features")
    logger.info("  2. Re-run pipeline with corrected data:")
    logger.info("     ..\\.venv\\Scripts\\python.exe scripts\\run_deepfake_detection.py \\")
    logger.info("       --device cuda --test-per-class 250 \\")
    logger.info("       --parquet-path data\\sea_spoof_en_hi_metadata_v2.parquet")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
