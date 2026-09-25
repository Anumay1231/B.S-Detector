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
import json
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


def generate_metadata_parquet(output_path: Path, audio_cache_dir: Optional[Path] = None, dataset_name: str = "Jack-ppkdczgx/SEA-Spoof") -> Path:
    """Fetch targeted Hindi bonafide and spoof samples from SEA-Spoof shards.

    Reads 200 Hindi spoof rows from train shard 0 and 200 Hindi bonafide rows
    from train shard 2 without downloading the full dataset. Also caches the audio.

    Args:
        output_path: Where to save the parquet file.
        audio_cache_dir: Optional directory to save audio .pt files directly.
        dataset_name: HuggingFace dataset identifier.

    Returns:
        Path to the saved parquet file.
    """
    import io
    import fsspec
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.info(f"Fetching targeted Hindi samples (spoof & bonafide) from {dataset_name} shards...")

    fs = fsspec.filesystem("hf")
    if audio_cache_dir:
        audio_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch spoof from Shard 0, RG 0 and RG 1 (400 rows total)
    fpath_spoof = f"datasets/{dataset_name}/data/train/train-00000.parquet"
    logger.info("  Reading Hindi spoof from Shard 0 (RG 0 & 1)...")
    with fs.open(fpath_spoof, "rb") as f:
        pf = pq.ParquetFile(f)
        df_spoof_0 = pf.read_row_group(0).to_pandas()
        df_spoof_1 = pf.read_row_group(1).to_pandas()
        df_spoof = pd.concat([df_spoof_0, df_spoof_1], ignore_index=True)

    # 2. Fetch bonafide from Shard 2, RG 2 and RG 3 (400 rows total)
    fpath_bonafide = f"datasets/{dataset_name}/data/train/train-00002.parquet"
    logger.info("  Reading Hindi bonafide from Shard 2 (RG 2 & 3)...")
    with fs.open(fpath_bonafide, "rb") as f:
        pf = pq.ParquetFile(f)
        df_bonafide_2 = pf.read_row_group(2).to_pandas()
        df_bonafide_3 = pf.read_row_group(3).to_pandas()
        df_bonafide = pd.concat([df_bonafide_2, df_bonafide_3], ignore_index=True)

    meta_rows = []
    for df in [df_spoof, df_bonafide]:
        for idx, row in df.iterrows():
            rid = row["row_id"]
            if audio_cache_dir:
                audio_file = audio_cache_dir / f"{rid}.pt"
                if not audio_file.exists():
                    audio_dict = row["audio"]
                    waveform, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
                    wf_tensor = torch.tensor(waveform, dtype=torch.float32)
                    torch.save({"waveform": wf_tensor, "sampling_rate": sr}, audio_file)
            meta_row = {k: v for k, v in row.items() if k != "audio"}
            meta_rows.append(meta_row)

    df_all = pd.DataFrame(meta_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    logger.info(f"Targeted Hindi metadata ({len(df_all)} rows) saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

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
        "--test-per-class",
        type=int,
        default=None,
        help="Number of test samples per class (default: 50 for quick base tests, or 250).",
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
        logger.error(
            f"Parquet not found at {parquet_path}.\n"
            f"Run the corrected sampling script first:\n"
            f"  .venv\\Scripts\\python.exe scripts\\generate_corrected_hindi_subset.py\n"
            f"Then re-run this script with:\n"
            f"  --parquet-path data\\sea_spoof_en_hi_metadata_v2.parquet"
        )
        sys.exit(1)

    df_hindi = load_hindi_metadata(parquet_path)

    # ---- Step 2: Create splits ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Create test and few-shot splits")
    logger.info("=" * 70)

    bonafide_avail = int((df_hindi["label"] == "bonafide").sum())
    spoof_avail = int((df_hindi["label"] == "spoof").sum())
    min_avail = min(bonafide_avail, spoof_avail)

    if args.test_per_class is not None:
        test_per_class = args.test_per_class
    elif min_avail >= 300:
        test_per_class = 250
    else:
        test_per_class = min(50, max(10, min_avail - 60))

    logger.info(f"Using test_per_class={test_per_class} (available bonafide: {bonafide_avail}, spoof: {spoof_avail})")
    test_ids, test_df = create_test_split(df_hindi, test_per_class=test_per_class)
    fewshot_pools = create_fewshot_pools(df_hindi, test_ids, fewshot_sizes=[10, 50], num_seeds=3)
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

    # ---- Filter to only IDs with cached features ----
    cached_feats = {p.stem for p in feature_cache_dir.glob("*.pt")}
    missing_feats = all_needed_ids - cached_feats
    if missing_feats:
        logger.warning(f"{len(missing_feats)} IDs have no cached features and will be excluded.")
        test_ids -= missing_feats
        for key in list(fewshot_pools.keys()):
            fewshot_pools[key]["row_ids"] -= missing_feats
    logger.info(f"Effective test set: {len(test_ids)} samples")

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

    # Save to outputs directory
    outputs_dir = _PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_path = outputs_dir / "results_full_250.json"

    def _serialize(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    clean_results = {}
    for exp_name, m_list in results.items():
        clean_results[exp_name] = []
        for m in m_list:
            clean_m = {}
            for k, v in m.items():
                if hasattr(v, "tolist"):
                    clean_m[k] = v.tolist()
                elif hasattr(v, "item"):
                    clean_m[k] = v.item()
                else:
                    clean_m[k] = v
            clean_results[exp_name].append(clean_m)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("All experiments complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
