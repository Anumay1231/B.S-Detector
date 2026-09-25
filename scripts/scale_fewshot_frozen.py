#!/usr/bin/env python3
"""
Scale Few-Shot Adaptation on Frozen XLS-R 300M Backbone.

Evaluates SLS Head adaptation across scaling regimes:
    N in [10, 25, 50, 100, 200] per class across 3 deterministic seeds.

Extracts features on GPU once, then runs fast head-only fine-tuning
in seconds, computing full Accuracy, EER, Precision, Recall, F1 curves.

Usage:
    python scripts/scale_fewshot_frozen.py --device cuda --fewshot-sizes 10 25 50 100 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

# Ensure project root is in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.experiment import evaluate_zeroshot, run_fewshot_experiment
from scripts.features import (
    FEATURE_CACHE_DIR,
    extract_and_cache_features,
)
from scripts.metrics import print_comparison_table
from scripts.model import load_backbone
from scripts.splits import (
    create_fewshot_pools,
    create_test_split,
    get_all_needed_ids,
    get_labels_for_ids,
    load_hindi_metadata,
)
from scripts.streaming import CACHE_DIR

logger = logging.getLogger("scale_fewshot")


def main():
    parser = argparse.ArgumentParser(description="Scale Few-Shot Adaptation on Frozen XLS-R Backbone")
    parser.add_argument("--parquet-path", type=str, default="data/sea_spoof_en_hi_metadata.parquet")
    parser.add_argument("--audio-cache-dir", type=str, default="data/deepfake_cache/audio")
    parser.add_argument("--feature-cache-dir", type=str, default="data/deepfake_cache/features")
    parser.add_argument("--fewshot-sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200], help="List of N values to test")
    parser.add_argument("--test-per-class", type=int, default=250, help="Test samples per class (default: 250)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for SLS head (default: 10)")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="outputs/results_scale_n_frozen.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parquet_path = _PROJECT_ROOT / args.parquet_path
    audio_cache_dir = _PROJECT_ROOT / args.audio_cache_dir
    feature_cache_dir = _PROJECT_ROOT / args.feature_cache_dir
    output_json = _PROJECT_ROOT / args.output_json

    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("FEW-SHOT SCALING ON FROZEN XLS-R 300M BACKBONE")
    logger.info("=" * 75)
    logger.info(f"Parquet path:      {parquet_path}")
    logger.info(f"Audio cache dir:   {audio_cache_dir}")
    logger.info(f"Feature cache dir: {feature_cache_dir}")
    logger.info(f"Fewshot sizes (N): {args.fewshot_sizes}")
    logger.info(f"Epochs:            {args.epochs}")
    logger.info(f"Device:            {args.device}")

    # 1. Load Metadata & Generate Splits
    df_hindi = load_hindi_metadata(parquet_path)
    test_ids, test_df = create_test_split(df_hindi, test_per_class=args.test_per_class, seed=42)
    fewshot_pools = create_fewshot_pools(
        df_hindi,
        test_ids=test_ids,
        fewshot_sizes=args.fewshot_sizes,
        num_seeds=args.num_seeds,
    )
    all_needed_ids = get_all_needed_ids(test_ids, fewshot_pools)
    labels_dict = get_labels_for_ids(df_hindi, all_needed_ids)

    # 2. Extract & Cache Features on GPU for all cached audio
    # Check which audio clips are available
    available_audio_ids = [
        rid for rid in all_needed_ids if (audio_cache_dir / f"{rid}.pt").exists()
    ]
    logger.info(f"Audio clips available on disk: {len(available_audio_ids)}/{len(all_needed_ids)}")

    # Extract 24-layer features using frozen XLS-R backbone (GPU accelerated)
    logger.info("\nExtracting & verifying XLS-R 300M hidden features...")
    backbone = load_backbone(device=args.device)
    extract_and_cache_features(
        row_ids=available_audio_ids,
        backbone=backbone,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
        audio_cache_dir=audio_cache_dir,
    )
    # Free backbone memory
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Filter test set and pools to verified cached features
    cached_feats = {
        rid for rid in available_audio_ids if (feature_cache_dir / f"{rid}.pt").exists()
    }
    effective_test_ids = [rid for rid in test_ids if rid in cached_feats]
    logger.info(f"Effective test set size: {len(effective_test_ids)} samples.")

    # 3. Step A: Zero-Shot Evaluation (Baseline)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Zero-Shot Baseline Evaluation")
    logger.info("=" * 70)
    zeroshot_metrics = evaluate_zeroshot(
        test_row_ids=effective_test_ids,
        labels_dict=labels_dict,
        device=args.device,
        feature_cache_dir=feature_cache_dir,
    )

    # 4. Step B: Scaling Few-Shot Adaptation (N = 10, 25, 50, 100, 200)
    all_results = {"Zero-shot": [zeroshot_metrics]}

    for n in args.fewshot_sizes:
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 2: Few-Shot Experiment (N={n} per class, 3 Seeds)")
        logger.info("=" * 70)

        # Filter pools for this N to available cached features
        filtered_pools = {}
        for (pool_n, s_idx), data in fewshot_pools.items():
            if pool_n == n:
                valid_ids = {rid for rid in data["row_ids"] if rid in cached_feats}
                filtered_pools[(pool_n, s_idx)] = {"row_ids": valid_ids}

        n_metrics = run_fewshot_experiment(
            n=n,
            fewshot_pools=filtered_pools,
            test_row_ids=effective_test_ids,
            labels_dict=labels_dict,
            num_seeds=args.num_seeds,
            device=args.device,
            feature_cache_dir=feature_cache_dir,
            epochs=args.epochs,
            batch_size=4,
        )
        all_results[f"Few-shot N={n}"] = n_metrics

    # 5. Step C: Print Consolidated Comparison Table
    logger.info("\n" + "=" * 75)
    logger.info("FINAL CONSOLIDATED RESULTS ACROSS SCALING REGIMES")
    logger.info("=" * 75)
    print("\n")
    print_comparison_table(all_results)

    # 6. Save JSON
    clean_results = {}
    for exp_name, m_list in all_results.items():
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

    with open(output_json, "w") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"\nScaling results successfully saved to {output_json}")


if __name__ == "__main__":
    main()
