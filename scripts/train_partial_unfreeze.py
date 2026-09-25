#!/usr/bin/env python3
"""
Experiment 2: Partial Backbone Unfreezing on NVIDIA L4 GPU.

Fine-tunes the top N transformer layers of XLS-R 300M (e.g., layers 21-24)
together with the SLS classification head on the stratified V2 Hindi benchmark.

Uses differential learning rates:
- Unfrozen XLS-R backbone layers: 1e-6 (conservative fine-tuning)
- SLS Head: 1e-5 (rapid adaptation)

Usage:
    python scripts/train_partial_unfreeze.py --device cuda --unfreeze-layers 4 --fewshot-n 50
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.metrics import compute_all_metrics, print_comparison_table, print_metrics
from scripts.model import INPUT_LENGTH, pad_or_truncate
from scripts.splits import (
    create_fewshot_pools,
    create_test_split,
    get_all_needed_ids,
    get_labels_for_ids,
    load_hindi_metadata,
)
from scripts.streaming import CACHE_DIR, load_cached_audio
from scripts.unfreeze_model import PartialUnfreezeSLSModel

logger = logging.getLogger("partial_unfreeze")


def load_waveforms_batch(
    row_ids: Sequence[str],
    audio_cache_dir: Path = CACHE_DIR,
    target_length: int = INPUT_LENGTH,
) -> torch.Tensor:
    """Load cached audio waveforms, pad/truncate to fixed length, and batch them."""
    waveforms = []
    for rid in row_ids:
        wf = load_cached_audio(rid, cache_dir=audio_cache_dir)
        wf = pad_or_truncate(wf, target_length=target_length)
        waveforms.append(wf)
    return torch.stack(waveforms, dim=0)  # (B, 64600)


def evaluate_end_to_end(
    model: PartialUnfreezeSLSModel,
    test_row_ids: Sequence[str],
    labels_dict: dict,
    device: str = "cuda",
    batch_size: int = 8,
    audio_cache_dir: Path = CACHE_DIR,
) -> Dict[str, Any]:
    """Evaluate end-to-end model on the held-out test set."""
    model.eval()
    test_ids_list = list(test_row_ids)

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []

    logger.info(f"Evaluating end-to-end on {len(test_ids_list)} test samples...")
    with torch.no_grad():
        for start in range(0, len(test_ids_list), batch_size):
            batch_ids = test_ids_list[start : start + batch_size]
            waveforms = load_waveforms_batch(batch_ids, audio_cache_dir=audio_cache_dir).to(device)
            batch_labels = [labels_dict[rid] for rid in batch_ids]

            log_probs = model(waveforms)  # (B, 2)
            bonafide_scores = log_probs[:, 1].cpu().numpy().tolist()
            preds = torch.argmax(log_probs, dim=-1).cpu().numpy().tolist()

            all_labels.extend(batch_labels)
            all_scores.extend(bonafide_scores)
            all_preds.extend(preds)

    metrics = compute_all_metrics(
        labels=np.array(all_labels),
        scores=np.array(all_scores),
        predictions=np.array(all_preds),
    )
    return metrics


def train_single_seed(
    train_row_ids: Sequence[str],
    test_row_ids: Sequence[str],
    labels_dict: dict,
    unfreeze_layers: int = 4,
    device: str = "cuda",
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    backbone_lr: float = 1e-6,
    head_lr: float = 1e-5,
    audio_cache_dir: Path = CACHE_DIR,
    loss_weight: Sequence[float] = (0.1, 0.9),
) -> Dict[str, Any]:
    """Train partial unfreezing model on a single few-shot split and evaluate."""
    # 1. Initialize Model
    model = PartialUnfreezeSLSModel(
        unfreeze_layers=unfreeze_layers,
        pretrained_head_version="v1",
        gradient_checkpointing=True,
    ).to(device)

    # 2. Configure Optimizer with differential learning rates
    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=1e-4,
    )
    optimizer = torch.optim.AdamW(param_groups)

    weight_tensor = torch.tensor(loss_weight, dtype=torch.float32).to(device)
    criterion = nn.NLLLoss(weight=weight_tensor)

    train_ids_list = list(train_row_ids)
    train_labels = [labels_dict[rid] for rid in train_ids_list]
    indices = np.arange(len(train_ids_list))

    logger.info(
        f"Starting few-shot fine-tuning: {len(train_ids_list)} samples, "
        f"{epochs} epochs, batch_size={batch_size} (effective={batch_size * grad_accum_steps})."
    )

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_steps = 0

        optimizer.zero_grad()
        for step, start in enumerate(range(0, len(indices), batch_size)):
            batch_idx = indices[start : start + batch_size]
            batch_ids = [train_ids_list[i] for i in batch_idx]
            waveforms = load_waveforms_batch(batch_ids, audio_cache_dir=audio_cache_dir).to(device)
            targets = torch.tensor([train_labels[i] for i in batch_idx], dtype=torch.long).to(device)

            log_probs = model(waveforms)
            loss = criterion(log_probs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (start + batch_size >= len(indices)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:02d}/{epochs:02d} — avg loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s.")

    # 3. Evaluate
    metrics = evaluate_end_to_end(
        model=model,
        test_row_ids=test_row_ids,
        labels_dict=labels_dict,
        device=device,
        batch_size=8,
        audio_cache_dir=audio_cache_dir,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Partial Backbone Unfreezing")
    parser.add_argument("--parquet-path", type=str, default="data/sea_spoof_en_hi_metadata_v2.parquet")
    parser.add_argument("--audio-cache-dir", type=str, default="data/deepfake_cache/audio")
    parser.add_argument("--unfreeze-layers", type=int, default=4, help="Top N transformer layers to unfreeze (default: 4)")
    parser.add_argument("--fewshot-n", type=int, default=50, help="Few-shot samples per class (default: 50)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--backbone-lr", type=float, default=1e-6, help="Learning rate for unfrozen XLS-R layers")
    parser.add_argument("--head-lr", type=float, default=1e-5, help="Learning rate for SLS head")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="outputs/results_unfreeze_l4.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parquet_path = _PROJECT_ROOT / args.parquet_path
    audio_cache_dir = _PROJECT_ROOT / args.audio_cache_dir
    output_json = _PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("EXPERIMENT 2: PARTIAL XLS-R 300M BACKBONE UNFREEZING")
    logger.info("=" * 75)
    logger.info(f"Parquet path:      {parquet_path}")
    logger.info(f"Audio cache dir:   {audio_cache_dir}")
    logger.info(f"Unfreeze layers:   Top {args.unfreeze_layers} layers of 24")
    logger.info(f"Few-shot N:        {args.fewshot_n} per class")
    logger.info(f"Backbone LR:       {args.backbone_lr}")
    logger.info(f"Head LR:           {args.head_lr}")
    logger.info(f"Device:            {args.device}")

    # 1. Load Metadata & Generate Splits
    df_hindi = load_hindi_metadata(parquet_path)
    test_ids, test_df = create_test_split(df_hindi, test_per_class=250, seed=42)
    fewshot_pools = create_fewshot_pools(df_hindi, test_ids=test_ids, fewshot_sizes=[args.fewshot_n], num_seeds=args.num_seeds)
    all_needed_ids = get_all_needed_ids(test_ids, fewshot_pools)
    labels_dict = get_labels_for_ids(df_hindi, all_needed_ids)

    # Filter to clips that are cached
    cached_ids = set()
    for rid in all_needed_ids:
        if (audio_cache_dir / f"{rid}.pt").exists():
            cached_ids.add(rid)

    logger.info(f"Found {len(cached_ids)}/{len(all_needed_ids)} audio clips cached on disk.")
    effective_test_ids = [rid for rid in test_ids if rid in cached_ids]
    logger.info(f"Effective test set size: {len(effective_test_ids)} samples.")

    # 2. Run Multi-Seed Experiment
    seed_metrics = []
    for s_idx in range(args.num_seeds):
        seed_key = (args.fewshot_n, s_idx)
        train_pool = fewshot_pools[seed_key]["row_ids"]
        effective_train_ids = [rid for rid in train_pool if rid in cached_ids]

        logger.info(f"\n--- SEED {s_idx + 1}/{args.num_seeds} (N={args.fewshot_n}, {len(effective_train_ids)} train clips) ---")
        metrics = train_single_seed(
            train_row_ids=effective_train_ids,
            test_row_ids=effective_test_ids,
            labels_dict=labels_dict,
            unfreeze_layers=args.unfreeze_layers,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            audio_cache_dir=audio_cache_dir,
        )
        seed_metrics.append(metrics)
        print_metrics(metrics, title=f"Seed {s_idx} Results (Unfrozen {args.unfreeze_layers} Layers, N={args.fewshot_n})")

    # 3. Print Consolidated Summary Table
    results_dict = {
        f"Few-Shot N={args.fewshot_n} (Unfrozen {args.unfreeze_layers}L)": seed_metrics
    }
    print("\n" + "=" * 75)
    print("FINAL CONSOLIDATED RESULTS")
    print("=" * 75)
    print_comparison_table(results_dict)

    # 4. Save to JSON
    with open(output_json, "w") as f:
        # Convert non-serializable items
        clean_res = {}
        for k, v in results_dict.items():
            clean_res[k] = []
            for m in v:
                m_copy = dict(m)
                if "confusion_matrix" in m_copy and isinstance(m_copy["confusion_matrix"], np.ndarray):
                    m_copy["confusion_matrix"] = m_copy["confusion_matrix"].tolist()
                clean_res[k].append(m_copy)
        json.dump(clean_res, f, indent=2)
    logger.info(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()
