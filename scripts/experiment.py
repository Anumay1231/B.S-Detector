"""Experiment functions for zero-shot and few-shot deepfake detection.

Provides:
  - evaluate_zeroshot: Evaluate a pretrained SLS head on a test set.
  - train_fewshot: Fine-tune an SLS head on a small labelled set and evaluate.
  - run_fewshot_experiment: Run few-shot training across multiple seeds.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_CACHE_DIR, collate_features, load_features_batch
from .metrics import compute_all_metrics, print_metrics
from .model import SLSHead, load_sls_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(
    sls_head: SLSHead,
    row_ids: list,
    labels_dict: dict,
    device: str = "cpu",
    batch_size: int = 16,
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
) -> Dict[str, Any]:
    """Run inference on a set of IDs and return metrics.

    Args:
        sls_head: SLSHead module (already loaded with weights).
        row_ids: List of row_id strings to evaluate.
        labels_dict: Mapping row_id -> int label (1=bonafide, 0=spoof).
        device: Torch device string.
        batch_size: Number of clips per forward pass.
        feature_cache_dir: Directory with cached feature .pt files.

    Returns:
        Dict from compute_all_metrics (accuracy, eer, precision, recall, f1, confusion_matrix).
    """
    sls_head = sls_head.to(device)
    sls_head.eval()

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []

    row_ids_list = list(row_ids)

    for start in range(0, len(row_ids_list), batch_size):
        batch_ids = row_ids_list[start : start + batch_size]
        batch_labels = [labels_dict[rid] for rid in batch_ids]

        # Load pre-cached features and collate into batched tensors
        features_list = load_features_batch(batch_ids, feature_cache_dir)
        batched_hidden = collate_features(features_list)  # List of 24 x (B, T, 1024)

        # Move to device
        batched_hidden = [h.to(device) for h in batched_hidden]

        with torch.no_grad():
            log_probs = sls_head(batched_hidden)  # (B, 2) log-probs [spoof, bonafide]

        # Predictions: argmax over [spoof=0, bonafide=1]
        preds = log_probs.argmax(dim=-1).cpu().tolist()
        # Score: log-probability of bonafide class (index 1) — higher = more bonafide
        scores = log_probs[:, 1].cpu().tolist()

        all_labels.extend(batch_labels)
        all_scores.extend(scores)
        all_preds.extend(preds)

    metrics = compute_all_metrics(all_labels, all_scores, all_preds)
    return metrics


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------

def evaluate_zeroshot(
    test_row_ids: set,
    labels_dict: dict,
    device: str = "cpu",
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate the pretrained SLS head zero-shot on a Hindi test set.

    Loads the pretrained checkpoint (English-trained), runs inference on the
    test set without any fine-tuning, and reports metrics.

    Args:
        test_row_ids: Set of row_id strings in the test split.
        labels_dict: Mapping row_id -> int label (1=bonafide, 0=spoof).
        device: Torch device string.
        feature_cache_dir: Directory with cached feature .pt files.
        batch_size: Inference batch size.

    Returns:
        Metrics dict from compute_all_metrics.
    """
    logger.info("=== Zero-Shot Evaluation ===")

    # Create and load pretrained SLS head
    sls_head = SLSHead()
    load_sls_checkpoint(sls_head)
    sls_head = sls_head.to(device)

    test_ids_list = sorted(test_row_ids)
    logger.info(f"Evaluating on {len(test_ids_list)} test samples...")

    metrics = _run_inference(
        sls_head, test_ids_list, labels_dict,
        device=device, batch_size=batch_size,
        feature_cache_dir=feature_cache_dir,
    )

    print_metrics(metrics, title="Zero-Shot (Pretrained English -> Hindi)")
    return metrics


# ---------------------------------------------------------------------------
# Few-shot training
# ---------------------------------------------------------------------------

def train_fewshot(
    train_row_ids: Sequence[str],
    labels_dict: dict,
    test_row_ids: Sequence[str],
    device: str = "cpu",
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
    epochs: int = 10,
    lr: float = 1e-5,
    batch_size: int = 4,
    loss_weight: Optional[List[float]] = None,
    eval_batch_size: int = 16,
) -> Dict[str, Any]:
    """Fine-tune a fresh SLS head on a few-shot training set, then evaluate.

    Initialises a new SLSHead with pretrained checkpoint weights, fine-tunes
    on the given training IDs using pre-cached features (no backbone pass),
    and evaluates on the test set.

    Args:
        train_row_ids: Row IDs for the few-shot training set.
        labels_dict: Mapping row_id -> int label (1=bonafide, 0=spoof).
        test_row_ids: Row IDs for the held-out test set.
        device: Torch device.
        feature_cache_dir: Directory with cached features.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        batch_size: Training mini-batch size.
        loss_weight: Class weights for NLLLoss [spoof_w, bonafide_w].
        eval_batch_size: Batch size for test evaluation.

    Returns:
        Metrics dict from test-set evaluation after fine-tuning.
    """
    if loss_weight is None:
        loss_weight = [0.1, 0.9]

    # Fresh SLS head with pretrained weights
    sls_head = SLSHead()
    load_sls_checkpoint(sls_head)
    sls_head = sls_head.to(device)
    sls_head.train()

    # Loss and optimizer
    weight_tensor = torch.tensor(loss_weight, dtype=torch.float32).to(device)
    criterion = nn.NLLLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(sls_head.parameters(), lr=lr, weight_decay=1e-4)

    train_ids_list = list(train_row_ids)
    train_labels = [labels_dict[rid] for rid in train_ids_list]

    # Training loop
    indices = np.arange(len(train_ids_list))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_ids = [train_ids_list[i] for i in batch_idx]
            batch_features = load_features_batch(batch_ids, feature_cache_dir)
            batch_labels = torch.tensor(
                [train_labels[i] for i in batch_idx], dtype=torch.long
            ).to(device)

            # Collate and move to device
            batched_hidden = collate_features(batch_features)
            batched_hidden = [h.to(device) for h in batched_hidden]

            optimizer.zero_grad()
            log_probs = sls_head(batched_hidden)  # (B, 2)
            loss = criterion(log_probs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

    # Evaluate on test set
    test_ids_list = sorted(test_row_ids)
    metrics = _run_inference(
        sls_head, test_ids_list, labels_dict,
        device=device, batch_size=eval_batch_size,
        feature_cache_dir=feature_cache_dir,
    )
    return metrics


# ---------------------------------------------------------------------------
# Multi-seed few-shot experiment
# ---------------------------------------------------------------------------

def run_fewshot_experiment(
    n: int,
    fewshot_pools: dict,
    test_row_ids: set,
    labels_dict: dict,
    num_seeds: int = 3,
    device: str = "cpu",
    feature_cache_dir: Path = FEATURE_CACHE_DIR,
    epochs: int = 10,
    lr: float = 1e-5,
    batch_size: int = 4,
    loss_weight: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Run few-shot fine-tuning across multiple seeds for a given N.

    For each seed, re-initialises the SLS head from the pretrained checkpoint,
    fine-tunes on the corresponding pool, and evaluates on the test set.

    Args:
        n: Number of samples per class (e.g. 10 or 50).
        fewshot_pools: Dict mapping (N, seed_idx) -> {'row_ids': set, 'df': DataFrame}.
        test_row_ids: Set of test row IDs.
        labels_dict: Mapping row_id -> int label.
        num_seeds: Number of seeds to run.
        device: Torch device.
        feature_cache_dir: Directory with cached features.
        epochs: Training epochs per seed.
        lr: Learning rate.
        batch_size: Training batch size.
        loss_weight: Class weights for NLLLoss.

    Returns:
        List of metrics dicts, one per seed.
    """
    logger.info(f"\n=== Few-Shot Experiment N={n} ({num_seeds} seeds) ===")

    all_metrics: List[Dict[str, Any]] = []

    for seed_idx in range(num_seeds):
        pool_key = (n, seed_idx)
        if pool_key not in fewshot_pools:
            logger.error(f"Pool {pool_key} not found in fewshot_pools, skipping")
            continue

        pool = fewshot_pools[pool_key]
        train_ids = pool["row_ids"]
        seed_value = 1000 + seed_idx

        logger.info(
            f"\n--- Seed {seed_idx} (seed_value={seed_value}), "
            f"N={n} per class, {len(train_ids)} training samples ---"
        )

        # Set numpy seed for reproducibility of shuffling in training
        np.random.seed(seed_value)

        metrics = train_fewshot(
            train_row_ids=train_ids,
            labels_dict=labels_dict,
            test_row_ids=test_row_ids,
            device=device,
            feature_cache_dir=feature_cache_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            loss_weight=loss_weight,
        )

        print_metrics(metrics, title=f"Few-Shot N={n}, Seed {seed_idx}")
        all_metrics.append(metrics)

    # Summary across seeds
    if all_metrics:
        metric_keys = ["accuracy", "eer", "precision", "recall", "f1"]
        logger.info(f"\n--- N={n} Summary (mean +/- std across {len(all_metrics)} seeds) ---")
        for key in metric_keys:
            vals = [m[key] * 100.0 for m in all_metrics if key in m]
            if vals:
                logger.info(f"  {key}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}%")

    return all_metrics
