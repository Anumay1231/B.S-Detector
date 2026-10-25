"""Module for creating train/test splits from the SEA-Spoof Hindi metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np
import pandas as pd

# Constants
PARQUET_PATH = Path(
    "/Users/yashgarg/Documents/bsdetector/B.S-Detector/data/sea_spoof_en_hi_metadata.parquet"
)
TEST_PER_CLASS = 250
FEWSHOT_SIZES = [10, 50]
NUM_SEEDS = 3


def load_hindi_metadata(
    parquet_path: Union[str, Path] = PARQUET_PATH,
) -> pd.DataFrame:
    """Load metadata parquet file and filter for Hindi language examples.

    Args:
        parquet_path: Path to the parquet metadata file. Defaults to PARQUET_PATH.

    Returns:
        Filtered DataFrame containing only Hindi examples.
    """
    df = pd.read_parquet(parquet_path)
    df_hindi = df[df["language"] == "hi"].copy()
    bonafide_count = int((df_hindi["label"] == "bonafide").sum())
    spoof_count = int((df_hindi["label"] == "spoof").sum())
    print(
        f"Loaded Hindi metadata: {len(df_hindi)} total rows "
        f"({bonafide_count} bonafide, {spoof_count} spoof)"
    )
    return df_hindi


def create_test_split(
    df_hindi: pd.DataFrame,
    test_per_class: int = TEST_PER_CLASS,
    seed: int = 42,
) -> tuple[set, pd.DataFrame]:
    """Create a balanced test split from the Hindi metadata.

    From the Hindi dataframe, samples `test_per_class` bonafide and
    `test_per_class` spoof rows using numpy random with the specified seed
    for reproducibility.

    Args:
        df_hindi: DataFrame containing Hindi metadata.
        test_per_class: Number of samples per class. Defaults to TEST_PER_CLASS.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (set of test row_ids, test DataFrame).
    """
    bonafide_df = df_hindi[df_hindi["label"] == "bonafide"]
    spoof_df = df_hindi[df_hindi["label"] == "spoof"]

    if len(bonafide_df) < test_per_class:
        raise ValueError(
            f"Not enough bonafide samples ({len(bonafide_df)}) to sample {test_per_class}"
        )
    if len(spoof_df) < test_per_class:
        raise ValueError(
            f"Not enough spoof samples ({len(spoof_df)}) to sample {test_per_class}"
        )

    rng = np.random.RandomState(seed)
    test_bonafide = bonafide_df.sample(n=test_per_class, random_state=rng)
    test_spoof = spoof_df.sample(n=test_per_class, random_state=rng)

    test_df = pd.concat([test_bonafide, test_spoof])
    if "row_id" in test_df.columns:
        test_ids = set(test_df["row_id"])
    else:
        test_ids = set(test_df.index)

    return test_ids, test_df


def create_fewshot_pools(
    df_hindi: pd.DataFrame,
    test_ids: set,
    fewshot_sizes: list[int] = FEWSHOT_SIZES,
    num_seeds: int = NUM_SEEDS,
) -> dict:
    """Create few-shot pools from rows not included in the test set.

    For each seed (0, 1, 2) and each N in fewshot_sizes:
      - Sample N bonafide and N spoof rows (disjoint from test)
      - Use np.random.RandomState(seed_value) where seed_value = 1000 + seed_idx

    Args:
        df_hindi: DataFrame containing Hindi metadata.
        test_ids: Set of row_ids already allocated to the test split.
        fewshot_sizes: List of few-shot pool sizes (N bonafide, N spoof).
        num_seeds: Number of random seeds to evaluate.

    Returns:
        Dict mapping (N, seed_idx) to {'row_ids': set, 'df': pd.DataFrame}.
    """
    if "row_id" in df_hindi.columns:
        candidate_df = df_hindi[~df_hindi["row_id"].isin(test_ids)]
    else:
        candidate_df = df_hindi[~df_hindi.index.isin(test_ids)]

    bonafide_pool = candidate_df[candidate_df["label"] == "bonafide"]
    spoof_pool = candidate_df[candidate_df["label"] == "spoof"]

    pools = {}
    for seed_idx in range(num_seeds):
        seed_value = 1000 + seed_idx
        for n in fewshot_sizes:
            if len(bonafide_pool) < n:
                raise ValueError(
                    f"Not enough bonafide candidates ({len(bonafide_pool)}) to sample {n}"
                )
            if len(spoof_pool) < n:
                raise ValueError(
                    f"Not enough spoof candidates ({len(spoof_pool)}) to sample {n}"
                )

            rng = np.random.RandomState(seed_value)
            sample_bonafide = bonafide_pool.sample(n=n, random_state=rng)
            sample_spoof = spoof_pool.sample(n=n, random_state=rng)

            pool_df = pd.concat([sample_bonafide, sample_spoof])
            if "row_id" in pool_df.columns:
                pool_ids = set(pool_df["row_id"])
            else:
                pool_ids = set(pool_df.index)

            pools[(n, seed_idx)] = {
                "row_ids": pool_ids,
                "df": pool_df,
            }

    print("Few-shot pools summary:")
    for (n, seed_idx), pool_info in pools.items():
        b_count = (pool_info["df"]["label"] == "bonafide").sum()
        s_count = (pool_info["df"]["label"] == "spoof").sum()
        print(
            f"  Pool (N={n}, seed_idx={seed_idx}): {len(pool_info['df'])} rows "
            f"({b_count} bonafide, {s_count} spoof, {len(pool_info['row_ids'])} unique IDs)"
        )

    return pools


def get_all_needed_ids(test_ids: set, fewshot_pools: dict) -> set:
    """Return the union of test_ids and all fewshot pool row_ids.

    Args:
        test_ids: Set of row_ids from the test split.
        fewshot_pools: Dict of few-shot pools mapping (N, seed_idx) to dict with 'row_ids'.

    Returns:
        Union of test_ids and all fewshot pool row_ids.
    """
    needed_ids = set(test_ids)
    for pool in fewshot_pools.values():
        needed_ids.update(pool["row_ids"])
    print(f"Total unique IDs needed: {len(needed_ids)}")
    return needed_ids


def get_labels_for_ids(df_hindi: pd.DataFrame, row_ids: Iterable[Any]) -> dict:
    """Return dict mapping row_id -> 1 (bonafide) or 0 (spoof).

    Args:
        df_hindi: DataFrame containing Hindi metadata with 'label' column.
        row_ids: Iterable of row_ids to look up.

    Returns:
        Dict mapping row_id -> 1 (bonafide) or 0 (spoof).
    """
    row_ids_set = set(row_ids)
    if "row_id" in df_hindi.columns:
        subset = df_hindi[df_hindi["row_id"].isin(row_ids_set)]
        id_series = subset["row_id"]
    else:
        subset = df_hindi.loc[df_hindi.index.intersection(row_ids_set)]
        id_series = subset.index

    labels = {}
    for rid, lbl in zip(id_series, subset["label"]):
        is_bonafide = (
            lbl == "bonafide"
            or lbl == 1
            or str(lbl).strip().lower() in ("bonafide", "1")
        )
        labels[rid] = 1 if is_bonafide else 0
    return labels


if __name__ == "__main__":
    if PARQUET_PATH.exists():
        print(f"Loading metadata from {PARQUET_PATH}...")
        df_hindi = load_hindi_metadata()
        test_ids, test_df = create_test_split(df_hindi)
        fewshot_pools = create_fewshot_pools(df_hindi, test_ids)
        needed_ids = get_all_needed_ids(test_ids, fewshot_pools)
        labels = get_labels_for_ids(df_hindi, needed_ids)
        print(f"Extracted labels for {len(labels)} IDs.")
    else:
        print(f"Parquet file not found at {PARQUET_PATH}.")
