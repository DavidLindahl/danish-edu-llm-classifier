"""Merge and balance English educational datasets.

This script combines locally classified English samples with two public
FineWeb educational datasets from Hugging Face.  After merging the samples,
the resulting data is balanced across the available ``int_score`` classes.

The final, balanced dataset is saved to ``../../data/english_fineweb_merged_data.csv``.

The defaults favour loading more samples from ``fineweb-edu-score-2`` as these
contain lower scores, helping balance the classes.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset


HF_DATASET_SCORE_3_NAME = "HuggingFaceFW/fineweb-edu"
HF_CONFIG_SCORE_3 = "CC-MAIN-2024-22"
HF_DATASET_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
HF_CONFIG_SCORE_2 = "CC-MAIN-2024-18"

# Path to locally classified FineWeb samples
LOCAL_CSV_PATH = "../../data/english_classified_samples.csv"
# Output path for the merged, balanced dataset
OUTPUT_CSV_PATH = "../../data/english_fineweb_merged_data.csv"

COMMON_COLUMNS = ["text", "language_score", "token_count", "int_score"]


def _load_hf_samples(
    dataset_name: str,
    config_name: str,
    num_samples: int,
    split: str = "train",
) -> pd.DataFrame:
    """Stream ``num_samples`` rows from a Hugging Face dataset.

    Parameters
    ----------
    dataset_name:
        Name of the dataset on the Hub.
    config_name:
        Config to load from the dataset.
    num_samples:
        Maximum number of samples to collect.
    split:
        Split to load.  Defaults to ``"train"``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested samples.  Missing columns are filled
        with ``None``.
    """

    print(
        f"Streaming {num_samples} samples from {dataset_name} ({config_name}) ..."
    )
    try:
        stream = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=True,
        )
    except Exception as exc:  # pragma: no cover - network errors
        print(f"Failed to load {dataset_name}: {exc}")
        return pd.DataFrame(columns=COMMON_COLUMNS)

    collected: List[dict] = []
    for idx, sample in enumerate(stream):
        if idx >= num_samples:
            break
        row = {col: sample.get(col) for col in COMMON_COLUMNS}
        if row.get("int_score") is None and sample.get("score") is not None:
            score = sample["score"]
            row["int_score"] = int(round(max(0, min(float(score), 5))))
        collected.append(row)

    print(f"Collected {len(collected)} samples from {dataset_name}")
    return pd.DataFrame(collected, columns=COMMON_COLUMNS)


def _load_local_samples(num_samples: int) -> pd.DataFrame:
    """Load locally classified samples from ``LOCAL_CSV_PATH``."""

    print(f"Loading {num_samples} local samples from {LOCAL_CSV_PATH} ...")
    if not os.path.exists(LOCAL_CSV_PATH):
        print("Local CSV not found, returning empty DataFrame")
        return pd.DataFrame(columns=COMMON_COLUMNS)

    try:
        df = pd.read_csv(LOCAL_CSV_PATH, on_bad_lines="skip")
    except Exception as exc:  # pragma: no cover - file errors
        print(f"Failed to load local CSV: {exc}")
        return pd.DataFrame(columns=COMMON_COLUMNS)

    subset = df[COMMON_COLUMNS].head(num_samples)
    print(f"Loaded {len(subset)} local samples")
    return subset


def _balance_int_scores(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Balance the DataFrame so each ``int_score`` occurs equally often."""

    score_counts = df["int_score"].value_counts().to_dict()
    if not score_counts:
        return df

    target = max(score_counts.values())
    balanced_parts: List[pd.DataFrame] = []
    for score, group in df.groupby("int_score"):
        if len(group) < target:
            balanced = group.sample(
                target, replace=True, random_state=random_state
            )
        else:
            balanced = group.sample(target, random_state=random_state)
        balanced_parts.append(balanced)

    balanced_df = pd.concat(balanced_parts).sample(
        frac=1.0, random_state=random_state
    )
    print("Balanced dataset distribution:")
    print(balanced_df["int_score"].value_counts().sort_index())
    return balanced_df.reset_index(drop=True)


def load_and_process_dataset(
    num_samples_score_3: int = 1500,
    num_samples_score_2: int = 2500,
    num_samples_csv: int = 1000,
) -> pd.DataFrame:
    """Load, merge and balance the English educational datasets."""

    df_score_3 = _load_hf_samples(
        HF_DATASET_SCORE_3_NAME, HF_CONFIG_SCORE_3, num_samples_score_3
    )
    df_score_2 = _load_hf_samples(
        HF_DATASET_SCORE_2_NAME, HF_CONFIG_SCORE_2, num_samples_score_2
    )
    df_local = _load_local_samples(num_samples_csv)

    merged = pd.concat([df_local, df_score_2, df_score_3], ignore_index=True)
    print(f"Merged dataset contains {len(merged)} samples")

    return _balance_int_scores(merged)


def plot_score_distribution(df: pd.DataFrame) -> None:
    """Plot a histogram of ``int_score`` values."""

    plt.figure(figsize=(8, 5))
    sns.histplot(df["int_score"], bins=range(0, 7), discrete=True)
    plt.xlabel("int_score")
    plt.ylabel("count")
    plt.title("Distribution of int_score")
    plt.tight_layout()
    plt.show()


def save_to_csv(df: pd.DataFrame, filename: str = OUTPUT_CSV_PATH) -> None:
    """Save ``df`` to ``filename``."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved merged dataset to {filename}")


if __name__ == "__main__":
    merged_df = load_and_process_dataset()
    plot_score_distribution(merged_df)
    save_to_csv(merged_df)

