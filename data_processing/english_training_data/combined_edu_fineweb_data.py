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

# Base directory for data files (relative to this script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

# Path to locally classified FineWeb samples
LOCAL_CSV_PATH = os.path.join(BASE_DIR, "english_classified_samples_5000.csv")
# Output path for the merged, balanced dataset
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "english_fineweb_merged_data.csv")

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


def load_and_process_dataset(
    num_samples_score_3: int = 3500,
    num_samples_score_2: int = 2500,
    num_samples_csv: int = 5000,
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
    return merged


def plot_score_distribution(df: pd.DataFrame) -> None:
    """Plot a histogram of ``int_score`` values."""

    plt.figure(figsize=(8, 5))
    # Count occurrences for each int_score 0-5
    value_counts = df['int_score'].value_counts().reindex(range(0, 6), fill_value=0)
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
    plt.xlabel("int_score")
    plt.ylabel("count")
    plt.title("Distribution of int_score (0-5)")
    plt.xticks(range(0, 6))
    plt.tight_layout()
    plt.show()


def save_to_csv(df: pd.DataFrame, filename: str = OUTPUT_CSV_PATH) -> None:
    """Save ``df`` to ``filename``."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved merged dataset to {filename}")


def limit_int_score_1_2_3(df: pd.DataFrame, N_1: int, N_2: int, N_3: int) -> pd.DataFrame:
    """
    Limit the number of samples with int_score == 1, 2, and 3 to N_1, N_2, and N_3 respectively.
    All other int_score classes (0, 4, 5) are unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataset.
    N_1 : int
        Maximum number of samples to keep for int_score == 1.
    N_2 : int
        Maximum number of samples to keep for int_score == 2.
    N_3 : int
        Maximum number of samples to keep for int_score == 3.

    Returns
    -------
    pd.DataFrame
        DataFrame with limited samples for int_score 1, 2, and 3.
    """
    df_1 = df[df["int_score"] == 1].head(N_1)
    df_2 = df[df["int_score"] == 2].head(N_2)
    df_3 = df[df["int_score"] == 3].head(N_3)
    df_rest = df[df["int_score"].isin([0, 4, 5])]
    result = pd.concat([df_rest, df_1, df_2, df_3], ignore_index=True)
    return result


def convert_int_score_5_to_4(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all int_score 5 to 4 in the DataFrame."""
    df.loc[df['int_score'] == 5, 'int_score'] = 4
    return df

if __name__ == "__main__":
    merged_df = load_and_process_dataset()
    # Limit int_score 1, 2, and 3 to 1000 samples each
    merged_df = limit_int_score_1_2_3(merged_df, N_1=1000, N_2=1000, N_3=1000)
    # Convert int_score 5 to 4
    merged_df = convert_int_score_5_to_4(merged_df)
    plot_score_distribution(merged_df)
    save_to_csv(merged_df)



