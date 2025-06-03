import pandas as pd
import json
import os


def get_merged_dataset(
    english_data_amount=1000,
    danish_data_amount=0,
    english_path="data/english_fineweb_merged_data.csv",
    danish_path="data/danish_filtered_labelled_data.json",
    random_seed=42,
):
    """Load and merge English and Danish datasets.
    Args:
        english_data_amount (int): Number of English samples to load.
        danish_data_amount (int): Number of Danish samples to load.
        english_path (str): Path to the English dataset CSV file.
        danish_path (str): Path to the Danish dataset JSON file.
        random_seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: Merged DataFrame containing both English and Danish data.
    """
    # Load English data
    english_df = pd.read_csv(english_path)
    if english_data_amount > 0:
        english_df = english_df.sample(
            n=min(english_data_amount, len(english_df)), random_state=random_seed
        )
    # Load Danish data
    if danish_data_amount > 0:
        with open(danish_path, "r", encoding="utf-8") as f:
            danish_data = json.load(f)
        danish_df = pd.DataFrame(danish_data)
        danish_df = danish_df.sample(
            n=min(danish_data_amount, len(danish_df)), random_state=random_seed
        )
        # Harmonize columns if needed
        if "educational_score" in danish_df.columns:
            danish_df = danish_df.rename(columns={"educational_score": "score"})
        # Only keep columns present in English data
        danish_df = danish_df[
            [col for col in english_df.columns if col in danish_df.columns]
        ]
        merged_df = pd.concat([english_df, danish_df], ignore_index=True)
    else:
        merged_df = english_df.copy()
    return merged_df
