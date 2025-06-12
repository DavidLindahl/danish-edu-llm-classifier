import pandas as pd
import json

def get_merged_dataset(
    english_data_amount=1000,
    danish_data_amount=0,
    english_path="data/english_fineweb_merged_data.csv",
    danish_path="data/danish_filtered_labelled_data.json",
    random_seed=42,
):
    """Load and merge English and Danish datasets.
    Args:
        english_data_amount (int): Number of English samples to load (-1 for all).
        danish_data_amount (int): Number of Danish samples to load (-1 for all).
        english_path (str): Path to the English dataset CSV file.
        danish_path (str): Path to the Danish dataset JSON file.
        random_seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: Merged DataFrame containing both English and Danish data.
    """
    dataframes = []

    if english_data_amount != 0:
        df = pd.read_csv(english_path)
        if english_data_amount > 0:
            df = df.sample(n=min(english_data_amount, len(df)), random_state=random_seed)
        dataframes.append(df)

    if danish_data_amount != 0:
        with open(danish_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
        if danish_data_amount > 0:
            df = df.sample(n=min(danish_data_amount, len(df)), random_state=random_seed)
        if "educational_score" in df.columns:
            df.rename(columns={"educational_score": "int_score"}, inplace=True)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()