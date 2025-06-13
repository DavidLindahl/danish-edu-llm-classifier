import pandas as pd
import json

def get_merged_dataset(
    english_data_amount=1000,
    danish_data_amount=0,
    english_path="data/english_fineweb_merged_data.csv",
    danish_path="data/danish_filtered_labelled_data.csv",
    random_seed=42,
):
    """Load and merge English and Danish datasets.
    Args:
        english_data_amount (int): Number of English samples to load (-1 for all).
        danish_data_amount (int): Number of Danish samples to load (-1 for all).
        english_path (str): Path to the English dataset CSV file.
        danish_path (str): Path to the Danish dataset CSV file.
        random_seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: Merged DataFrame containing both English and Danish data.
    """
    dataframes = []

    if english_data_amount != 0:
        df = pd.read_csv(english_path) # if -1 then load all
        if english_data_amount > 0: 
            df = df.sample(n=min(english_data_amount, len(df)), random_state=random_seed) # sample english_data_amount if positive
        dataframes.append(df)

    if danish_data_amount != 0:
        df = pd.read_csv(danish_path)
        if danish_data_amount > 0:
            df = df.sample(n=min(danish_data_amount, len(df)), random_state=random_seed)
        dataframes.append(df)
  

    return pd.concat(dataframes, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    merged_df = get_merged_dataset(
        english_data_amount=1000,
        danish_data_amount=500,
        english_path="data/processed/english_fineweb_merged_data.csv",
        danish_path="data/processed/danish_filtered_labelled_data.csv"
    )
    print(f"Merged dataset contains {len(merged_df)} samples.")
    print(merged_df.head())