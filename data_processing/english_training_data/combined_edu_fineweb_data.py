import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

HF_DATASET_SCORE_3_NAME = "HuggingFaceFW/fineweb-edu"
HF_CONFIG_SCORE_3 = "CC-MAIN-2024-22"
HF_DATASET_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
HF_CONFIG_SCORE_2 = "CC-MAIN-2024-18"
CSV_FILE_PATH = "data/fineweb_train_classifier.csv"
COMMON_COLUMNS = ["text", "language_score", "token_count", "int_score"]


def _load_hf_samples(
    dataset_name: str, config_name: str, num_samples: int, split: str = "train"
) -> pd.DataFrame:
    """
    Helper function to stream and collect a specified number of samples from a HuggingFace dataset.
    Ensures the returned DataFrame has the COMMON_COLUMNS.
    """
    print(
        f"Streaming {num_samples} samples from HuggingFace dataset: {dataset_name} (config: {config_name})..."
    )

    try:
        hf_stream = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=True,
        )
    except Exception as e:
        print(
            f"Error loading HuggingFace dataset {dataset_name} (config: {config_name}): {e}"
        )
        print("Returning an empty DataFrame for this source.")
        return pd.DataFrame(
            columns=COMMON_COLUMNS
        )  # Return empty df with expected columns

    collected_samples = []
    for i, sample in enumerate(hf_stream):
        if i >= num_samples:
            break
        collected_samples.append(sample)

    print(f"Collected {len(collected_samples)} samples from {dataset_name}.")

    df = pd.DataFrame(collected_samples)

    return df[COMMON_COLUMNS]


# --- Main Function ---
def load_and_process_dataset(
    num_samples_score_3: int = 300,  # Default based on original code's hardcoded value
    num_samples_score_2: int = 350,  # Default based on original code's hardcoded value
    num_samples_csv: int = 500,  # Renamed from num_smaples_score_full, default based on original
) -> pd.DataFrame:
    """
    Loads and processes data from multiple sources (HuggingFace datasets and a CSV file)
    and merges them into a single Pandas DataFrame.

    Args:
        num_samples_score_3 (int): Number of samples to collect from the 'fineweb-edu' HuggingFace dataset.
                                   Defaults to 300.
        num_samples_score_2 (int): Number of samples to collect from the 'fineweb-edu-score-2' HuggingFace dataset.
                                   Defaults to 350.
        num_samples_csv (int): Number of samples to read from the local CSV file.
                               Defaults to 500.

    Returns:
        pd.DataFrame: A merged DataFrame containing processed samples from all sources.
                      Columns include: 'text', 'language_score', 'token_count', 'score'.
                      Returns an empty DataFrame with correct columns if no data is loaded.
    """

    # 1. Load HuggingFace datasets
    hf_df_score_3 = _load_hf_samples(
        HF_DATASET_SCORE_3_NAME, HF_CONFIG_SCORE_3, num_samples_score_3
    )
    hf_df_score_2 = _load_hf_samples(
        HF_DATASET_SCORE_2_NAME, HF_CONFIG_SCORE_2, num_samples_score_2
    )

    # 2. Load samples from the CSV file
    csv_df = pd.DataFrame(
        columns=COMMON_COLUMNS
    )  # Initialize empty with expected columns
    print(f"Loading samples from CSV: {CSV_FILE_PATH}...")
    if not os.path.exists(CSV_FILE_PATH):
        print(
            f"Warning: CSV file not found at '{CSV_FILE_PATH}'. Skipping CSV loading."
        )
    else:
        try:
            csv_df_raw = pd.read_csv(CSV_FILE_PATH)

            # Ensure common columns exist, add missing with NaN/None if necessary

            # Select relevant columns and limit samples
            csv_df = csv_df_raw[COMMON_COLUMNS].head(num_samples_csv)
            print(f"Loaded {len(csv_df)} samples from CSV.")
        except pd.errors.EmptyDataError:
            print(
                f"Warning: CSV file '{CSV_FILE_PATH}' is empty. Skipping CSV loading."
            )
        except Exception as e:
            print(
                f"Error loading CSV file '{CSV_FILE_PATH}': {e}. Skipping CSV loading."
            )

    # 3. Concatenate all DataFrames
    # Filter out any empty DataFrames before concatenation to prevent errors
    dataframes_to_concat = [
        df for df in [hf_df_score_2, hf_df_score_3, csv_df] if not df.empty
    ]

    if not dataframes_to_concat:
        print("Warning: No dataframes to concatenate. Returning an empty DataFrame.")
        return pd.DataFrame(
            columns=COMMON_COLUMNS
        )  # Ensure empty df has expected columns

    merged_df = pd.concat(dataframes_to_concat, ignore_index=True)

    print(f"Total number of samples in merged DataFrame: {len(merged_df)}")

    return merged_df


def plot_score_distribution(merged_df):

    # Plot the distribution of scores
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["score"], bins=30, kde=True)
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()


def save_merged_df_to_csv(df, filename="merged_fineweb_samples.csv"):
    """
    Saves the merged DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the output CSV file. Defaults to "merged_fineweb_samples.csv".
    """
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", os.path.basename(filename))
    df.to_csv(output_path, index=False)
    print(f"Merged DataFrame saved to {output_path}")


if __name__ == "__main__":
    num_samples_score_3 = 500
    num_samples_score_2 = 500
    num_samples_csv = 1000

    merged_df = load_and_process_dataset(
        num_samples_score_3=num_samples_score_3,
        num_samples_score_2=num_samples_score_2,
        num_samples_csv=num_samples_csv,
    )

    plot_score_distribution(merged_df)
    save_merged_df_to_csv(merged_df, "data/merged_fineweb_samples.csv")
