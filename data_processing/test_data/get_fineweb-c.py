"""
load_fineweb-c_danish.py

A script to download the Danish subset of the Fineweb-C dataset
and save it as a CSV file.
"""
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
with open(".venv/huggingface_login_token.txt", "r") as f:
    token = f.read().strip()
login(token=token)
def download_danish_fineweb_c(to_csv = False, output_filename = "data/fineweb-c_danish_raw.csv"):
    """
    Loads the Danish (dan_Latn) configuration of the Fineweb-C dataset,
    selects the 'text' and 'score' columns, and saves them to a CSV.
    """

    print("Loading Danish Fineweb-C dataset from the Hugging Face Hub...")

    dataset = load_dataset("data-is-better-together/fineweb-c", name="dan_Latn")    
    # Save to a CSV file.
    if to_csv:
        dataset.to_csv(output_filename, index=False, encoding="utf-8")
    print("--- Download complete. ---")
    df = dataset["train"].to_pandas()

    # Step 2: Filter out the problematic rows
    clean_df = df[df["problematic_content_label_present"] == False]

    return clean_df


if __name__ == "__main__":
    df = download_danish_fineweb_c()