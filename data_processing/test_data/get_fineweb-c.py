"""
load_fineweb-c_danish.py

A script to download the Danish subset of the Fineweb-C dataset
and save it as a CSV file.
"""
import pandas as pd
from datasets import load_dataset
import numpy as np

import os
def download_danish_fineweb_c(to_csv = False, path_to_save = os.path.join("data", "fineweb-c_danish.csv")):
    """
    Loads the Danish (dan_Latn) configuration of the Fineweb-C dataset,
    selects the 'text' and 'score' columns, and saves them to a CSV.
    """

    print("Loading Danish Fineweb-C dataset from the Hugging Face Hub...")

    dataset = load_dataset("data-is-better-together/fineweb-c", name="dan_Latn")    

    print("--- Download complete. ---")
    df = dataset["train"].to_pandas()

    # Step 1: Filter out the problematic rows
    df = df[df["problematic_content_label_present"] == False]



    # Step 2: Translate the score from strings to numbers
    score_mapping = {
    'None': 0,
    'Minimal': 1,
    'Basic': 2,
    'Good': 3,
    'Excellent': 4
            }
    
    # Step 3: Calculate 'int_score' by rounding up the average of annotator scores.
    print("Calculating final score by rounding up the average of annotations...")
    df['int_score'] = df['educational_value_labels'].apply(
        lambda labels: round(np.mean([score_mapping[l] for l in labels]))
    )

    # Step 4: Select only the relevant columns.
    df = df[['text', 'int_score', 'educational_value_labels', 'problematic_content_label_present', 'problematic_content_label_agreement']]

    # Optional: Save to a CSV file.
    if to_csv:
        df.to_csv(path_to_save, index=False, encoding="utf-8")

    return df

if __name__ == "__main__":
    df = download_danish_fineweb_c(to_csv=True)