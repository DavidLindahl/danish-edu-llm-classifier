"""
Convert newline-delimited JSON (.jsonl) with Gemini scores → CSV.
Keeps 'text' and renames 'gemini_prediction' to 'int_score'.
"""

import argparse
import os
import pandas as pd

def convert_jsonl_to_csv(jsonl_path: str, csv_path: str):
    # --- 1. Check if the input file exists ---
    if not os.path.exists(jsonl_path):
        print(f"❌ ERROR: Input file not found at '{jsonl_path}'")
        return

    # --- 2. Check if the input file is empty ---
    if os.path.getsize(jsonl_path) == 0:
        print(f"❌ ERROR: The file '{jsonl_path}' is empty.")
        return

    print(f"Reading data from '{jsonl_path}' …")
    
    # --- 3. Load the data ---
    df = pd.read_json(jsonl_path, lines=True)

    # --- 4. DEBUG: Print what was actually loaded ---
    print("\n--- DEBUG INFO ---")
    print(f"Columns found in the DataFrame: {df.columns.tolist()}")
    print("First 2 rows of the loaded data:")
    print(df.head(2))
    print("------------------\n")

    # --- 5. Process the data with clear error handling ---
    required_columns = ["text", "gemini_prediction"]
    
    # Check if all required columns are present
    if not all(col in df.columns for col in required_columns):
        print(f"❌ ERROR: One or more required columns {required_columns} not found in the file.")
        print(f"Please check that your JSONL file contains these keys on every line.")
        return

    # Keep wanted columns and rename
    df_out = (
        df[required_columns]
        .rename(columns={"gemini_prediction": "int_score"})
    )

    # 6. Save the output
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✓ CSV written to '{csv_path}'")
    print("\nFinal Preview:")
    print(df_out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", default="archive/old_results/gemini_predictions.jsonl")
    parser.add_argument("--csv_path",  default="results/gemini_predictions.csv")
    args = parser.parse_args()

    convert_jsonl_to_csv(args.jsonl_path, args.csv_path)