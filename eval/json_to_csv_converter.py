#!/usr/bin/env python3
"""
Convert newline-delimited JSON (.jsonl) with Gemini scores → CSV.
Keeps 'text' and renames 'gemini_prediction' to 'int_score'.
"""

import argparse, os, pandas as pd

def convert_jsonl_to_csv(jsonl_path: str, csv_path: str):
    print(f"Reading data from '{jsonl_path}' …")

    # 1. Load .jsonl (lines=True is the key)
    df = pd.read_json(jsonl_path, lines=True)

    # 2. Keep wanted columns and rename
    df_out = (
        df[["text", "gemini_prediction"]]
        .rename(columns={"gemini_prediction": "int_score"})
    )

    # 3. Save
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✓ CSV written to '{csv_path}'")
    print("\nPreview:")
    print(df_out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", default="test/gemini_predictions.jsonl")
    parser.add_argument("--csv_path",  default="test/gemini_predictions.csv")
    args = parser.parse_args()

    convert_jsonl_to_csv(args.jsonl_path, args.csv_path)