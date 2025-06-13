"""
json_to_csv_converter.py

A minimal script to convert a JSON file into a CSV file.
It extracts the 'text' and 'educational_score' fields, and renames
the latter to 'int_score'.

Usage:
    python json_to_csv_converter.py --json_path path/to/your/file.json
"""
import pandas as pd
import argparse
import os

def convert_json_to_csv(json_path: str, csv_path: str):
    """
    Reads a JSON file, processes it, and saves it as a CSV.
    
    Args:
        json_path (str): Path to the input .json file.
        csv_path (str): Path for the output .csv file.
    """
    print(f"Reading data from '{json_path}'...")
    
    # 1. Read the JSON file into a pandas DataFrame
    df = pd.read_json(json_path)
    
    # 2. Select the 'text' and 'educational_score' columns
    df_processed = df[['text', 'educational_score']]
    
    # 3. Rename 'educational_score' to 'int_score'
    df_processed = df_processed.rename(columns={'educational_score': 'int_score'})
    
    # 4. Save to a CSV file without the pandas index
    df_processed.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"Successfully created '{csv_path}'.")
    print("\nFirst 5 rows of the output file:")
    print(df_processed.head())


if __name__ == "__main__":
    json_path = "data/interim/danish_filtered_labelled_data.json"
    output_path = "data/processed/danish_filtered_labelled_data.csv"
    convert_json_to_csv(json_path, output_path)