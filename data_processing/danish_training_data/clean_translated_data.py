import pandas as pd
import os

# --- Configuration ---
INPUT_CSV_PATH = "data/danish_train.csv" # Adjust to your actual input file
OUTPUT_CLEANED_CSV_PATH = "data/processed/danish_train.csv"


if __name__ == "__main__":
    print(f"--- Starting Data Cleaning Process ---")

    # Load the CSV file
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        exit()
    print(f"Loading data from: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Validate required columns
    if 'tran_text' not in df.columns or 'int_score' not in df.columns:
        print("Error: Missing required columns. Expected 'tran_text' (Danish text) and 'int_score'.")
        print("Available columns:", df.columns.tolist())
        exit()

    # --- Explicitly select and rename the Danish text column ---
    # This ensures only 'tran_text' (the Danish one) is used and the original 'text' (English) is dropped.
    cleaned_df = df[['tran_text', 'int_score']].copy()
    cleaned_df = cleaned_df.rename(columns={'tran_text': 'text'})
    print(f"Selected 'tran_text' and 'int_score' columns, renamed 'tran_text' to 'text'.")
    print(f"Resulting sample count: {len(cleaned_df)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CLEANED_CSV_PATH), exist_ok=True)
    
    # Save the cleaned DataFrame
    cleaned_df.to_csv(OUTPUT_CLEANED_CSV_PATH, index=False, encoding='utf-8')
    print(f"Cleaned data saved to: {OUTPUT_CLEANED_CSV_PATH}")
    
    print(f"--- Cleaning Process Complete ---")