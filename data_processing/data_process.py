import pandas as pd
import json
import os
import numpy as np # Ensure numpy is imported

def get_merged_dataset(
    english_data_amount=500,
    danish_data_amount=500,
    english_path="data/english_classified_samples_5000.csv",
    danish_path="data/danish_filtered_labelled_data.json", # Updated to your Danish JSON
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
    print(f"Loading English data from: {english_path}")
    english_df = pd.read_csv(english_path)
    
    # --- Start of score column cleaning for English data ---
    # 1. Standardize score column name
    if "int_score" in english_df.columns:
        english_df["score"] = english_df["int_score"] 
    
    # 2. Convert 'score' to numeric, coercing errors to NaN
    english_df['score'] = pd.to_numeric(english_df['score'], errors='coerce')
    
    # 3. Round and clip the scores to be between 0 and 5, handling potential NaNs
    english_df['score'] = english_df['score'].round().clip(0, 5)
    
    # 4. Drop rows where 'text' or 'score' is NaN, as these are essential
    english_df.dropna(subset=['text', 'score'], inplace=True)
    
    # 5. Finally, cast to integer type (after NaNs are removed)
    english_df['score'] = english_df['score'].astype(int)
    # --- End of score column cleaning for English data ---

    # Sample English data if specified
    if english_data_amount >= 0: # Use >= 0 for samples, -1 for all
        english_df = english_df.sample(
            n=min(english_data_amount, len(english_df)) if english_data_amount != -1 else len(english_df), 
            random_state=random_seed
        )
    else: # If english_data_amount is 0 or less, ensure it's an empty DataFrame with correct columns
        english_df = pd.DataFrame(columns=['text', 'score'])

    print(f"English data loaded. Sample size: {len(english_df)}, Columns: {english_df.columns.tolist()}, Score Dtype: {english_df['score'].dtype}")

    print(f"Loading Danish data from: {danish_path}")
    danish_df = pd.DataFrame() # Initialize empty in case no Danish data is loaded or errors occur
    if danish_data_amount > 0 or danish_data_amount == -1: # Load if amount is positive or -1
        try:
            with open(danish_path, "r", encoding="utf-8") as f:
                danish_data = json.load(f)
            danish_df = pd.DataFrame(danish_data)
            
            # --- Start of score column cleaning for Danish data ---
            # 1. Standardize score column name
            if "educational_score" in danish_df.columns:
                danish_df["score"] = danish_df["educational_score"] 
            
            # 2. Convert 'score' to numeric, coercing errors to NaN
            danish_df['score'] = pd.to_numeric(danish_df['score'], errors='coerce')
            
            # 3. Round and clip the scores to be between 0 and 5
            danish_df['score'] = danish_df['score'].round().clip(0, 5)
            
            # 4. Drop rows where 'text' or 'score' is NaN
            danish_df.dropna(subset=['text', 'score'], inplace=True)
            
            # 5. Finally, cast to integer type
            danish_df['score'] = danish_df['score'].astype(int)
            # --- End of score column cleaning for Danish data ---

            # Sample Danish data if specified
            danish_df = danish_df.sample(
                n=min(danish_data_amount, len(danish_df)) if danish_data_amount != -1 else len(danish_df), 
                random_state=random_seed
            )
            
            print(f"Danish data loaded. Sample size: {len(danish_df)}, Columns: {danish_df.columns.tolist()}, Score Dtype: {danish_df['score'].dtype}")

        except FileNotFoundError:
            print(f"Warning: Danish data file not found at {danish_path}. Skipping Danish data.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {danish_path}. Skipping Danish data.")
        # If an error occurred, danish_df remains an empty DataFrame, which is fine

    # --- Combine dataframes ---
    essential_cols = ['text', 'score']
    
    # Ensure both dataframes only contain essential columns before concatenation
    english_df = english_df[essential_cols]
    if not danish_df.empty:
        danish_df = danish_df[essential_cols]
    else: # If danish_df is empty, ensure it has the expected columns for concat
        danish_df = pd.DataFrame(columns=essential_cols)

    # Concatenate dataframes
    merged_df = pd.concat([english_df, danish_df], ignore_index=True)
    
    # Final dropna just in case (e.g., if 'text' was missing in original data after previous steps)
    merged_df.dropna(subset=['text', 'score'], inplace=True)

    print(f"Merged dataset size: {len(merged_df)}")
    if not merged_df.empty:
        print(f"Final Merged Score Dtype: {merged_df['score'].dtype}")
        print("Value counts for score column (merged):\n", merged_df['score'].value_counts().sort_index())
    else:
        print("Merged DataFrame is empty. No data for training.")
    
    return merged_df

# For local testing of get_merged_dataset (Optional: you can run this file directly to test)
if __name__ == "__main__":
    # Create dummy data files for testing if you don't have real ones
    # if not os.path.exists("data"):
    #     os.makedirs("data")
    # if not os.path.exists("data/english_fineweb_merged_data.csv"):
    #     pd.DataFrame({
    #         'text': ["english text 1", "english text 2", "english text 3", "english text with invalid score", "another text", "a score of 5"],
    #         'int_score': [1, 3, 5, "invalid", None, 5] # Added None for testing missing values
    #     }).to_csv("data/english_fineweb_merged_data.csv", index=False)
    #     print("Created dummy English CSV.")

    # if not os.path.exists("data/danish_filtered_labelled_data.json"):
    #     with open("data/danish_filtered_labelled_data.json", "w", encoding="utf-8") as f:
    #         json.dump([
    #             {"text": "danish text 1", "educational_score": 0},
    #             {"text": "danish text 2", "educational_score": 2},
    #             {"text": "danish text 3", "educational_score": 4},
    #             {"text": "danish text with missing score": "no_score"} # Non-numeric score
    #         ], f)
    #     print("Created dummy Danish JSON.")

    merged_dataset = get_merged_dataset(
        english_data_amount=-1, # Use all English for test
        danish_data_amount=-1,  # Use all Danish for test
    )
    print("\n--- Merged Dataset Head ---")
    print(merged_dataset.head())
    print(f"Total samples: {len(merged_dataset)}")
    print(f"Columns: {merged_dataset.columns.tolist()}")
    if not merged_dataset.empty:
        print(f"Score Dtype: {merged_dataset['score'].dtype}")
        print("Value counts for score column:\n", merged_dataset['score'].value_counts().sort_index())