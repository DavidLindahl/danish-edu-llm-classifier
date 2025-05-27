"""
Script to generate training data for the Danish educational content classifier.
This script uses the dataloader to fetch multiple batches of Danish text data
from FineWeb-2 for training purposes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loading'))

from dataloader import dataloader
import pandas as pd

def main():
    """Generate training data in multiple batches."""
    
    # Training data configuration
    BATCH_SIZE = 250
    NUM_BATCHES = 4
    TOTAL_SAMPLES = BATCH_SIZE * NUM_BATCHES
    
    print(f"Generating training data in {NUM_BATCHES} batches of {BATCH_SIZE} samples each...")
    print(f"Total samples: {TOTAL_SAMPLES}")
    
    all_batches = []
    
    # Generate data in batches
    for batch_num in range(1, NUM_BATCHES + 1):
        print(f"\nProcessing batch {batch_num}/{NUM_BATCHES}...")
        
        batch_df = dataloader(
            samples=BATCH_SIZE,
            to_csv=True,
            csv_filename=f"training_data_batch_{batch_num}.csv"
        )
        
        all_batches.append(batch_df)
        print(f"Batch {batch_num} completed: {batch_df.shape[0]} samples")
    
    # Combine all batches into a single DataFrame
    print("\nCombining all batches...")
    combined_df = pd.concat(all_batches, ignore_index=True)
    
    # Save combined dataset
    combined_csv_path = os.path.join("data", "training_data_combined.csv")
    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")
    
    print(f"\nTraining data generation completed!")
    print(f"Total dataset shape: {combined_df.shape}")
    print(f"Individual batches saved as: training_data_batch_1.csv to training_data_batch_{NUM_BATCHES}.csv")
    print(f"Combined dataset saved as: training_data_combined.csv")

if __name__ == "__main__":
    main()