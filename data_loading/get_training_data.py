"""
Script to generate training data for the Danish educational content classifier.
This script uses the dataloader to fetch a specific amount of Danish text data
from FineWeb-2 for training purposes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loading'))

from dataloader import dataloader

def main():
    """Generate training data with specified parameters."""
    
    # Training data configuration
    TRAINING_SAMPLES = 1000  # Adjust based on your needs
    CSV_FILENAME = "training_data.csv"
    
    print(f"Generating training data with {TRAINING_SAMPLES} samples...")
    
    # Generate training data
    training_df = dataloader(
        samples=TRAINING_SAMPLES,
        to_csv=True,
        csv_filename=CSV_FILENAME
    )
    
    print(f"Training data generated successfully!")
    print(f"Dataset shape: {training_df.shape}")
    print(f"Saved to: data/{CSV_FILENAME}")

if __name__ == "__main__":
    main()