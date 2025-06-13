"""
test.py

Script for evaluating a fine-tuned transformer model on the Danish educational content task.

This script loads a trained model, runs inference on a provided test set, and calculates
a comprehensive suite of metrics to evaluate performance from both a regression and
classification perspective.

Usage:
    python test.py --model_path path/to/your/model --test_data_path path/to/test.csv

Arguments:
    --model_path (str): Required. Path to the directory containing the saved model,
                        tokenizer, and configuration files.
    --test_data_path (str): Required. Path to the test data CSV file. The CSV
                            must contain 'text' and 'int_score' columns.
    --device (str): Optional. Device to run inference on ('cuda' or 'cpu').
                    Defaults to 'cuda' if available.
    --batch_size (int): Optional. Batch size for inference. Defaults to 32.
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
# To calculate Krippendorff's Alpha, you might need to install a library.
# `simpledorff` is a great choice: pip install simpledorff
# If you don't have it, you can comment out the related lines.
# try:
#     from simpledorff import calculate_krippendorffs_alpha_for_df
# except ImportError:
#     print("Warning: `simpledorff` is not installed. Skipping Krippendorff's Alpha calculation.")
#     print("To install, run: pip install simpledorff")
#     calculate_krippendorffs_alpha_for_df = None


def run_inference(model, tokenizer, test_dataset, device, batch_size):
    """
    Runs inference on the test dataset and returns raw predictions and true labels.
    """
    # Set up DataLoader
    data_collator = lambda data: {'input_ids': torch.stack([s['input_ids'] for s in data]),
                                  'attention_mask': torch.stack([s['attention_mask'] for s in data])}
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    model.eval()
    all_raw_preds = []
    
    print("Running inference on the test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Move batch to the correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # The output for regression is a single logit per sample
            raw_preds = outputs.logits.squeeze(-1).cpu().numpy()
            all_raw_preds.extend(raw_preds)
            
    return np.array(all_raw_preds)


def main(args):
    """Main function to run the evaluation."""
    print("--- Starting Model Evaluation ---")
    
    # 1. Setup Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Model and Tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3. Load and Prepare Test Data
    print(f"Loading test data from: {args.test_data_path}")
    test_df = pd.read_csv(args.test_data_path)
    
    # Ensure necessary columns exist
    if 'text' not in test_df.columns or 'int_score' not in test_df.columns:
        print("Error: Test CSV must contain 'text' and 'int_score' columns.")
        return

    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text', 'int_score'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    true_labels = np.array(test_df['int_score'])

    # 4. Run Inference
    raw_predictions = run_inference(model, tokenizer, test_dataset, device, args.batch_size)

    # 5. Post-Process Predictions
    # The model was trained as a regression model, so we clip and round the outputs.
    clipped_predictions = np.clip(raw_predictions, 0, 4)
    final_predictions = np.round(clipped_predictions).astype(int)

    # 6. Calculate and Display Metrics
    print("\n--- Evaluation Metrics ---")
    labels = sorted(np.unique(true_labels))
    if len(labels) != 5:
        labels = [0,1,2,3,4] # Ensure we have 5 classes for classification metrics    
    target_names = [f"Class {i}" for i in labels]
    print(f"target_names: {target_names}")

    # a. Regression Metric
    mse = mean_squared_error(true_labels, raw_predictions)
    print(f"\n[Regression Metric]")
    print(f"  - Mean Squared Error (MSE/MSD): {mse:.4f}\n")

    # b. Classification Metrics
    print("[Classification Metrics]")
    accuracy = accuracy_score(true_labels, final_predictions)
    f1_macro = f1_score(true_labels, final_predictions, average='macro')
    f1_weighted = f1_score(true_labels, final_predictions, average='weighted')
    
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1-Score (Macro): {f1_macro:.4f}")
    print(f"  - F1-Score (Weighted): {f1_weighted:.4f}\n")

    # c. Full Classification Report
    print("[Classification Report]")
    report = classification_report(true_labels, final_predictions, target_names=target_names, zero_division=0)
    print(report)

    # d. Confusion Matrix
    print("[Confusion Matrix]")
    cm = confusion_matrix(true_labels, final_predictions, labels=labels)
    print("        " + " ".join([f"{lbl:<5}" for lbl in labels]))
    print("       " + "-"*35)
    for i, row in enumerate(cm):
        print(f"True {labels[i]} |", " ".join([f"{val:<5}" for val in row]))
    print("\n")


    # # e. Krippendorff's Alpha
    # if calculate_krippendorffs_alpha_for_df:
    #     print("[Agreement Metric]")
    #     # Create a DataFrame in the format simpledorff expects
    #     kripp_df = pd.DataFrame({
    #         'doc_id': test_df.index,
    #         'text': test_df['text'],
    #         'model': final_predictions,
    #         'human': true_labels
    #     })
        
    #     try:
    #         k_alpha = calculate_krippendorffs_alpha_for_df(kripp_df,
    #                                                         text_col='text',
    #                                                         unit_col='doc_id',
    #                                                         annotator_cols=['model', 'human'])
    #         print(f"  - Krippendorff's Alpha (Model vs. Human): {k_alpha:.4f}\n")
    #     except Exception as e:
    #         print(f"Could not calculate Krippendorff's Alpha. Error: {e}")


    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned educational content model.")
    
    parser.add_argument(
        "--model_path",
        default = "Davidozito/xlm-roberta-danish-educational-scorer-zeroshot",
        type=str,
        help="Path to the directory containing the saved model and tokenizer."
    )
    parser.add_argument(
        "--test_data_path",
        default = "self_annotation/test_uCarl.csv",
        type=str,
        help= "Path to you test data CSV file. The CSV must contain 'text' and 'int_score' columns."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to use for inference ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference."
    )
    
    args = parser.parse_args()
    main(args)