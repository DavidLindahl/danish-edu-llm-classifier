"""
test.py

Evaluates a model, prints metrics, and optionally saves the predictions to a CSV file.
It automatically handles both regression and classification models.

Usage:
    # Run and print metrics
    python test/test.py --model_path "YourUsername/your-model"

    # Run, print metrics, AND save results
    python test/test.py --model_path "YourUsername/your-model" --output_file "test/inference/zeroshot_results.csv"
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

def run_inference(model, test_dataset, device, batch_size):
    """Runs inference and returns the raw model logits."""
    data_collator = lambda data: {'input_ids': torch.stack([s['input_ids'] for s in data]),
                                  'attention_mask': torch.stack([s['attention_mask'] for s in data])}
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.extend(outputs.logits.cpu().numpy())
    return np.array(all_logits)


def main(args):
    """Main function to run the evaluation."""
    print("--- Starting Model Evaluation ---")
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model and Data
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test_df = pd.read_csv(args.test_data_path)
    
    # Prepare dataset for tokenization
    test_dataset = Dataset.from_pandas(test_df)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_df.columns.tolist())
    
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    true_labels = np.array(test_df['int_score'])
    
    # 2. Run Inference
    raw_logits = run_inference(model, test_dataset, device, args.batch_size)

    # 3. Process Predictions based on model type
    if model.config.num_labels == 1:
        print("Detected Regression Model (num_labels=1).")
        raw_predictions = raw_logits.squeeze(-1)
        final_predictions = np.round(np.clip(raw_predictions, 0, 4)).astype(int)
    else:
        print(f"Detected Classification Model (num_labels={model.config.num_labels}).")
        final_predictions = np.argmax(raw_logits, axis=1)
        raw_predictions = np.full_like(true_labels, np.nan, dtype=float)

    # 4. Calculate and Display Metrics (this part is the same)
    print("\n--- Evaluation Metrics ---")
    labels = list(range(5))
    target_names = [f"Class {i}" for i in labels]
    
    mse = mean_squared_error(true_labels, raw_predictions)
    accuracy = accuracy_score(true_labels, final_predictions)
    f1_macro = f1_score(true_labels, final_predictions, average='macro', zero_division=0)
    
    print(f"  - MSE: {mse:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Macro: {f1_macro:.4f}\n")
    print(classification_report(true_labels, final_predictions, target_names=target_names, zero_division=0))
    
    print("[Confusion Matrix]")
    cm = confusion_matrix(true_labels, final_predictions, labels=labels)
    print("        " + " ".join([f"{lbl:<5}" for lbl in labels]))
    print("       " + "-"*35)
    for i, row in enumerate(cm):
        print(f"True {labels[i]} |", " ".join([f"{val:<5}" for val in row]))
    print("\n")

 # --- NEW: Automatically save results to a dynamic path ---
    # 1. Get the base name of the model from its path
    model_basename = os.path.basename(args.model_path)
    # 2. Define the output directory and create it if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    # 3. Create the full dynamic path for the output file
    output_file_path = os.path.join(output_dir, f"{model_basename}_results.csv")
    
    print(f"\nSaving inference results to {output_file_path}...")
    results_df = pd.DataFrame({
        'text': test_df['text'],
        'true_label': true_labels,
        'predicted_label': final_predictions,
        'raw_prediction': raw_predictions
    })
    results_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print("Results saved successfully.")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned educational content model.")
    
    # We no longer need an argument for the output file, as it's generated automatically
    parser.add_argument(
        "--model_path",
        default="Davidozito/xlm-roberta-danish-educational-scorer-zeroshot",
        type=str,
        help="Path or Hub ID of the model to evaluate."
    )
    parser.add_argument(
        "--test_data_path",
        default="data/fineweb-c_danish.csv",
        type=str,
        help="Path to the prepared test data CSV file."
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