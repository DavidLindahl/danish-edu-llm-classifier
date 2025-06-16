"""
test_all_models.py

Script for evaluating multiple fine-tuned models and summarizing the results.
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


def run_inference(model, test_loader, device):
    """Runs inference on the test dataloader and returns raw predictions."""
    model.eval()
    all_raw_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            raw_preds = outputs.logits.squeeze(-1).cpu().numpy()
            all_raw_preds.extend(raw_preds)
    return np.array(all_raw_preds)


def evaluate_single_model(model_path, test_data_path, device_str, batch_size):
    """Loads a single model, evaluates it, and returns a dictionary of metrics."""
    print(f"\n--- Evaluating Model: {model_path} ---")
    
    # 1. Setup Device, Model, and Tokenizer
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Load and Prepare Test Data
    test_df = pd.read_csv(test_data_path)
    test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_df.columns.tolist())
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    data_collator = lambda data: {'input_ids': torch.stack([s['input_ids'] for s in data]),
                                  'attention_mask': torch.stack([s['attention_mask'] for s in data])}
    test_loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    true_labels = np.array(test_df['int_score'])

    # 3. Run Inference and Process Predictions
    raw_predictions = run_inference(model, test_loader, device)
    final_predictions = np.round(np.clip(raw_predictions, 0, 4)).astype(int)

    # 4. Calculate and Display Metrics
    print("\n[Classification Report]")
    print(classification_report(true_labels, final_predictions, labels=list(range(5)), zero_division=0))
    
    # --- ADDED: Display Confusion Matrix ---
    print("[Confusion Matrix]")
    labels = list(range(5))
    cm = confusion_matrix(true_labels, final_predictions, labels=labels)
    print("        " + " ".join([f"Pred {lbl}" for lbl in labels]))
    print("       " + "-" * 37)
    for i, row in enumerate(cm):
        print(f"True {labels[i]} |", " ".join([f"{val:<5}" for val in row]))
    # --- END OF ADDITION ---

    # 5. Return summary metrics as a dictionary
    return {
        "model_name": os.path.basename(model_path),
        "mse": mean_squared_error(true_labels, raw_predictions),
        "accuracy": accuracy_score(true_labels, final_predictions),
        "f1_macro": f1_score(true_labels, final_predictions, average='macro', zero_division=0),
        "f1_weighted": f1_score(true_labels, final_predictions, average='weighted', zero_division=0)
    }


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATHS = [
    "Davidozito/xlm-roberta-danish-educational-scorer-zeroshot",
    "Davidozito/xlm-roberta-danish-educational-scorer-full-finetuning",
    "Davidozito/xlm-roberta-danish-educational-scorer-zeroshot-fewshot-2500",
    "Davidozito/xlm-roberta-danish-educational-scorer-zeroshot-fewshot-1000",
    "Davidozito/xlm-roberta-danish-educational-scorer-zeroshot-fewshot-250",
    # "Davidozito/xlm-roberta-danish-educational-scorer-fewshot-250",
    # "Davidozito/xlm-roberta-danish-educational-scorer-fewshot-1000",
    # "Davidozito/xlm-roberta-danish-educational-scorer-fewshot-2500",
    # "Davidozito/xlm-roberta-danish-educational-scorer-fewshot-5000",
]
        
    TEST_DATA_PATH = "self_annotation/test_final.csv"
    OUTPUT_CSV_PATH = "test/test_results_100.csv"
    DEVICE = "cpu"
    BATCH_SIZE = 32

    # --- Main Loop ---
    all_results = []
    for path in MODEL_PATHS:    
        metrics = evaluate_single_model(path, TEST_DATA_PATH, DEVICE, BATCH_SIZE)
        all_results.append(metrics)

    # --- Summarize and Save Results ---
    if not all_results:
        print("No models were evaluated. Exiting.")
        exit()
        
    results_df = pd.DataFrame(all_results)

    print("\n\n--- 📊 Summary of All Model Results ---")
    print(results_df.to_string())

    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Results successfully saved to: {OUTPUT_CSV_PATH}")