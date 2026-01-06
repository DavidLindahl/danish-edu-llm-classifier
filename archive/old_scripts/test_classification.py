"""
test_classification.py

Script for evaluating multiple fine-tuned models with robust error handling for classification tasks.
"""

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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Handle different output shapes
            logits = outputs.logits
            if len(logits.shape) == 2 and logits.shape[1] == 1:
                # Regression case: squeeze the last dimension
                raw_preds = logits.squeeze(-1).cpu().numpy()
            elif len(logits.shape) == 2 and logits.shape[1] > 1:
                # Classification case: take argmax or use softmax
                raw_preds = torch.argmax(logits, dim=-1).cpu().numpy().astype(float)
            else:
                # Fallback: flatten and hope for the best
                raw_preds = logits.flatten().cpu().numpy()

            all_raw_preds.extend(raw_preds)

    return np.array(all_raw_preds)


def safe_classification_report(y_true, y_pred, labels=None):
    """Safely generate classification report with error handling."""
    try:
        # Ensure both arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Ensure they have the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        return classification_report(y_true, y_pred, labels=labels, zero_division=0)
    except Exception as e:
        return f"Error generating classification report: {str(e)}\ny_true shape: {y_true.shape if 'y_true' in locals() else 'undefined'}\ny_pred shape: {y_pred.shape if 'y_pred' in locals() else 'undefined'}"


def safe_confusion_matrix(y_true, y_pred, labels=None):
    """Safely generate confusion matrix with error handling."""
    try:
        # Ensure both arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Ensure they have the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        return confusion_matrix(y_true, y_pred, labels=labels)
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")
        return None


def evaluate_single_model(model_path, test_data_path, device_str, batch_size):
    """Loads a single model, evaluates it, and returns predictions with test data."""
    print(f"\n--- Evaluating Model: {model_path} ---")

    try:
        # 1. Setup Device, Model, and Tokenizer
        device = torch.device(
            device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Print model info for debugging
        print(
            f"Model config: num_labels={model.config.num_labels}, problem_type={getattr(model.config, 'problem_type', 'not specified')}"
        )

        # 2. Load and Prepare Test Data
        test_df = pd.read_csv(test_data_path)
        test_dataset = Dataset.from_pandas(test_df)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=512
            )

        tokenized_dataset = test_dataset.map(
            tokenize_function, batched=True, remove_columns=test_df.columns.tolist()
        )
        tokenized_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )

        data_collator = lambda data: {
            "input_ids": torch.stack([s["input_ids"] for s in data]),
            "attention_mask": torch.stack([s["attention_mask"] for s in data]),
        }
        test_loader = DataLoader(
            tokenized_dataset, batch_size=batch_size, collate_fn=data_collator
        )

        true_labels = np.array(test_df["int_score"])

        # 3. Run Inference and Process Predictions
        raw_predictions = run_inference(model, test_loader, device)

        # Debug: Print raw prediction info
        print(f"Raw predictions shape: {raw_predictions.shape}")
        print(f"Raw predictions sample: {raw_predictions[:5]}")
        print(
            f"Raw predictions range: [{raw_predictions.min():.3f}, {raw_predictions.max():.3f}]"
        )

        # Ensure raw_predictions is 1D and same length as true_labels
        raw_predictions = raw_predictions.flatten()
        min_len = min(len(true_labels), len(raw_predictions))
        true_labels = true_labels[:min_len]
        raw_predictions = raw_predictions[:min_len]

        # Process final predictions
        final_predictions = np.round(np.clip(raw_predictions, 0, 4)).astype(int)

        # 4. Calculate and Display Metrics
        print("\n[Classification Report]")
        report = safe_classification_report(
            true_labels, final_predictions, labels=list(range(5))
        )
        print(report)

        print("[Confusion Matrix]")
        labels = list(range(5))
        cm = safe_confusion_matrix(true_labels, final_predictions, labels=labels)
        if cm is not None:
            print("        " + " ".join([f"Pred {lbl}" for lbl in labels]))
            print("       " + "-" * 37)
            for i, row in enumerate(cm):
                print(f"True {labels[i]} |", " ".join([f"{val:<5}" for val in row]))

        # 5. Create DataFrame with test data and predictions
        model_name = os.path.basename(model_path)
        result_df = test_df[["id", "text", "int_score"]].copy()[
            :min_len
        ]  # Ensure same length
        result_df["real_label"] = true_labels
        result_df[f"predicted_label_{model_name}"] = final_predictions
        result_df[f"raw_prediction_{model_name}"] = raw_predictions

        # 6. Calculate summary metrics with error handling
        try:
            mse = mean_squared_error(true_labels, raw_predictions)
            accuracy = accuracy_score(true_labels, final_predictions)
            f1_macro = f1_score(
                true_labels, final_predictions, average="macro", zero_division=0
            )
            f1_weighted = f1_score(
                true_labels, final_predictions, average="weighted", zero_division=0
            )
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            mse = accuracy = f1_macro = f1_weighted = -1.0

        metrics = {
            "model_name": model_name,
            "mse": mse,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "num_samples": len(true_labels),
        }

        return result_df, metrics

    except Exception as e:
        print(f"ERROR evaluating model {model_path}: {str(e)}")
        model_name = os.path.basename(model_path)
        return None, {
            "model_name": model_name,
            "mse": -1.0,
            "accuracy": -1.0,
            "f1_macro": -1.0,
            "f1_weighted": -1.0,
            "num_samples": 0,
            "error": str(e),
        }


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATHS = [
        # "Davidozito/zeroshot",
        "Davidozito/zeroshot-classification",
        "Davidozito/fewshot-CDW-CE-250-samples",
        "Davidozito/fewshot-CDW-CE-1000-samples",
        "Davidozito/fewshot-CDW-CE-2500-samples",
        "Davidozito/full-finetuning-classification",
        # "Davidozito/Full-finetune",
        # "Davidozito/fewshot-250-samples",
        # "Davidozito/fewshot-1000-samples",
        # "Davidozito/fewshot-2500-samples",
    ]

    TEST_DATA_PATH = "src/annotation/test_final.csv"
    OUTPUT_CSV_PATH = "archive/test_clas/test_classification_results.csv"
    METRICS_CSV_PATH = "archive/test_clas/test_classification_metrics.csv"
    DEVICE = "cpu"
    BATCH_SIZE = 32

    # --- Main Loop ---
    all_results = []
    all_metrics = []

    # Load test data once to create the base DataFrame
    test_df = pd.read_csv(TEST_DATA_PATH)
    combined_results = test_df[["id", "text", "int_score"]].copy()
    combined_results["real_label"] = test_df["int_score"]

    for path in MODEL_PATHS:
        result_df, metrics = evaluate_single_model(
            path, TEST_DATA_PATH, DEVICE, BATCH_SIZE
        )

        if result_df is not None:
            model_name = os.path.basename(path)

            # Ensure combined_results has the same length as result_df
            min_len = min(len(combined_results), len(result_df))
            combined_results = combined_results.iloc[:min_len].copy()
            result_df = result_df.iloc[:min_len].copy()

            # Add predictions to combined results
            combined_results[f"predicted_label_{model_name}"] = result_df[
                f"predicted_label_{model_name}"
            ]
            combined_results[f"raw_prediction_{model_name}"] = result_df[
                f"raw_prediction_{model_name}"
            ]

        all_metrics.append(metrics)

    # --- Save Results ---
    if not all_metrics:
        print("No models were evaluated successfully. Exiting.")
        exit()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save combined predictions
    combined_results.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Predictions successfully saved to: {OUTPUT_CSV_PATH}")

    # Save metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"✅ Metrics summary saved to: {METRICS_CSV_PATH}")

    print("\n\n--- 📊 Summary of All Model Results ---")
    print(metrics_df.to_string())

    print("\n--- 📋 Preview of Combined Results ---")
    print(f"Shape: {combined_results.shape}")
    print(combined_results.head())
