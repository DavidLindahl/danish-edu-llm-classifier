import sys
import os
from sklearn.metrics import mean_squared_error
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
from sklearn.model_selection import train_test_split
import yaml
from data_processing.data_process import get_merged_dataset

# Load config
print("Loading config...")
with open("training/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

val_split = 0.1  # Proportion of data to use for validation

model_name = config["model_name"]
model_dir = config["model_dir"]
results_dir = config["results_dir"]
num_labels = config.get("num_labels", 1)

num_danish_samples = config["num_danish_samples"]  # Number of Danish samples to include
num_english_samples = config[
    "num_english_samples"
]  # Number of English samples to include

# Load data
print(
    f"Loading {num_english_samples} English and {num_danish_samples} Danish samples..."
)
df = get_merged_dataset(
    english_data_amount=num_english_samples,
    danish_data_amount=num_danish_samples,
)
print(f"Loaded dataset with {len(df)} samples.")

# Assume columns: text, score
texts = df["text"].astype(str).tolist()
labels = df["score"].astype(float).tolist()

# Split train/eval
print(f"Splitting data into train and validation sets (val_split={val_split})...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=val_split, random_state=42
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch, truncation=True, padding="max_length", max_length=500)


class EnglishDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=500
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


print("Preparing datasets...")
train_dataset = EnglishDataset(train_texts, train_labels)
val_dataset = EnglishDataset(val_texts, val_labels)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="regression"
)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=config.get("num_train_epochs", 3),
    per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
    per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
    weight_decay=config.get("weight_decay", 0.01),
    eval_strategy=config.get("evaluation_strategy", "epoch"),
    logging_strategy=config.get("logging_strategy", "epoch"),
    save_strategy=config.get("save_strategy", "epoch"),
    logging_first_step=True,
    logging_dir=os.path.join(results_dir, "logs"),
    report_to="wandb",
    run_name="danish-edu-llm-run", 
)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse}


print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    # 1. Print config & splits (sanity checks)
    # 2. Initialize tokenizer, datasets, model (with problem_type="regression")
    # 3. Build TrainingArguments (with logging_strategy="epoch")
    # 4. Create Trainer (with compute_metrics)
    print("Starting training…")
    train_result = trainer.train()
    print("Training complete.")
    
    print("Evaluating on validation set…")
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")
