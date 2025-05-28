import sys
import os

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
import os
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
    save_strategy=config.get("save_strategy", "epoch"),
    logging_dir=os.path.join(results_dir, "logs"),
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training complete. Saving model...")
    # Save the trained model to the model directory
    trainer.save_model(model_dir)

    # Save training and validation metrics to CSV
    print("Saving training/validation metrics...")
    metrics = trainer.state.log_history
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(results_dir, "train_eval_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved training/validation metrics to {metrics_path}")
