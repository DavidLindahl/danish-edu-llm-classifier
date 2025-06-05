"""Training script for the Danish educational score model."""

import sys
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from datasets import Dataset, ClassLabel

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
num_english_samples = config["num_english_samples"]  # Number of English samples to include

learning_rate = config.get("learning_rate", 2e-5)





# Load data
print(
    f"Loading {num_english_samples} English and {num_danish_samples} Danish samples..."
)
df = get_merged_dataset(
    english_data_amount=num_english_samples,
    danish_data_amount=num_danish_samples,
)
print(f"Loaded dataset with {len(df)} samples.")

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["text", "score"]])
dataset = dataset.map(
    lambda x: {"score": int(np.clip(round(float(x["score"])), 0, 5))}
)
dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(6)]))
dataset = dataset.train_test_split(
    train_size=1 - val_split, seed=42, stratify_by_column="score"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess(examples):
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = np.array(examples["score"], dtype=np.float32)
    return batch

dataset = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="regression"
)

for param in model.base_model.parameters():
    param.requires_grad = False

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=config.get("num_train_epochs", 3),
    per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
    per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
    learning_rate=learning_rate,
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
    """Compute precision, recall, F1 and accuracy."""

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)

    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")[
        "f1"
    ]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)[
        "accuracy"
    ]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


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

    # save model in the model directory
    # save to model to the "model_dir+model_weights" directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, "model_weights")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Saving model...")
    trainer.save_model(model_dir)
    print(f"Model saved to {model_dir}")
    print("Training and evaluation complete.")
    print("Done.")
