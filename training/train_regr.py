"""Training script for the Danish educational score model."""
import sys
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import evaluate
import time
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

# path setup to import data processing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset

# Load config
print("Loading config...")
with open("training/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

val_split = config.get("val_split", 0.1) # Use config for validation split

model_name = config["model_name"]
model_dir = config["model_dir"]
results_dir = config["results_dir"]
# num_labels should be 1 for regression
# num_labels = config.get("num_labels", 1) # Redundant, will be set by problem_type

num_danish_samples = config.get("num_danish_samples", 0)  # Number of Danish samples to include
num_english_samples = config.get("num_english_samples", 0)  # Number of English samples to include

learning_rate = float(config.get("learning_rate", 2e-5))
num_train_epochs = config.get("num_train_epochs", 3)
per_device_train_batch_size = config.get("per_device_train_batch_size", 8)
per_device_eval_batch_size = config.get("per_device_eval_batch_size", 8)
evaluation_strategy = config.get("evaluation_strategy", "epoch")
logging_strategy = config.get("logging_strategy", "epoch")
save_strategy = config.get("save_strategy", "epoch")
weight_decay = config.get("weight_decay", 0.01)

# Load data
# print(
#     f"Loading {num_english_samples} English and {num_danish_samples} Danish samples..."
# )
# df = get_merged_dataset(
#     english_data_amount=num_english_samples,
#     danish_data_amount=num_danish_samples,
# )
# print(f"Loaded dataset with {len(df)} samples.")

df = pd.read_csv("data/english_fineweb_merged_data.csv")

# pick out 100 random samples
df = df.sample(n=100, random_state=42)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["text", "int_score"]])

# Ensure score is an integer between 0 and 4
dataset = dataset.map(
    lambda x: {"score": int(np.clip(round(float(x["int_score"])), 0, 4))}
)

# Cast to ClassLabel *after* clipping/rounding if you want stratification based on the final integer values
# This is primarily for stratify_by_column. The actual labels for regression training will be float.
dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))

dataset = dataset.train_test_split(
    train_size=1 - val_split, seed=42, stratify_by_column="score"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    # Some models like GPT2 don't have a default pad token, eos_token is often used instead
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    # Tokenize the text
    # Added padding=True here - DataCollatorWithPadding will handle variable lengths
    # but adding padding=True *can* sometimes help with older tokenizers, though less necessary with DataCollator
    # truncation=True is important to handle long texts
    batch = tokenizer(examples["text"], truncation=True)

    # Prepare labels as float32 for regression loss
    # The 'score' column contains the integer value after clipping/casting
    #batch["labels"] = np.array(examples["score"], dtype=np.float32).?reshape(-1, 1)
    batch["labels"] = np.float32(examples["score"])  # Reshape for single output regression
    return batch

dataset = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

print("Loading model...")
# --- REG-MODIFICATION START ---
# Set problem_type="regression" and num_labels=1
# Add dropout = 0.0 as seen in the example
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1, # 1 output neuron for regression
    # problem_type="regression", # Explicitly set problem type
    classifier_dropout=0.0, # As in the inspiring example
    hidden_dropout_prob=0.0, # As in the inspiring example (applies to base model encoder)
    output_hidden_states=False # Keep this False unless you need them
)
# --- REG-MODIFICATION END ---

# Freezing base model parameters
# This looks correct for freezing the core transformer layers

#might need changes...:
print("Freezing base model parameters...")
for param in model.base_model.parameters():
    param.requires_grad = False
print("Base model parameters frozen.")

# create a timestamp for the run
timestamp = time.strftime("%Y%m%d-%H%M%S")

print("Setting up training arguments...")
training_args = TrainingArguments(
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=100,
    learning_rate=3e-4,
    num_train_epochs=20,
    seed=0,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    eval_on_start=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=True,
)

"""
    output_dir=model_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    eval_strategy=evaluation_strategy,
    logging_strategy=logging_strategy,
    save_strategy=save_strategy,
    logging_first_step=True,
    logging_dir=os.path.join(results_dir, "logs"),
    report_to="wandb",
    run_name=config.get("run_name", "Regression_model" + timestamp), # Use config for run name
    load_best_model_at_end=True if evaluation_strategy != "no" else False, # Added common practice
    metric_for_best_model="f1_macro" if evaluation_strategy != "no" else None, # Added common practice
    greater_is_better=True if evaluation_strategy != "no" else None, # Added common practice
    # bf16=True, # Add this if your hardware supports it and you want faster training/less memory
)
"""


def compute_metrics(eval_pred):
    """
    Compute precision, recall, F1, and accuracy by rounding regression predictions.
    """
    # logits is the single output value from the regression head
    logits, labels = eval_pred

    # Squeeze to remove the dimension of size 1 (from num_labels=1)
    # Round to get integer predictions and labels
    # Clip predictions to the valid range [0, 4]
    # Cast to int for classification metrics
    preds = np.round(logits.squeeze()).clip(0, 4).astype(int)
    labels = np.round(labels.squeeze()).astype(int) # labels should already be integers, but rounding is safe

    # Load evaluation metrics - already done in the example, keep as is
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    #MeanSquaredError = evaluate.load("mse")

    # Compute metrics using macro average for multi-class classification
    # Handle cases where a class might not be present in predictions or labels to avoid errors
    # Example: zero_division=0 returns 0 for metrics if there are no true samples for a class

    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    # Accuracy metric does not have average and does not need/support zero_division
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    # Mean Squared Error for regression evaluation
    mse = ((logits.squeeze() - labels) ** 2).mean()

    # Print detailed classification report and confusion matrix
    # Ensure target_names match the possible score values (0-5)
    target_names = [str(i) for i in range(6)]
    try:
        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        print("Validation Report:\n" + report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("Labels:", np.unique(labels))
        print("Predictions:", np.unique(preds))
        report = "Error generating report."

    try:
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:\n" + str(cm))
    except ValueError as e:
         print(f"Could not generate confusion matrix: {e}")
         cm = "Error generating matrix."


    # Return metrics as a dictionary
    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
        "mse": mse,
        "classification_report": report,
    }


print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer, # Pass tokenizer for data collator
    data_collator=data_collator, # Explicitly use data collator
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    # Added some print statements from the example's structure
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 20)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 20)

    print("Starting training…")
    train_result = trainer.train()
    print("Training complete.")

    print("Evaluating on validation set…")
    # Evaluate again after loading the best model if load_best_model_at_end is True
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    # save model in the model directory
    # save to model to the "model_dir/model_weights" directory
    final_model_save_path = os.path.join(model_dir, "model_weights")

    print(f"Saving model to {final_model_save_path}...")
    # Create directory if it doesn't exist
    os.makedirs(final_model_save_path, exist_ok=True)
    trainer.save_model(final_model_save_path)
    # Also save tokenizer and trainer state if needed for resuming or inference
    tokenizer.save_pretrained(final_model_save_path)
    trainer.state.save_to_json(os.path.join(final_model_save_path, "trainer_state.json"))

    print("Training and evaluation complete.")
    print("Done.")