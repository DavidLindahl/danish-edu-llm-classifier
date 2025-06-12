"""Training script using Deep Ordinal Regression with Optimal Transport Loss."""
import sys
import os
import numpy as np
import time
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, ClassLabel
import yaml

from metrics import compute_metrics

# path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset


def load_config(config_path):
    """Load configuration from YAML file."""
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess(examples, tokenizer):
    """Preprocess examples for classification."""
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = np.int64(examples["score"])
    return batch


def main(val_split, model_name, model_dir, num_danish_samples, 
         num_english_samples, learning_rate, num_train_epochs, 
         per_device_train_batch_size, per_device_eval_batch_size, 
         evaluation_strategy, eval_steps, save_strategy, config):
    """Main training function that handles the entire training pipeline."""
    
    print(f"Loading {num_english_samples} English and {num_danish_samples} Danish samples...")
    df = get_merged_dataset(
        english_data_amount=num_english_samples,
        danish_data_amount=num_danish_samples,
    )
    print(f"Loaded dataset with {len(df)} samples.")

    dataset = Dataset.from_pandas(df[["text", "int_score"]])
    dataset = dataset.map(
        lambda x: {"score": int(np.clip(round(float(x["int_score"])), 0, 4))}
    )
    # This remains a 5-class problem
    num_classes = 5
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(num_classes)]))
    
    dataset = dataset.train_test_split(
        train_size=1 - val_split, seed=42, stratify_by_column="score"
    )
    
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # --- 1. Define the Ordinal Optimal Transport Loss Function ---
    def ordinal_ot_loss_func(outputs, labels, num_items_in_batch=None):
        """
        Calculates the Earth Mover's Distance (1-Wasserstein distance) between
        the predicted and true cumulative distribution functions (CDFs).
        This naturally penalizes "far-away" incorrect predictions more heavily.
        """
        # Extract logits and get predicted probabilities via softmax
        logits = outputs.get("logits")
        pred_probs = torch.softmax(logits, dim=-1)
        
        # Calculate the predicted CDF
        # Shape: (batch_size, num_classes)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)

        # Create the true label CDF.
        # For a label `k`, the CDF is [0, ..., 0, 1, ..., 1] (0 up to k-1, 1 from k onwards)
        # We achieve this by first creating a one-hot encoding of the labels...
        with torch.no_grad():
            true_one_hot = nn.functional.one_hot(labels, num_classes=num_classes).float()
            # ...and then calculating the cumulative sum.
            true_cdf = torch.cumsum(true_one_hot, dim=-1)

        # Calculate the loss: the L1 norm (absolute difference) between the two CDFs.
        # We sum the absolute differences across the class dimension and then
        # average over the batch.
        loss = torch.mean(torch.sum(torch.abs(pred_cdf - true_cdf), dim=-1))
        
        return loss

    # --- 2. Set up Training Arguments ---
    timestamp = time.strftime("%m.%d-%H.%M")
    output_dir = f'./results/{timestamp}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=50,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        seed=42,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    # --- 3. Configure Model for Classification ---
    print("Loading model for CLASSIFICATION (with Ordinal Loss)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        problem_type="single_label_classification",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    # Process datasets
    train_dataset = train_dataset.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Freezing base model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    print("Base model parameters frozen.")
    
    # --- 4. Initialize the Trainer with the Ordinal OT loss function ---
    print("Initializing Trainer with Optimal Transport Ordinal Loss function...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        compute_loss_func=ordinal_ot_loss_func, # Pass the new loss function here
    )

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 20)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 20)

    print("Starting training…")
    trainer.train()
    print("Training complete.")

    print("Evaluating on validation set…")
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    model_save_name = f"{model_name.split('/')[-1][:4]}_Danish={num_danish_samples}_{timestamp}_OrdinalOT"
    final_model_save_path = os.path.join(model_dir, model_save_name)
    print(f"Saving model to {final_model_save_path}...")
    os.makedirs(final_model_save_path, exist_ok=True)
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    trainer.state.save_to_json(os.path.join(final_model_save_path, "trainer_state.json"))
    
    print("Training and evaluation complete.")
    print("Done.")


if __name__ == "__main__":
    config_path = "training/config/base.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_config(config_path)
    
    # Extract config parameters
    val_split = config.get("val_split", 0.1)
    model_name = config["model_name"]
    model_dir = config["model_dir"]
    num_danish_samples = config.get("num_danish_samples", 0)
    num_english_samples = config.get("num_english_samples", 0)
    learning_rate = float(config.get("learning_rate", 3e-4))
    num_train_epochs = config.get("num_train_epochs", 3)
    per_device_train_batch_size = config.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = config.get("per_device_eval_batch_size", 32)
    evaluation_strategy = config.get("evaluation_strategy", "steps")
    save_strategy = config.get("save_strategy", "steps")
    eval_steps = config.get("eval_steps", 50)
    
    main(
        val_split, model_name, model_dir, num_danish_samples,
        num_english_samples, learning_rate, num_train_epochs,
        per_device_train_batch_size, per_device_eval_batch_size,
        evaluation_strategy, eval_steps, save_strategy,
        config
    )