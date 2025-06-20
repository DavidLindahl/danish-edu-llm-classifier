"""Training script for the Danish educational score model."""
import sys
import os
import numpy as np
import time
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import yaml

from metrics import compute_metrics
from utils import set_seed

# path setup to import data processing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset

seed_ = 42
set_seed(seed_)  # Set random seed for reproducibility

def load_config(config_path):
    """Load configuration from YAML file."""
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess(examples, tokenizer):
    """Preprocess examples by tokenizing text and converting scores to float."""
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = np.float32(examples["score"]) 
    return batch


def main(val_split, model_name, model_dir, num_danish_samples, 
         num_english_samples, learning_rate, num_train_epochs, 
         per_device_train_batch_size, per_device_eval_batch_size, 
         evaluation_strategy, eval_steps, save_strategy, config):
    """Main training function that handles the entire training pipeline."""
    # All config parameters are now passed directly as arguments

    # Load data
    print(f"Loading {num_english_samples} English and {num_danish_samples} Danish samples...")
    df = get_merged_dataset(
        english_data_amount=num_english_samples,
        danish_data_amount=num_danish_samples,
    )

    # multiply all int_score with 4/5 to convert to 0-4 scale
    df["int_score"] = (df["int_score"] * 4 / 5)


    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df[["text", "int_score"]])

    # rename columns to match expected format
    dataset = dataset.rename_column("int_score", "score")

    # Cast to ClassLabel for stratification
    # dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))

    # Split dataset
    dataset = dataset.train_test_split(
        train_size=1 - val_split, seed=42
    )

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # 1 output neuron for regression
        problem_type="regression"

    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Process dataset
    dataset = dataset.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # Freeze base model parameters
    print("Freezing base model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    print("Base model parameters frozen.")

    # Create a timestamp for the run
    timestamp = time.strftime("%m.%d-%H.%M")

    # Set up training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        # --- Training Parameters ---
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        # --- Learning Rate Scheduling ---
        lr_scheduler_type="linear", # or "cosine"
        warmup_ratio=0.1, # 10% of training steps used for linear warmup


        # --- Evaluation and Logging ---
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        eval_steps=eval_steps,
        logging_steps=50,
        eval_on_start=True, # Baseline
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_mps_device=True,
        # --- other parameters ---
        seed=seed_,
        bf16=False,
    )

    # Initialize trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
    )

    # Print configuration and dataset info
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 20)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 20)

    # Train the model
    print("Starting training…")
    train_result = trainer.train()
    print("Training complete.")

    # Evaluate the model
    print("Evaluating on validation set…")
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    # Save the model
    model_save_name = f"{model_name[:4]}_Danish={num_danish_samples}_{timestamp}" 
    final_model_save_path = os.path.join(model_dir, model_save_name)
    print(f"Saving model to {final_model_save_path}...")
    os.makedirs(final_model_save_path, exist_ok=True)
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    trainer.state.save_to_json(os.path.join(final_model_save_path, "trainer_state.json"))
    
    print("Training and evaluation complete.")
    print("Done.")
    
    return trainer, eval_metrics


if __name__ == "__main__":
    # Parse command line arguments to allow for different config files
    config_path = "training/config/base.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract config parameters here in __main__
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
    
    # Run main training function with extracted parameters
    trainer, metrics = main(
        val_split, model_name, model_dir, num_danish_samples,
        num_english_samples, learning_rate, num_train_epochs,
        per_device_train_batch_size, per_device_eval_batch_size,
        evaluation_strategy, eval_steps, save_strategy,
        config
    )