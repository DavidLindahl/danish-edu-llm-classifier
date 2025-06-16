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
from datasets import Dataset, ClassLabel
import yaml

from metrics import compute_metrics

# path setup to import data processing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset


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

    print(f"Loaded dataset with {len(df)} samples.")

    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df[["text", "int_score"]])

    # Ensure score is an integer between 0 and 4
    dataset = dataset.map(
        lambda x: {"score": int(np.clip(round(float(x["int_score"])), 0, 4))}
    )

    # Cast to ClassLabel for stratification
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))

    # Split dataset
    dataset = dataset.train_test_split(
        train_size=1 - val_split, seed=42, stratify_by_column="score"
    )

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # 1 output neuron for regression
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0, 
        output_hidden_states=False 
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

    # Ensure the classifier head is trainable
    print("Ensuring classifier head parameters are trainable...")
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Classifier head parameters ensured trainable.")


    # Create a timestamp for the run
    timestamp = time.strftime("%m.%d-%H.%M")

    run_name = f"Danish_{num_danish_samples}_{timestamp}"


    # Set up training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        # --- Training Parameters ---
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        # --- Learning Rate Scheduling ---
        lr_scheduler_type="linear", # Added: Linear scheduler
        warmup_ratio=0.1, # Added: 10% of steps for warmup

        # --- Evaluation and Logging ---
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        eval_steps=eval_steps,
        eval_on_start=True, # Changed to True for baseline eval
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse_loss", # Changed to eval_mse_loss
        greater_is_better=False, # Changed to False for loss
        
        # --- Mixed Precision ---
        fp16=True, # Changed to True for V100 as per PDF
    
        # --- Other parameters ---
        seed=42,
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
    config_path = "training/config/fewshot.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Load configuration
    base_config = load_config(config_path) # Renamed to base_config

    # Define few-shot sample sizes
    # These are the N Danish samples you add for fine-tuning
    few_shot_danish_samples = [50, 100, 200, 500, 1000, 2500, 5000]
    
    all_results = {}

    for dan_samples in few_shot_danish_samples:
        print(f"\n--- Starting training for {dan_samples} Danish samples ---")
        
        # Create a mutable config for the current run
        current_config = base_config.copy()

        # If dan_samples is 5000, train on the XLM-RoBERTa base model
        if dan_samples == 5000:
            current_config["model_name"] = "FacebookAI/xlm-roberta-base"

        current_config["num_danish_samples"] = dan_samples
        current_config["num_english_samples"] = 0 # Explicitly set to 0 as per few-shot strategy

        # Extract config parameters for the current run
        val_split = current_config.get("val_split", 0.1)
        model_name = current_config["model_name"] # This is your English-trained model
        model_dir = current_config["model_dir"]
        num_danish_samples = current_config.get("num_danish_samples", 0)
        num_english_samples = current_config.get("num_english_samples", 0)
        learning_rate = float(current_config.get("learning_rate", 3e-4))
        num_train_epochs = current_config.get("num_train_epochs", 3) # Increased default epochs
        per_device_train_batch_size = current_config.get("per_device_train_batch_size", 16)
        per_device_eval_batch_size = current_config.get("per_device_eval_batch_size", 32)
        evaluation_strategy = current_config.get("evaluation_strategy", "steps") # Use steps
        save_strategy = current_config.get("save_strategy", "steps") # Use steps
        # eval_steps is not used with epoch strategy, no need to extract
        
        # Run main training function with extracted parameters
        trainer, metrics = main(
            val_split, model_name, model_dir, num_danish_samples,
            num_english_samples, learning_rate, num_train_epochs,
            per_device_train_batch_size, per_device_eval_batch_size,
            evaluation_strategy, save_strategy, # Pass without eval_steps
            current_config # Pass the current config for logging/callbacks
        )
        if metrics:
            all_results[f"Danish_{dan_samples}_samples"] = metrics
            print(f"Results for {dan_samples} samples: {metrics}")
        
    print("\n--- All few-shot training runs complete ---")
    print("Summary of all results:")
    for size, res in all_results.items():
        print(f"{size}: {res}")