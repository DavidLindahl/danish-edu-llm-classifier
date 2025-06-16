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
import wandb # <--- ADD THIS LINE

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


def main(val_split, model_name, hub_repo_id, num_danish_samples, 
         num_english_samples, learning_rate, max_steps, 
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
    output_dir = f"./results-temp/{hub_repo_id.split('/')[-1]}" # Temp dir for checkpoints

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        # --- Training Parameters ---
        learning_rate=learning_rate,
        output_dir=output_dir, # <-- CHANGE to the new temp directory

        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        # --- Learning Rate Scheduling ---
        lr_scheduler_type="linear",
        warmup_ratio=0.1,

        # --- Evaluation and Logging ---
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        eval_steps=eval_steps,
        logging_steps=eval_steps, # <--- THIS IS THE CORRECTED LINE LOCATION
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        save_steps = eval_steps,  # Save every eval_steps
        # --- Mixed Precision ---
        fp16=False,
        use_mps_device=True,        
        # --- Other parameters (including pushing to hub or not) ---
        # Note: push_to_hub, hub_model_id, hub_strategy are not explicitly
        # set here as this is for local saving in this specific version.
        # If you want to push to hub, uncomment those lines from previous versions.
        push_to_hub=True,
        hub_model_id=hub_repo_id,
        hub_strategy="end",        
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
    print("Starting training...")
    train_result = trainer.train()
    print("Training complete.")

    # Evaluate the model
    print("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    # # Save the model
    # model_save_name = f"{model_name[:4]}_Danish={num_danish_samples}_{timestamp}" 
    # final_model_save_path = os.path.join(model_dir, model_save_name)
    # print(f"Saving model to {final_model_save_path}...")
    # os.makedirs(final_model_save_path, exist_ok=True)
    # trainer.save_model(final_model_save_path)
    # tokenizer.save_pretrained(final_model_save_path)
    # trainer.state.save_to_json(os.path.join(final_model_save_path, "trainer_state.json"))
    trainer.push_to_hub()    
    print("Training and evaluation complete.")
    print("Done.")
    
    return trainer, eval_metrics



if __name__ == "__main__":
    config_path = "training/config/fewshot.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    base_config = load_config(config_path)
    hub_username = base_config.get("hub_username")
    if not hub_username:
        print("ERROR: 'hub_username' must be set in your config YAML file.")
        exit()
    # Define your few-shot sizes
    few_shot_danish_samples = [250, 1000, 2500, 5000] # Adjust as needed
    
    all_results = {}
    MAX_TRAINING_STEPS = 300  # A reasonable number for all few-shot runs
    EVAL_STEPS = 30           # Evaluate 10 times during the training run
    # Define a group name for this entire experiment in W&B
    experiment_group_name = f"FewShot-Danish-{time.strftime('%m.%d')}"

    for dan_samples in few_shot_danish_samples:
        print(f"\n--- Starting training for {dan_samples} Danish samples ---")
        
        current_config = base_config.copy()
        current_config["num_danish_samples"] = dan_samples
        # Your logic for few-shot vs full training
        if dan_samples == 5000:
            current_config["model_name"] = "FacebookAI/xlm-roberta-base"
            current_config["num_english_samples"] = 0

        # Extract config parameters
        val_split = current_config.get("val_split", 0.1)
        model_name = current_config["model_name"]
        model_dir = current_config["model_dir"]
        num_danish_samples = current_config.get("num_danish_samples", 0)
        num_english_samples = current_config.get("num_english_samples", 0)
        learning_rate = float(current_config.get("learning_rate", 3e-4))
        num_train_epochs = current_config.get("num_train_epochs", 3)
        per_device_train_batch_size = current_config.get("per_device_train_batch_size", 16)
        per_device_eval_batch_size = current_config.get("per_device_eval_batch_size", 32)
        evaluation_strategy = current_config.get("evaluation_strategy", "steps")
        save_strategy = current_config.get("save_strategy", "steps")

        # --- DYNAMICALLY CALCULATE EVAL_STEPS FOR 50 EVALUATIONS ---
        total_train_samples = num_english_samples + num_danish_samples
        train_set_size = int(total_train_samples * (1 - val_split))
        steps_per_epoch = train_set_size // per_device_train_batch_size
        total_training_steps = steps_per_epoch * num_train_epochs
        eval_steps = max(1, total_training_steps // 25) # Divide total steps by 25
        print(f"Dynamic eval_steps calculated: {eval_steps} (for ~25 evaluations)")

        # --- START A NEW W&B RUN FOR THIS ITERATION ---
        run = wandb.init(
            project="danish-educational-scorer", # Or your project name
            group=experiment_group_name,
            name=f"fewshot-{dan_samples}-samples",
            config=current_config,
            reinit=True # Important for loops
        )
        base_model_name = model_name.split('/')[-1]
        repo_name = f"xlm-roberta-danish-educational-scorer-fewshot-{dan_samples}"
        hub_repo_id = f"{hub_username}/{repo_name}"        
        # Run main training function with the dynamically calculated eval_steps
        trainer, metrics = main(
            val_split, model_name, hub_repo_id, num_danish_samples,
            num_english_samples, learning_rate, MAX_TRAINING_STEPS,
            per_device_train_batch_size, per_device_eval_batch_size,
            evaluation_strategy, EVAL_STEPS, save_strategy,
            current_config
        )
        if metrics:
            all_results[f"Danish_{dan_samples}_samples"] = metrics
            print(f"Results for {dan_samples} samples: {metrics}")
        
        # --- END THE CURRENT W&B RUN ---
        run.finish()
        
    print("\n--- All few-shot training runs complete ---")
    print("Summary of all results:")
    for size, res in all_results.items():
        print(f"{size}: {res}")