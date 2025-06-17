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
import wandb
from datasets import Dataset, ClassLabel
import yaml

from metrics import compute_metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset

def load_config(config_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess(examples, tokenizer):
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = np.float32(examples["score"]) 
    return batch

def main(val_split, model_name, hub_repo_id, num_danish_samples, 
         num_english_samples, learning_rate, num_train_epochs, 
         per_device_train_batch_size, per_device_eval_batch_size, 
         evaluation_strategy, eval_steps, save_strategy, config):
    
    # Load and process data
    df = get_merged_dataset(english_data_amount=num_english_samples, danish_data_amount=num_danish_samples)
    dataset = Dataset.from_pandas(df[["text", "int_score"]])
    dataset = dataset.map(lambda x: {"score": int(np.clip(round(float(x["int_score"])), 0, 4))})
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))
    dataset = dataset.train_test_split(train_size=1 - val_split, seed=42, stratify_by_column="score")

    # Load model, tokenizer, and prepare datasets
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, val_dataset = dataset["train"], dataset["test"]

    # Freeze base model layers
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results-temp/{hub_repo_id.split('/')[-1]}",
        num_train_epochs=num_train_epochs, # Use epochs instead of max_steps
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        save_total_limit=1,
        fp16=False,
        use_mps_device=True,
        push_to_hub=True,
        hub_model_id=hub_repo_id,
        hub_strategy="end",
        seed=42,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer, 
        data_collator=data_collator, compute_metrics=compute_metrics,
    )

    print(f"Starting training for {num_train_epochs} epochs...")
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"Final validation metrics: {eval_metrics}")
    trainer.push_to_hub()
    return trainer, eval_metrics

if __name__ == "__main__":
    config_path = "training/config/fewshot.yaml"
    base_config = load_config(config_path)
    hub_username = base_config.get("hub_username")
    if not hub_username:
        print("ERROR: 'hub_username' must be set in your config YAML file.")
        exit()

    few_shot_danish_samples = [250, 1000, 2500]
    all_results = {}
    experiment_group_name = f"FewShot-Danish-Epochs-{time.strftime('%m.%d')}"

    for dan_samples in few_shot_danish_samples:
        print(f"\n--- Starting training for {dan_samples} Danish samples ---")
        
        current_config = base_config.copy()
        current_config["num_danish_samples"] = dan_samples
        current_config["num_english_samples"] = 0

        if dan_samples == 5000:
            current_config["model_name"] = "FacebookAI/xlm-roberta-base"
            current_config["num_english_samples"] = 0

        # Extract config parameters
        model_name = current_config["model_name"]
        num_train_epochs = current_config.get("num_train_epochs", 3)
        per_device_train_batch_size = current_config.get("per_device_train_batch_size", 16)
        val_split = current_config.get("val_split", 0.1)
        num_english_samples = current_config.get("num_english_samples", 0)

        # --- Dynamically calculate eval_steps for 4 evaluations per epoch ---
        train_set_size = int((num_english_samples + dan_samples) * (1 - val_split))
        steps_per_epoch = max(1, train_set_size // per_device_train_batch_size)
        eval_steps = max(1, steps_per_epoch // 4) # Evaluate 4 times per epoch
        print(f"Dynamic eval_steps calculated: {eval_steps}")

        run = wandb.init(
            project="danish-educational-scorer",
            group=experiment_group_name,
            name=f"fewshot-{dan_samples}-samples",
            config=current_config,
            reinit=True
        )

        repo_name = f"{model_name.split('/')[-1]}-fewshot-{dan_samples}"
        hub_repo_id = f"{hub_username}/{repo_name}"
        
        trainer, metrics = main(
            val_split=val_split,
            model_name=model_name,
            hub_repo_id=hub_repo_id,
            num_danish_samples=dan_samples,
            num_english_samples=num_english_samples,
            learning_rate=float(current_config.get("learning_rate", 3e-4)),
            num_train_epochs=num_train_epochs, # Pass epochs
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=current_config.get("per_device_eval_batch_size", 32),
            evaluation_strategy="steps",
            eval_steps=eval_steps, # Pass dynamic eval steps
            save_strategy="steps",
            config=current_config
        )
        if metrics:
            all_results[repo_name] = metrics
        
        run.finish()
        
    print("\n--- All few-shot training runs complete ---")
    print("Summary of all results:")
    for repo, res in all_results.items():
        print(f"{repo}: {res}")