"""Training script for the Danish educational score model."""

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
import wandb
from datasets import Dataset, ClassLabel
import yaml

# path setup to import data processing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.data_process import get_merged_dataset

# from training.metrics import compute_metrics # Assuming metrics.py is in training/
# New compute metrics, that supports classification
from archive.train_clas import compute_metrics

from training.utils import set_seed  # Assuming utils.py is in training/

seed_ = 42
set_seed(seed_)  # Set random seed for reproducibility


class CustomCDWTrainer(Trainer):
    def __init__(self, *args, class_weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights_tensor.to(logits.device)
        )

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def load_config(config_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess(examples, tokenizer):
    batch = tokenizer(examples["text"], truncation=True)
    batch["labels"] = np.int64(
        examples["score"]
    )  # Changed to np.int64 for classification
    return batch


def main(
    val_split,
    model_name,
    hub_repo_id,
    num_danish_samples,
    num_english_samples,
    learning_rate,
    num_train_epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    evaluation_strategy,
    eval_steps,
    save_strategy,
    config,
):
    # Load and process data
    df = get_merged_dataset(
        english_data_amount=num_english_samples, danish_data_amount=num_danish_samples
    )
    # Removed df["int_score"] = (df["int_score"] * 4 / 5) as it's for regression scaling
    dataset = Dataset.from_pandas(df[["text", "int_score"]])
    dataset = dataset.rename_column("int_score", "score")

    # Cast to ClassLabel for stratification and classification
    dataset = dataset.map(
        lambda x: {"score": int(np.clip(round(float(x["score"])), 0, 4))}
    )
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))

    dataset = dataset.train_test_split(
        train_size=1 - val_split, seed=42, stratify_by_column="score"
    )

    train_dataset, val_dataset = dataset["train"], dataset["test"]

    # Calculate Class Weights for CDW-CE Loss
    print("Calculating class weights for CDW-CE loss...")
    class_counts = pd.Series(train_dataset["score"]).value_counts().sort_index()
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).tolist()
    print(f"Calculated Class Weights: {class_weights}")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Load model, tokenizer, and prepare datasets
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,  # 5 classes for classification (0-4)
        problem_type="single_label_classification",  # Changed for classification
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=512
    )  # Added model_max_length

    # Process datasets
    train_dataset = train_dataset.map(
        lambda examples: preprocess(examples, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: preprocess(examples, tokenizer), batched=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Freeze base model layers
    print("Freezing base model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    print("Base model parameters frozen.")
    # Removed freezing/unfreezing classifier.parameters(), it's trainable by default.

    # Set up training arguments
    training_args = TrainingArguments(
        # Training Parameters
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        # Learning Rate Scheduling
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # Evaluation and Logging
        save_strategy=save_strategy,
        save_steps=eval_steps,
        save_total_limit=1,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        # Removed eval_on_start=True as it's not a standard TrainingArguments parameter
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # Changed for classification
        greater_is_better=True,  # Changed for classification
        use_mps_device=True,
        # other parameters
        seed=seed_,
        bf16=False,
        # WandB and Hub Integration
        output_dir=f"./results-temp/{hub_repo_id.split('/')[-1]}",
        push_to_hub=True,
        hub_model_id=hub_repo_id,
        hub_strategy="end",
    )

    trainer = CustomCDWTrainer(  # Changed to CustomCDWTrainer
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights_tensor=class_weights_tensor,  # Pass class weights
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
        sys.exit(1)  # Use sys.exit(1) for error exit

    few_shot_danish_samples = [250]
    all_results = {}
    experiment_group_name = f"FewShot-Danish-Epochs-{time.strftime('%m.%d')}"

    for dan_samples in few_shot_danish_samples:
        print(f"\n--- Starting training for {dan_samples} Danish samples ---")

        current_config = base_config.copy()
        current_config["num_danish_samples"] = dan_samples
        current_config["num_english_samples"] = (
            0  # Ensure only Danish samples are used for fine-tuning here
        )

        # Removed this block as it contradicts few-shot purpose
        # if dan_samples == 5000:
        #     current_config["model_name"] = "FacebookAI/xlm-roberta-base"
        #     current_config["num_english_samples"] = 0

        # Extract config parameters
        model_name = current_config[
            "model_name"
        ]  # This should be the pre-trained EN-only model
        num_train_epochs = current_config.get("num_train_epochs", 3)
        per_device_train_batch_size = current_config.get(
            "per_device_train_batch_size", 16
        )
        val_split = current_config.get("val_split", 0.1)
        num_english_samples = current_config.get(
            "num_english_samples", 0
        )  # This is 0 for few-shot fine-tuning phase

        # Dynamically calculate eval_steps for 4 evaluations per epoch
        # Use total samples (EN + DA) for training size calculation if it was a combined dataset scenario.
        # But for few-shot, train_set_size should reflect the *danish* samples used for fine-tuning.
        train_set_size = int(
            dan_samples * (1 - val_split)
        )  # Using dan_samples for relevant train_set_size
        steps_per_epoch = max(1, train_set_size // per_device_train_batch_size)
        eval_steps = max(1, steps_per_epoch // 4)  # Evaluate 4 times per epoch
        print(f"Dynamic eval_steps calculated: {eval_steps}")

        run = wandb.init(
            project="danish-educational-scorer",
            group=experiment_group_name,
            name=f"fewshot-{dan_samples}-classification",  # Changed run name for clarity
            config=current_config,
            reinit=True,
        )

        hub_username = base_config["hub_username"].strip("/")
        repo_name = (
            f"fewshot-CDW-CE-{dan_samples}-samples"  # Changed repo name for clarity
        )
        hub_repo_id = f"{hub_username}/{repo_name}"
        trainer, metrics = main(
            val_split=val_split,
            model_name=model_name,  # This is the warm-start model
            hub_repo_id=hub_repo_id,
            num_danish_samples=dan_samples,
            num_english_samples=num_english_samples,  # This is 0
            learning_rate=float(current_config.get("learning_rate", 3e-4)),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=current_config.get(
                "per_device_eval_batch_size", 32
            ),
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            config=current_config,
        )
        if metrics:
            all_results[repo_name] = metrics

        run.finish()

    print("\n--- All few-shot classification training runs complete ---")
    print("Summary of all results:")
    for repo, res in all_results.items():
        print(f"{repo}: {res}")
