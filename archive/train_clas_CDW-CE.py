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
    """Preprocess examples for CLASSIFICATION."""
    batch = tokenizer(examples["text"], truncation=True)
    # Ensure labels are integers for CrossEntropyLoss
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
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(5)]))
    
    # Split dataset FIRST to calculate weights ONLY on the training set
    dataset = dataset.train_test_split(
        train_size=1 - val_split, seed=42, stratify_by_column="score"
    )
    
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # --- 1. Calculate Class Weights for CDW-CE Loss ---
    print("Calculating class weights for CDW-CE loss...")
    class_counts = pd.Series(train_dataset['score']).value_counts().sort_index()
    # Inverse frequency weighting
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).tolist()
    print(f"Calculated Class Weights: {class_weights}")

    # --- 2. Set up Training Arguments (needed for device info) ---
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
        bf16=False,
    )

    # --- 3. Define the Custom Loss Function using a Closure ---
    # We define it here to "capture" the class_weights variable.
    # The Trainer will handle moving the weights tensor to the correct device.
    class_weights_tensor = torch.tensor(class_weights, device=training_args.device)

    def cdw_ce_loss_func(outputs, labels, num_items_in_batch=None):
        """
        Custom loss function for Class-Distribution-Weighted Cross-Entropy.
        This function signature matches what `Trainer.compute_loss` expects.
        """
        # Extract logits from the model outputs
        logits = outputs.get("logits")
        # Define the standard Cross-Entropy loss function, but with our calculated weights
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # Compute the loss
        loss = loss_fct(logits.view(-1, 5), labels.view(-1))
        return loss

    # --- 4. Configure Model for CLASSIFICATION ---
    print("Loading model for CLASSIFICATION...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,  # 5 classes for classification (0-4)
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
    
    # --- 5. Initialize the standard Trainer with the custom loss function ---
    print("Initializing Trainer with custom CDW-CE loss function...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics,
        compute_loss_func=cdw_ce_loss_func, # Pass the custom function here
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

    model_save_name = f"{model_name.split('/')[-1][:4]}_Danish={num_danish_samples}_{timestamp}_CDW-CE"
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
    config_path = "training/config/base.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_config(config_path)
    
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