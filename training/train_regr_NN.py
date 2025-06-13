"""Training script for the Danish educational score model."""
import sys
import os
import numpy as np
import time
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, ClassLabel
import yaml
import torch.nn as nn
from metrics import compute_metrics
from transformers import PretrainedConfig

# path setup to import data processing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.data_process import get_merged_dataset
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig


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


class SimpleNNConfig(PretrainedConfig):
    def __init__(self, input_size=768, hidden_size=128, num_labels=1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels

class SimpleRegressionNN(PreTrainedModel):
    config_class = SimpleNNConfig

    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None, **kwargs):
        # input_ids: (batch_size, input_size)
        x = self.fc1(input_ids.float())  # Ensure input is float for MPS/Linear
        x = self.relu(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(), labels.squeeze())

        return {"loss": loss, "logits": logits}


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

    df = get_merged_dataset(
        english_data_amount=num_english_samples,
        danish_data_amount=num_danish_samples,
    )

    # df = pd.read_csv("data/english_fineweb_merged_data.csv") 
    # df = df.sample(100, random_state=42)  # For testing, use a smaller sample
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
    configs = SimpleNNConfig(input_size=512, hidden_size=128, num_labels=1)
    model = SimpleRegressionNN(configs)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

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

        # --- Evaluation and Logging ---
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        eval_steps=eval_steps,
        save_steps=50,
        logging_steps=50,
        eval_on_start=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        # --- other parameters ---
        seed=42,
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