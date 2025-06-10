from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoModel, # --- CHANGE 1: Import AutoModel for the base ---
)
from datasets import Dataset, Value
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
from data_processing.data_process import get_merged_dataset
import torch # --- CHANGE 2: Import torch for nn.Module ---
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# --- CHANGE 3: Create a custom model class ---
class CustomRegressionModel(nn.Module):
    def __init__(self, model_name, freeze_base=True):
        """
        Initializes the custom model.
        Args:
            model_name (str): The name of the base model from Hugging Face.
            freeze_base (bool): If True, freezes the parameters of the base model.
        """
        super().__init__()
        # Load the base embedding model. `trust_remote_code` is needed for this model.
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Add a regression head. The input size is the hidden size of the base model.
        # The output size is 1 for our single regression score.
        self.regression_head = nn.Linear(self.base_model.config.hidden_size, 1)
        
        # Freeze the base model layers if requested
        if freeze_base:
            print("Freezing the base model...")
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Base model frozen.")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Defines the forward pass of the model.
        """
        # Get the outputs from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # The embedding for the `[CLS]` token is typically used for classification/regression.
        # It's the first token's embedding in the last hidden state.
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        # Pass the CLS embedding through our regression head to get the final score (logit)
        logits = self.regression_head(cls_embedding)
        
        loss = None
        # If labels are provided, calculate the loss (needed for training)
        if labels is not None:
            # Squeeze logits and labels to be 1D for the loss function
            # The Trainer expects a regression model to use Mean Squared Error loss.
            loss = F.mse_loss(logits.squeeze(), labels.squeeze())
        
        # The Trainer expects a tuple-like output where the first element is the loss (if calculated)
        # and subsequent elements are other outputs like logits.
        return (loss, logits) if loss is not None else (logits,)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # The output from our custom model might be a tuple, so we handle that
    if isinstance(logits, tuple):
        logits = logits[0]
        
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)

    precision = evaluate.load("precision").compute(predictions=preds, references=labels, average="macro", zero_division=0)["precision"]
    recall = evaluate.load("recall").compute(predictions=preds, references=labels, average="macro", zero_division=0)["recall"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="macro", zero_division=0)["f1"]
    accuracy = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    df = get_merged_dataset(1000, 1000)
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column(
        args.target_column, Value("float32")
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42
    )

    # --- CHANGE 4: Instantiate our new custom model ---
    # We pass the model name and tell it to freeze the base, as requested.
    model = CustomRegressionModel(args.base_model_name, freeze_base=True)
    
    # The tokenizer is loaded as usual
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        trust_remote_code=True, # Also good practice to add here
        model_max_length=min(model.base_model.config.max_position_embeddings, 512),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # The freezing logic is now inside our custom model class, so we can remove it from here.
    # for param in model.bert.embeddings.parameters():
    #     param.requires_grad = False
    # for param in model.bert.encoder.parameters():
    #     param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        logging_steps=10,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True, # Use fp16 on GPU
        bf16=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model, # Pass our custom model instance to the Trainer
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-m-v2.0"
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model_checkpoints",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="my-local-snowflake-scorer"
    )
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)