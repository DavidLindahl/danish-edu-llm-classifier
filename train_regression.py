from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
from data_processing.data_process import get_merged_dataset
from datasets import Dataset, Value


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }

def training_function(args):
    # All the code from your original main() function goes here
    # Starting from get_merged_dataset() all the way to trainer.train()
    df = get_merged_dataset(1000, 1000)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column(args.target_column, Value("float32"))
    dataset = dataset.train_test_split(train_size=0.9, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,
        output_hidden_states=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(model.config.max_position_embeddings, 512),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=10,
        save_steps=40,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting Trainer.train() via notebook_launcher ---")
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))

# 2. Your __main__ block now calls notebook_launcher
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-m"
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model_checkpoints",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="my-local-fineweb-scorer"
    )
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 3. Import notebook_launcher and use it
    from accelerate import notebook_launcher
    
    # This will handle the distributed setup for the TPU
    notebook_launcher(training_function, args=(args,), num_processes=8) # For a v2-8 TPU