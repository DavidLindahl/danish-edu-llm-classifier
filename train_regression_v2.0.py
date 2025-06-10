from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, Value
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
# Make sure this path is correct for your project structure
from data_processing.data_process import get_merged_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    
    # Using zero_division=0 to prevent warnings if a class has no predictions
    precision = evaluate.load("precision").compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = evaluate.load("recall").compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="macro")["f1"]
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

    # This rename is fine if your get_merged_dataset sometimes leaves 'int_score'
    # df = df.rename(columns={"int_score": "score"})
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column(
        args.target_column, Value("float32")
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42
    )

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

    # --- MINIMAL CHANGE #2: Changed `model.bert` to `model.roberta` ---
    # This is the correct attribute for XLM-Roberta models.
    print("Freezing the body of the model...")
    for param in model.roberta.parameters():
        param.requires_grad = False
    print("Model body frozen. Only the classifier head will be trained.")

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False, # Recommended for GPU performance
        bf16=False,
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

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- MINIMAL CHANGE #1: Changed the default model name ---
    parser.add_argument(
        "--base_model_name", type=str, default="xlm-roberta-base"
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model_checkpoints",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="my-local-xlm-roberta-scorer"
    )
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)