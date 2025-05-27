from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import torch

# Define model and tokenizer
MODEL_NAME = "FacebookAI/xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)  # Adjust based on label count

# Load dataset (FineWeb2 for training, FineWeb-C for testing)
test_path = "train-00000-of-00001.parquet"
test_df = pd.read_parquet(test_path)

train_path = ""
#NOTE: missing


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

df = df.map(tokenize_function, batched=True)
df = df.rename_column("educational_value_labels", "labels")
df.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_df = test_df.map(tokenize_function, batched=True)
test_df = test_df.rename_column("educational_value_labels", "labels")
test_df.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df,
    eval_dataset=test_df
)

# Train model
trainer.train()

# Save fine-tuned model
trainer.save_model("./fine_tuned_xlm_roberta")