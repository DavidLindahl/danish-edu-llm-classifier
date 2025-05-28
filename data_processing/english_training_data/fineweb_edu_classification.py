import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Load 1k datapoints from the dataset
raw_dataset = load_dataset("HuggingFaceFW/fineweb", split="train[:10]")

# Load model and tokenizer
model_name = "HuggingFaceTB/fineweb-edu-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Tokenize and predict
scores = []
for example in tqdm(raw_dataset):
    inputs = tokenizer(
        example["text"], truncation=True, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()  # Assuming binary classification, take class 1
    scores.append(score)

# Add scores to dataset
raw_dataset = raw_dataset.add_column("score", scores)

# Save to disk (optional)
import os

os.makedirs("data", exist_ok=True)
raw_dataset.to_csv("data/fineweb_with_scores.csv")
print("Done! Saved as data/fineweb_with_scores.csv")
