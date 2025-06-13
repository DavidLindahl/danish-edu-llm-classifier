# Load N samples from the streaming dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd


N = 10000  # Number of samples to load (change as needed)
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2024-10",
    split="train",
    streaming=True
)

# straming the first N samples from the dataset
streamed_samples = []
for idx, sample in enumerate(dataset):
    if idx >= N:
        break
    streamed_samples.append(sample)
print(f"Loaded {len(streamed_samples)} samples.")

# load the tokenizer and model for classification
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")



# Classify streamed samples and add score and int_score
classified_samples = []
for sample in tqdm(streamed_samples, desc="Classifying samples"):
    text = sample.get("text", "")
    if not text:
        continue
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    int_score = int(round(max(0, min(score, 5)))) # ensure score is between 0 and 5
    sample["score"] = score
    sample["int_score"] = int_score
    classified_samples.append(sample)

df = pd.DataFrame(classified_samples)

# Save the classified samples to a CSV file
import pandas as pd
output_file = "data/english_classified_samples_10000.csv"
df.to_csv(output_file, index=False)
print(f"Classified samples saved to {output_file}.")
