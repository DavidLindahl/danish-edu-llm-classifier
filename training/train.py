import sys
import os
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score # Added classification metrics
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
from torch import nn # Import nn for loss function
from sklearn.model_selection import train_test_split
import yaml
from data_processing.data_process import get_merged_dataset
import torch.nn.functional as F # Import F for softmax

# Load config
print("Loading config...")
with open("training/config/base.yaml", "r") as f:
    config = yaml.safe_load(f)

val_split = 0.1  # Proportion of data to use for validation

model_name = config["model_name"]
model_dir = config["model_dir"]
results_dir = config["results_dir"]
num_labels = config.get("num_labels", 5) # CHANGED: Default to 5 for your ordinal scale (0-4)

num_danish_samples = config["num_danish_samples"]
num_english_samples = config["num_english_samples"]

learning_rate = config.get("learning_rate", 2e-5)

# Load data
print(
    f"Loading {num_english_samples} English and {num_danish_samples} Danish samples..."
)
df = get_merged_dataset(
    english_data_amount=num_english_samples,
    danish_data_amount=num_danish_samples,
)
print(f"Loaded dataset with {len(df)} samples.")

# Assume columns: text, score
texts = df["text"].astype(str).tolist()
print(f"Number of texts: {len(texts)}")

# CHANGED: Ensure labels are integers for classification (e.g., 0, 1, 2, 3, 4)
# Make sure your get_merged_dataset() and subsequent processing
# results in integer labels corresponding to your U-scale.
labels = df["score"].astype(int).tolist() # CHANGED: to int

# Split train/eval
print(f"Splitting data into train and validation sets (val_split={val_split})...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=val_split, random_state=42, stratify=labels if len(set(labels)) > 1 and num_labels > 1 else None # Added stratify for classification
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

class OrdinalClassificationDataset(torch.utils.data.Dataset): # Renamed for clarity
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=500
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # CHANGED: Labels should be long (integer) for CrossEntropyLoss
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


print("Preparing datasets...")
train_dataset = OrdinalClassificationDataset(train_texts, train_labels)
val_dataset = OrdinalClassificationDataset(val_texts, val_labels)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="single_label_classification", # CHANGED: for classification
)

# --- CDW Cross Entropy Loss Implementation ---
class CDWCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_counts, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(CDWCrossEntropyLoss, self).__init__()
        print(f"CDWLoss __init__: Received num_classes = {num_classes}, device = {device}")
        self.num_classes = num_classes
        self.device = device # Store the intended device

        # 1. Calculate class weights for imbalance
        total_samples = sum(class_counts.values())
        # Ensure class_counts keys are sorted or map to 0..num_classes-1
        counts_list = [class_counts.get(i, 1e-6) for i in range(num_classes)] # Handle missing classes
        if not counts_list: # Should not happen if num_classes > 0
            counts_list = [1e-6] * num_classes
            
        counts_tensor = torch.tensor(counts_list, dtype=torch.float) # Create on CPU first
        
        # Ensure self.class_weights_balance is initialized correctly based on num_classes
        if num_classes > 0 and len(counts_tensor) > 0:
            self.class_weights_balance = total_samples / (num_classes * counts_tensor)
            # Normalize weights
            self.class_weights_balance = self.class_weights_balance / self.class_weights_balance.sum() * num_classes
        else: # Handle edge case if num_classes is 0 or counts_tensor is empty
            self.class_weights_balance = torch.ones(num_classes, dtype=torch.float)

        # 2. Create distance matrix D for ordinal penalty
        # Ensure self.distance_matrix is initialized correctly
        if num_classes > 0:
            self.distance_matrix = torch.zeros((num_classes, num_classes)) # Create on CPU first
            for i in range(num_classes):
                for j in range(num_classes):
                    self.distance_matrix[i, j] = (i - j) ** 2
        else: # Handle edge case
            self.distance_matrix = torch.zeros((0,0))


        print(f"CDWLoss: Initialized class_weights_balance (on CPU): {self.class_weights_balance}")
        print(f"CDWLoss: Initialized Distance matrix (on CPU): \n{self.distance_matrix}")
        # Move to the specified device during init if desired, but moving in forward is safer
        self.class_weights_balance = self.class_weights_balance.to(self.device)
        self.distance_matrix = self.distance_matrix.to(self.device)
        print(f"CDWLoss: Moved class_weights_balance to device: {self.class_weights_balance.device}")
        print(f"CDWLoss: Moved Distance matrix to device: {self.distance_matrix.device}")


    def forward(self, logits, labels):
        # logits: (batch_size, num_classes) - should be on the correct training device
        # labels: (batch_size) - should be on the correct training device

        # Ensure weights and distance matrix are on the same device as logits/labels
        # This is crucial if the loss object was created before the model was moved to GPU/MPS
        current_device = logits.device
        class_weights_balance_device = self.class_weights_balance.to(current_device)
        distance_matrix_device = self.distance_matrix.to(current_device)

        # Softmax to get probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Gather the imbalance weights for the true labels in the batch
        # labels tensor should be on current_device
        gathered_class_weights = class_weights_balance_device[labels] # (batch_size)

        # Expand distance matrix based on true labels in the batch
        # distance_penalty_for_batch_labels[i, k] = distance_matrix[labels[i], k]
        distance_penalty_for_batch_labels = distance_matrix_device[labels] # (batch_size, num_classes)
        
        effective_weights = gathered_class_weights.unsqueeze(1) * distance_penalty_for_batch_labels
        
        loss_per_sample = -torch.sum(effective_weights * log_probs, dim=1)

        return loss_per_sample.mean()

# You'll need to get class counts for your training data
# Example: class_counts = {0: 1278, 1: 696, 2: 185, 3: 34, 4: 10} # Based on your test set distribution
# IMPORTANT: Use class counts from your *training set* for cdw_loss instantiation
train_class_counts = pd.Series(train_labels).value_counts().sort_index().to_dict()
print(f"Training class counts: {train_class_counts}")
# Ensure all classes 0 to num_labels-1 are present, even if with count 0 (though cdw_loss handles 1e-6)
for i in range(num_labels):
    if i not in train_class_counts:
        train_class_counts[i] = 0 # Or a small number if 1e-6 in loss is not preferred
        print(f"Warning: Class {i} not present in training data. Setting count to 0 for CDW Loss.")


cdw_loss_fn = CDWCrossEntropyLoss(num_classes=num_labels, class_counts=train_class_counts, device=model.device)

# Custom Trainer to use CDW Loss


class CDWTrainer(Trainer):
    def __init__(self, *args, cdw_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cdw_loss_fn = cdw_loss_fn

    # THIS IS THE CRITICAL PART - ensure the signature includes **kwargs
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # num_items_in_batch = kwargs.get("num_items_in_batch") # You can access it if needed
        # print(f"compute_loss called with num_items_in_batch: {num_items_in_batch}") # For debugging
        
        labels = inputs.pop("labels") # .pop("labels") is standard in Trainer
        outputs = model(**inputs)
        logits = outputs.get("logits") # Or outputs.logits if that's the structure

        if self.cdw_loss_fn:
            loss = self.cdw_loss_fn(logits, labels)
        else: 
            # Fallback to standard CrossEntropyLoss
            # Ensure model.config.num_labels is correct
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-len(labels))) # Use -len(labels) to match batch size if labels are 1D

        return (loss, outputs) if return_outputs else loss
# --- End of CDW Cross Entropy Loss Implementation ---


print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=config.get("num_train_epochs", 3),
    per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
    per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
    weight_decay=config.get("weight_decay", 0.01),
    eval_strategy=config.get("evaluation_strategy", "epoch"),
    logging_strategy=config.get("logging_strategy", "epoch"),
    save_strategy=config.get("save_strategy", "epoch"),
    logging_first_step=True,
    logging_dir=os.path.join(results_dir, "logs"),
    report_to="wandb",
    run_name="danish-edu-llm-ordinal-classification-run", # Updated run name
    # load_best_model_at_end=True, # Optional: if you want to load the best model
    # metric_for_best_model="f1", # Optional: specify metric for best model
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    # Using 'weighted' f1 for imbalanced classes. Could also use 'macro'.
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # You can also calculate Mean Squared Distance (MSD) if desired
    # For ordinal tasks, this can be more informative than pure accuracy/F1 sometimes
    # Ensure predictions and labels are treated as numerical values for MSD
    msd = mean_squared_error(labels, predictions) # Note: this treats class labels as numerical

    return {"accuracy": acc, "f1_weighted": f1, "msd_on_classes": msd}


print("Initializing Trainer...")
trainer = CDWTrainer( # CHANGED: Using custom CDWTrainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    cdw_loss_fn=cdw_loss_fn # Pass the custom loss function
)

if __name__ == "__main__":
    print("Starting training…")
    train_result = trainer.train()
    print("Training complete.")

    print("Evaluating on validation set…")
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics: {eval_metrics}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_model_path = os.path.join(model_dir, "model_weights_ordinal") # Changed save path slightly
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    print("Saving model...")
    trainer.save_model(save_model_path)
    print(f"Model saved to {save_model_path}")
    print("Training and evaluation complete.")
    print("Done.")