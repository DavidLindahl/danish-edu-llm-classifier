import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import evaluate


def compute_metrics(eval_pred):
    """
    Compute precision, recall, F1, and accuracy by rounding regression predictions.
    """
    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 4).astype(int)
    labels = labels.squeeze().astype(int)

    # Load evaluation metrics - already done in the example, keep as is
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")


    # Compute metrics using macro average for multi-class classification
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro")["precision"]
    
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro")["recall"]
    
    f1 = f1_metric.compute(
        predictions=preds, references=labels, average="macro")["f1"]

    accuracy = accuracy_metric.compute(
        predictions=preds, references=labels)["accuracy"]

    mse_score = mean_squared_error(labels, preds)

    target_names = ["None", "Minimal", "Basic", "Good", "Excellent"]

    report = classification_report(labels, preds, target_names=target_names, labels=list(range(5)), zero_division=0)
    cm = confusion_matrix(labels, preds)

    print("Classification Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
        "classification_report": report,
        "mse": mse_score,
    }