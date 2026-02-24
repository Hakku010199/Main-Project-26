import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.preprocessing.dataset_loader import load_dataset

MODEL_PATH = "models/propaganda_model"

def evaluate_model(csv_path="data/raw/dataset.csv"):
    dataset = load_dataset(csv_path)
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    predictions = []
    true_labels = []

    for example in test_dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred)
        true_labels.append(example["label"])

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    evaluate_model()
