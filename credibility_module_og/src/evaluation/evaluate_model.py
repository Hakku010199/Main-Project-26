from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing.dataset_loader import load_datasets
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np

def evaluate():
    _, _, test_dataset = load_datasets()

    model_path = "models/credibility_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize(example):
        return tokenizer(
            example["statement"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    test_dataset = test_dataset.map(tokenize, batched=True)

    model.eval()
    predictions = []
    true_labels = []

    for item in test_dataset:
        inputs = {
            "input_ids": torch.tensor([item["input_ids"]]),
            "attention_mask": torch.tensor([item["attention_mask"]])
        }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        predictions.append(pred)
        true_labels.append(item["label"])

    print("Accuracy:", accuracy_score(true_labels, predictions))
    print("Classification Report:")
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    evaluate()
