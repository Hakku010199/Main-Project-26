import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_PATH = "models/bias_model"
DATASET_PATH = "data/processed/bias_dataset.csv"
OUTPUT_PATH = "results/metrics/bias_metrics.txt"

df = pd.read_csv(DATASET_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

true_labels = []
pred_labels = []

model.eval()

for _, row in df.sample(min(500, len(df))).iterrows():
    inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    true_labels.append(row["label"])
    pred_labels.append(pred)

accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")

with open(OUTPUT_PATH, "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

print("Evaluation complete")