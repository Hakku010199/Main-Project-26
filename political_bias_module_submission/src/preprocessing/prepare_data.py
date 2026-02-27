import os
import json
import pandas as pd

RAW_PATH = "data/raw/Article-Bias-Prediction/data/jsons"
OUTPUT_PATH = "data/processed/bias_dataset.csv"

data = []
for file in os.listdir(RAW_PATH):
    if file.endswith(".json"):
        with open(os.path.join(RAW_PATH, file), "r", encoding="utf-8") as f:
            article = json.load(f)
            text = article.get("content", "")
            bias = article.get("bias", None)
            if text and bias is not None:
                data.append({"text": text, "label": bias})

df = pd.DataFrame(data)
df.dropna(inplace=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset prepared:", OUTPUT_PATH)