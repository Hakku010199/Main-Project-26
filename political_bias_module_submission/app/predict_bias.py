import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article

MODEL_PATH = "models/bias_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

labels = ["Left", "Center", "Right"]

def extract_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def predict_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    return {"bias": labels[pred], "confidence": float(probs[0][pred])}

def predict_bias_from_url(url):
    text = extract_article(url)
    return predict_bias(text)

if __name__ == "__main__":
    url = input("Enter news URL: ")
    print(predict_bias_from_url(url))