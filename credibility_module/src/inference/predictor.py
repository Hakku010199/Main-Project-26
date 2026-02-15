from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.utils.label_map import label_names

model_path = "models/credibility_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_credibility(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()

    return {
        "prediction": label_names[pred_class],
        "confidence": round(confidence, 3)
    }
