from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_path = "models/propaganda_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_propaganda(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    label = "Propaganda" if pred == 1 else "Non-propaganda"
    return {"label": label, "confidence": round(confidence, 3)}

if __name__ == "__main__":
    sample = "This corrupt government is destroying our nation!"
    print(predict_propaganda(sample))
