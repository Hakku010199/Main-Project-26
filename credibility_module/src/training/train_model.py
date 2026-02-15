from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from src.preprocessing.dataset_loader import load_datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def train():
    train_dataset, valid_dataset, test_dataset = load_datasets()

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["statement"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6
    )

    training_args = TrainingArguments(
        output_dir="models/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="models/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Test Evaluation:")
    print(trainer.evaluate(test_dataset))

    model.save_pretrained("models/credibility_model")
    tokenizer.save_pretrained("models/credibility_model")

if __name__ == "__main__":
    train()
