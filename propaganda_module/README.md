# Propaganda Detection Module

This project detects propaganda in news articles using a fine‑tuned DistilBERT model.

## Features
- Train DistilBERT on a propaganda dataset
- Predict propaganda from text
- Extract article text from URL
- Detect propaganda techniques

## Project Structure
```
propaganda_module/
│
├── data/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── results/
├── requirements.txt
└── README.md
```

## Installation
```
pip install -r requirements.txt
```

## Training
```
python src/training/train.py
```

## Inference
```
python src/training/predict.py
```
