# Credibility Module

This module performs AI-based credibility analysis of news articles using DistilBERT.

## Steps
1. Train model on LIAR dataset
2. Extract article text from URL
3. Predict credibility

## Setup
pip install -r requirements.txt

## Training
python src/training/train_model.py

## Run
python main.py
