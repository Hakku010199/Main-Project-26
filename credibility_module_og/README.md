# Credibility Module

AI-based news credibility classification using DistilBERT.

## Project Structure
data/raw/        -> Place LIAR dataset files here
models/          -> Saved trained model
src/             -> Source code

## Dataset Required
Download LIAR dataset and place:
- train.tsv
- valid.tsv
- test.tsv

inside data/raw/

## Install
pip install -r requirements.txt

## Train
python src/training/train_model.py

## Evaluate
python src/evaluation/evaluate_model.py

## Run Prediction
python main.py
