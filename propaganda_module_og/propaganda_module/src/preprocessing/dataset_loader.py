import pandas as pd
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']]
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.1)
