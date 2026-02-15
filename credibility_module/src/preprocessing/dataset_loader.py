import pandas as pd
from datasets import Dataset
from src.utils.label_map import label_map

columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_counts",
    "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context"
]

def load_datasets(data_path="data/raw/"):
    train_df = pd.read_csv(data_path + "train.tsv", sep="\t", names=columns)
    valid_df = pd.read_csv(data_path + "valid.tsv", sep="\t", names=columns)
    test_df = pd.read_csv(data_path + "test.tsv", sep="\t", names=columns)

    for df in [train_df, valid_df, test_df]:
        df["label"] = df["label"].map(label_map)

    train_df = train_df[["statement", "label"]]
    valid_df = valid_df[["statement", "label"]]
    test_df = test_df[["statement", "label"]]

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, valid_dataset, test_dataset
