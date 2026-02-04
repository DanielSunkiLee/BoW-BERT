import pandas as pd
from pathlib import Path

def load_dataset(csv_path):
    names = ('polarity', 'id', 'date', 'query', 'author', 'text')
    df = pd.read_csv(
        csv_path,
        encoding='latin1',
        names=names)
    df["label"] = df["polarity"].map({0:0,4:1})
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels