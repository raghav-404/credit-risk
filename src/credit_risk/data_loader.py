import pandas as pd
from sklearn.datasets import fetch_openml

from credit_risk.config import OPENML_DATASET_NAME, OPENML_VERSION


def load_data() -> pd.DataFrame:
    print("Downloading data...")
    data = fetch_openml(
        name=OPENML_DATASET_NAME,
        version=OPENML_VERSION,
        as_frame=True,
        parser='auto'
    )
    df = data.frame
    df['target'] = df['class'].apply(lambda x: 0 if x == 'good' else 1)
    df = df.drop(columns=['class'])
    print("Success! Data loaded. Shape:", df.shape)
    return df
