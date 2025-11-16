import json, pandas as pd, numpy as np
from pathlib import Path

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_parquet(df: pd.DataFrame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_npy(arr: np.ndarray, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def load_npy(path) -> np.ndarray:
    return np.load(path)
