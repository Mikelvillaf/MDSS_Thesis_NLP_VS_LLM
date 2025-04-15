# data_loader.py

import pandas as pd
import os
import json
import weave
from tqdm import tqdm
from typing import List, Optional

@weave.op()
def load_reviews(
    filepath: str,
    year_range: Optional[List[int]] = None,
    max_rows: Optional[int] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📥 Loading reviews from: {filepath}")
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)

    if 'timestamp' in df.columns:
        df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year

    if year_range and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    if max_rows is not None:
        df = df.sample(n=min(max_rows, len(df)), random_state=seed)

    print(f"✅ Loaded {len(df)} reviews after filtering and sampling")
    return df

@weave.op()
def temporal_split(df: pd.DataFrame, train_years: List[int], val_year: int, test_year: int):
    train = df[df['year'].isin(train_years)]
    val = df[df['year'] == val_year]
    test = df[df['year'] == test_year]
    return train, val, test