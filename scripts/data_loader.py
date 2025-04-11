# Data Loader
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
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load reviews from a .jsonl file and optionally filter by year and row count.

    Args:
        filepath (str): Path to the JSONL file.
        year_range (list, optional): [start_year, end_year] for filtering.
        max_rows (int, optional): Max number of rows to return.

    Returns:
        pd.DataFrame: Filtered reviews.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📥 Loading reviews from: {filepath}")
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)

    # Convert Unix timestamp to year
    if 'timestamp' in df.columns:
        df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year

    # Year filtering
    if year_range and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    print(f"✅ Loaded {len(df)} reviews after filtering")

    return df