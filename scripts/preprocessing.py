import pandas as pd
import weave
from typing import Optional

@weave.op()
def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

@weave.op()
def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [
        "rating", "title", "text", "parent_asin",
        "user_id", "timestamp", "verified_purchase", "helpful_vote"
    ]
    df = df[[col for col in columns_to_keep if col in df.columns]]
    df = df.dropna(subset=columns_to_keep)

    df["year"] = pd.to_datetime(df["timestamp"], unit="ms").dt.year
    df["clean_text"] = df["text"].str.strip().str.lower()
    df["clean_title"] = df["title"].str.strip().str.lower()
    df["full_text"] = df["clean_title"] + ". " + df["clean_text"]
    df["review_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["review_char_count"] = df["clean_text"].apply(len)

    return df