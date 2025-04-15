# Preprocessing
import pandas as pd
import weave
from typing import Optional


@weave.op()
def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    return text

@weave.op()
def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    # 🔍 Keep only relevant columns
    columns_to_keep = [
        "rating",
        "title",
        "text",
        "parent_asin",
        "user_id",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
        "label"
    ]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # 🧹 Drop rows missing critical fields
    df = df.dropna(subset=["title", "text", "parent_asin", "rating", "verified_purchase", "helpful_vote", "timestamp"])

    # 🗓️ Extract year from Unix timestamp
    df["year"] = pd.to_datetime(df["timestamp"], unit="ms").dt.year

    # 🧽 Clean title and text
    df["clean_text"] = df["text"].str.strip().str.lower()
    df["clean_title"] = df["title"].str.strip().str.lower()

    # 📝 Combine title + text
    df["full_text"] = df["clean_title"] + ". " + df["clean_text"]

    # 🔢 Add review length features
    df["review_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["review_char_count"] = df["clean_text"].apply(len)

    return df