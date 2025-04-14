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
    # Drop rows without core fields
    df = df.dropna(subset=['text', 'title'])

    # Clean review text and titles
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_title'] = df['title'].apply(clean_text)

    # Combine for full-text modeling (optional)
    df['full_text'] = df['clean_title'] + ". " + df['clean_text']

    # Add length-based features
    df['review_word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['review_char_count'] = df['clean_text'].apply(len)

    return df