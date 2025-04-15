# Feature Engineering
import pandas as pd
import numpy as np
import weave
from typing import Tuple, Literal, List
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler


@weave.op()
def extract_sentiment(text: str) -> float:
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0


@weave.op()
def build_features(
    df: pd.DataFrame,
    feature_set: Literal["structured", "nlp", "hybrid"] = "hybrid"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Add derived structured features
    df["verified_encoded"] = df["verified_purchase"].astype(int) if "verified_purchase" in df.columns else 0
    df["sentiment_score"] = df["full_text"].apply(extract_sentiment)
    df["review_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["review_char_count"] = df["clean_text"].apply(len)

    structured_cols = [
        "rating",
        "verified_encoded",
        "review_word_count",
        "review_char_count",
        "sentiment_score"
    ]

    if feature_set == "structured":
        featurizer = ColumnTransformer(
            transformers=[
                ("struct", StandardScaler(), structured_cols)
            ],
            remainder="drop"
        )
    elif feature_set == "nlp":
        featurizer = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(max_features=1000, stop_words="english"), "full_text")
            ],
            remainder="drop"
        )
    elif feature_set == "hybrid":
        featurizer = ColumnTransformer(
            transformers=[
                ("struct", StandardScaler(), structured_cols),
                ("text", TfidfVectorizer(max_features=1000, stop_words="english"), "full_text")
            ],
            remainder="drop"
        )
    else:
        raise ValueError("Invalid feature_set. Choose from 'structured', 'nlp', or 'hybrid'.")

    X = featurizer.fit_transform(df)
    y = df["label"].values

    return X, y, structured_cols + ["tfidf"]