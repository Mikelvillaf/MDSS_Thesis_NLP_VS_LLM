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
    # Make sure label exists
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in DataFrame.")

    features = []

    if feature_set in ["structured", "hybrid"]:
        df['verified_encoded'] = df['verified_purchase'].astype(int) if 'verified_purchase' in df.columns else 0
        df['sentiment_score'] = df['full_text'].apply(extract_sentiment)

        structured_cols = [
            'rating',
            'verified_encoded',
            'review_word_count',
            'review_char_count',
            'sentiment_score'
        ]

        df[structured_cols] = df[structured_cols].fillna(0)
        features.append(df[structured_cols])

    if feature_set in ["nlp", "hybrid"]:
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        tfidf_matrix = tfidf.fit_transform(df['full_text'].fillna(""))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
        features.append(tfidf_df)

    # Concatenate all features
    X = pd.concat(features, axis=1)
    X = StandardScaler().fit_transform(X)  # Optional: normalize
    y = df['label'].astype(int).values

    return X, y, list(X.columns) if isinstance(X, pd.DataFrame) else []