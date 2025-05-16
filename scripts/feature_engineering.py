# scripts/feature_engineering.py
import pandas as pd
import numpy as np
import weave # Ensure weave is imported
from typing import Tuple, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib # For saving/loading the featurizer
import os

# Safely extracts sentiment polarity using TextBlob. Returns 0.0 for non-strings or errors.
@weave.op()
def extract_sentiment(text: Optional[str]) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        # print(f"Warning: TextBlob error for text '{text[:50]}...': {e}") # Optional debug
        return 0.0

# Internal helper: Creates the ColumnTransformer based on the feature set config.
@weave.op()
def _create_featurizer(feature_set: str = "hybrid", text_max_features: int = 1000) -> ColumnTransformer:
    # Define structured columns expected to be created by _add_derived_features
    # This couples this function to the output of _add_derived_features.
    structured_cols = [
        "rating",
        "verified_encoded",
        "review_word_count",
        "title_sentiment_score",
        # "review_char_count",
        "sentiment_score"
    ]
    # Define the text column expected from preprocessing
    text_col = "full_text"

    # Define transformers
    structured_transformer = StandardScaler()
    # Consider language if not 'english'
    text_transformer = TfidfVectorizer(max_features=text_max_features, stop_words="english")

    transformers = []
    if feature_set in ["structured", "hybrid"]:
        transformers.append(("struct", structured_transformer, structured_cols))
    if feature_set in ["nlp", "hybrid"]:
        transformers.append(("text", text_transformer, text_col))

    if not transformers:
        raise ValueError("Invalid feature_set. Choose 'structured', 'nlp', or 'hybrid'.")

    # Create ColumnTransformer
    featurizer = ColumnTransformer(
        transformers=transformers,
        remainder="drop", # Drop columns not specified
        verbose_feature_names_out=True # Get clear feature names
    )
    return featurizer

# Internal helper: Adds features derived from existing columns. Runs before featurizer.
@weave.op()
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    # Check for base columns and apply fallbacks if missing (original behavior)
    # Consider raising error instead if preprocessing guarantees these columns.
    if 'verified_purchase' not in df_out.columns: df_out['verified_purchase'] = False
    if 'full_text' not in df_out.columns: df_out['full_text'] = ""
    if 'clean_text' not in df_out.columns: df_out['clean_text'] = ""
    if 'rating' not in df_out.columns: df_out['rating'] = np.nan # Add rating fallback if needed

    # Add derived features
    df_out["verified_encoded"] = df_out["verified_purchase"].apply(lambda x: 1 if x is True else 0)
    df_out["sentiment_score"] = df_out["clean_text"].apply(extract_sentiment)
    df_out["title_sentiment_score"] = df_out["clean_title"].apply(extract_sentiment)
    # Ensure clean_text is string before applying len/split
    df_out["review_word_count"] = df_out["clean_text"].astype(str).apply(lambda x: len(x.split()))
    # df_out["review_char_count"] = df_out["clean_text"].astype(str).apply(len)

    return df_out

# Fits the feature extractor (ColumnTransformer) on training data.
@weave.op()
def fit_feature_extractor(
    df_train: pd.DataFrame,
    feature_set: str = "hybrid",
    text_max_features: int = 1000
) -> Tuple[ColumnTransformer, List[str]]:
    print(f"🛠️ Fitting feature extractor (feature_set='{feature_set}')...")

    # Ensure label exists for fitting checks (though not used as feature)
    if "label" not in df_train.columns:
        raise ValueError("Training data must include 'label' column.")

    df_train_clean = df_train.dropna(subset=["label"]).copy()
    if len(df_train_clean) < len(df_train):
        print(f"   Dropped {len(df_train) - len(df_train_clean)} rows with missing labels before fitting.")
    if df_train_clean.empty:
        raise ValueError("Training data is empty after dropping rows with missing labels.")

    # Add derived features needed by the featurizer
    df_train_derived = _add_derived_features(df_train_clean)

    # Define columns required based on feature_set for fitting check
    required_cols_for_fit = set()
    if feature_set in ["structured", "hybrid"]:
        required_cols_for_fit.update(["rating", "verified_encoded", "review_word_count", "title_sentiment_score", "sentiment_score"])
    if feature_set in ["nlp", "hybrid"]:
        required_cols_for_fit.add("full_text")

    missing_cols = required_cols_for_fit - set(df_train_derived.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in training data for feature set '{feature_set}': {missing_cols}")

    # Create and fit the featurizer structure
    featurizer = _create_featurizer(feature_set, text_max_features)
    print(f"   Fitting featurizer on {len(df_train_derived)} training samples.")
    try:
        # Fit on features (X), labels (y) are passed but ColumnTransformer ignores y by default during fit
        X_fit = df_train_derived.drop(columns="label", errors='ignore') # Drop label if exists
        y_fit = df_train_derived["label"]
        fitted_featurizer = featurizer.fit(X_fit, y_fit)
    except Exception as e:
        print(f"❌ Error during featurizer fitting: {e}")
        print(f"   Columns available for fitting: {list(X_fit.columns)}")
        raise e

    # Get feature names
    try:
        feature_names = list(fitted_featurizer.get_feature_names_out())
        print(f"   Fit successful. Features created: {len(feature_names)}")
    except Exception as e:
        print(f"⚠️ Fit successful, but failed to get feature names: {e}. Returning empty list.")
        feature_names = []

    return fitted_featurizer, feature_names


# Transforms data using a pre-fitted featurizer and extracts labels.
@weave.op()
def transform_features(
    df: pd.DataFrame,
    featurizer: ColumnTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    if df.empty:
        print("⚠️ Input DataFrame for transform_features is empty. Returning empty arrays.")
        num_features = 0
        try: # Attempt to get feature count from featurizer
             num_features = len(featurizer.get_feature_names_out())
        except: pass # Ignore error if featurizer hasn't been fit etc.
        return np.empty((0, num_features)), np.empty((0,))

    if "label" not in df.columns:
         raise ValueError("Input DataFrame for transform_features is missing 'label' column.")

    df_clean = df.dropna(subset=["label"]).copy()
    if len(df_clean) < len(df):
         print(f"   Dropped {len(df) - len(df_clean)} rows with missing labels before transforming.")
    if df_clean.empty:
         print("⚠️ DataFrame empty after dropping NA labels in transform_features.")
         num_features = len(featurizer.get_feature_names_out()) if featurizer else 0
         return np.empty((0, num_features)), np.empty((0,))

    # Add derived features needed for transformation
    df_derived = _add_derived_features(df_clean)

    # Extract labels BEFORE transforming features
    y = df_derived["label"].astype(int).values

    # Transform features using the FITTED featurizer
    print(f"   Transforming {len(df_derived)} samples...")
    try:
        X = featurizer.transform(df_derived.drop(columns="label", errors='ignore')) # Drop label if exists
        # print(f"   Transform successful. Output shape: {X.shape}") # Optional verbose
    except ValueError as e:
        print(f"❌ Error during feature transformation: {e}")
        print("   Check if columns expected by the fitted featurizer are present.")
        print(f"   Input columns available: {list(df_derived.columns)}")
        # To debug, inspect featurizer.transformers_
        raise e # Re-raise

    return X, y


# --- Featurizer Saving/Loading Utilities ---

# Saves a fitted featurizer to disk using joblib.
def save_featurizer(featurizer: ColumnTransformer, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(featurizer, path)
        print(f"✅ Featurizer saved to {path}")
    except Exception as e:
        print(f"❌ Error saving featurizer to {path}: {e}")

# Loads a fitted featurizer from disk using joblib.
def load_featurizer(path: str) -> ColumnTransformer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Featurizer file not found: {path}")
    try:
        featurizer = joblib.load(path)
        print(f"✅ Featurizer loaded from {path}")
        return featurizer
    except Exception as e:
        print(f"❌ Error loading featurizer from {path}: {e}")
        raise e