# scripts/feature_engineering.py
import pandas as pd
import numpy as np
import weave
from typing import Tuple, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib # For potentially saving the featurizer
import os

@weave.op()
def extract_sentiment(text: Optional[str]) -> float:
    """Safely extracts sentiment polarity using TextBlob."""
    if not isinstance(text, str) or not text:
        return 0.0 # Return neutral for non-strings or empty strings
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        # Log unexpected errors if any
        # print(f"Warning: TextBlob error for text '{text[:50]}...': {e}")
        return 0.0

@weave.op()
def _create_featurizer(feature_set: str = "hybrid", text_max_features: int = 1000) -> ColumnTransformer:
    """Creates the ColumnTransformer based on the feature set."""

    # Define structured columns that should exist after preprocessing
    # Ensure these columns are present in the data passed for fitting/transforming
    structured_cols = [
        "rating", # Should exist
        "verified_encoded", # Created in _add_derived_features
        "review_word_count", # Created in _add_derived_features
        "review_char_count", # Created in _add_derived_features
        "sentiment_score" # Created in _add_derived_features
    ]

    # Define transformers
    structured_transformer = StandardScaler()
    # Note: TF-IDF stop words can be language-specific. 'english' is common but consider dataset language.
    text_transformer = TfidfVectorizer(max_features=text_max_features, stop_words="english")

    # Define ColumnTransformer based on feature_set
    if feature_set == "structured":
        featurizer = ColumnTransformer(
            transformers=[
                ("struct", structured_transformer, structured_cols)
            ],
            remainder="drop", # Drop columns not specified
            verbose_feature_names_out=True # <-- Set to True
        )
    elif feature_set == "nlp":
        featurizer = ColumnTransformer(
            transformers=[
                ("text", text_transformer, "full_text") # Assumes 'full_text' column exists
            ],
            remainder="drop",
            verbose_feature_names_out=True # <-- Set to True
        )
        # Set feature names manually for NLP-only case if needed later
        # featurizer.set_output(transform="pandas") # Easier feature name handling potentially
    elif feature_set == "hybrid":
        featurizer = ColumnTransformer(
            transformers=[
                ("struct", structured_transformer, structured_cols),
                ("text", text_transformer, "full_text")
            ],
            remainder="drop", # Important: drops columns not used in struct or text
            verbose_feature_names_out=True # <-- Set to True
        )
        # featurizer.set_output(transform="pandas") # Optional: returns DataFrame
    else:
        raise ValueError("Invalid feature_set. Choose from 'structured', 'nlp', or 'hybrid'.")

    return featurizer

@weave.op()
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features derived from existing columns. Expected to run before featurizer."""
    df_out = df.copy()
    # Ensure required base columns exist from preprocessing
    if 'verified_purchase' not in df_out.columns: df_out['verified_purchase'] = False # Assume False if missing
    if 'full_text' not in df_out.columns: df_out['full_text'] = "" # Assume empty if missing
    if 'clean_text' not in df_out.columns: df_out['clean_text'] = "" # Assume empty if missing

    # Convert verified_purchase to int, handling potential non-boolean values
    df_out["verified_encoded"] = df_out["verified_purchase"].apply(lambda x: 1 if x is True else 0)

    # Apply sentiment extraction safely
    df_out["sentiment_score"] = df_out["full_text"].apply(extract_sentiment)
    # Ensure clean_text is string before splitting/len
    df_out["review_word_count"] = df_out["clean_text"].astype(str).apply(lambda x: len(x.split()))
    df_out["review_char_count"] = df_out["clean_text"].astype(str).apply(len)

    return df_out

@weave.op()
def fit_feature_extractor(
    df_train: pd.DataFrame,
    feature_set: str = "hybrid",
    text_max_features: int = 1000
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Adds derived features, fits the ColumnTransformer (featurizer) on the training data,
    and returns the fitted featurizer and feature names.

    Args:
        df_train: Training DataFrame (must include 'label' and columns needed for features).
        feature_set: Type of features to use ('structured', 'nlp', 'hybrid').
        text_max_features: Max features for TF-IDF if used.

    Returns:
        Tuple containing:
            - fitted_featurizer: The scikit-learn ColumnTransformer fitted on df_train.
            - feature_names: List of feature names generated by the featurizer.
    """
    print(f"🛠️ Fitting feature extractor (feature_set='{feature_set}')...")

    # Drop rows with missing labels if any slipped through (should not happen ideally)
    df_train_clean = df_train.dropna(subset=["label"]).copy()
    if len(df_train_clean) < len(df_train):
        print(f"⚠️ Dropped {len(df_train) - len(df_train_clean)} rows with missing labels before fitting features.")
    if df_train_clean.empty:
        raise ValueError("Training data is empty after dropping rows with missing labels.")


    # Add derived features needed by the featurizer
    df_train_derived = _add_derived_features(df_train_clean)

    # Create the featurizer structure
    featurizer = _create_featurizer(feature_set, text_max_features)

    # Fit the featurizer ONLY on the training data (excluding the label)
    # Ensure all columns needed by the featurizer exist in df_train_derived
    required_cols_base = ['label'] # Base requirement
    required_cols_struct = ["rating", "verified_encoded", "review_word_count", "review_char_count", "sentiment_score"]
    required_cols_nlp = ["full_text"]
    required_cols = list(required_cols_base) # Start with base

    if feature_set in ["structured", "hybrid"]:
        required_cols.extend(required_cols_struct)
    if feature_set in ["nlp", "hybrid"]:
        required_cols.append("full_text")

    missing_cols = [col for col in required_cols if col not in df_train_derived.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in training data for feature engineering: {missing_cols}. Available: {list(df_train_derived.columns)}")


    print(f"   Fitting featurizer on {len(df_train_derived)} training samples.")
    try:
      # Ensure label is not passed to fit if dropping it
      if 'label' in df_train_derived.columns:
            fitted_featurizer = featurizer.fit(df_train_derived.drop(columns="label"), df_train_derived["label"])
      else: # Should not happen if checks above pass, but safeguard
            fitted_featurizer = featurizer.fit(df_train_derived, None) # Or raise error
    except Exception as e:
        print(f"❌ Error during featurizer fitting: {e}")
        raise e


    # Get feature names
    try:
        feature_names = list(fitted_featurizer.get_feature_names_out())
        print(f"   Fit successful. Number of features: {len(feature_names)}")
    except Exception as e:
        # Catching error getting names, but fit was successful
        print(f"⚠️ Fit successful, but could not get feature names: {e}. Returning empty list.")
        feature_names = []

    return fitted_featurizer, feature_names


@weave.op()
def transform_features(
    df: pd.DataFrame,
    featurizer: ColumnTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds derived features and applies a pre-fitted ColumnTransformer to transform data.
    Extracts the label column.

    Args:
        df: DataFrame to transform (train, val, or test). Must contain 'label'.
        featurizer: The pre-fitted ColumnTransformer object.

    Returns:
        Tuple containing:
            - X: Transformed features (NumPy array).
            - y: Labels (NumPy array).
    """
    if df.empty:
        print("⚠️ Input DataFrame for transform_features is empty. Returning empty arrays.")
        # Determine expected feature count from featurizer if possible
        try:
            # Get feature names to infer count; handle potential error
            num_features = len(featurizer.get_feature_names_out())
        except:
            print("   Could not determine feature count from featurizer for empty input.")
            num_features = 0 # Fallback, might cause issues downstream if not consistent
        return np.empty((0, num_features)), np.empty((0,))

    # Ensure label exists before dropping NAs based on it
    if "label" not in df.columns:
         raise ValueError("Input DataFrame for transform_features is missing the 'label' column.")

    df_clean = df.dropna(subset=["label"]).copy()
    if len(df_clean) < len(df):
         print(f"⚠️ Dropped {len(df) - len(df_clean)} rows with missing labels before transforming features.")
    if df_clean.empty:
         print("⚠️ DataFrame is empty after dropping rows with missing labels in transform_features. Returning empty arrays.")
         num_features = len(featurizer.get_feature_names_out()) if featurizer else 0
         return np.empty((0, num_features)), np.empty((0,))


    # Add derived features needed for the transformation
    df_derived = _add_derived_features(df_clean)

    # Extract labels BEFORE transforming features
    y = df_derived["label"].astype(int).values

    # Transform features using the FITTED featurizer
    # Ensure columns expected by the featurizer are present
    # (ColumnTransformer handles selecting the right ones based on its definition)
    print(f"   Transforming {len(df_derived)} samples...")
    try:
        # Ensure label is not passed to transform if present
        if 'label' in df_derived.columns:
            X = featurizer.transform(df_derived.drop(columns="label"))
        else: # Should not happen if label check passed, but safeguard
            X = featurizer.transform(df_derived)
        print(f"   Transform successful. Output shape: {X.shape}")
    except ValueError as e:
        print(f"❌ Error during feature transformation: {e}")
        print("   This might happen if columns expected by the fitted featurizer are missing in the input DataFrame.")
        print(f"   Input columns: {list(df_derived.columns)}")
        # You might want to inspect featurizer.transformers_ to see expected columns
        raise e # Re-raise for now

    return X, y


# --- Optional: Function to save/load featurizer ---
def save_featurizer(featurizer: ColumnTransformer, path: str):
    """Saves a fitted featurizer to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(featurizer, path)
        print(f"✅ Featurizer saved to {path}")
    except Exception as e:
        print(f"❌ Error saving featurizer to {path}: {e}")


def load_featurizer(path: str) -> ColumnTransformer:
    """Loads a fitted featurizer from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Featurizer file not found: {path}")
    try:
        featurizer = joblib.load(path)
        print(f"✅ Featurizer loaded from {path}")
        return featurizer
    except Exception as e:
        print(f"❌ Error loading featurizer from {path}: {e}")
        raise e