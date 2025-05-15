# scripts/preprocessing.py

import pandas as pd
import weave # Ensure weave is imported
from typing import Optional, Dict
import os

@weave.op()
def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

@weave.op()
def preprocess_reviews(
    df: pd.DataFrame, 
    metadata_df: Optional[pd.DataFrame] = None,
    labeling_mode: Optional[str] = None # Added labeling_mode parameter
) -> pd.DataFrame:
    # The DataFrame 'df' passed to this function will be modified in place.

    if metadata_df is not None and not metadata_df.empty and \
        'parent_asin' in metadata_df.columns and 'price' in metadata_df.columns:
        if 'parent_asin' in df.columns: 
            df['parent_asin'] = df['parent_asin'].astype(str)
        metadata_df['parent_asin'] = metadata_df['parent_asin'].astype(str)
        df = pd.merge(df, metadata_df, on='parent_asin', how='left')

    # This check must happen before attempting to calculate total_vote
    if "helpful_vote" not in df.columns:
        raise ValueError("Dataset missing 'helpful_vote' column, which is essential.")
    if "parent_asin" not in df.columns: # Also essential for grouping
        raise ValueError("Dataset missing 'parent_asin' column, which is essential.")

    # Conditionally calculate total_vote
    if labeling_mode == "threshold":
        print("⚙️ Calculating total_vote per product (parent_asin) for threshold labeling...")
        df['total_vote'] = df.groupby('parent_asin')['helpful_vote'].transform(lambda x: x.fillna(0).sum())
    else:
        print(f"ℹ️ Skipping total_vote calculation as labeling_mode is '{labeling_mode}' (not 'threshold').")

    # Define base essential columns and dynamically add total_vote if it was created
    base_essential_cols = [
        "rating", "title", "text", "parent_asin",
        "user_id", "timestamp", "verified_purchase",
        "helpful_vote" 
    ]
    
    actual_essential_cols = base_essential_cols[:]
    columns_to_keep_final = base_essential_cols[:]

    if 'total_vote' in df.columns: # Check if 'total_vote' was actually created
        actual_essential_cols.append("total_vote")
        columns_to_keep_final.append("total_vote")
    
    # Add price if it exists (as in original logic)
    if 'price' in df.columns:
        columns_to_keep_final.append('price')
    
    # Ensure all columns listed in actual_essential_cols actually exist before trying to use them in dropna
    # This check is mainly for the base_essential_cols, as total_vote is handled.
    missing_from_base = [col for col in base_essential_cols if col not in df.columns]
    if missing_from_base:
        raise ValueError(f"DataFrame missing base essential columns: {missing_from_base}")

    # Ensure columns_to_keep_final only contains columns that currently exist in df
    # This also ensures the order is somewhat preserved if that matters, though set operations lose order.
    current_df_cols_set = set(df.columns)
    final_cols_to_select = [col for col in columns_to_keep_final if col in current_df_cols_set]
    
    # Reassign df with only the selected columns
    df = df[final_cols_to_select]

    # Drop rows with NA in the *actual* essential columns
    # (actual_essential_cols will include 'total_vote' only if it was created and is in final_cols_to_select)
    cols_for_dropna_check = [col for col in actual_essential_cols if col in df.columns]

    original_count = len(df)
    df = df.dropna(subset=cols_for_dropna_check) 
    if len(df) < original_count:
        print(f"   Dropped {original_count - len(df)} rows with NA in essential columns: {cols_for_dropna_check}.")

    if df.empty:
        print("⚠️ DataFrame empty after dropping NAs from essential columns.")
        return df 

    # --- Create derived features (these will be added to 'df' directly) ---
    df["year"] = pd.to_datetime(df["timestamp"], unit="ms").dt.year
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)
    df["full_text"] = df["clean_title"] + ". " + df["clean_text"]
    df["review_word_count"] = df["clean_text"].apply(lambda x: len(x.split())) # Was present in original
    df["review_char_count"] = df["clean_text"].apply(len) # Was present in original


    print(f"✅ Preprocessing complete. Final shape: {df.shape}")
    return df