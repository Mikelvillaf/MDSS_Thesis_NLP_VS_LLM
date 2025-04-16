# scripts/preprocessing.py

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
    df = df.copy() # Work on a copy to avoid modifying the original DataFrame passed to the function

    # --- Start: Correct Total Vote Calculation ---
    # Ensure necessary columns exist
    if "helpful_vote" not in df.columns:
        raise ValueError("Dataset missing 'helpful_vote' column, cannot calculate total votes.")
    if "parent_asin" not in df.columns:
        raise ValueError("Dataset missing 'parent_asin' column, cannot group by product.")

    print("⚙️ Calculating total_votes_on_product by summing helpful_vote per parent_asin...")
    # Calculate the sum of helpful_vote for each product (parent_asin)
    # Use transform to broadcast the sum back to each review row for that product
    # Handle potential NaNs in helpful_vote before summing, fillna(0) assumes NaN means 0 votes
    df['total_votes_on_product'] = df.groupby('parent_asin')['helpful_vote'].transform(lambda x: x.fillna(0).sum())
    print(f"✅ Calculated total_votes_on_product. Min: {df['total_votes_on_product'].min()}, Max: {df['total_votes_on_product'].max()}")

    # Rename this to 'total_vote' for consistency with label_generation.py OR
    # update label_generation.py to use 'total_votes_on_product'
    # Let's rename for simplicity here:
    df.rename(columns={'total_votes_on_product': 'total_vote'}, inplace=True)
    # --- End: Correct Total Vote Calculation ---


    # Define columns to keep (ensure 'total_vote' is now included from calculation)
    columns_to_keep = [
        "rating", "title", "text", "parent_asin",
        "user_id", "timestamp", "verified_purchase", "helpful_vote", "total_vote"
    ]
    # Ensure all essential columns are present before selecting/dropping NAs
    essential_cols = ["rating", "title", "text", "parent_asin", "user_id", "timestamp", "verified_purchase", "helpful_vote", "total_vote"]
    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        raise ValueError(f"DataFrame missing essential columns after preprocessing: {missing_essential}")

    # Select relevant columns (those that exist in the dataframe)
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Drop rows where any of the essential columns have NA values AFTER calculating total_vote
    # Important: Decide how to handle reviews for products where total_vote might be 0 if all helpful_votes were NaN/0.
    # The label_generation script filters for total_vote >= min_total_votes later.
    print(f"Shape before dropping NAs from essential columns: {df.shape}")
    df = df.dropna(subset=essential_cols)
    print(f"Shape after dropping NAs: {df.shape}")


    # Proceed with other preprocessing
    df["year"] = pd.to_datetime(df["timestamp"], unit="ms").dt.year
    df["clean_text"] = df["text"].apply(clean_text) # Use apply for safety with potential non-strings
    df["clean_title"] = df["title"].apply(clean_text)
    df["full_text"] = df["clean_title"] + ". " + df["clean_text"]
    # These counts are less critical now but can stay
    df["review_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["review_char_count"] = df["clean_text"].apply(len)

    # Ensure required columns for labeling exist before returning
    if "helpful_vote" not in df.columns or "total_vote" not in df.columns:
        raise ValueError("Missing helpful_vote or total_vote column before returning from preprocess_reviews")


    return df