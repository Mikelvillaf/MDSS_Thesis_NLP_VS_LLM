# scripts/label_generation.py

import pandas as pd
import weave

@weave.op()
def generate_labels(
    df: pd.DataFrame,
    mode: str = "percentile",
    top_percentile: float = 0.2,
    bottom_percentile: float = 0.4,
    helpful_ratio_min: float = 0.75,
    unhelpful_ratio_max: float = 0.35,
    min_total_votes: int = 10,
    # Length filter params added
    use_length_filter: bool = False,
    min_review_words: int = 10,
    max_review_words: int = 1000
) -> pd.DataFrame:
    """
    Generates helpfulness labels (1 for helpful, 0 for unhelpful) based on
    the specified mode and parameters, with an optional word count filter.

    Args:
        df (pd.DataFrame): Input DataFrame with review data. Requires 'total_vote',
                        'helpful_vote', and 'review_word_count' columns
                        depending on mode and filters enabled.
        mode (str): Labeling mode ('threshold' or 'percentile').
        # ... (other parameters as before) ...
        min_total_votes (int): Minimum total votes required.
        use_length_filter (bool): Whether to apply the word count filter.
        min_review_words (int): Minimum words if length filter is active.
        max_review_words (int): Maximum words if length filter is active.


    Returns:
        pd.DataFrame: A DataFrame containing ONLY the rows that were labeled
                    as helpful (1) or unhelpful (0), with the 'label' column added.
                    Returns an empty DataFrame if no rows are labeled.
    """
    df_processed = df.copy() # Work on a copy

    print(f"--- generate_labels --- Mode: {mode}")
    original_count_before_any_filtering = len(df_processed)

    # --- Optional: Apply Length Filter FIRST ---
    if use_length_filter:
        if 'review_word_count' not in df_processed.columns:
            raise ValueError("Missing 'review_word_count' column needed for length filter. Ensure preprocessing calculates it.")

        print(f"   Applying length filter (min: {min_review_words}, max: {max_review_words} words)...")
        # Ensure column is numeric before filtering
        df_processed['review_word_count'] = pd.to_numeric(df_processed['review_word_count'], errors='coerce')
        df_processed = df_processed.dropna(subset=['review_word_count']) # Drop if conversion failed
        df_processed['review_word_count'] = df_processed['review_word_count'].astype(int)

        initial_len = len(df_processed)
        df_processed = df_processed[
            (df_processed['review_word_count'] >= min_review_words) &
            (df_processed['review_word_count'] <= max_review_words)
        ]
        print(f"   Length filter kept {len(df_processed)} out of {initial_len} rows.")
        if df_processed.empty:
            print("   ⚠️ No reviews remaining after length filter.")
            if 'label' not in df.columns: df['label'] = None # Use original df for column structure
            return df[0:0] # Return empty frame with original columns + label


    # --- Apply Labeling Logic (Threshold or Percentile) ---
    labeled_df = pd.DataFrame() # Initialize empty dataframe

    if mode == "threshold":
        if "helpful_vote" not in df_processed.columns or "total_vote" not in df_processed.columns:
            raise ValueError("Missing 'helpful_vote' or 'total_vote' column for threshold labeling")

        # 1. Filter by min_total_votes (on potentially length-filtered data)
        count_before_min_votes = len(df_processed)
        df_filtered = df_processed[df_processed["total_vote"] >= min_total_votes].copy()
        print(f"   Threshold Mode: Applied min_total_votes >= {min_total_votes}. Kept {len(df_filtered)} out of {count_before_min_votes} rows.")

        if df_filtered.empty:
            print("   ⚠️ No reviews met the min_total_votes requirement in threshold mode.")
            if 'label' not in df_processed.columns: df_processed['label'] = None
            return df_processed[0:0]

        # 2. Calculate helpful ratio
        df_safe = df_filtered[df_filtered["total_vote"] > 0].copy()
        if len(df_safe) < len(df_filtered):
            print(f"   ⚠️ Warning: {len(df_filtered) - len(df_safe)} rows had total_vote=0 after min_votes filter.")
        df_safe["helpful_ratio"] = df_safe["helpful_vote"] / df_safe["total_vote"]

        print("   📊 Helpful ratio stats (after filters):")
        print(df_safe["helpful_ratio"].describe())

        # 3. Assign labels
        df_filtered["label"] = None
        df_filtered.loc[df_safe[df_safe["helpful_ratio"] >= helpful_ratio_min].index, "label"] = 1
        df_filtered.loc[df_safe[df_safe["helpful_ratio"] <= unhelpful_ratio_max].index, "label"] = 0

        # 4. Filter for labeled rows
        labeled_df = df_filtered[df_filtered["label"].isin([0, 1])].copy()


    elif mode == "percentile":
        if "helpful_vote" not in df_processed.columns:
            raise ValueError("Missing 'helpful_vote' column for percentile labeling")
        if "total_vote" not in df_processed.columns:
            raise ValueError("Missing 'total_vote' column required for min_total_votes filter in percentile mode")

        # 1. Filter by min_total_votes (on potentially length-filtered data)
        count_before_min_votes = len(df_processed)
        df_filtered = df_processed[df_processed["total_vote"] >= min_total_votes].copy()
        print(f"   Percentile Mode: Applied min_total_votes >= {min_total_votes}. Kept {len(df_filtered)} out of {count_before_min_votes} rows.")

        if df_filtered.empty:
            print("   ⚠️ No reviews met the min_total_votes requirement in percentile mode.")
            if 'label' not in df_processed.columns: df_processed['label'] = None
            return df_processed[0:0]

        # 2. Sort the *filtered* DataFrame by helpful_vote
        df_sorted = df_filtered.sort_values("helpful_vote", ascending=False).reset_index(drop=True)
        total_filtered = len(df_sorted)

        # 3. Calculate percentile indices
        top_n = int(total_filtered * top_percentile)
        bottom_n = int(total_filtered * bottom_percentile)
        print(f"   Percentile Mode: Top {top_percentile*100:.1f}% = {top_n} rows, Bottom {bottom_percentile*100:.1f}% = {bottom_n} rows (based on {total_filtered} filtered rows).")

        if top_n == 0 and bottom_n == 0:
            print("   ⚠️ Calculated top_n and bottom_n are both zero. No rows will be labeled.")
            if 'label' not in df_filtered.columns: df_filtered['label'] = None
            return df_filtered[0:0]

        # 4. Assign labels
        df_sorted["label"] = None
        if top_n > 0:
            df_sorted.iloc[:top_n, df_sorted.columns.get_loc("label")] = 1
        if bottom_n > 0:
            actual_bottom_n = min(bottom_n, total_filtered)
            df_sorted.iloc[-actual_bottom_n:, df_sorted.columns.get_loc("label")] = 0

        # 5. Filter for labeled rows
        labeled_df = df_sorted[df_sorted["label"].isin([0, 1])].copy()

    else:
        raise ValueError(f"Invalid labeling mode: '{mode}'. Choose 'percentile' or 'threshold'.")

    # --- Final processing ---
    if not labeled_df.empty:
        labeled_df["label"] = labeled_df["label"].astype(int)
        h_count = (labeled_df['label'] == 1).sum()
        u_count = (labeled_df['label'] == 0).sum()
        print(f"✅ Labeled dataset: {len(labeled_df)} rows (Helpful: {h_count}, Unhelpful: {u_count}) (out of {original_count_before_any_filtering} initially)")
    else:
        print(f"✅ Labeled dataset: 0 rows (out of {original_count_before_any_filtering} initially)")

    return labeled_df