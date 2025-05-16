# scripts/label_generation.py

import pandas as pd
import weave 
from typing import Any 

@weave.op()
def generate_labels(
    df: pd.DataFrame, 
    mode: str, 
    top_percentile: float, 
    bottom_percentile: float, 
    helpful_ratio_min: float, 
    unhelpful_ratio_max: float, 
    min_total_votes: int, 
    min_helpful_votes: int,
    use_length_filter: bool, 
    min_review_words: int, 
    max_review_words: int 
) -> pd.DataFrame:

    if mode not in ["threshold", "percentile"]:
        raise ValueError(f"Invalid labeling mode provided: '{mode}'. Choose 'percentile' or 'threshold'.")

    original_count = len(df)
    print(f"--- generate_labels --- Mode: {mode}, Initial rows: {original_count}")

    df_for_processing = df 

    if use_length_filter:
        if min_review_words is None or max_review_words is None:
            raise ValueError("Length filter enabled but min/max_review_words not provided.")
        if 'review_word_count' not in df_for_processing.columns: 
            raise ValueError("Missing 'review_word_count' column for length filter.")
        df_for_processing['review_word_count'] = pd.to_numeric(df_for_processing['review_word_count'], errors='coerce')
        df_for_processing.dropna(subset=['review_word_count'], inplace=True) 
        df_for_processing['review_word_count'] = df_for_processing['review_word_count'].astype(int)
        initial_len = len(df_for_processing)
        df_for_processing = df_for_processing[
            (df_for_processing['review_word_count'] >= min_review_words) &
            (df_for_processing['review_word_count'] <= max_review_words)
        ]
        print(f"   Length filter kept {len(df_for_processing)} of {initial_len} rows.")
        if df_for_processing.empty:
            print("   ⚠️ No reviews remaining after length filter.")
            return pd.DataFrame(columns=list(df.columns) + ['label'] if 'label' not in df.columns else df.columns).iloc[0:0]

    # --- FILTERING LOGIC FOR HELPFUL VOTES ---
    if min_helpful_votes is not None and min_helpful_votes > 0:
        if "helpful_vote" in df_for_processing.columns:
            # Ensure 'helpful_vote' is numeric before filtering
            df_for_processing.loc[:, 'helpful_vote'] = pd.to_numeric(df_for_processing['helpful_vote'], errors='coerce').fillna(0)
            
            count_before_min_indiv_votes = len(df_for_processing)
            df_for_processing = df_for_processing[df_for_processing["helpful_vote"] >= min_helpful_votes]
            print(f"   Applied min_helpful_votes >= {min_helpful_votes} (on 'helpful_vote' column). Kept {len(df_for_processing)} of {count_before_min_indiv_votes} rows.")
        else:
            print(f"   ⚠️ 'helpful_vote' column not found for min_helpful_votes filter. Skipping this filter.") # Should not happen if preprocessing is correct
    
    if df_for_processing.empty: # Check if empty after this new filter
        print("   ⚠️ No reviews remaining after min_helpful_votes filter (and possibly length filter).")
        out_cols = list(df.columns)
        if 'label' not in out_cols: out_cols.append('label')
        return pd.DataFrame(columns=out_cols).iloc[0:0]

    # df_filtered will be the DataFrame used for labeling logic after this point
    df_filtered = df_for_processing # Start with the (potentially) length-filtered data

    # Conditionally apply min_total_votes filter
    # This filter is relevant if min_total_votes is specified AND (typically) if we are in threshold mode where total_vote is meaningful
    if min_total_votes is not None and min_total_votes > 0: # Check if filter is intended
        if "total_vote" in df_filtered.columns:
            # Ensure total_vote is numeric before filtering
            if pd.api.types.is_numeric_dtype(df_filtered["total_vote"]):
                count_before_min_votes = len(df_filtered)
                df_filtered = df_filtered[df_filtered["total_vote"] >= min_total_votes]
                print(f"   Applied min_total_votes >= {min_total_votes} (using 'total_vote' column). Kept {len(df_filtered)} of {count_before_min_votes} rows.")
            else:
                print(f"   ⚠️ 'total_vote' column found but is not numeric. Skipping min_total_votes filter.")
        else:
            # If total_vote column isn't there, this filter cannot be applied as originally intended.
            # For percentile mode, min_total_votes from config might be intended to apply to something else,
            # or it's a leftover. The current logic strictly ties it to the "total_vote" column.
            print(f"   ⚠️ min_total_votes filter ({min_total_votes}) specified, but 'total_vote' column not found. Skipping this filter.")
    # else: (min_total_votes is 0 or None, so no filter based on it)


    if df_filtered.empty:
        print("   ⚠️ No reviews remaining after length and/or min_total_votes filters.")
        # Ensure columns match potential expectation even if empty
        return pd.DataFrame(columns=list(df.columns) + ['label'] if 'label' not in df.columns else df.columns).iloc[0:0]


    labeled_df_output = pd.DataFrame() 

    if mode == "threshold":
        if "helpful_vote" not in df_filtered.columns: # Should be guaranteed by preprocessing
            raise ValueError("Missing 'helpful_vote' for threshold mode.")
        if "total_vote" not in df_filtered.columns: # CRITICAL: This column is now conditional
            raise ValueError("Missing 'total_vote' column for threshold mode. It should have been created in preprocessing if mode is 'threshold'.")
        if helpful_ratio_min is None or unhelpful_ratio_max is None:
            raise ValueError("helpful_ratio_min/unhelpful_ratio_max not provided for threshold mode.")

        # Ensure 'total_vote' is numeric here before division
        if not pd.api.types.is_numeric_dtype(df_filtered["total_vote"]):
            raise TypeError("'total_vote' column must be numeric for threshold mode calculations.")
        if not pd.api.types.is_numeric_dtype(df_filtered["helpful_vote"]):
            raise TypeError("'helpful_vote' column must be numeric for threshold mode calculations.")


        df_filtered_copy = df_filtered.copy() # Work on a copy to add columns safely
        df_filtered_copy.loc[:, 'helpful_ratio'] = pd.NA 
        mask_safe_div = df_filtered_copy["total_vote"] > 0
        
        # Ensure dtypes for division
        helpful_votes_numeric = pd.to_numeric(df_filtered_copy["helpful_vote"], errors='coerce')
        total_votes_numeric = pd.to_numeric(df_filtered_copy["total_vote"], errors='coerce')

        df_filtered_copy.loc[mask_safe_div, 'helpful_ratio'] = helpful_votes_numeric[mask_safe_div] / total_votes_numeric[mask_safe_div]
        
        df_filtered_copy.loc[:, "label"] = pd.NA 
        df_filtered_copy.loc[df_filtered_copy["helpful_ratio"] >= helpful_ratio_min, "label"] = 1
        df_filtered_copy.loc[df_filtered_copy["helpful_ratio"] <= unhelpful_ratio_max, "label"] = 0
        labeled_df_output = df_filtered_copy[df_filtered_copy["label"].isin([0, 1])].copy() # .copy() already here

    elif mode == "percentile":
        if "helpful_vote" not in df_filtered.columns: # Should be guaranteed
            raise ValueError("Missing 'helpful_vote' for percentile mode sorting.")
        if top_percentile is None or bottom_percentile is None:
            raise ValueError("top/bottom_percentile not provided for percentile mode.")
        
        # Ensure 'helpful_vote' is numeric for sorting
        if not pd.api.types.is_numeric_dtype(df_filtered["helpful_vote"]):
            df_filtered["helpful_vote"] = pd.to_numeric(df_filtered["helpful_vote"], errors='coerce')
            df_filtered.dropna(subset=["helpful_vote"], inplace=True) # Drop if became NaN after coercion
            if df_filtered.empty:
                print("   ⚠️ No reviews remaining after ensuring 'helpful_vote' is numeric for percentile sort.")
                return pd.DataFrame(columns=list(df.columns) + ['label'] if 'label' not in df.columns else df.columns).iloc[0:0]


        df_sorted = df_filtered.sort_values("helpful_vote", ascending=False, kind='mergesort', na_position='last').reset_index(drop=True)
        total_filtered_rows = len(df_sorted)
        top_n = int(total_filtered_rows * top_percentile)
        bottom_n = int(total_filtered_rows * bottom_percentile)

        if top_n == 0 and bottom_n == 0 and total_filtered_rows > 0 : # Avoid warning if df is already empty
            print("   ⚠️ Calculated top_n and bottom_n are both zero for percentile mode, but rows exist. No rows will be labeled.")
            # Still proceed to return an empty labeled_df_output with correct columns
        elif total_filtered_rows == 0:
            print("   ⚠️ No rows available for percentile labeling after filters.")


        df_sorted.loc[:, "label"] = pd.NA 
        if top_n > 0:
            df_sorted.iloc[:top_n, df_sorted.columns.get_loc("label")] = 1
        if bottom_n > 0:
            actual_bottom_n = min(bottom_n, total_filtered_rows) # Ensure we don't try to index beyond df length
            if actual_bottom_n > 0 : # Only assign if there are rows to take
                df_sorted.iloc[-actual_bottom_n:, df_sorted.columns.get_loc("label")] = 0
        
        labeled_df_output = df_sorted[df_sorted["label"].isin([0, 1])].copy() # .copy() already here

    if not labeled_df_output.empty:
        labeled_df_output.loc[:, "label"] = labeled_df_output["label"].astype(int)
        h_count = (labeled_df_output['label'] == 1).sum()
        u_count = (labeled_df_output['label'] == 0).sum()
        print(f"✅ Labeled: {len(labeled_df_output)} rows (H: {h_count}, U: {u_count}) from {original_count} initial.")
    else:
        # Construct an empty DataFrame with the expected columns if labeled_df_output is empty
        # This ensures consistency in what's returned, especially the 'label' column.
        expected_cols = list(df.columns) # Start with original df columns
        if 'label' not in expected_cols: expected_cols.append('label')
        if 'helpful_ratio' not in expected_cols and mode == "threshold": # helpful_ratio is specific to threshold
            expected_cols.append('helpful_ratio')
        
        print(f"✅ Labeled: 0 rows from {original_count} initial (mode: {mode}).")
        return pd.DataFrame(columns=expected_cols).iloc[0:0]


    return labeled_df_output