# scripts/data_loader.py

import pandas as pd
import os
import json
import numpy as np
import weave
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
import gzip
# Correctly import from label_generation module
from scripts.label_generation import generate_labels # <--- Ensure this IMPORT is present

@weave.op()
def load_reviews(filepath: str, year_range: Optional[List[int]] = None, max_rows: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Loads reviews from a JSON Lines file, treating it as plain text
    regardless of the '.gz' extension. Filters by year range after loading.
    Optionally samples max_rows.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📥 Loading reviews from: {filepath} (treating as plain JSON Lines text)")
    data = []
    line_count = 0
    skipped_lines = 0
    # Use standard open in text mode ('r' or 'rt') with UTF-8 encoding
    try:
        # Ensure tqdm description is helpful
        base_filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use tqdm for progress bar
            for line in tqdm(f, desc=f"Reading {base_filename}"):
                line_count += 1
                try:
                    # Skip empty lines
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    record = json.loads(stripped_line)
                    data.append(record)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    # Optionally log the line number or content causing the error
                    # print(f"Warning: Skipping malformed JSON on line {line_count}")
                    continue # Skip lines that fail to parse

    except Exception as e:
        print(f"❌ Critical error reading file {filepath}: {e}")
        return pd.DataFrame() # Return empty DataFrame on critical read error

    if skipped_lines > 0:
        print(f"⚠️ Skipped {skipped_lines} lines due to JSON parsing errors.")

    if not data:
        print("⚠️ No valid JSON records loaded.")
        return pd.DataFrame() # Return empty if no data loaded

    # Create DataFrame from the loaded data
    df = pd.DataFrame(data)
    print(f"✅ Initial load successful: {len(df)} records.")

    # --- Post-loading Filtering and Sampling ---

    # Apply year filtering (if required)
    if "timestamp" in df.columns:
         # Convert timestamp to datetime safely, coercing errors
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df['year'] = df['timestamp_dt'].dt.year # Extract year

        if year_range and isinstance(year_range, list) and len(year_range) == 2:
            original_count = len(df)
            # Drop rows where timestamp was invalid or year extraction failed
            df = df.dropna(subset=['year'])
            # Ensure year is integer before filtering
            if not df.empty: # Check if dataframe is not empty after dropping NAs
                 df['year'] = df['year'].astype(int)
                 df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
                 print(f"🗓️ Filtered by year range {year_range}. Kept {len(df)} out of {original_count} valid timestamp records.")
            else:
                 print("🗓️ DataFrame empty after dropping invalid timestamps. No year filtering applied.")
        elif year_range:
             print("⚠️ Warning: year_range provided but not in expected format [start_year, end_year]. Skipping year filter.")
         # Drop the temporary datetime column if no longer needed
         # df = df.drop(columns=['timestamp_dt']) # Optional cleanup
    elif year_range:
        print(f"⚠️ Warning: year_range specified but 'timestamp' column missing or invalid. Cannot filter by year.")


    # Apply max_rows sampling (if required and valid)
    if max_rows and isinstance(max_rows, int) and max_rows > 0:
        if len(df) > max_rows:
            print(f"🎲 Sampling down to {max_rows} reviews...")
            # Use specified seed for reproducibility if provided
            df = df.sample(n=max_rows, random_state=seed)
        else:
            print(f"ℹ️ Dataset has {len(df)} rows, less than or equal to max_rows={max_rows}. Using all.")
    elif max_rows:
        print(f"⚠️ Warning: Invalid max_rows value ({max_rows}). Skipping sampling.")


    print(f"✅ Final DataFrame ready: {len(df)} reviews.")
    return df


@weave.op()
def identify_label_candidates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Identifies potential label candidates (Helpful=1, Unhelpful=0) by calling
    the generate_labels function based on the configuration.

    Args:
        df (pd.DataFrame): DataFrame after preprocessing.
        config (dict): The overall configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame containing only rows labeled 0 or 1.
    """
    if 'labeling' not in config:
        raise ValueError("Configuration missing 'labeling' section.")

    label_cfg = config['labeling']
    mode = label_cfg.get('mode', 'threshold') # Default to threshold if not specified

    print(f"🏷️ Identifying label candidates using mode: '{mode}' (via generate_labels)")

    # Prepare arguments for generate_labels by extracting from config
    # Add defaults for length filter params if they are not in config
    label_args = {
        'df': df, # Pass the original df, generate_labels makes its own copy
        'mode': mode,
        'top_percentile': label_cfg.get('top_percentile', 0.2),
        'bottom_percentile': label_cfg.get('bottom_percentile', 0.4),
        'helpful_ratio_min': label_cfg.get('helpful_ratio_min', 0.75),
        'unhelpful_ratio_max': label_cfg.get('unhelpful_ratio_max', 0.35),
        'min_total_votes': label_cfg.get('min_total_votes', 10),
        # Add length filter args with defaults
        'use_length_filter': label_cfg.get('use_length_filter', False),
        'min_review_words': label_cfg.get('min_review_words', 10),
        'max_review_words': label_cfg.get('max_review_words', 1000)
    }

    # Basic checks before calling generate_labels (somewhat redundant, but safe)
    if mode == "threshold":
        if "helpful_vote" not in df.columns or "total_vote" not in df.columns:
            raise ValueError("Missing 'helpful_vote' or 'total_vote' for threshold labeling.")
    elif mode == "percentile":
         if "helpful_vote" not in df.columns:
             raise ValueError("Missing 'helpful_vote' column for percentile labeling.")
         # Percentile mode now also needs total_vote if min_total_votes filter is applied within it
         if "total_vote" not in df.columns:
             raise ValueError("Missing 'total_vote' column required for min_total_votes filter in percentile mode")

    # Call the actual labeling function from label_generation.py
    try:
        # --- CORRECTED INDENTATION HERE ---
        labeled_df = generate_labels(**label_args)
    except Exception as e:
        print(f"❌ Error calling generate_labels function: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        # Return an empty frame matching original columns + label if possible
        cols = list(df.columns)
        if 'label' not in cols: cols.append('label')
        # Ensure columns are unique before creating DataFrame
        cols = list(dict.fromkeys(cols)) # Preserves order, removes duplicates
        return pd.DataFrame(columns=cols)


    # The generate_labels function handles printing its own stats.

    if labeled_df.empty:
        print("⚠️ No reviews were labeled as helpful (1) or unhelpful (0).")
         # Return empty frame with correct columns
        cols = list(df.columns)
        if 'label' not in cols: cols.append('label')
        cols = list(dict.fromkeys(cols))
        return pd.DataFrame(columns=cols)

    elif 'label' not in labeled_df.columns or not pd.api.types.is_numeric_dtype(labeled_df['label']):
         print("⚠️ generate_labels did not return a valid DataFrame with a numeric 'label' column.")
         # Return empty frame with correct columns
         cols = list(df.columns)
         if 'label' not in cols: cols.append('label')
         cols = list(dict.fromkeys(cols))
         return pd.DataFrame(columns=cols)

    elif labeled_df['label'].isin([0, 1]).sum() == 0:
         print("⚠️ generate_labels returned rows, but none had label 0 or 1.")
         # Return empty frame with correct columns (already includes 'label')
         return labeled_df[0:0]

    # Ensure label is integer type (already done in generate_labels, but safe)
    labeled_df['label'] = labeled_df['label'].astype(int)

    print(f"✅ identify_label_candidates returning {len(labeled_df)} labeled rows.")
    return labeled_df


@weave.op()
def create_balanced_temporal_splits(
    df: pd.DataFrame,
    config: dict,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates train, validation, and test splits based on year.
    Applies optional balancing or max sample limits based on config.

    If config['balanced_sampling']['use_strict_balancing'] is True, it applies
    strict balancing by sampling min(target, available_helpful, available_unhelpful)
    from each class within each split period, using 'samples_per_class' limits.

    If config['balanced_sampling']['use_strict_balancing'] is False, it uses
    all data for the period by default, BUT if 'max_total_samples_imbalanced'
    is set in the config, it samples down to that total limit while preserving
    the natural class ratio.

    Args:
        df (pd.DataFrame): DataFrame containing *only* rows labeled 0 or 1.
        config (dict): Overall configuration dictionary.
        seed (int): Random seed for sampling.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test splits.
    """
    split_cfg = config.get('temporal_split_years', {})
    sampling_cfg = config.get('balanced_sampling', {}) # Get sampling config safely
    use_balancing = sampling_cfg.get('use_strict_balancing', False) # Default to False if flag missing
    samples_per_class_target = sampling_cfg.get('samples_per_class', {}) # Get target samples for balancing
    max_total_cfg = sampling_cfg.get('max_total_samples_imbalanced', {}) # Get max total samples for imbalanced

    if 'year' not in df.columns:
        raise ValueError("Missing 'year' column for temporal splitting. Ensure preprocessing adds it.")
    if df.empty:
         raise ValueError("Received empty DataFrame in create_balanced_temporal_splits, cannot create splits.")
    if 'label' not in df.columns or not pd.api.types.is_numeric_dtype(df['label']) or not df['label'].isin([0, 1]).all():
         print("Label column status:")
         if 'label' not in df.columns: print(" - Missing 'label' column.")
         else:
              print(f" - 'label' column type: {df['label'].dtype}")
              print(f" - All labels are 0 or 1: {df['label'].isin([0, 1]).all()}")
              print(f" - Label value counts:\n{df['label'].value_counts()}")
         raise ValueError("Input DataFrame to create_balanced_temporal_splits must contain only rows with label 0 or 1.")


    df_candidates = df.copy() # Input df IS the candidates pool
    df_candidates['label'] = df_candidates['label'].astype(int) # Ensure int type

    # --- Map Years to Splits ---
    splits = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    split_map = {}

    # Handle train_years: can be list [start, end] or list of years
    train_years_config = split_cfg.get('train_years', [])
    train_years_list = []
    if isinstance(train_years_config, list):
        if len(train_years_config) == 2 and all(isinstance(y, int) for y in train_years_config):
             train_years_list = list(range(train_years_config[0], train_years_config[1] + 1))
        elif all(isinstance(y, int) for y in train_years_config):
             train_years_list = train_years_config
        else:
             print("⚠️ Warning: 'train_years' format not recognized. No training years defined.")
    else:
         print("⚠️ Warning: 'train_years' not found or not a list. No training years defined.")

    for yr in train_years_list: split_map[yr] = 'train'

    # Handle val_year and test_year
    val_year = split_cfg.get('val_year')
    test_year = split_cfg.get('test_year')
    if val_year and isinstance(val_year, int): split_map[val_year] = 'val'
    else: print("⚠️ Warning: 'val_year' not found or not an integer. No validation split.")
    if test_year and isinstance(test_year, int): split_map[test_year] = 'test'
    else: print("⚠️ Warning: 'test_year' not found or not an integer. No test split.")

    # Add 'split_period' column based on the map
    df_candidates['split_period'] = df_candidates['year'].map(split_map)

    # Keep only rows that fall into one of the defined periods
    original_candidate_count = len(df_candidates)
    df_candidates = df_candidates.dropna(subset=['split_period'])
    print(f"ℹ️ Kept {len(df_candidates)} out of {original_candidate_count} candidates within defined train/val/test years.")


    if df_candidates.empty:
        print("⚠️ No candidates found within the specified train/val/test years after filtering.")
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    # --- Create Splits (Balanced or Imbalanced) ---
    defined_periods = [p for p in ['train', 'val', 'test'] if any(df_candidates['split_period'] == p)]

    for period in defined_periods:
        print(f"\n⚖️ Processing split for: {period}")
        period_df = df_candidates[df_candidates['split_period'] == period].copy()

        if period_df.empty:
             print(f"   - No candidates found for period '{period}'. Skipping.")
             splits[period] = pd.DataFrame(columns=df_candidates.columns)
             continue

        if use_balancing:
            # --- Strict Balancing Logic ---
            print(f"   Applying STRICT balancing for '{period}'...")
            n_target_per_class = samples_per_class_target.get(period, 0) # Default target to 0

            if n_target_per_class <= 0:
                print(f"   ⚠️ Target samples per class for '{period}' is {n_target_per_class}. Balancing skipped. Returning empty split for '{period}'.")
                splits[period] = pd.DataFrame(columns=df_candidates.columns)
                continue

            helpful_pool = period_df[period_df['label'] == 1]
            unhelpful_pool = period_df[period_df['label'] == 0]
            n_helpful_avail = len(helpful_pool)
            n_unhelpful_avail = len(unhelpful_pool)

            n_sample_per_class = max(0, min(n_target_per_class, n_helpful_avail, n_unhelpful_avail))

            if n_sample_per_class == 0:
                print(f"   ⚠️ Cannot create balanced split for {period}: Not enough samples in at least one class (Available: H={n_helpful_avail}, U={n_unhelpful_avail}). Returning empty split.")
                splits[period] = pd.DataFrame(columns=df_candidates.columns)
                continue

            if n_sample_per_class < n_target_per_class and (n_helpful_avail > 0 and n_unhelpful_avail > 0):
                print(f"   ⚠️ Warning: Target was {n_target_per_class} per class, but only {n_sample_per_class} possible due to limited availability "
                      f"(Available: H={n_helpful_avail}, U={n_unhelpful_avail}) for {period}.")

            sampled_helpful = helpful_pool.sample(n=n_sample_per_class, random_state=seed)
            sampled_unhelpful = unhelpful_pool.sample(n=n_sample_per_class, random_state=seed)
            final_split_df = pd.concat([sampled_helpful, sampled_unhelpful], ignore_index=True)

            splits[period] = final_split_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"   ✅ Created BALANCED '{period}' split: {len(splits[period])} rows "
                  f"(Helpful: {len(sampled_helpful)}, Unhelpful: {len(sampled_unhelpful)})")

        else:
            # --- Imbalanced Logic (with optional max total size) ---
            print(f"   Strict balancing disabled.")
            max_samples = max_total_cfg.get(period) # Get limit for train/val/test

            if max_samples and isinstance(max_samples, int) and max_samples > 0 and len(period_df) > max_samples:
                print(f"   Limiting '{period}' to {max_samples} total samples (from {len(period_df)} available).")
                # Sample down to the max limit, maintaining imbalance (simple random sample)
                sampled_df = period_df.sample(n=max_samples, random_state=seed)
                # Shuffle final sample
                splits[period] = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            else:
                # Use all data if no limit set, limit is invalid, or data is already below limit
                if max_samples and isinstance(max_samples, int) and max_samples > 0:
                     print(f"   Using all {len(period_df)} candidates for '{period}' (within max limit of {max_samples}).")
                else:
                     print(f"   Using all {len(period_df)} candidates for '{period}' (no valid max limit specified).")
                # Just use all data for the period, shuffle it
                splits[period] = period_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            h_count = (splits[period]['label'] == 1).sum()
            u_count = (splits[period]['label'] == 0).sum()
            print(f"   ✅ Created IMBALANCED '{period}' split: {len(splits[period])} rows "
                  f"(Helpful: {h_count}, Unhelpful: {u_count})")


    # --- Finalize and Return Splits ---
    final_cols = list(df_candidates.columns) # Use columns from the candidates pool before splitting
    final_splits = []
    for period in ['train', 'val', 'test']:
         split_df = splits.get(period)
         if split_df is None or split_df.empty:
             print(f"   - No data generated for '{period}' split. Returning empty DataFrame.")
             final_splits.append(pd.DataFrame(columns=final_cols))
         else:
             # Ensure columns are consistent
             current_cols = list(split_df.columns)
             missing_cols = [col for col in final_cols if col not in current_cols]
             if missing_cols:
                 print(f"   Warning: Adding missing columns {missing_cols} to {period} split (filling with NaN).")
                 for col in missing_cols:
                     split_df[col] = np.nan
             # Ensure final column order and select only those columns
             final_splits.append(split_df[final_cols])

    # Check if exactly 3 splits were generated before returning
    if len(final_splits) != 3:
         raise RuntimeError(f"Expected 3 splits (train, val, test) but generated {len(final_splits)}")

    return tuple(final_splits) # Return train, val, test