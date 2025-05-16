# scripts/data_loader.py

import pandas as pd
import os
import json
import numpy as np
import random # For random.sample
import weave
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple, Any
from scripts.label_generation import generate_labels


@weave.op()
def load_and_clean_metadata(metadata_filepath: str) -> pd.DataFrame:
    if not os.path.exists(metadata_filepath):
        print(f"⚠️ Metadata file not found: {metadata_filepath}")
        return pd.DataFrame(columns=['parent_asin', 'price'])

    print(f"📥 Loading metadata: {os.path.basename(metadata_filepath)}")
    metadata = []
    try:
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if 'parent_asin' in record and 'price' in record:
                    metadata.append({'parent_asin': record['parent_asin'], 'price': record['price']})
    except Exception as e:
        print(f"❌ Error reading metadata {metadata_filepath}: {e}")
        return pd.DataFrame(columns=['parent_asin', 'price'])

    if not metadata:
        print("⚠️ No metadata records with parent_asin and price loaded.")
        return pd.DataFrame(columns=['parent_asin', 'price'])

    meta_df = pd.DataFrame(metadata)
    meta_df['price'] = pd.to_numeric(meta_df['price'], errors='coerce')
    meta_df.dropna(subset=['price'], inplace=True)
    meta_df = meta_df[meta_df['price'] > 0]
    meta_df.drop_duplicates(subset=['parent_asin'], keep='first', inplace=True)

    if meta_df.empty:
        print("⚠️ No valid metadata after cleaning.")
        return pd.DataFrame(columns=['parent_asin', 'price'])
    return meta_df[['parent_asin', 'price']]


@weave.op()
def load_reviews(
    filepath: str,
    year_range: Optional[List[int]] = None,
    max_initial_load: Optional[int] = None, # For limiting initial JSON lines processed
    seed: Optional[int] = None
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📥 Loading reviews from: {filepath}")
    base_filename = os.path.basename(filepath)

    data_to_parse_strings = [] # Will hold string lines

    # Determine if we need to limit and sample
    should_limit_and_sample = max_initial_load and isinstance(max_initial_load, int) and max_initial_load > 0

    if should_limit_and_sample:
        print(f"   Attempting to load at most {max_initial_load} random records from {base_filename}.")
        all_lines_read_for_sampling = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading all lines for sampling from {base_filename}"):
                    stripped_line = line.strip()
                    if stripped_line:
                        all_lines_read_for_sampling.append(stripped_line)

            total_lines_in_file = len(all_lines_read_for_sampling)
            print(f"   Read {total_lines_in_file} non-empty lines for sampling.")

            if total_lines_in_file > max_initial_load:
                print(f"   Randomly selecting {max_initial_load} lines from {total_lines_in_file}...")
                if seed is not None:
                    random.seed(seed) # Ensure reproducibility
                data_to_parse_strings = random.sample(all_lines_read_for_sampling, max_initial_load)
            else:
                print(f"   Total lines ({total_lines_in_file}) not greater than limit ({max_initial_load}). Using all lines.")
                data_to_parse_strings = all_lines_read_for_sampling
            del all_lines_read_for_sampling # Free memory

        except MemoryError:
            print(f"❌ MemoryError: Could not load all lines from {filepath} into memory for random sampling.")
            print(f"   Consider increasing system memory or reducing max_initial_load.")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Critical error during line reading/sampling for {filepath}: {e}")
            return pd.DataFrame()
    else:
        # Load all lines if no limit is set
        print(f"   No valid max_initial_load limit. Reading all lines sequentially from {base_filename}.")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading all lines from {base_filename}"):
                    stripped_line = line.strip()
                    if stripped_line:
                        data_to_parse_strings.append(stripped_line)
        except Exception as e:
            print(f"❌ Critical error reading all lines from {filepath}: {e}")
            return pd.DataFrame()

    # Parse the selected lines (either all or sampled strings)
    data = []
    skipped_json_parse = 0
    for line_content in tqdm(data_to_parse_strings, desc="Parsing selected JSON lines"):
        try:
            data.append(json.loads(line_content))
        except json.JSONDecodeError:
            skipped_json_parse += 1
    del data_to_parse_strings # Free memory from the list of strings

    if skipped_json_parse > 0:
        print(f"⚠️ Skipped {skipped_json_parse} malformed JSON lines during parsing.")
    if not data:
        print("⚠️ No valid JSON records to form DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"✅ Initial DataFrame created with {len(df)} records (pre-year-filter).")

    # --- Year filtering ---
    if "timestamp" in df.columns and isinstance(year_range, list) and len(year_range) == 2:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        # Modify inplace to save memory if df is large
        df.dropna(subset=['timestamp_dt'], inplace=True)
        if not df.empty:
            df['year'] = df['timestamp_dt'].dt.year.astype(int)
            original_count_before_year_filter = len(df)
            # This operation creates a new DataFrame. Reassign to df.
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
            print(f"🗓️ Filtered by year {year_range}. Kept {len(df)} of {original_count_before_year_filter}.")
            if 'timestamp_dt' in df.columns: # Check if column exists before dropping
                df.drop(columns=['timestamp_dt'], inplace=True)
        else:
            print("🗓️ DataFrame empty after dropping invalid timestamps (before year filter).")
    elif year_range:
        print(f"⚠️ Invalid year_range or 'timestamp' missing. Year filtering skipped.")

    if df.empty:
        print("⚠️ DataFrame is empty after all loading and filtering steps.")
    else:
        print(f"✅ Reviews ready for preprocessing: {len(df)} records.")
    return df


# Identifies label candidates using generate_labels.
@weave.op()
def identify_label_candidates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # This function now receives a DataFrame that might have been modified by preprocess_reviews
    # if we remove .copy() there.
    if 'labeling' not in config:
        raise ValueError("Config missing 'labeling' section.")
    label_cfg = config['labeling']
    print(f"🏷️ Labeling mode: '{label_cfg.get('mode', 'N/A')}'")

    try:
        # generate_labels is expected to handle its own memory internally
        # and return a NEW DataFrame of labeled candidates.
        labeled_df = generate_labels(
            df=df, # Pass the potentially large df
            mode=label_cfg.get('mode'),
            top_percentile=label_cfg.get('top_percentile'),
            bottom_percentile=label_cfg.get('bottom_percentile'),
            helpful_ratio_min=label_cfg.get('helpful_ratio_min'),
            unhelpful_ratio_max=label_cfg.get('unhelpful_ratio_max'),
            min_total_votes=label_cfg.get('min_total_votes'),
            min_helpful_votes=label_cfg.get('min_helpful_votes'),
            use_length_filter=label_cfg.get('use_length_filter', False),
            min_review_words=label_cfg.get('min_review_words'),
            max_review_words=label_cfg.get('max_review_words')
        )
    except Exception as e:
        print(f"❌ Error in generate_labels: {e}")
        # Fallback to return empty DataFrame with expected columns
        cols_to_return = list(df.columns)
        if 'label' not in cols_to_return: cols_to_return.append('label')
        # Ensure price is also considered if it was part of the input df
        if 'price' in df.columns and 'price' not in cols_to_return: cols_to_return.append('price')
        return pd.DataFrame(columns=list(dict.fromkeys(cols_to_return))).iloc[0:0]


    if labeled_df.empty:
        print("⚠️ No reviews labeled.")
    elif 'label' not in labeled_df.columns or not labeled_df['label'].isin([0, 1]).any():
        print("⚠️ generate_labels returned invalid labels.")
        # Fallback for invalid labels
        cols_to_return = list(df.columns)
        if 'label' not in cols_to_return: cols_to_return.append('label')
        if 'price' in df.columns and 'price' not in cols_to_return: cols_to_return.append('price')
        return pd.DataFrame(columns=list(dict.fromkeys(cols_to_return))).iloc[0:0]

    return labeled_df


# Helper for sampling within create_balanced_temporal_splits.
def _sample_period_data(
    period_df: pd.DataFrame, period: str, sampling_cfg: dict, seed: int
) -> pd.DataFrame:
    use_balancing = sampling_cfg.get('use_strict_balancing', False)
    samples_per_class_target = sampling_cfg.get('samples_per_class', {})
    max_total_cfg = sampling_cfg.get('max_total_samples_imbalanced', {})

    if period_df.empty: return pd.DataFrame(columns=period_df.columns)

    if use_balancing:
        n_target = samples_per_class_target.get(period, 0)
        if n_target <= 0:
            print(f"   ⚠️ Target samples for '{period}' <= 0. Empty split.")
            return pd.DataFrame(columns=period_df.columns)
        h_pool = period_df[period_df['label'] == 1]
        u_pool = period_df[period_df['label'] == 0]
        n_sample = min(n_target, len(h_pool), len(u_pool))
        if n_sample <= 0:
            print(f"   ⚠️ Cannot balance {period}: H={len(h_pool)}, U={len(u_pool)}, Target={n_target}.")
            return pd.DataFrame(columns=period_df.columns)
        if n_sample < n_target:
            print(f"   ⚠️ {period}: Target={n_target}/class, possible={n_sample}.")
        s_h = h_pool.sample(n=n_sample, random_state=seed)
        s_u = u_pool.sample(n=n_sample, random_state=seed + 1)
        final_df = pd.concat([s_h, s_u], ignore_index=True)
        print(f"   ✅ Balanced '{period}': {len(final_df)} ({n_sample}/class)")
    else: # Imbalanced or no specific target per class
        max_s = max_total_cfg.get(period)
        if max_s and isinstance(max_s, int) and max_s > 0 and len(period_df) > max_s:
            print(f"   Limiting '{period}' to {max_s} samples.")
            # Sample from the whole period_df if imbalanced
            final_df = period_df.sample(n=max_s, random_state=seed)
        else:
            final_df = period_df # Use all data for the period
        h_c = (final_df['label'] == 1).sum(); u_c = (final_df['label'] == 0).sum()
        print(f"   ✅ Imbalanced '{period}': {len(final_df)} (H={h_c}, U={u_c})")
    # Shuffle the final selected samples for this period
    return final_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)


# Creates train, validation, and test splits.
@weave.op()
def create_balanced_temporal_splits(
    df: pd.DataFrame, config: dict, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = config.get('temporal_split_years', {})
    sampling_cfg = config.get('balanced_sampling', {}) # For _sample_period_data

    if 'year' not in df.columns:
        raise ValueError("Missing 'year' column for temporal splitting.")
    if df.empty:
        print("⚠️ Cannot create splits from empty DataFrame passed to create_balanced_temporal_splits.")
        empty_df_template = pd.DataFrame(columns=df.columns if not df.columns.empty else ['year', 'label']) # Ensure some columns for consistency
        return empty_df_template, empty_df_template, empty_df_template
    if 'label' not in df.columns or not df['label'].isin([0,1]).all(): # Check on input 'df'
        raise ValueError("Input df to create_balanced_temporal_splits must have labels 0 or 1.")

    # This copy is made on df_labeled_pool from main.py, which is already reduced.
    # This should be acceptable.
    df_candidates = df.copy()
    df_candidates['label'] = df_candidates['label'].astype(int)

    split_map = {}
    train_years_cfg = split_cfg.get('train_years', [])
    train_years = []
    if isinstance(train_years_cfg, list):
        if len(train_years_cfg) == 2 and all(isinstance(y, int) for y in train_years_cfg):
            train_years = list(range(train_years_cfg[0], train_years_cfg[1] + 1))
        elif all(isinstance(y, int) for y in train_years_cfg):
            train_years = train_years_cfg
    for yr in train_years: split_map[yr] = 'train'

    val_year_cfg = split_cfg.get('val_year')
    if val_year_cfg is not None: split_map[int(val_year_cfg)] = 'val'

    test_year_cfg = split_cfg.get('test_year')
    if test_year_cfg is not None: split_map[int(test_year_cfg)] = 'test'

    df_candidates['split_period'] = df_candidates['year'].map(split_map)
    # Modify inplace to save memory on df_candidates
    df_candidates.dropna(subset=['split_period'], inplace=True)

    if df_candidates.empty:
        print("⚠️ No candidates in specified train/val/test years after mapping.")
        empty_df_template = pd.DataFrame(columns=df.columns) # Use original df columns for template
        return empty_df_template, empty_df_template, empty_df_template

    splits_data = {}
    print(f"\n⚖️ Processing splits...")
    for p_name_key in ['train', 'val', 'test']: # Use consistent keys
        # Filter df_candidates to get data for the current period
        period_data_df = df_candidates[df_candidates['split_period'] == p_name_key]
        splits_data[p_name_key] = _sample_period_data(period_data_df, p_name_key, sampling_cfg, seed)

    # Ensure consistent columns for all output DataFrames
    # Use the columns from the input `df` as the base, plus 'split_period' if it was added and not dropped
    # However, df_candidates (a copy of input df) is a better source for final columns if it's not empty
    final_cols_order = list(df_candidates.columns if not df_candidates.empty else df.columns)
    # Ensure 'split_period' is not carried over unless explicitly desired later
    if 'split_period' in final_cols_order:
        final_cols_order.remove('split_period')


    final_tuple_list = []
    for p_name_key in ['train', 'val', 'test']:
        split_df_current = splits_data.get(p_name_key)
        if split_df_current is None or split_df_current.empty:
            # Create an empty DataFrame with the correct columns if a split is missing/empty
            empty_split_df = pd.DataFrame(columns=final_cols_order)
            final_tuple_list.append(empty_split_df)
        else:
            # Ensure all necessary columns are present and in order
            for col_name in final_cols_order:
                if col_name not in split_df_current.columns:
                    # Add missing column with appropriate NA type
                    # For 'label' specifically, it should be numeric if present, otherwise can be object
                    if col_name == 'label' and col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                        split_df_current[col_name] = np.nan
                    else:
                        split_df_current[col_name] = pd.NA
            final_tuple_list.append(split_df_current[final_cols_order]) # Enforce column order

    return tuple(final_tuple_list)