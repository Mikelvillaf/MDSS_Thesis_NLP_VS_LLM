# scripts/data_loader.py

# (Keep imports: pandas, os, json, numpy, weave, tqdm, typing, gzip)
import pandas as pd
import os
import json
import numpy as np
import weave
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
import gzip # Keep import in case needed elsewhere or future files are actually gzipped

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
		with open(filepath, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc=f"Reading {os.path.basename(filepath)}"):
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
		df["year"] = pd.to_datetime(df["timestamp"], unit="ms", errors='coerce').dt.year
		if year_range:
			original_count = len(df)
			df = df.dropna(subset=['year']) # Drop rows where timestamp was invalid
			df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
			print(f"🗓️ Filtered by year range {year_range}. Kept {len(df)} out of {original_count} records.")
	elif year_range:
		print(f"⚠️ Warning: year_range specified but 'timestamp' column missing. Cannot filter by year.")

	# Apply max_rows sampling (if required)
	if max_rows:
		if len(df) > max_rows:
			print(f"🎲 Sampling down to {max_rows} reviews...")
			df = df.sample(n=max_rows, random_state=seed)
		else:
			print(f"ℹ️ Dataset has {len(df)} rows, less than max_rows={max_rows}. Using all.")

	print(f"✅ Final DataFrame ready: {len(df)} reviews.")
	return df

# --- Keep identify_label_candidates and create_balanced_temporal_splits ---
# (Ensure these functions are still present below with correct tab indentation)
@weave.op()
def identify_label_candidates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
	# ...(Same tab-indented code as previous working version)...
	label_cfg = config['labeling']
	df = df.copy()

	if "helpful_vote" not in df.columns or "total_vote" not in df.columns:
		raise ValueError("Missing 'helpful_vote' or 'total_vote'. Run preprocessing first.")

	df['label'] = -1 # Initialize label column (-1 indicates rows to be discarded initially)
	df['total_vote'] = pd.to_numeric(df['total_vote'], errors='coerce')
	df.dropna(subset=['total_vote', 'helpful_vote'], inplace=True)

	mask_min_votes = (df["total_vote"] >= label_cfg['min_total_votes'])
	df_filtered_min_votes = df[mask_min_votes].copy()

	if df_filtered_min_votes.empty:
		print("⚠️ No reviews met the min_total_votes requirement after basic cleaning.")
		df['label'] = -1
		return df[0:0]

	mask_total_vote_positive = df_filtered_min_votes["total_vote"] > 0
	valid_df_for_ratio = df_filtered_min_votes[mask_total_vote_positive].copy()

	if valid_df_for_ratio.empty:
		print("⚠️ No reviews with total_vote > 0 after min_votes filtering.")
		df['label'] = -1
		return df[0:0]

	valid_df_for_ratio["helpful_ratio"] = valid_df_for_ratio["helpful_vote"] / valid_df_for_ratio["total_vote"]
	helpful_mask = valid_df_for_ratio["helpful_ratio"] >= label_cfg['helpful_ratio_min']
	unhelpful_mask = valid_df_for_ratio["helpful_ratio"] <= label_cfg['unhelpful_ratio_max']

	df.loc[valid_df_for_ratio[helpful_mask].index, 'label'] = 1
	df.loc[valid_df_for_ratio[unhelpful_mask].index, 'label'] = 0

	print(f"✅ Identified candidates: "
		f"Helpful={ (df['label'] == 1).sum() }, "
		f"Unhelpful={ (df['label'] == 0).sum() }, "
		f"Discarded={ (df['label'] == -1).sum() }")

	return df

@weave.op()
def create_balanced_temporal_splits(
	df: pd.DataFrame,
	config: dict,
	seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Creates balanced train, validation, and test splits based on year and sampling config.
	Enforces strict balance by sampling min(target, available_helpful, available_unhelpful)
	from each class.
	"""
	split_cfg = config['temporal_split_years']
	sampling_cfg = config['balanced_sampling']['samples_per_class']

	if 'year' not in df.columns:
		raise ValueError("Missing 'year' column for temporal splitting.")
	if 'label' not in df.columns:
		raise ValueError("Missing preliminary 'label' column. Run identify_label_candidates first.")

	df_candidates = df[df['label'].isin([0, 1])].copy()
	if not df_candidates.empty:
		df_candidates['label'] = df_candidates['label'].astype(int)
	else:
		print("⚠️ No valid candidates (label 0 or 1) found to create splits.")
		# Return empty DataFrames with original columns
		return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

	splits = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
	split_map = {}
	for yr in split_cfg['train_years']: split_map[yr] = 'train'
	split_map[split_cfg['val_year']] = 'val'
	split_map[split_cfg['test_year']] = 'test'

	df_candidates['split_period'] = df_candidates['year'].map(split_map)
	df_candidates.dropna(subset=['split_period'], inplace=True)

	if df_candidates.empty:
		print("⚠️ No candidates found within the specified train/val/test years.")
		# Return empty DataFrames with original columns
		return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

	# --- Start: Strict Balancing Logic ---
	for period in ['train', 'val', 'test']:
		print(f"\n⚖️ Creating STRICTLY balanced split for: {period}")
		period_df = df_candidates[df_candidates['split_period'] == period]
		# Use .get() with a default of 0 for the target samples
		n_target_per_class = sampling_cfg.get(period, 0)

		if n_target_per_class == 0:
			print(f"⚠️ Target samples for '{period}' is 0. Skipping.")
			# Ensure created empty DataFrame has the correct columns
			splits[period] = pd.DataFrame(columns=df_candidates.columns)
			continue

		helpful_pool = period_df[period_df['label'] == 1]
		unhelpful_pool = period_df[period_df['label'] == 0]
		n_helpful_avail = len(helpful_pool)
		n_unhelpful_avail = len(unhelpful_pool)

		# Determine the maximum number of samples we can take *from each class*
		# while maintaining balance and respecting the target.
		n_sample_per_class = max(0, min(n_target_per_class, n_helpful_avail, n_unhelpful_avail))

		if n_sample_per_class == 0:
			print(f"⚠️ Cannot create balanced split for {period}: Not enough samples in at least one class (Available: H={n_helpful_avail}, U={n_unhelpful_avail}).")
			splits[period] = pd.DataFrame(columns=df_candidates.columns) # Ensure correct columns
			continue

		# Print warnings if the actual sample size is less than the original target
		# Only warn if there was at least *some* availability in both classes but less than target
		if n_sample_per_class < n_target_per_class and (n_helpful_avail > 0 and n_unhelpful_avail > 0):
			print(f"⚠️ Warning: Target was {n_target_per_class} per class, but only {n_sample_per_class} possible due to limited availability "
				f"(Available: H={n_helpful_avail}, U={n_unhelpful_avail}) for {period}.")

		# Sample exactly n_sample_per_class from each pool
		sampled_helpful = helpful_pool.sample(n=n_sample_per_class, random_state=seed)
		sampled_unhelpful = unhelpful_pool.sample(n=n_sample_per_class, random_state=seed)

		final_split_df = pd.concat([sampled_helpful, sampled_unhelpful], ignore_index=True)

		# Shuffle the final balanced split
		splits[period] = final_split_df.sample(frac=1, random_state=seed).reset_index(drop=True)
		print(f"✅ Created '{period}' split: {len(splits[period])} rows "
			f"(Helpful: {len(sampled_helpful)}, Unhelpful: {len(sampled_unhelpful)})") # Counts should be equal

	# --- End: Strict Balancing Logic ---

	# Ensure consistent columns in returned dataframes
	# Use columns from df_candidates as it contains all necessary cols after labeling/splitting
	final_cols = list(df_candidates.columns)
	for period in ['train', 'val', 'test']:
		if splits[period].empty:
			# Ensure empty dataframes have the correct columns
			splits[period] = pd.DataFrame(columns=final_cols)
		else:
			# Check for missing columns and add them if necessary before reordering
			missing_cols = [col for col in final_cols if col not in splits[period].columns]
			if missing_cols:
				print(f"Warning: Adding missing columns {missing_cols} to {period} split.")
				for col in missing_cols:
					# Assign a default value suitable for the likely column type, e.g., NaN
					splits[period][col] = np.nan
			# Ensure column order matches
			splits[period] = splits[period][final_cols]

	return splits['train'], splits['val'], splits['test']