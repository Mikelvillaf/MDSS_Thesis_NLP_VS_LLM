# scripts/llm_prediction.py
import os
import time
import random
from typing import List, Optional, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import warnings

# Attempt to import openai, provide instructions if missing
try:
    from openai import OpenAI, APIError, Timeout, RateLimitError
except ImportError:
    raise ImportError("OpenAI library not found. Please install it: pip install openai")

# --- Constants ---
# WARNING: Hardcoding keys is insecure. Use environment variables for production.
# For initial testing as requested, using the key from the PDF.
# Replace with None or os.getenv("YOUR_ENV_VAR") in production.
TESTING_API_KEY = "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"

LABEL_MAP_TO_TEXT = {1: "Helpful", 0: "Unhelpful"}
LABEL_MAP_FROM_TEXT = {"helpful": 1, "unhelpful": 0}

# --- Helper Functions ---

def _setup_openai_client(api_key: Optional[str] = None, env_var_name: Optional[str] = "OPENAI_API_KEY") -> Optional[OpenAI]:
    """Initializes the OpenAI client."""
    key_to_use = None
    if api_key:
        key_to_use = api_key
        # print("   LLM: Using provided API key.")
    elif env_var_name and os.getenv(env_var_name):
        key_to_use = os.getenv(env_var_name)
        # print(f"   LLM: Using API key from environment variable '{env_var_name}'.")
    elif TESTING_API_KEY:
         key_to_use = TESTING_API_KEY
         warnings.warn(
             "Using hardcoded TESTING_API_KEY from llm_prediction.py. "
             "This is insecure and only for initial testing. Set the environment variable."
         )
         print("   LLM: WARNING - Using hardcoded testing API key.")

    if not key_to_use:
        print(f"❌ LLM Error: OpenAI API key not found. Provide 'api_key' or set environment variable '{env_var_name}'.")
        return None

    try:
        client = OpenAI(api_key=key_to_use)
        # Optional: Test connection (costs minimal tokens)
        # client.models.list()
        # print("   LLM: OpenAI client initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ LLM Error: Failed to initialize OpenAI client: {e}")
        return None

def _parse_llm_response(response_content: str) -> Optional[int]:
    """Parses the LLM's response ('Helpful'/'Unhelpful') into 1/0."""
    if not response_content or not isinstance(response_content, str):
        return None
    # Clean whitespace and convert to lower case for robust matching
    cleaned_response = response_content.strip().lower()
    # Remove potential punctuation
    cleaned_response = ''.join(filter(str.isalpha, cleaned_response))

    return LABEL_MAP_FROM_TEXT.get(cleaned_response, None) # Return None if not 'helpful' or 'unhelpful'

def _build_zero_shot_prompt(template: str, review_text: str) -> str:
    """Builds the zero-shot prompt using the template."""
    return template.format(review_text=review_text)

def _build_few_shot_prompt(template: str, examples_str: str, review_text: str) -> str:
    """Builds the few-shot prompt using the template."""
    return template.format(examples=examples_str, review_text=review_text)

def _format_examples(examples_df: pd.DataFrame, format_template: str) -> str:
    """Formats the selected few-shot examples into a single string."""
    example_strings = []
    for _, row in examples_df.iterrows():
        label_text = LABEL_MAP_TO_TEXT.get(row['label'], 'Unknown') # Convert 0/1 to text
        # Ensure review_text exists, fallback to empty string if missing
        review_text = row.get('full_text', '')
        try:
            example_strings.append(
                format_template.format(review_text=review_text, label_text=label_text).strip()
            )
        except KeyError as e:
             print(f"⚠️ Warning: Missing key '{e}' in format_template or examples_df row. Skipping example.")
             continue

    return "\n\n".join(example_strings)

def select_few_shot_examples(
    train_df: pd.DataFrame,
    num_examples: int,
    strategy: str,
    seed: int
) -> Optional[pd.DataFrame]:
    """Selects examples from the training data for few-shot prompting."""
    if 'label' not in train_df.columns or 'full_text' not in train_df.columns:
        print("❌ LLM Error: train_df missing 'label' or 'full_text' for few-shot example selection.")
        return None
    if train_df.empty:
         print("❌ LLM Error: train_df is empty, cannot select few-shot examples.")
         return None

    # Ensure num_examples is positive
    if num_examples <= 0:
        print(f"❌ LLM Error: num_examples must be positive, got {num_examples}.")
        return None

    if strategy == 'balanced_random':
        n_each = num_examples // 2
        if n_each == 0: # Handle case where num_examples is 1
             print("⚠️ Warning: num_examples=1 for balanced strategy. Using random sampling instead.")
             return select_few_shot_examples(train_df, num_examples, 'random', seed)

        if num_examples % 2 != 0:
             print(f"⚠️ Warning: num_examples ({num_examples}) is odd for balanced strategy. Using {n_each} per class (will result in {n_each*2} examples).")

        helpful_pool = train_df[train_df['label'] == 1]
        unhelpful_pool = train_df[train_df['label'] == 0]

        # Sample, allowing replacement only if pool size is less than required sample size
        helpful_examples = helpful_pool.sample(n=n_each, random_state=seed, replace=len(helpful_pool) < n_each)
        unhelpful_examples = unhelpful_pool.sample(n=n_each, random_state=seed+1, replace=len(unhelpful_pool) < n_each)

        if len(helpful_examples) < n_each or len(unhelpful_examples) < n_each:
             print(f"⚠️ Warning: Could not sample {n_each} examples for each class. Available H: {len(helpful_pool)}, U: {len(unhelpful_pool)}. Using available.")

        # Combine and shuffle
        examples_df = pd.concat([helpful_examples, unhelpful_examples]).sample(frac=1, random_state=seed+2).reset_index(drop=True)
        # Ensure we don't exceed num_examples (although balanced might return n_each*2 if num_examples was odd)
        examples_df = examples_df.head(num_examples)

    elif strategy == 'random':
        examples_df = train_df.sample(n=num_examples, random_state=seed, replace=len(train_df) < num_examples)
        if len(examples_df) < num_examples:
            print(f"⚠️ Warning: Could not sample {num_examples} random examples. Available: {len(train_df)}. Using available.")
    else:
        print(f"❌ LLM Error: Unknown few-shot example selection strategy: '{strategy}'.")
        return None

    if examples_df.empty:
         print("⚠️ Warning: No few-shot examples were selected.")
         return None # Return None if empty after selection attempt

    print(f"   Selected {len(examples_df)} few-shot examples using '{strategy}' strategy.")
    return examples_df


# --- Main Prediction Function ---

def get_llm_predictions(
    client: OpenAI,
    texts_to_classify: List[str],
    model_name: str,
    mode: str, # 'zero_shot' or 'few_shot'
    prompt_template: str,
    request_timeout: int = 30,
    max_retries: int = 3,
    retry_delay: int = 5,
    few_shot_examples_df: Optional[pd.DataFrame] = None,
    few_shot_example_format: Optional[str] = None
) -> Tuple[List[Optional[int]], int]:
    """
    Gets predictions from an OpenAI model for a list of texts using the specified mode.
    (Indentation Corrected)
    """
    predictions = []
    failed_count = 0
    examples_str = "" # Initialize

    # --- Corrected Indentation Block Starts Here ---
    if mode == 'few_shot':
        if few_shot_examples_df is None or few_shot_examples_df.empty or not few_shot_example_format:
            print("❌ LLM Error: Few-shot mode requires valid examples DataFrame and format template.")
            # Return lists matching the length of input texts, with appropriate failure count
            return [None] * len(texts_to_classify), len(texts_to_classify)

        examples_str = _format_examples(few_shot_examples_df, few_shot_example_format)
        if not examples_str:
            # Check if examples_str is empty *after* trying to format
            print("❌ LLM Error: Failed to format few-shot examples (result was empty string).")
            return [None] * len(texts_to_classify), len(texts_to_classify)
    # --- End Corrected Indentation Block ---

    print(f"   LLM: Getting predictions for {len(texts_to_classify)} texts using '{model_name}' ({mode})...")
    # Using tqdm for progress bar
    for text in tqdm(texts_to_classify, desc=f"LLM {mode} Prediction"):
        # Build the appropriate prompt
        if mode == 'zero_shot':
            prompt = _build_zero_shot_prompt(prompt_template, text)
        elif mode == 'few_shot':
            # Ensure examples_str was successfully created above
            if not examples_str: # Double check, though should be caught earlier
                 print(f"❌ LLM Error: examples_str is empty for few-shot mode text: '{text[:50]}...'. Skipping.")
                 predictions.append(None)
                 failed_count += 1
                 continue
            prompt = _build_few_shot_prompt(prompt_template, examples_str, text)
        else:
            print(f"❌ LLM Error: Invalid mode '{mode}'.")
            predictions.append(None)
            failed_count += 1
            continue

        # Make API call with retries
        prediction = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10, # Adjust if needed, but should be enough for "Helpful"/"Unhelpful"
                    timeout=request_timeout
                )
                response_content = response.choices[0].message.content
                prediction = _parse_llm_response(response_content)
                if prediction is None:
                     # Avoid printing entire response content if it's very long/irrelevant
                     print(f"\n   ⚠️ LLM Warning: Failed to parse response: '{response_content[:100]}' for text: '{text[:50]}...'")
                break # Success

            except (Timeout, RateLimitError, APIError) as e:
                print(f"\n   ⚠️ LLM Warning: API Error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    # Implement exponential backoff or just simple delay
                    actual_delay = retry_delay * (attempt + 1) # Simple exponential backoff
                    print(f"      Retrying in {actual_delay} seconds...")
                    time.sleep(actual_delay)
                else:
                    print(f"❌ LLM Error: Max retries exceeded for text: '{text[:50]}...'")
                    # Do not increment failed_count here, let the outer check handle it based on prediction being None
            except Exception as e:
                print(f"\n❌ LLM Error: Unexpected error during API call for text '{text[:50]}...': {e}")
                # Do not increment failed_count here, let the outer check handle it
                break # Unexpected error, don't retry

        # Append prediction (will be None if all retries failed or parsing failed)
        predictions.append(prediction)
        if prediction is None:
            failed_count += 1 # Increment fail count if prediction is None after loop

        # Optional: Add a small delay between requests to respect rate limits
        time.sleep(0.05) # 50ms delay

    if failed_count > 0:
        print(f"   LLM: Completed predictions with {failed_count} failures out of {len(texts_to_classify)}.")
    else:
        print(f"   LLM: Completed predictions successfully for all {len(texts_to_classify)} texts.")

    return predictions, failed_count