# scripts/llm_prediction.py
import os
import time
import random
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod

# For Google AI Client SDK (as per your provided, working snippet)
# NO ALIAS. STRAIGHT IMPORT.
from google import genai

# For OpenAI
try: # Keep try-except for OpenAI as it's pre-existing and allows optional install
    from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError
except ImportError:
    OpenAI = None
    APIError = RateLimitError = AuthenticationError = APIConnectionError = APITimeoutError = Exception

# --- Constants ---
LABEL_MAP_TO_TEXT = {1: "1", 0: "0"}
LABEL_MAP_FROM_TEXT = {"1": 1, "0": 0}

# --- LLM Wrapper Base Class ---
class LLMWrapper(ABC):
    def __init__(self, api_key_env_var: str, model_id: str, **kwargs):
        self.model_id = model_id
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(f"API key from env var '{api_key_env_var}' not found for {self.__class__.__name__} (model: {self.model_id}).")
        self._setup_client(**kwargs)

    @abstractmethod
    def _setup_client(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str, request_timeout: int) -> Optional[str]:
        pass

# --- OpenAI Wrapper (Assumed correct) ---
class OpenAIWrapper(LLMWrapper):
    def _setup_client(self, **kwargs):
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed.")
        self.client = OpenAI(api_key=self.api_key)
        print(f"   OpenAI client initialized for model {self.model_id}.")

    def generate(self, prompt: str, request_timeout: int) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
                timeout=request_timeout
            )
            return response.choices[0].message.content
        except AuthenticationError as e:
            print(f"\n❌ OpenAI Auth Error for {self.model_id}: {e}")
            raise
        except Exception:
            return None

# --- Google AI Client Wrapper (Corrected to use YOUR genai.Client pattern) ---
class GoogleWrapper(LLMWrapper):
    def _setup_client(self, **kwargs):
        # The 'genai' module is imported directly at the top of the file.
        # We check if it was successfully imported and if it has the Client attribute.
        if 'genai' not in globals() or genai is None or not hasattr(genai, "Client"):
             raise ImportError("The 'genai' module from 'google' import does not seem to be correctly loaded or does not have a 'Client' attribute. Ensure the correct Google AI SDK is installed.")
        
        self.client = genai.Client(api_key=self.api_key) # Using the imported 'genai' directly
        print(f"   Google AI Client initialized for model {self.model_id}.")

    def generate(self, prompt: str, request_timeout: int) -> Optional[str]:
        if not self.client:
             return None
        try:
            api_model_id = self.model_id
            if not self.model_id.startswith("models/"):
                 api_model_id = f"models/{self.model_id}"
            
            response = self.client.models.generate_content(
                model=api_model_id,
                contents=prompt,
            )
            if hasattr(response, 'text'):
                return response.text
            # print(f"   Warning: Response object from Google AI Client for model {self.model_id} does not have a .text attribute. Response: {response}")
            return None
        except Exception:
            return None

# --- Helper Functions (Unchanged) ---
def _parse_llm_response(response_content: Optional[str]) -> Optional[int]:
    if not isinstance(response_content, str): return None
    cleaned_response = response_content.strip()
    return LABEL_MAP_FROM_TEXT.get(cleaned_response, None)

def _build_zero_shot_prompt(template: str, review_text: Any) -> str:
    review_text_str = str(review_text) if review_text is not None else ""
    return template.format(review_text=review_text_str)

def _build_few_shot_prompt(template: str, examples_str: str, review_text: Any) -> str:
    review_text_str = str(review_text) if review_text is not None else ""
    return template.format(examples=examples_str, review_text=review_text_str)

def _format_examples(examples_df: pd.DataFrame, format_template: str) -> str:
    example_strings = []
    label_map = LABEL_MAP_TO_TEXT
    for _, row in examples_df.iterrows():
        try:
            label_val = row.get('label')
            if isinstance(label_val, np.generic): label_val = label_val.item()
            label_text = label_map.get(label_val, '?')
            review_text = str(row.get('full_text', ''))
            example_strings.append(
                format_template.format(review_text=review_text, label_text=label_text).strip()
            )
        except KeyError: continue
        except Exception: continue
    return "\n\n".join(example_strings)

def select_few_shot_examples(
    train_df: pd.DataFrame, num_examples: int, strategy: str, seed: int
) -> Optional[pd.DataFrame]:
    required_cols = {'label', 'full_text'}
    if strategy == 'extreme_helpful_vote': required_cols.add('helpful_vote')
    if not required_cols.issubset(train_df.columns) or train_df.empty or num_examples <= 0: return None
    examples_df = pd.DataFrame()
    n_each = num_examples // 2
    n_h_target = n_each + (num_examples % 2); n_u_target = n_each
    train_df_copy = train_df.copy()
    if strategy == 'random':
        actual_n = min(num_examples, len(train_df_copy))
        if actual_n > 0: examples_df = train_df_copy.sample(n=actual_n, random_state=seed)
    elif strategy in ['balanced_random', 'extreme_helpful_vote']:
        h_pool = train_df_copy[train_df_copy['label'] == 1].copy()
        u_pool = train_df_copy[train_df_copy['label'] == 0].copy()
        s_h = pd.DataFrame(); s_u = pd.DataFrame()
        actual_n_h = min(n_h_target, len(h_pool)); actual_n_u = min(n_u_target, len(u_pool))
        if strategy == 'balanced_random':
            if actual_n_h > 0: s_h = h_pool.sample(n=actual_n_h, random_state=seed)
            if actual_n_u > 0: s_u = u_pool.sample(n=actual_n_u, random_state=seed + 1)
        elif strategy == 'extreme_helpful_vote':
            h_pool['helpful_vote'] = pd.to_numeric(h_pool['helpful_vote'], errors='coerce').fillna(0)
            u_pool['helpful_vote'] = pd.to_numeric(u_pool['helpful_vote'], errors='coerce').fillna(0)
            if actual_n_h > 0: s_h = h_pool.nlargest(actual_n_h, 'helpful_vote', keep='first')
            if actual_n_u > 0: s_u = u_pool.nsmallest(actual_n_u, 'helpful_vote', keep='first')
        to_concat = [df for df in [s_h, s_u] if not df.empty]
        if to_concat: examples_df = pd.concat(to_concat, ignore_index=True)
    else: return None
    if examples_df.empty: return None
    if strategy != 'random': examples_df = examples_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    return examples_df

def get_llm_predictions(
    client_wrapper: LLMWrapper,
    texts_to_classify: List[str],
    mode: str,
    prompt_template: str,
    request_timeout: int,
    max_retries: int,
    retry_delay: int,
    few_shot_examples_df: Optional[pd.DataFrame] = None,
    few_shot_example_format: Optional[str] = None
) -> Tuple[List[Optional[int]], int]:
    predictions: List[Optional[int]] = []
    failed_count = 0
    examples_str = ""
    if mode == 'few_shot':
        if few_shot_examples_df is None or few_shot_examples_df.empty or not few_shot_example_format:
            return [None] * len(texts_to_classify), len(texts_to_classify)
        examples_str = _format_examples(few_shot_examples_df, few_shot_example_format)
        if not examples_str: return [None] * len(texts_to_classify), len(texts_to_classify)

    model_id_for_logging = client_wrapper.model_id
    for text in tqdm(texts_to_classify, desc=f"LLM {model_id_for_logging} {mode}", unit=" text"):
        try:
            if mode == 'zero_shot': prompt = _build_zero_shot_prompt(prompt_template, text)
            elif mode == 'few_shot': prompt = _build_few_shot_prompt(template=prompt_template, examples_str=examples_str, review_text=text) # Ensure kwargs match
            else: predictions.append(None); failed_count += 1; continue
        except Exception: predictions.append(None); failed_count += 1; continue
        prediction: Optional[int] = None
        raw_response_content: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                raw_response_content = client_wrapper.generate(prompt, request_timeout)
                if raw_response_content is not None:
                    prediction = _parse_llm_response(raw_response_content)
                    if prediction is not None: break
                if attempt < max_retries: time.sleep(retry_delay * (2 ** attempt))
            except AuthenticationError: break
            except Exception:
                if attempt < max_retries: time.sleep(retry_delay * (2 ** attempt)); continue
                else: break
        if prediction is None: failed_count += 1
        predictions.append(prediction)
    return predictions, failed_count