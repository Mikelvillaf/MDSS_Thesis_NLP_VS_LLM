# scripts/llm_prediction.py
import os
import time
import random
from typing import List, Optional, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

# For Google AI Client SDK
from google import genai
try:
    # Explicitly import 'types' from 'google.genai' based on YOUR documentation snippet
    from google.genai import types as GoogleGenAITypes
except ImportError:
    GoogleGenAITypes = None # Will be checked before use
    print("Warning: Could not import 'types' from 'google.genai'. "
        "Advanced generation configuration for Google models (max_output_tokens, temperature) via GenerateContentConfig will not be applied.")

# For OpenAI
try:
    from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError, APITimeoutError
except ImportError:
    OpenAI = None # type: ignore
    APIError = RateLimitError = AuthenticationError = APIConnectionError = APITimeoutError = Exception # type: ignore

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
    def generate(self, prompt: str, request_timeout: int, max_output_tokens: Optional[int] = None) -> Optional[str]:
        pass

# --- OpenAI Wrapper ---
class OpenAIWrapper(LLMWrapper):
    def _setup_client(self, **kwargs):
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed.")
        self.client = OpenAI(api_key=self.api_key)
        print(f"   OpenAI client initialized for model {self.model_id}.")

    def generate(self, prompt: str, request_timeout: int, max_output_tokens: Optional[int] = None) -> Optional[str]:
        try:
            completion_params: Dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "timeout": request_timeout
            }
            if max_output_tokens is not None:
                completion_params["max_tokens"] = max_output_tokens
            else:
                completion_params["max_tokens"] = 10 # Fallback
            
            response = self.client.chat.completions.create(**completion_params)
            return response.choices[0].message.content
        except AuthenticationError as e:
            print(f"\n❌ OpenAI Auth Error for {self.model_id}: {e}")
            raise
        except Exception as e:
            print(f"\n❌ OpenAI API Error for {self.model_id} (attempt): {type(e).__name__} - {e}")
            return None

# --- Google AI Client Wrapper ---
class GoogleWrapper(LLMWrapper):
    def _setup_client(self, **kwargs):
        if 'genai' not in globals() or genai is None or not hasattr(genai, "Client"):
            raise ImportError("The 'genai' module (for genai.Client) does not seem to be correctly loaded.")
        if GoogleGenAITypes is None:
            print("   Warning: 'google.genai.types' could not be imported. Max output tokens and temperature for Google models will use defaults.")
        elif not hasattr(GoogleGenAITypes, 'GenerateContentConfig'):
            print("   Warning: 'google.genai.types.GenerateContentConfig' not found. Max output tokens and temperature for Google models will use defaults.")


        self.client = genai.Client(api_key=self.api_key)
        print(f"   Google AI Client initialized for model {self.model_id}.")

    def generate(self, prompt: str, request_timeout: int, max_output_tokens: Optional[int] = None) -> Optional[str]:
        # request_timeout is not directly used by this specific client.models.generate_content()
        if not self.client:
            return None
        try:
            api_model_id = self.model_id
            if not self.model_id.startswith("models/"):
                api_model_id = f"models/{self.model_id}"
            
            config_for_api = None # Initialize to None
            if GoogleGenAITypes is not None and hasattr(GoogleGenAITypes, 'GenerateContentConfig'):
                # Only create config object if 'types' and 'GenerateContentConfig' are available
                config_params: Dict[str, Any] = {
                    "temperature": 0.0, # Default temperature
                }
                if max_output_tokens is not None:
                    config_params["max_output_tokens"] = max_output_tokens
                
                # Instantiate types.GenerateContentConfig if we have parameters for it
                if "max_output_tokens" in config_params or "temperature" in config_params:
                    try:
                        config_for_api = GoogleGenAITypes.GenerateContentConfig(**config_params)
                    except Exception as e_cfg:
                        print(f"  Warning: Failed to instantiate google.genai.types.GenerateContentConfig with params {config_params}: {e_cfg}. API call will use defaults for config.")
                        config_for_api = None # Ensure it's None if instantiation failed
            
            # Use 'config=' parameter as per your documentation snippet
            response = self.client.models.generate_content(
                model=api_model_id,
                contents=prompt,
                config=config_for_api # Pass the GenerateContentConfig object (or None)
            )

            if hasattr(response, 'text') and response.text is not None:
                return response.text
            if hasattr(response, 'parts') and response.parts:
                text_parts = [part.text for part in response.parts if hasattr(part, 'text') and part.text]
                if text_parts:
                    return "".join(text_parts)
            print(f"   Warning: Response from Google AI for model {self.model_id} had no '.text' or parsable '.parts'. Response: {response}")
            return None
        except Exception as e:
            print(f"\n❌ Google AI API Error for {self.model_id} during generate: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
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
    llm_max_output_tokens: Optional[int] = None, 
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
        if not examples_str:
            return [None] * len(texts_to_classify), len(texts_to_classify)

    model_id_for_logging = client_wrapper.model_id
    for text_idx, text in enumerate(tqdm(texts_to_classify, desc=f"LLM {model_id_for_logging} {mode}", unit=" text")):
        current_prompt: Optional[str] = None
        try:
            if mode == 'zero_shot': current_prompt = _build_zero_shot_prompt(prompt_template, text)
            elif mode == 'few_shot': current_prompt = _build_few_shot_prompt(template=prompt_template, examples_str=examples_str, review_text=text)
            else:
                predictions.append(None); failed_count += 1; continue
            if not current_prompt:
                predictions.append(None); failed_count += 1; continue
        except Exception:
            predictions.append(None); failed_count += 1; continue
        
        prediction: Optional[int] = None
        raw_response_content: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                raw_response_content = client_wrapper.generate(current_prompt, request_timeout, max_output_tokens=llm_max_output_tokens)
                
                if raw_response_content is not None:
                    prediction = _parse_llm_response(raw_response_content)
                    if prediction is not None:
                        break 
                
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                
            except AuthenticationError:
                print(f"\n❌ Authentication Error for {model_id_for_logging}. Stopping retries for this item.")
                break 
            except Exception as e:
                print(f"\n❌ Error during generate/parse attempt {attempt + 1} for {model_id_for_logging}: {type(e).__name__} - {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    print(f"\n❌ Max retries reached for item after general error. Error: {e}")
                    break 
        
        if prediction is None:
            failed_count += 1
        predictions.append(prediction)
        
    return predictions, failed_count