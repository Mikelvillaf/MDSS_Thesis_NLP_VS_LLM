# main.py
import os
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import random
import time
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Union, Any
from scripts.utils import load_config, load_hyperparameters
from scripts.data_loader import load_reviews, identify_label_candidates, create_balanced_temporal_splits, load_and_clean_metadata
from scripts.feature_engineering import fit_feature_extractor, transform_features, save_featurizer
from scripts.evaluation import (
    analyze_price_quantiles,
    generate_accuracy_table,
    summarize_evaluations,
    plot_all_average_cms,
    plot_f1_comparison_chart # Added new import
)
from scripts import preprocessing, model_training, evaluation, llm_prediction

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from google import genai
except ImportError:
    genai = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if tf:
        tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

MODEL_DISPATCH = {
    "random_forest": model_training.train_random_forest,
    "svm": model_training.train_svm,
    "gradient_boosting": model_training.train_gbm,
    "cnn": model_training.train_cnn,
    "rcnn": model_training.train_rcnn
}

# Modified signature to accept all_hyperparameters
def run_pipeline_for_seed(config: dict, seed: int, main_run_dir: str, all_hyperparameters: dict) -> List[dict]:
    set_seed(seed)
    all_results_this_seed = []
    seed_output_dir = os.path.join(main_run_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    print(f"--- Seed: {seed} | Output Dir: {seed_output_dir} ---")

    categories_to_run = config.get("categories", [])
    data_path_template = config.get("data_path_template", "data/{category}.jsonl")
    year_range = config.get("year_range")
    metadata_path_template = config.get("metadata_path_template", "data/meta_{category}.jsonl")

    # Get max_initial_rows_per_category from config for data loading
    max_initial_rows_per_category = config.get("max_initial_rows_per_category", None)

    models_config = config.get("models_to_run", {})
    ml_models_to_run = models_config.get("ml", [])
    dl_models_to_run = models_config.get("dl", [])
    # LLM models will be read from provider-specific keys like "llm_openai", "llm_google"

    ml_feature_set = config.get("feature_set", "hybrid")
    text_max_features = config.get("text_max_features", 1000)

    dl_feature_set = config.get("dl_feature_set", "hybrid")
    dl_num_structured = config.get("dl_num_structured_features", 5)
    # Base DL hyperparameters from experiment_config.yaml, can be overridden by tuned ones
    base_dl_hypers = {
        "max_words": config.get("dl_max_words", 10000),
        "max_len": config.get("dl_max_len", 300),
        "embedding_dim": config.get("dl_embedding_dim", 64),
        "epochs": config.get("dl_epochs", 5),
        "batch_size": config.get("dl_batch_size", 64),
        "conv1d_filters": config.get("dl_conv1d_filters", 64), # Default if not in tuned
        "lstm_units": config.get("dl_lstm_units", 64),       # Default if not in tuned
        "dense_units": config.get("dl_dense_units", 32),      # Default if not in tuned
        "dropout_cat": config.get("dl_dropout_cat", 0.5),     # Default if not in tuned
        "learning_rate_pow": config.get("dl_learning_rate_pow", -3) # Default if not in tuned
    }

    llm_overall_config = config.get("llm_evaluation", {}) # Contains API keys and general LLM params
    # Specific API key env var names will be pulled from llm_overall_config later
    llm_test_sample_size = llm_overall_config.get("test_sample_size")
    prompting_modes = llm_overall_config.get("prompting_modes", ['zero_shot'])
    # Generic LLM parameters from llm_overall_config, passed to get_llm_predictions
    request_timeout = llm_overall_config.get("request_timeout", 30)
    max_retries = llm_overall_config.get("max_retries", 3)
    retry_delay = llm_overall_config.get("retry_delay", 5)


    if not categories_to_run:
        print("⚠️ No categories specified in config. Exiting seed run.")
        return []

    for category in categories_to_run:
        print(f"\n📦 Processing Category: {category} | Seed: {seed}")
        if max_initial_rows_per_category is not None:
            print(f"   Configured to load at most {max_initial_rows_per_category} initial records for this category.")
        category_start_time = time.time()
        path = data_path_template.format(category=category)
        actual_metadata_path = metadata_path_template.format(category=category)

        # Initialize train_df here to ensure it's available for LLM few-shot example selection
        # It will be populated in the data loading/splitting block
        train_df = pd.DataFrame()
        test_df = pd.DataFrame() # Also initialize test_df for LLM eval set creation
        val_df = pd.DataFrame()  # And val_df for completeness, though not directly used by LLM block

        try:
            print("\n--- 1. Data Loading ---")
            df_raw = load_reviews(
                filepath=path,
                year_range=year_range,
                max_initial_load=max_initial_rows_per_category, # Pass the limit
                seed=seed
            )
            if df_raw.empty: print(f"⚠️ No data loaded for {category}. Skipping."); continue

            meta_df_loaded = load_and_clean_metadata(actual_metadata_path)
            if meta_df_loaded.empty:
                print(f"⚠️ No valid meta data loaded for {category}. Price-related features/analysis might be affected.")

            print("\n--- 2. Preprocessing ---")
                        # Get labeling_mode from config to pass to preprocessing
            labeling_config_main = config.get("labeling", {}) # Use a distinct name
            labeling_mode_main = labeling_config_main.get("mode")
            df_processed = preprocessing.preprocess_reviews(
                df_raw,
                metadata_df=meta_df_loaded,
                labeling_mode=labeling_mode_main # Pass the labeling mode
            )
            if not df_raw.empty: del df_raw # Memory management
            if df_processed.empty: print(f"⚠️ Data empty after preprocessing. Skipping."); continue

            print("\n--- 3. Label Generation ---")
            df_labeled_pool = identify_label_candidates(df_processed, config)
            if not df_processed.empty: del df_processed # Memory management
            if df_labeled_pool.empty: print(f"⚠️ No labeled candidates found. Skipping."); continue

            print("\n--- 4. Temporal Splitting & Sampling ---")
            # Assign to the train_df, val_df, test_df initialized above
            train_df, val_df, test_df = create_balanced_temporal_splits(df_labeled_pool, config, seed)
            if not df_labeled_pool.empty: del df_labeled_pool # Memory management

            if train_df.empty or test_df.empty:
                print(f"⚠️ Train or Test split empty. Skipping category."); continue
            if train_df['label'].nunique() < 2 or test_df['label'].nunique() < 2:
                print(f"⚠️ Train or Test split lacks samples from both classes. Skipping category."); continue
            print(f"   Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"❌ Error during data preparation for {category}: {e}")
            traceback.print_exc()
            continue

        featurizer = None
        X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml = [None] * 6
        X_train_structured_dl, X_val_structured_dl, X_test_structured_dl = [None] * 3
        needs_ml_features = bool(ml_models_to_run)
        needs_structured_for_dl = bool(dl_models_to_run) and dl_feature_set == 'hybrid'

        if needs_ml_features or needs_structured_for_dl:
            try:
                print(f"\n--- 5. Feature Engineering (ML features: '{ml_feature_set}') ---")
                featurizer, _ = fit_feature_extractor(
                    train_df, feature_set=ml_feature_set, text_max_features=text_max_features
                )
                featurizer_path = os.path.join(seed_output_dir, f"{category}_featurizer.joblib")
                save_featurizer(featurizer, featurizer_path)

                print("   Transforming data splits...")
                X_train_ml, y_train_ml = transform_features(train_df, featurizer)
                X_test_ml, y_test_ml = transform_features(test_df, featurizer)
                if not val_df.empty:
                    X_val_ml, y_val_ml = transform_features(val_df, featurizer)
                else:
                    feat_dim = X_train_ml.shape[1] if X_train_ml is not None and X_train_ml.ndim == 2 else 0
                    X_val_ml, y_val_ml = np.empty((0, feat_dim)), np.empty((0,))
                print(f"   ML Feature Shapes: Train={X_train_ml.shape}, Val={X_val_ml.shape}, Test={X_test_ml.shape}")

                if needs_structured_for_dl:
                    if ml_feature_set == 'nlp':
                        print(f"   ⚠️ Warning: ML feature_set is '{ml_feature_set}'. Cannot extract structured features for Hybrid DL.")
                    elif X_train_ml is None or X_train_ml.shape[1] < dl_num_structured:
                        print(f"   ⚠️ Warning: Not enough ML features ({X_train_ml.shape[1] if X_train_ml is not None else 0}) to extract {dl_num_structured} for Hybrid DL.")
                    else:
                        print(f"   Extracting first {dl_num_structured} columns for Hybrid DL...")
                        X_train_structured_dl = X_train_ml[:, :dl_num_structured]
                        X_test_structured_dl = X_test_ml[:, :dl_num_structured]
                        if X_val_ml.shape[0] > 0:
                            X_val_structured_dl = X_val_ml[:, :dl_num_structured]
                        else:
                            X_val_structured_dl = np.empty((0, dl_num_structured))
                        print(f"      DL Structured Shapes: Train={X_train_structured_dl.shape}, Val={X_val_structured_dl.shape}, Test={X_test_structured_dl.shape}")

                if X_train_ml is None or X_test_ml is None or y_train_ml is None or y_test_ml is None or X_train_ml.shape[0] == 0 or X_test_ml.shape[0] == 0:
                    print("⚠️ Feature Engineering yielded empty/invalid features for ML. Skipping ML models.")
                    ml_models_to_run = []

            except Exception as e:
                print(f"❌ Error during Feature Engineering for {category}: {e}")
                traceback.print_exc()
                ml_models_to_run = []

        print(f"\n--- 6. Model Training & Evaluation ---")

        if ml_models_to_run:
            ml_class_weight_default = 'balanced' if not config.get('balanced_sampling', {}).get('use_strict_balancing', False) else None
            for model_name in ml_models_to_run:
                print(f"\n🚀 Training ML Model: {model_name}...")
                if model_name not in MODEL_DISPATCH:
                    print(f"   ⚠️ Model '{model_name}' not found. Skipping."); continue
                try:
                    train_fn = MODEL_DISPATCH[model_name]
                    model_specific_hyperparams = all_hyperparameters.get(model_name, {})
                    train_args = {
                        'seed': seed,
                        'class_weight': ml_class_weight_default
                    }
                    train_args.update(model_specific_hyperparams)
                    print(f"   Using FINAL hyperparameters for {model_name}: {train_args}")
                    model = train_fn(X_train_ml, y_train_ml, **train_args)

                    print("   Evaluating ML model on test set...")
                    y_pred = model.predict(X_test_ml)
                    y_proba = None
                    if hasattr(model, "predict_proba"): y_proba = model.predict_proba(X_test_ml)[:, 1]
                    elif hasattr(model, "decision_function"): y_proba = model.decision_function(X_test_ml)

                    # Construct the model identifier part for the name
                    # For ML models, model_name is already the pure identifier (e.g., "random_forest")
                    model_identifier_for_filename = model_name
                    eval_model_name = f"{category}_{model_identifier_for_filename}_seed{seed}"

                    eval_results = evaluation.evaluate_model(
                        y_true=y_test_ml, y_pred=y_pred, y_proba=y_proba,
                        model_name=eval_model_name, output_dir=seed_output_dir,
                        # Pass category and model_identifier for easier parsing later
                        category=category,
                        model_identifier=model_identifier_for_filename
                    )
                    all_results_this_seed.append(eval_results)

                    try:
                        if test_df is not None and not test_df.empty and \
                        'price' in test_df.columns and len(y_test_ml) == len(test_df):
                            preds_save_df = pd.DataFrame({
                                'y_true': y_test_ml,
                                'y_pred': y_pred,
                                'price': test_df['price'].values
                            })
                            if y_proba is not None:
                                preds_save_df['y_proba'] = y_proba
                            # Save predictions with a name that includes category and model_identifier
                            preds_filename = f"{category}_{model_identifier_for_filename}_seed{seed}_predictions.csv"
                            preds_save_path = os.path.join(seed_output_dir, preds_filename)
                            preds_save_df.to_csv(preds_save_path, index=False)
                        else:
                            print(f"   ⚠️ Cannot save predictions with price for {eval_model_name}: test_df missing, 'price' column missing, or length mismatch.")
                    except Exception as save_err:
                        print(f"   ⚠️ Error saving predictions for {eval_model_name}: {save_err}")
                except Exception as e:
                    print(f"❌ Error during ML model {model_name}: {e}")
                    traceback.print_exc()
        else:
            print("   No ML models specified or FE yielded no features.")

        if dl_models_to_run:
            print(f"\n🧠 Preparing data for DL models (Configured mode: {dl_feature_set})...")
            try:
                texts_train_dl = train_df["full_text"].tolist()
                labels_train_dl = train_df["label"].astype(int).values
                texts_test_dl = test_df["full_text"].tolist()
                labels_test_dl = test_df["label"].astype(int).values

                temp_tokenizer_for_val_prep = None
                if tf:
                    from tensorflow.keras.preprocessing.text import Tokenizer
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    # Use base_dl_hypers for this temporary tokenizer's parameters
                    temp_tokenizer_for_val_prep = Tokenizer(num_words=base_dl_hypers["max_words"], oov_token="<OOV>")
                    temp_tokenizer_for_val_prep.fit_on_texts(texts_train_dl)

                dl_validation_data = None
                actual_dl_hybrid_mode = dl_feature_set == 'hybrid' and \
                                        X_train_structured_dl is not None and \
                                        X_train_structured_dl.shape[0] == len(labels_train_dl) and \
                                        X_train_structured_dl.shape[1] == dl_num_structured

                if not val_df.empty and temp_tokenizer_for_val_prep:
                    texts_val_dl_prep = val_df["full_text"].tolist()
                    labels_val_dl_prep = val_df["label"].astype(int).values
                    padded_val_dl = pad_sequences(temp_tokenizer_for_val_prep.texts_to_sequences(texts_val_dl_prep), maxlen=base_dl_hypers["max_len"])
                    keras_val_inputs_dl = [padded_val_dl]
                    if actual_dl_hybrid_mode:
                        if X_val_structured_dl is not None and X_val_structured_dl.shape[0] == len(labels_val_dl_prep):
                            keras_val_inputs_dl.append(X_val_structured_dl)
                        else:
                            print("   ⚠️ Hybrid DL: Validation structured features mismatch or unavailable. Using text-only validation for Keras.")
                            keras_val_inputs_dl = padded_val_dl # Revert to text-only for val if mismatch
                    # else: # No change, keras_val_inputs_dl is already just padded_val_dl
                    dl_validation_data = (keras_val_inputs_dl, labels_val_dl_prep)
                    print(f"   Prepared Keras validation data (Inputs type: {type(keras_val_inputs_dl)}).")


                dl_class_weight_val = None
                if not config.get('balanced_sampling',{}).get('use_strict_balancing', False):
                    from sklearn.utils.class_weight import compute_class_weight
                    unique_labels_train_dl_calc = np.unique(labels_train_dl)
                    if len(unique_labels_train_dl_calc) > 1:
                        weights_dl = compute_class_weight('balanced', classes=unique_labels_train_dl_calc, y=labels_train_dl)
                        dl_class_weight_val = dict(enumerate(weights_dl))

                # ... (inside run_pipeline_for_seed method, within the DL models block) ...

                for model_name in dl_models_to_run: # model_name here is "cnn", "rcnn"
                    # Define final_dl_mode_str based on actual_dl_hybrid_mode for clarity in print output
                    final_dl_mode_str = "Hybrid" if actual_dl_hybrid_mode else "Text-Only"

                    # Construct model_identifier for filenames, hyperparameter lookup, and logging
                    if actual_dl_hybrid_mode:
                        # For hybrid DL models, append "_Hybrid" to the base model name (e.g., "cnn_Hybrid")
                        model_identifier_for_filename = f"{model_name}_Hybrid"
                    else:
                        # For text-only DL models, use the base model name (e.g., "rcnn")
                        model_identifier_for_filename = model_name

                    print(f"\n🧠 Training DL Model: {model_identifier_for_filename} (Configured feature set: {dl_feature_set}, Effective mode for this run: {final_dl_mode_str})...")

                    if model_name not in MODEL_DISPATCH: # Check base model name in dispatch
                        print(f"   ⚠️ Model '{model_name}' not found in MODEL_DISPATCH. Skipping."); continue
                    try:
                        train_fn = MODEL_DISPATCH[model_name] # Use base name for dispatch

                        # Hyperparameters: try model_identifier_for_filename first (e.g. cnn_Hybrid),
                        # then base model_name as fallback (e.g. cnn)
                        dl_model_specific_hyperparams = all_hyperparameters.get(model_identifier_for_filename, {})
                        if not dl_model_specific_hyperparams and model_identifier_for_filename != model_name:
                            # Fallback for cases like "cnn_Hybrid" not found, try "cnn"
                            dl_model_specific_hyperparams = all_hyperparameters.get(model_name, {})


                        current_dl_model_params = base_dl_hypers.copy()
                        current_dl_model_params.update(dl_model_specific_hyperparams)
                        print(f"   Final effective DL parameters for {model_identifier_for_filename}: {current_dl_model_params}")

                        model, tokenizer = train_fn(
                            texts_train=texts_train_dl,
                            labels_train=labels_train_dl,
                            structured_train=X_train_structured_dl if actual_dl_hybrid_mode else None,
                            validation_data=dl_validation_data,
                            num_structured_features=dl_num_structured if actual_dl_hybrid_mode else 0,
                            **current_dl_model_params,
                            class_weight=dl_class_weight_val,
                            seed=seed
                        )

                        print("   Evaluating DL model on test set...")
                        padded_test_dl = pad_sequences(tokenizer.texts_to_sequences(texts_test_dl), maxlen=current_dl_model_params["max_len"])
                        predict_inputs_dl = [padded_test_dl]
                        if actual_dl_hybrid_mode:
                            if X_test_structured_dl is not None and X_test_structured_dl.shape[0] == len(labels_test_dl):
                                predict_inputs_dl.append(X_test_structured_dl)
                            else:
                                print("   ⚠️ Cannot perform hybrid prediction: Test structured data mismatch. Skipping DL eval.")
                                continue

                        probas_test_dl = model.predict(predict_inputs_dl)
                        preds_test_dl = (probas_test_dl > 0.5).astype("int32").flatten()

                        eval_model_name_dl = f"{category}_{model_identifier_for_filename}_seed{seed}"
                        eval_results_dl = evaluation.evaluate_model(
                            y_true=labels_test_dl, y_pred=preds_test_dl, y_proba=probas_test_dl.flatten(),
                            model_name=eval_model_name_dl, output_dir=seed_output_dir,
                            category=category,
                            model_identifier=model_identifier_for_filename # Correct identifier
                        )
                        all_results_this_seed.append(eval_results_dl)

                        try:
                            if test_df is not None and not test_df.empty and \
                            'price' in test_df.columns and len(labels_test_dl) == len(test_df):
                                preds_save_df_dl = pd.DataFrame({
                                    'y_true': labels_test_dl,
                                    'y_pred': preds_test_dl,
                                    'price': test_df['price'].values,
                                    'y_proba': probas_test_dl.flatten()
                                })
                                # Use the consistent model_identifier_for_filename here
                                preds_filename_dl = f"{category}_{model_identifier_for_filename}_seed{seed}_predictions.csv"
                                preds_save_path_dl = os.path.join(seed_output_dir, preds_filename_dl)
                                preds_save_df_dl.to_csv(preds_save_path_dl, index=False)
                            # ... (rest of error handling for saving) ...
                        except Exception as save_err:
                            print(f"   ⚠️ Error saving DL predictions for {eval_model_name_dl}: {save_err}")
                    # ... (rest of error handling for model training) ...
                    except Exception as e:
                        print(f"❌ Error during DL model {model_identifier_for_filename}: {e}")
                        traceback.print_exc()
            # ... (rest of run_pipeline_for_seed) ...
            except Exception as e: # This except corresponds to the try around DL data preparation
                print(f"❌ Error during DL data preparation or outer loop for {category}: {e}")
                traceback.print_exc()
        else:
            print("\n   No DL models specified.")

        llm_providers_configured = any(
            key.startswith("llm_") and models_config.get(key) for key in models_config
        )

        if llm_providers_configured:
            print(f"\n🤖 Evaluating LLM Providers...")

            eval_test_df_llm = test_df
            if llm_test_sample_size and isinstance(llm_test_sample_size, int) and 0 < llm_test_sample_size < len(test_df):
                print(f"   Sampling {llm_test_sample_size} reviews for LLM test (from test_df)...")
                try:
                    n_each_llm = llm_test_sample_size // 2; n_helpful_target = n_each_llm + (llm_test_sample_size % 2); n_unhelpful_target = n_each_llm
                    h_pool_llm = test_df[test_df['label'] == 1]; u_pool_llm = test_df[test_df['label'] == 0]
                    h_llm = h_pool_llm.sample(n=min(n_helpful_target, len(h_pool_llm)), random_state=seed)
                    u_llm = u_pool_llm.sample(n=min(n_unhelpful_target, len(u_pool_llm)), random_state=seed + 1)
                    eval_test_df_llm = pd.concat([h_llm, u_llm]).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
                    print(f"   LLM test sample size: {len(eval_test_df_llm)} (Helpful: {len(h_llm)}, Unhelpful: {len(u_llm)})")
                except ValueError as sample_err:
                    print(f"   LLM sampling failed ({sample_err}). Using full test set of {len(test_df)} rows for LLM.")
                    eval_test_df_llm = test_df
            else:
                print(f"   Using full test set of {len(test_df)} rows for LLM.")

            if eval_test_df_llm.empty:
                print("   ⚠️ Test set for LLM evaluation is empty. Skipping LLMs for this category.")
            else:
                test_texts_llm_val = eval_test_df_llm['full_text'].tolist()
                test_labels_llm_val_true = eval_test_df_llm['label'].values

                provider_map = {}
                if hasattr(llm_prediction, "OpenAIWrapper"):
                    provider_map["llm_openai"] = {
                        "wrapper": llm_prediction.OpenAIWrapper,
                        "api_key_env_var": llm_overall_config.get("openai_api_key_env_var", "OPENAI_API_KEY")
                    }
                if hasattr(llm_prediction, "GoogleWrapper") and genai is not None:
                    provider_map["llm_google"] = {
                        "wrapper": llm_prediction.GoogleWrapper,
                        "api_key_env_var": llm_overall_config.get("google_api_key_env_var", "GOOGLE_API_KEY")
                    }

                for provider_key_from_config, model_ids_for_provider in models_config.items():
                    if not provider_key_from_config.startswith("llm_") or not model_ids_for_provider:
                        continue

                    if provider_key_from_config not in provider_map:
                        print(f"⚠️ LLM: Provider key '{provider_key_from_config}' found in config but no corresponding wrapper/setup in main.py provider_map. Skipping.")
                        continue

                    provider_details = provider_map[provider_key_from_config]
                    api_key_env = provider_details["api_key_env_var"]
                    WrapperClass = provider_details["wrapper"]

                    if not os.getenv(api_key_env):
                        print(f"⚠️ LLM: Environment variable '{api_key_env}' for {provider_key_from_config} not set. Skipping all {provider_key_from_config} models.")
                        continue

                    print(f"\n🤖 Processing LLM Provider: {provider_key_from_config.replace('llm_', '').capitalize()}")

                    for model_id_llm in model_ids_for_provider: # e.g., "gpt-3.5-turbo"
                        print(f"\n   --- LLM Model: {model_id_llm} ---")
                        client_wrapper_instance = None
                        try:
                            client_wrapper_instance = WrapperClass(api_key_env_var=api_key_env, model_id=model_id_llm)
                        except ImportError as e:
                            print(f"❌ LLM Import Error for {model_id_llm} ({provider_key_from_config}): {e}. Ensure required libraries are installed. Skipping model.")
                            continue
                        except ValueError as e:
                            print(f"❌ Error setting up client wrapper for {model_id_llm}: {e}. Skipping this model.")
                            continue
                        except Exception as e:
                            print(f"❌ Unexpected error setting up client wrapper for {model_id_llm}: {type(e).__name__} - {e}. Skipping model.")
                            traceback.print_exc()
                            continue

                        few_shot_examples_df_val = None
                        if 'few_shot' in prompting_modes and not train_df.empty:
                            fs_config = llm_overall_config.get('few_shot', {})
                            num_examples_fs = fs_config.get('num_examples')
                            strategy_fs = fs_config.get('example_selection_strategy')
                            if num_examples_fs and strategy_fs:
                                few_shot_examples_df_val = llm_prediction.select_few_shot_examples(
                                    train_df=train_df, num_examples=num_examples_fs, strategy=strategy_fs, seed=seed
                                )

                        for mode in prompting_modes: # e.g., "zero_shot", "few_shot"
                            # Construct a model identifier that includes provider, model_id, and mode
                            # e.g., llm_openai_gpt-3.5-turbo_zero_shot
                            model_identifier_for_filename = f"{provider_key_from_config}_{model_id_llm}_{mode}"


                            print(f"      Mode: {mode}")
                            current_prompt_template_val = None; current_example_format_val = None
                            current_few_shot_examples_for_mode = None
                            try:
                                if mode == 'zero_shot':
                                    current_prompt_template_val = llm_overall_config.get('zero_shot_prompt_template')
                                elif mode == 'few_shot':
                                    fs_config_mode = llm_overall_config.get('few_shot', {})
                                    current_example_format_val = fs_config_mode.get('example_format')
                                    current_prompt_template_val = fs_config_mode.get('prompt_template')
                                    current_few_shot_examples_for_mode = few_shot_examples_df_val
                                    if not all([current_example_format_val, current_prompt_template_val]):
                                        raise ValueError(f"Few-shot config incomplete for mode '{mode}'.")
                                    if current_few_shot_examples_for_mode is None or current_few_shot_examples_for_mode.empty:
                                        print("      ⚠️ Valid few-shot examples unavailable. Skipping few-shot mode.")
                                        continue
                                else: print(f"      ⚠️ Unknown prompting mode: {mode}. Skipping."); continue
                                if not current_prompt_template_val: print(f"      ⚠️ Prompt template not found for '{mode}'. Skipping."); continue

                                predictions_llm_raw_val, failed_count_val = llm_prediction.get_llm_predictions(
                                    client_wrapper=client_wrapper_instance,
                                    texts_to_classify=test_texts_llm_val,
                                    mode=mode, prompt_template=current_prompt_template_val,
                                    request_timeout=request_timeout, max_retries=max_retries, retry_delay=retry_delay,
                                    few_shot_examples_df=current_few_shot_examples_for_mode,
                                    few_shot_example_format=current_example_format_val
                                )
                                valid_indices_llm_val = [i for i, p_val in enumerate(predictions_llm_raw_val) if p_val is not None]
                                if not valid_indices_llm_val: print(f"      ⚠️ No valid predictions for {mode}."); continue
                                y_true_eval_llm_val = test_labels_llm_val_true[valid_indices_llm_val]
                                y_pred_eval_llm_val = [predictions_llm_raw_val[i] for i in valid_indices_llm_val]
                                if failed_count_val > 0 : print(f"      Evaluating on {len(valid_indices_llm_val)} valid predictions ({failed_count_val} failed).")

                                eval_model_name_llm_val = f"{category}_{model_identifier_for_filename}_seed{seed}"
                                eval_results_llm_val = evaluation.evaluate_model(
                                    y_true=y_true_eval_llm_val, y_pred=y_pred_eval_llm_val, y_proba=None,
                                    model_name=eval_model_name_llm_val, output_dir=seed_output_dir,
                                    category=category, # Pass category
                                    model_identifier=model_identifier_for_filename # Pass clean model id
                                )
                                all_results_this_seed.append(eval_results_llm_val)
                                try:
                                    if eval_test_df_llm is not None and not eval_test_df_llm.empty and \
                                    'price' in eval_test_df_llm.columns and 'label' in eval_test_df_llm.columns and len(valid_indices_llm_val) > 0:
                                        eval_df_subset_llm = eval_test_df_llm.iloc[valid_indices_llm_val].copy()
                                        eval_df_subset_llm['y_pred'] = y_pred_eval_llm_val
                                        preds_save_df_llm_val = eval_df_subset_llm[['label', 'y_pred', 'price']].rename(columns={'label': 'y_true'})
                                        preds_filename_llm_val = f"{category}_{model_identifier_for_filename}_seed{seed}_predictions.csv"
                                        preds_save_path_llm_val = os.path.join(seed_output_dir, preds_filename_llm_val)
                                        preds_save_df_llm_val.to_csv(preds_save_path_llm_val, index=False)
                                except Exception as save_err: print(f"   ⚠️ Error saving LLM predictions for {eval_model_name_llm_val}: {save_err}")
                            except Exception as e: print(f"❌ Error during LLM {mode} for {model_identifier_for_filename}: {e}"); traceback.print_exc()
        else:
            print("\n   No LLM models specified in config under provider keys (e.g., llm_openai, llm_google).")

        if 'train_df' in locals() and isinstance(train_df, pd.DataFrame): del train_df
        if 'val_df' in locals() and isinstance(val_df, pd.DataFrame): del val_df
        if 'test_df' in locals() and isinstance(test_df, pd.DataFrame): del test_df
        if 'X_train_ml' in locals() and X_train_ml is not None: del X_train_ml, y_train_ml
        if 'X_val_ml' in locals() and X_val_ml is not None: del X_val_ml, y_val_ml
        if 'X_test_ml' in locals() and X_test_ml is not None: del X_test_ml, y_test_ml
        if 'X_train_structured_dl' in locals() and X_train_structured_dl is not None: del X_train_structured_dl
        if 'X_val_structured_dl' in locals() and X_val_structured_dl is not None: del X_val_structured_dl
        if 'X_test_structured_dl' in locals() and X_test_structured_dl is not None: del X_test_structured_dl

        category_duration = time.time() - category_start_time
        print(f"✅ Finished Category: {category} | Duration: {category_duration:.2f}s")

    print(f"--- Finished Seed: {seed} ---")
    return all_results_this_seed

# main.py
# ... (all previous imports and function definitions like set_seed, MODEL_DISPATCH, run_pipeline_for_seed) ...

def main():
    start_time = datetime.now()
    run_timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    print(f"\n======== Starting Run: {run_timestamp} ========")

    loaded_hyperparameters = {}
    config = None # Initialize config to ensure it's in scope
    try:
        config = load_config()
        hyperparameters_file_path = config.get("hyperparameters_file", "configs/hyperparameters.yaml")
        loaded_hyperparameters = load_hyperparameters(hyperparameters_file_path)
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"❌ Critical Error: Failed to load configuration or hyperparameters - {e}")
        traceback.print_exc()
        return

    base_output_dir = config.get("output_dir", "results/")
    main_run_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
    try:
        os.makedirs(main_run_dir, exist_ok=True)
        print(f"Output directory: {main_run_dir}")
        config_save_path = os.path.join(main_run_dir, 'config_used.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"   Configuration saved to: {config_save_path}")
        hyperparams_save_path = os.path.join(main_run_dir, 'hyperparameters_used.yaml')
        with open(hyperparams_save_path, 'w') as f:
            yaml.dump(loaded_hyperparameters, f, default_flow_style=False, sort_keys=False)
        print(f"   Hyperparameters saved to: {hyperparams_save_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create output directory or save config/hyperparameters: {e}")

    seeds_to_run = config.get("random_seeds", [42])
    if not isinstance(seeds_to_run, list) or not seeds_to_run:
        print("⚠️ Invalid 'random_seeds' format in config. Using default seed [42].")
        seeds_to_run = [42]

    all_run_results_accumulator = []
    print(f"\n--- Running pipeline for seeds: {seeds_to_run} ---")
    for seed_val in seeds_to_run:
        try:
            current_seed_val = int(seed_val)
            seed_results_list = run_pipeline_for_seed(config, current_seed_val, main_run_dir, loaded_hyperparameters)
            if seed_results_list:
                all_run_results_accumulator.extend(seed_results_list)
        except KeyboardInterrupt:
            print("\n🚫 Run interrupted by user (KeyboardInterrupt). Stopping...")
            break
        except Exception as e:
            print(f"\n❌❌❌ Unhandled Error during pipeline for seed {seed_val} ❌❌❌")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print(f"--- Attempting to continue with next seed if any ---")

    # --- Final Summary Generation ---
    print("\n--- Final Summary Generation ---")
    current_run_summary = {}  # Initialize current_run_summary
    if not all_run_results_accumulator:
        print("⚠️ No results were generated across any seeds/categories. Cannot summarize.")
        # current_run_summary remains empty if no results
    else:
        try:
            # Pass all_run_results_accumulator directly to summarize_evaluations
            current_run_summary = evaluation.summarize_evaluations(
                all_results_data=all_run_results_accumulator, # Pass the collected results
                result_dir=main_run_dir # Still pass result_dir for saving summary.json
            )

            print(f"\n📋 Final Summary (Run ID: {run_timestamp}):")
            if isinstance(current_run_summary, dict) and current_run_summary:
                if "error" in current_run_summary and len(current_run_summary) == 1:
                    print(f"  ❌ Summary generation failed: {current_run_summary['error']}")
                else:
                    # Iterate through model types in the summary
                    for model_type, summary_details in sorted(current_run_summary.items()):
                        print(f"  Model Type: {model_type}")
                        if isinstance(summary_details, dict):
                            overall_metrics = summary_details.get('metrics_across_all_categories_and_seeds', {})
                            print(f"    Overall (across categories & seeds):")
                            
                            acc_val = overall_metrics.get('avg_accuracy')
                            acc_display_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                            print(f"      Avg Accuracy: {acc_display_str}")

                            f1_val = overall_metrics.get('avg_f1')
                            f1_display_str = f"{f1_val:.4f}" if f1_val is not None else "N/A"
                            print(f"      Avg F1 Score: {f1_display_str}")
                            
                            roc_auc_val = overall_metrics.get('avg_roc_auc')
                            roc_auc_display_str = f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A"
                            print(f"      Avg ROC AUC:  {roc_auc_display_str}")
                            
                            print(f"      Total Runs (Cat x Seed combinations): {overall_metrics.get('total_runs', 'N/A')}")
                            if overall_metrics.get('_avg_cm_ndarray') is not None:
                                print(f"      Avg CM: {overall_metrics['_avg_cm_ndarray'].tolist()}")

                            category_specifics = summary_details.get('metrics_per_category', {})
                            if category_specifics:
                                print(f"    Metrics per category (averaged over seeds):")
                                for cat, cat_metrics in sorted(category_specifics.items()):
                                    print(f"      Category: {cat}")
                                    avg_acc_cat = cat_metrics.get('avg_accuracy')
                                    avg_acc_cat_str = f"{avg_acc_cat:.4f}" if avg_acc_cat is not None else "N/A"
                                    print(f"        Avg Accuracy: {avg_acc_cat_str}")
                                    
                                    avg_f1_cat = cat_metrics.get('avg_f1')
                                    avg_f1_cat_str = f"{avg_f1_cat:.4f}" if avg_f1_cat is not None else "N/A"
                                    print(f"        Avg F1 Score: {avg_f1_cat_str}")
                        else:
                            print(f"    Unexpected data structure for model type {model_type}")
            elif not current_run_summary: # Handles if summarize_evaluations returned empty
                print("  No models found in summary results (summary is empty).")
            else: # Handles if summarize_evaluations returned non-dict or other unexpected
                print(f"  Summary generation returned unexpected result: {current_run_summary}")

            # Plotting calls (CMs and F1 chart)
            if isinstance(current_run_summary, dict) and current_run_summary and \
                not ("error" in current_run_summary and len(current_run_summary) == 1):
                
                mldl_keys_for_cm_plot = []
                llm_keys_for_cm_plot = []
                
                mldl_name_map_for_cm_plot = {
                    "svm": "SVM", "random_forest": "RF", "gradient_boosting": "GBM",
                    "cnn": "CNN", "rcnn": "RCNN", 
                    "cnn_Hybrid": "CNN", "rcnn_Hybrid": "RCNN"
                }
                llm_name_map_for_cm_plot = {}

                for model_key in current_run_summary.keys():
                    if model_key.startswith("llm_"):
                        llm_keys_for_cm_plot.append(model_key)
                        name_parts = model_key.replace("llm_", "").split('_')
                        provider = name_parts[0]
                        model_id_str = name_parts[1] if len(name_parts) > 1 else "unknown_model"
                        mode_suffix = "_".join(name_parts[2:]) if len(name_parts) > 2 else "unknown_mode"
                        
                        display_model_acronym = model_id_str 
                        if "gpt-4o-mini" == model_id_str: display_model_acronym = "GPT 4o M."
                        elif "gpt-3.5-turbo" == model_id_str: display_model_acronym = "GPT 3.5"
                        elif "gemini-2.0-flash-lite" == model_id_str: display_model_acronym = "Gemini 2.0"
                        elif "gemini-1.5-pro" == model_id_str: display_model_acronym = "Gemini 1.5"

                        display_mode_acronym = mode_suffix.replace("_", " ").title()
                        llm_name_map_for_cm_plot[model_key] = f"{display_model_acronym} {display_mode_acronym}".strip()
                    elif model_key in mldl_name_map_for_cm_plot:
                        mldl_keys_for_cm_plot.append(model_key)
                    else: 
                        if not model_key.startswith("llm_"):
                            mldl_keys_for_cm_plot.append(model_key)
                            if model_key not in mldl_name_map_for_cm_plot:
                                mldl_name_map_for_cm_plot[model_key] = model_key.replace("_", "-").capitalize()
                
                if mldl_keys_for_cm_plot:
                    evaluation.plot_all_average_cms(
                        summary_data=current_run_summary, main_run_dir=main_run_dir,
                        model_filter_keys=mldl_keys_for_cm_plot, model_name_map=mldl_name_map_for_cm_plot,
                        plot_filename="summary_avg_cms_mldl.png"
                    )
                else: print("   No ML/DL model data found in summary for CM plot.")

                if llm_keys_for_cm_plot:
                    evaluation.plot_all_average_cms(
                        summary_data=current_run_summary, main_run_dir=main_run_dir,
                        model_filter_keys=llm_keys_for_cm_plot, model_name_map=llm_name_map_for_cm_plot,
                        plot_filename="summary_avg_cms_llm.png"
                    )
                else: print("   No LLM model data found in summary for CM plot.")
                
                evaluation.plot_f1_comparison_chart(current_run_summary, main_run_dir)

                f1_chart_name_map = {} 
                # (Populate f1_chart_name_map similar to how master_model_name_map is populated,
                #  or directly use master_model_name_map if it's already created and in scope)

                # Example of populating f1_chart_name_map if master_model_name_map isn't ready:
                predefined_mldl_display_names_f1 = {
                    "svm": "SVM", "random_forest": "RF", "gradient_boosting": "GBM",
                    "cnn": "CNN", "rcnn": "RCNN", 
                    "cnn_Hybrid": "CNN-H", "rcnn_Hybrid": "RCNN-H"
                }
                for model_key_f1 in current_run_summary.keys():
                    if model_key_f1.startswith("llm_"):
                        name_parts_f1 = model_key_f1.replace("llm_", "").split('_')
                        model_id_str_f1 = name_parts_f1[1] if len(name_parts_f1) > 1 else "llm_model"
                        mode_suffix_f1 = "_".join(name_parts_f1[2:]) if len(name_parts_f1) > 2 else "mode"
                        
                        display_model_acronym_f1 = model_id_str_f1 
                        if "gpt-4o-mini" == model_id_str_f1: display_model_acronym_f1 = "GPT-4o-mini"
                        elif "gpt-3.5-turbo" == model_id_str_f1: display_model_acronym_f1 = "GPT-3.5T"
                        elif "gemini-pro" == model_id_str_f1: display_model_acronym_f1 = "Gemini-Pro"
                        elif "gemini-1.5-pro" == model_id_str_f1: display_model_acronym_f1 = "Gemini-1.5P"

                        display_mode_acronym_f1 = ""
                        if "zero_shot" == mode_suffix_f1: display_mode_acronym_f1 = "ZS"
                        elif "few_shot" == mode_suffix_f1: display_mode_acronym_f1 = "FS"
                        f1_chart_name_map[model_key_f1] = f"{display_model_acronym_f1} {display_mode_acronym_f1}".strip()
                    elif model_key_f1 in predefined_mldl_display_names_f1:
                        f1_chart_name_map[model_key_f1] = predefined_mldl_display_names_f1[model_key_f1]
                    else: 
                        if not model_key_f1.startswith("llm_"):
                            f1_chart_name_map[model_key_f1] = model_key_f1.replace("_", "-").capitalize()
                
                evaluation.plot_f1_comparison_chart(
                    summary_data=current_run_summary,
                    main_run_dir=main_run_dir,
                    model_name_map=f1_chart_name_map # Pass the map
                )

        except Exception as e:
            print(f"❌ Error during final result summarization or plotting in main: {e}")
            traceback.print_exc()
            # current_run_summary might be empty or partially filled if an error occurred

    # --- Define master_model_name_map (uses current_run_summary) ---
    # This block is now correctly placed AFTER the try-except for summary generation and plotting.
    master_model_name_map = {}
    # Check if current_run_summary is a valid dict and not an error placeholder
    if isinstance(current_run_summary, dict) and current_run_summary and not \
    ("error" in current_run_summary and len(current_run_summary) == 1):
        predefined_mldl_display_names = {
            "svm": "SVM", "random_forest": "RF", "gradient_boosting": "GBM",
            "cnn": "CNN", "rcnn": "RCNN", 
            "cnn_Hybrid": "CNN", "rcnn_Hybrid": "RCNN"
        }
        for model_key in current_run_summary.keys():
            if model_key.startswith("llm_"):
                name_parts = model_key.replace("llm_", "").split('_')
                # Ensure robust parsing for model_id_str and mode_suffix
                model_id_str = name_parts[1] if len(name_parts) > 1 else "llm_model"
                mode_suffix = "_".join(name_parts[2:]) if len(name_parts) > 2 else "mode"

                display_model_acronym = model_id_str 
                if "gpt-4o-mini" == model_id_str: display_model_acronym = "GPT 4o M."
                elif "gpt-3.5-turbo" == model_id_str: display_model_acronym = "GPT 3.5"
                elif "gemini-2.0-flash-lite" == model_id_str: display_model_acronym = "Gemini 2.0"
                elif "gemini-1.5-pro" == model_id_str: display_model_acronym = "Gemini 1.5"
                
                display_mode_acronym = ""
                if "zero_shot" == mode_suffix: display_mode_acronym = "ZS"
                elif "few_shot" == mode_suffix: display_mode_acronym = "FS"
                
                master_model_name_map[model_key] = f"{display_model_acronym} {display_mode_acronym}".strip()
            elif model_key in predefined_mldl_display_names:
                master_model_name_map[model_key] = predefined_mldl_display_names[model_key]
            else: 
                if not model_key.startswith("llm_"):
                    master_model_name_map[model_key] = model_key.replace("_", "-").capitalize()
    else:
        print("⚠️ current_run_summary not available or in error state for creating master_model_name_map; map may be incomplete or empty.")

    # --- Setup for analyze_price_quantiles (needs unique_known_model_ids_sorted) ---
    known_model_identifiers_list = []
    if config is not None: # Ensure config was loaded successfully earlier
        parsed_models_config = config.get("models_to_run", {})
        known_model_identifiers_list.extend(parsed_models_config.get("ml", []))
        dl_models_base = parsed_models_config.get("dl", [])
        dl_feature_set_config = config.get("dl_feature_set", "hybrid")
        for dl_model_name_base in dl_models_base:
            known_model_identifiers_list.append(dl_model_name_base)
            if dl_feature_set_config == 'hybrid':
                known_model_identifiers_list.append(f"{dl_model_name_base}_Hybrid")
        
        llm_overall_eval_config = config.get("llm_evaluation", {})
        llm_prompting_modes = llm_overall_eval_config.get("prompting_modes", ['zero_shot'])
        for llm_provider_key, llm_model_ids in parsed_models_config.items():
            if llm_provider_key.startswith("llm_") and llm_model_ids:
                for llm_model_id_str_cfg in llm_model_ids:
                    for p_mode in llm_prompting_modes:
                        known_model_identifiers_list.append(f"{llm_provider_key}_{llm_model_id_str_cfg}_{p_mode}")
    else:
        print("⚠️ Config object not loaded. Cannot determine known_model_identifiers for price quantile analysis.")

    unique_known_model_ids_sorted = sorted(list(set(known_model_identifiers_list)), key=len, reverse=True)
    if not unique_known_model_ids_sorted and known_model_identifiers_list: # Check if list had items but set is empty (should not happen)
        print("⚠️ An issue occurred creating unique_known_model_ids_sorted.")
    elif not known_model_identifiers_list : # Only print if truly no models configured
        print("⚠️ No models configured; known_model_identifiers list is empty. Price quantile parsing might be affected if prediction files exist from other means.")

    price_analysis_results_data = evaluation.analyze_price_quantiles(
        main_run_dir,
        known_model_identifiers=unique_known_model_ids_sorted
    )

    # --- Generate Tables ---
    generate_accuracy_table(
        summary_data=current_run_summary,
        analysis_results=None,
        main_run_dir=main_run_dir,
        analysis_type='overall_by_category',
        model_name_map=master_model_name_map
    )
    generate_accuracy_table(
        summary_data=None,
        analysis_results=price_analysis_results_data,
        main_run_dir=main_run_dir,
        analysis_type='price_quantile',
        model_name_map=master_model_name_map
    )
    generate_accuracy_table(
        summary_data=None,
        analysis_results=price_analysis_results_data,
        main_run_dir=main_run_dir,
        analysis_type='price_quantile_cm_metrics',
        model_name_map=master_model_name_map
    )

    end_time = datetime.now()
    print(f"\n======== Run {run_timestamp} Finished ========")
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()