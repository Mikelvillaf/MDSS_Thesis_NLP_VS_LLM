# main.py (Modified for LLM Integration - Zero/Few Shot & Balanced Sampling)
import os
import numpy as np
import pandas as pd
from datetime import datetime
import yaml # Ensure yaml is imported
from scripts.utils import load_config
from scripts.data_loader import load_reviews, identify_label_candidates, create_balanced_temporal_splits
from scripts.feature_engineering import fit_feature_extractor, transform_features, save_featurizer
from scripts import preprocessing, model_training, evaluation, llm_prediction # Import llm_prediction
import random
import tensorflow as tf
import time # Import time for delays
import warnings # To manage warnings if needed

def set_seed(seed=42):
    # Sets random seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # print(f"   Seed set to {seed}") # Optional: uncomment for verification


# MODEL_DISPATCH maps config strings to ML/DL training functions
MODEL_DISPATCH = {
    "random_forest": model_training.train_random_forest,
    "svm": model_training.train_svm,
    "gradient_boosting": model_training.train_gbm,
    "cnn": model_training.train_cnn,
    "rcnn": model_training.train_rcnn
    # LLMs are handled separately in the pipeline
}

# Main pipeline function executed for each seed
def run_pipeline_for_seed(config, seed, main_run_dir):
    set_seed(seed)
    all_results_this_seed = []
    seed_output_dir = os.path.join(main_run_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    print(f"   Output directory for this seed: {seed_output_dir}")

    categories_to_run = config.get("categories", [])
    if not categories_to_run:
        print("⚠️ No categories specified in config. Exiting seed run.")
        return []

    # --- Loop through specified categories ---
    for category in categories_to_run:
        print(f"\n📦 Processing category: {category} | Seed: {seed}")
        # Construct data path using template from config
        path = config.get("data_path_template", "data/{category}.jsonl").format(category=category)
        if not os.path.exists(path):
             print(f"⚠️ Data file not found: {path}. Skipping category.")
             continue

        # --- Steps 1-4: Load, Preprocess, Label, Split ---
        # Wrapped in try-except to handle errors gracefully per category
        try:
            print("\n--- Data Loading & Preprocessing ---")
            df_raw = load_reviews(
                filepath=path,
                year_range=config.get("year_range"),
                max_rows=config.get("max_reviews"), # max_reviews likely removed/null in current config
                seed=seed
            )
            if df_raw.empty: print(f"⚠️ No data loaded for {category}. Skipping."); continue

            df_processed = preprocessing.preprocess_reviews(df_raw)
            if df_processed.empty: print(f"⚠️ Data empty after preprocessing. Skipping."); continue

            print("\n--- Label Generation ---")
            df_labeled_pool = identify_label_candidates(df_processed, config)
            if df_labeled_pool.empty: print(f"⚠️ No labeled candidates found. Skipping."); continue

            print("\n--- Temporal Splitting & Sampling ---")
            # Crucially, keep train_df accessible for few-shot examples later
            train_df, val_df, test_df = create_balanced_temporal_splits(df_labeled_pool, config, seed)

            # Sanity checks on splits
            if train_df.empty: print(f"⚠️ Train split empty. Skipping category."); continue
            if test_df.empty: print(f"⚠️ Test split empty. Skipping category."); continue
            # Check if both classes present in train/test (needed for many metrics/models)
            if train_df['label'].nunique() < 2: print(f"⚠️ Train split lacks samples from both classes. Skipping category."); continue
            if test_df['label'].nunique() < 2: print(f"⚠️ Test split lacks samples from both classes. Skipping category."); continue

        except FileNotFoundError as e:
             print(f"❌ Error: Data file not found during data prep: {e}")
             continue # Skip to next category
        except ValueError as e:
             print(f"❌ Error during data preparation steps: {e}")
             continue # Skip to next category
        except Exception as e:
             print(f"❌ Unexpected error during data preparation: {e}")
             import traceback; traceback.print_exc()
             continue # Skip to next category

        print("\n✅ Data splits created successfully:")
        print(f"   Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
        # Display label distribution for context
        if not train_df.empty: print(f"   Train Label Dist:\n{train_df['label'].value_counts(normalize=True).to_string()}")
        if not test_df.empty: print(f"   Test Label Dist:\n{test_df['label'].value_counts(normalize=True).to_string()}")


        # --- Step 5: Feature Engineering (for ML and potentially Hybrid DL) ---
        models_config = config.get("models_to_run", {})
        ml_models_to_run = models_config.get("ml", [])
        dl_models_to_run = models_config.get("dl", [])
        llm_models_to_run = models_config.get("llm", []) # Get LLM models

        # Get feature set configs
        ml_feature_set = config.get("feature_set", "hybrid") # Used for ML and structured feature generation
        dl_feature_set = config.get("dl_feature_set", "hybrid") # Used for DL input type
        dl_num_structured = config.get("dl_num_structured_features", 5) # Expected count for hybrid DL

        # Determine if the feature engineering step (fitting featurizer) needs to run
        # Runs if ML models are selected OR if DL models are 'hybrid' (as they need structured features)
        needs_feature_engineering_step = bool(ml_models_to_run) or (bool(dl_models_to_run) and dl_feature_set == 'hybrid')

        # Initialize feature variables
        featurizer = None
        feature_names = []
        X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml = [None] * 6 # For ML models
        X_train_structured_dl, X_val_structured_dl, X_test_structured_dl = [None] * 3 # For Hybrid DL

        if needs_feature_engineering_step:
            try:
                print(f"\n--- Feature Engineering (ML feature_set='{ml_feature_set}') ---")
                # Fit featurizer on training data using the ML feature_set config
                featurizer, feature_names = fit_feature_extractor(
                    train_df,
                    feature_set=ml_feature_set,
                    text_max_features=config.get("text_max_features", 1000)
                )
                # Save the fitted featurizer for this seed
                featurizer_path = os.path.join(seed_output_dir, f"{category}_featurizer.joblib")
                save_featurizer(featurizer, featurizer_path)

                # Transform data splits for ML
                print("   Transforming data for ML models...")
                X_train_ml, y_train_ml = transform_features(train_df, featurizer)
                if not val_df.empty:
                    X_val_ml, y_val_ml = transform_features(val_df, featurizer)
                else:
                    # Create empty arrays with correct feature dimension if val_df is empty
                    feat_dim = X_train_ml.shape[1] if X_train_ml is not None else 0
                    X_val_ml, y_val_ml = np.empty((0, feat_dim)), np.empty((0,))

                X_test_ml, y_test_ml = transform_features(test_df, featurizer)
                print("✅ Feature Engineering completed for ML:")
                print(f"   X_train_ml shape: {X_train_ml.shape if X_train_ml is not None else 'None'}")
                print(f"   X_val_ml shape: {X_val_ml.shape if X_val_ml is not None else 'None'}")
                print(f"   X_test_ml shape: {X_test_ml.shape if X_test_ml is not None else 'None'}")

                # Extract structured features for Hybrid DL if needed
                if bool(dl_models_to_run) and dl_feature_set == 'hybrid':
                    # Structured features only generated if ML feature set wasn't 'nlp'
                    if ml_feature_set != 'nlp':
                        print(f"   Extracting first {dl_num_structured} columns as structured features for Hybrid DL...")
                        # Check if featurizer produced enough columns
                        if X_train_ml is not None and X_train_ml.shape[1] >= dl_num_structured:
                            X_train_structured_dl = X_train_ml[:, :dl_num_structured]
                            # Handle validation set existence
                            if X_val_ml is not None and X_val_ml.shape[0] > 0:
                                X_val_structured_dl = X_val_ml[:, :dl_num_structured]
                            else:
                                X_val_structured_dl = np.empty((0, dl_num_structured)) # Empty with correct feature dim
                            X_test_structured_dl = X_test_ml[:, :dl_num_structured]
                            print(f"      DL Structured Shapes: Train={X_train_structured_dl.shape}, Val={X_val_structured_dl.shape}, Test={X_test_structured_dl.shape}")
                        else:
                            # Not enough columns generated by featurizer
                            print(f"   ⚠️ Warning: Cannot extract {dl_num_structured} structured features for Hybrid DL. " \
                                  f"Featurizer (ml_feature_set='{ml_feature_set}') produced "
                                  f"{X_train_ml.shape[1] if X_train_ml is not None else 0} columns. Hybrid DL models may fail.")
                            # Set structured arrays to indicate failure (e.g., shape (N, 0))
                            num_train_samples = X_train_ml.shape[0] if X_train_ml is not None else train_df.shape[0]
                            num_val_samples = X_val_ml.shape[0] if X_val_ml is not None else val_df.shape[0]
                            num_test_samples = X_test_ml.shape[0] if X_test_ml is not None else test_df.shape[0]
                            X_train_structured_dl = np.empty((num_train_samples, 0))
                            X_val_structured_dl = np.empty((num_val_samples, 0))
                            X_test_structured_dl = np.empty((num_test_samples, 0))
                    else:
                         # ML feature set was 'nlp', so no structured features generated
                        print(f"   ⚠️ Warning: Cannot extract structured features for Hybrid DL because " \
                              f"ML feature_set was '{ml_feature_set}'. Hybrid DL models may fail.")
                        X_train_structured_dl = np.empty((train_df.shape[0], 0))
                        X_val_structured_dl = np.empty((val_df.shape[0], 0))
                        X_test_structured_dl = np.empty((test_df.shape[0], 0))

                # Check if ML features are usable after transformation
                if X_train_ml is None or X_test_ml is None or y_train_ml is None or y_test_ml is None or \
                   X_train_ml.shape[0] == 0 or X_test_ml.shape[0] == 0 or X_train_ml.shape[1] == 0:
                    print("⚠️ Empty arrays or zero features after FE. Skipping ML models.")
                    ml_models_to_run = [] # Cannot run ML if features are invalid

            except Exception as e:
                print(f"❌ Error during feature engineering: {e}")
                import traceback; traceback.print_exc()
                ml_models_to_run = [] # Cannot run ML if FE fails
                # DL 'hybrid' mode will also fail if FE failed
                if dl_feature_set == 'hybrid':
                    print("   Hybrid DL models will likely fail due to FE error.")
                    # Clear DL models requiring hybrid input if FE failed
                    # dl_models_to_run = [] # Or let it fail later

        # --- Step 6: Model Training & Evaluation ---
        print(f"\n--- Model Training & Evaluation ---")

        # --- ML Models ---
        if ml_models_to_run:
            if X_train_ml is None or X_test_ml is None:
                 print("⚠️ Skipping ML models as features (X_train_ml/X_test_ml) are not available.")
            else:
                 for model_name in ml_models_to_run:
                    print(f"\n🚀 Training ML model: {model_name} (using '{ml_feature_set}' features)")
                    if model_name not in MODEL_DISPATCH:
                        print(f"   ⚠️ Model '{model_name}' not found in MODEL_DISPATCH. Skipping.")
                        continue
                    try:
                        model_fn = MODEL_DISPATCH[model_name]
                        # Determine class weight setting based on balancing config
                        class_weight_config = 'balanced' if not config.get('balanced_sampling', {}).get('use_strict_balancing', False) else None

                        # Pass ML-specific features and labels
                        # Assuming ML training functions handle class_weight argument
                        _, model = model_fn(X_train_ml, y_train_ml, class_weight=class_weight_config)

                        print(f"   Fitting {model_name} completed.")
                        print("   Evaluating ML model on test set...")
                        y_pred_ml = model.predict(X_test_ml)
                        y_proba_ml = None

                        # Attempt to get probabilities or decision function values
                        if hasattr(model, "predict_proba"):
                            try: probas = model.predict_proba(X_test_ml)
                            except Exception as e: print(f"   ⚠️ Could not get predict_proba: {e}"); probas = None
                            if probas is not None:
                                if probas.ndim == 2 and probas.shape[1] == 2: y_proba_ml = probas[:, 1] # Standard case
                                elif probas.ndim == 1: y_proba_ml = probas
                                elif probas.ndim == 2 and probas.shape[1] == 1: y_proba_ml = probas[:, 0]
                                else: print(f"   ⚠️ Unexpected predict_proba shape {probas.shape}")
                        elif hasattr(model, "decision_function"):
                             try: y_proba_ml = model.decision_function(X_test_ml)
                             except Exception as e: print(f"   ⚠️ Could not get decision_function: {e}")
                        else: print("   ⚠️ Model has neither predict_proba nor decision_function.")


                        # Evaluate using the ML test labels (y_test_ml)
                        eval_results = evaluation.evaluate_model(
                            y_true=y_test_ml,
                            y_pred=y_pred_ml,
                            y_proba=y_proba_ml,
                            model_name=f"{category}_{model_name}_seed{seed}",
                            output_dir=seed_output_dir
                        )
                        all_results_this_seed.append(eval_results)

                    except Exception as e:
                        print(f"❌ Error during ML model {model_name}: {e}")
                        import traceback; traceback.print_exc()
        else:
            print("   No ML models specified or runnable.")


        # --- DL Models ---
        if dl_models_to_run:
            print(f"\n🧠 Preparing data for DL models (Mode: {dl_feature_set})...")
            try: # Prepare text and labels (always needed from original splits)
                texts_train = train_df["full_text"].tolist()
                labels_train = train_df["label"].astype(int).values
                texts_val = val_df["full_text"].tolist() if not val_df.empty else []
                labels_val = val_df["label"].astype(int).values if not val_df.empty else np.array([])
                texts_test = test_df["full_text"].tolist()
                labels_test = test_df["label"].astype(int).values # Use original labels from test_df

                if not texts_train: raise ValueError("Training text list is empty.")
                if not texts_test: raise ValueError("Test text list is empty.")

            except (KeyError, ValueError) as e:
                print(f"❌ Error preparing text/labels for DL: {e}. Skipping DL models.")
                continue # Skip DL for this category if base data prep fails

            # --- Prepare DL Inputs based on dl_feature_set ---
            dl_train_struct_input = None
            dl_val_struct_input = None
            dl_test_struct_input = None
            num_struct_features_for_model = 0 # Default to 0 for text-only

            if dl_feature_set == 'hybrid':
                print("   Hybrid DL mode selected. Verifying structured features...")
                # Check if structured features were successfully extracted and have correct dimensions
                valid_train_struct = (X_train_structured_dl is not None and
                                      X_train_structured_dl.shape[0] == len(texts_train) and
                                      X_train_structured_dl.shape[1] == dl_num_structured)
                valid_test_struct = (X_test_structured_dl is not None and
                                     X_test_structured_dl.shape[0] == len(texts_test) and
                                     X_test_structured_dl.shape[1] == dl_num_structured)
                # Validation struct check (only matters if val text exists)
                valid_val_struct = True # Assume true if no val text
                if texts_val:
                     valid_val_struct = (X_val_structured_dl is not None and
                                         X_val_structured_dl.shape[0] == len(texts_val) and
                                         X_val_structured_dl.shape[1] == dl_num_structured)

                if not valid_train_struct or not valid_test_struct:
                     print(f"❌ Required structured features (shape[1]={dl_num_structured}) not available or shape mismatch " \
                           f"(Train valid: {valid_train_struct}, Test valid: {valid_test_struct}). Skipping Hybrid DL models.")
                     continue # Skip DL if required structured features aren't ready/valid

                if not valid_val_struct and texts_val:
                      print("   ⚠️ Mismatch in validation structured features shape or count. Validation during Keras training might be skipped.")
                      dl_val_struct_input = None # Ensure Keras gets None if invalid
                elif texts_val: # only assign if val text exists and struct is valid
                      dl_val_struct_input = X_val_structured_dl
                else: # No val text, so no val struct needed
                    dl_val_struct_input = None


                dl_train_struct_input = X_train_structured_dl
                dl_test_struct_input = X_test_structured_dl
                num_struct_features_for_model = dl_num_structured
                print(f"   Structured features verified for Hybrid DL.")

            elif dl_feature_set == 'text':
                print("   Text-only DL mode selected. Structured features will not be used.")
            else:
                print(f"⚠️ Invalid dl_feature_set: '{dl_feature_set}'. Skipping DL models.")
                continue

            # --- Prepare DL Training Params ---
            dl_params = {
                "max_words": config.get("dl_max_words", 10000),
                "max_len": config.get("dl_max_len", 300),
                "embedding_dim": config.get("dl_embedding_dim", 64),
                "epochs": config.get("dl_epochs", 5),
                "batch_size": config.get("dl_batch_size", 64),
                "seed": seed,
                "num_structured_features": num_struct_features_for_model, # Pass 0 if text-only
                "class_weight": None # Calculated below
            }

            # Calculate class weights if needed
            if not config.get('balanced_sampling',{}).get('use_strict_balancing', False):
                 from sklearn.utils.class_weight import compute_class_weight
                 try:
                      unique_labels_train = np.unique(labels_train)
                      if len(unique_labels_train) > 1:
                           weights = compute_class_weight('balanced', classes=unique_labels_train, y=labels_train)
                           dl_params["class_weight"] = dict(enumerate(weights))
                           print(f"   Calculated class weights for DL: {dl_params['class_weight']}")
                      else: print("   Skipping class weight calculation: only one class present in DL training labels.")
                 except Exception as e: print(f"   ⚠️ Could not compute class weights for DL: {e}")

            # --- DL Training Loop ---
            for model_name in dl_models_to_run:
                print(f"\n🧠 Training DL model: {model_name} (Input Type: {dl_feature_set})")
                if model_name not in MODEL_DISPATCH:
                    print(f"   ⚠️ Model '{model_name}' not found in MODEL_DISPATCH. Skipping.")
                    continue
                try:
                    model_fn = MODEL_DISPATCH[model_name]
                    # Prepare inputs dictionary for model training function call
                    train_inputs = {
                        "texts_train": texts_train,
                        "labels_train": labels_train,
                        "structured_train": dl_train_struct_input, # Will be None for text-only
                        "texts_val": texts_val,
                        "labels_val": labels_val,
                        "structured_val": dl_val_struct_input # Will be None for text-only or if val failed check
                    }

                    # Train model, passing combined params and conditional inputs
                    # Assuming DL training functions handle None for structured inputs
                    model, tokenizer = model_fn(**train_inputs, **dl_params)
                    print(f"   Fitting {model_name} completed.")

                    # --- DL Evaluation ---
                    print("   Evaluating DL model on test set...")
                    print("      Preprocessing test text data using fitted tokenizer...")
                    sequences_test = tokenizer.texts_to_sequences(texts_test)
                    padded_test = tf.keras.preprocessing.sequence.pad_sequences(
                        sequences_test, maxlen=dl_params["max_len"]
                    )

                    # Prepare list of inputs for prediction based on dl_feature_set
                    predict_inputs = [padded_test] # Text input always first
                    if dl_feature_set == 'hybrid':
                        if dl_test_struct_input is not None:
                             predict_inputs.append(dl_test_struct_input)
                        else:
                             # This case should ideally be caught earlier, but double-check
                             print("   ⚠️ Cannot perform hybrid prediction as test structured data is missing/invalid. Skipping evaluation.")
                             continue # Skip evaluation for this model

                    print(f"      Predicting using {'Hybrid' if len(predict_inputs)>1 else 'Text-Only'} input...")
                    probas_test = model.predict(predict_inputs)
                    preds_test = (probas_test > 0.5).astype("int32").flatten()
                    proba_test_for_eval = probas_test.flatten() # Use probabilities for ROC AUC

                    # Evaluate using the DL test labels (labels_test)
                    eval_results = evaluation.evaluate_model(
                        y_true=labels_test,
                        y_pred=preds_test,
                        y_proba=proba_test_for_eval,
                        model_name=f"{category}_{model_name}_{dl_feature_set}_seed{seed}", # Add mode to name
                        output_dir=seed_output_dir
                    )
                    all_results_this_seed.append(eval_results)

                except Exception as e:
                    print(f"❌ Error during DL model {model_name} ({dl_feature_set} mode): {e}")
                    import traceback; traceback.print_exc()
        else:
            print("   No DL models specified or runnable.")


        # --- LLM Evaluation ---
        if llm_models_to_run:
            print(f"\n🤖 Evaluating LLM models...")
            llm_config = config.get("llm_evaluation", {})

            # --- Setup Client ---
            api_key_env_var = llm_config.get("api_key_env_var", "OPENAI_API_KEY")
            # Pass None as api_key to force check of env var / fallback testing key
            client = llm_prediction._setup_openai_client(api_key=None, env_var_name=api_key_env_var)
            if not client:
                 print("   Skipping LLM evaluation due to API client initialization failure.")
                 continue # Skip LLM part for this category/seed if client fails

            # --- Handle Test Set Sampling (With Balanced Sampling Logic) ---
            llm_test_sample_size = llm_config.get("test_sample_size")
            eval_test_df = pd.DataFrame() # Initialize empty DataFrame

            if llm_test_sample_size and isinstance(llm_test_sample_size, int) and llm_test_sample_size > 0:
                # Ensure test_df has enough data for sampling
                if test_df.empty:
                     print("   ⚠️ Original test_df is empty. Cannot sample for LLM evaluation. Skipping LLM.")
                     continue

                if llm_test_sample_size < len(test_df):
                    print(f"   Attempting BALANCED sampling of {llm_test_sample_size} reviews from the test set for LLM evaluation.")
                    n_each = llm_test_sample_size // 2
                    # Handle odd sample size
                    if llm_test_sample_size % 2 != 0:
                        print(f"   ⚠️ Sample size ({llm_test_sample_size}) is odd for balanced sampling. Aiming for {n_each + 1} from one class and {n_each} from the other.")
                        # Add the extra sample to the helpful class (label=1)
                        n_helpful = n_each + 1
                        n_unhelpful = n_each
                    else:
                        n_helpful = n_each
                        n_unhelpful = n_each

                    helpful_pool_test = test_df[test_df['label'] == 1]
                    unhelpful_pool_test = test_df[test_df['label'] == 0]

                    # Check if pools are empty
                    if helpful_pool_test.empty or unhelpful_pool_test.empty:
                         print(f"   ⚠️ Cannot perform balanced sampling: One or both classes are empty in the test set (H={len(helpful_pool_test)}, U={len(unhelpful_pool_test)}). Skipping LLM.")
                         continue

                    # Ensure we don't request more than available
                    actual_n_helpful = min(n_helpful, len(helpful_pool_test))
                    actual_n_unhelpful = min(n_unhelpful, len(unhelpful_pool_test))

                    if actual_n_helpful < n_helpful or actual_n_unhelpful < n_unhelpful:
                        print(f"   ⚠️ Could not sample requested number per class. Sampling available: H={actual_n_helpful}, U={actual_n_unhelpful}.")
                    # Only proceed if we can sample at least one from each requested class (or if only one was requested)
                    if actual_n_helpful == 0 and n_helpful > 0:
                        print(f"   ⚠️ No helpful samples available in test set to sample from. Skipping LLM balanced sample.")
                        continue
                    if actual_n_unhelpful == 0 and n_unhelpful > 0:
                        print(f"   ⚠️ No unhelpful samples available in test set to sample from. Skipping LLM balanced sample.")
                        continue


                    # Sample with different random states for safety
                    sampled_helpful = helpful_pool_test.sample(n=actual_n_helpful, random_state=seed)
                    sampled_unhelpful = unhelpful_pool_test.sample(n=actual_n_unhelpful, random_state=seed + 1)

                    # Combine and shuffle
                    eval_test_df = pd.concat([sampled_helpful, sampled_unhelpful]).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
                    print(f"   Actual sampled size: {len(eval_test_df)} (Helpful: {actual_n_helpful}, Unhelpful: {actual_n_unhelpful})")

                else:
                    # Sample size >= test set size, use full set (likely unbalanced)
                    print(f"   Sample size ({llm_test_sample_size}) >= test set size ({len(test_df)}). Using full test set (likely unbalanced).")
                    eval_test_df = test_df
            else:
                 # No sample size specified, use full set (likely unbalanced)
                 print("   Using full test set for LLM evaluation (likely unbalanced).")
                 eval_test_df = test_df

            if eval_test_df.empty:
                 print("   ⚠️ Test DataFrame for LLM evaluation is empty after sampling/selection. Skipping LLM.")
                 continue

            # Prepare texts and labels from the selected evaluation DF
            test_texts_llm = eval_test_df['full_text'].tolist()
            test_labels_llm = eval_test_df['label'].values

            # --- Loop through configured LLMs and Prompting Modes ---
            prompting_modes = llm_config.get("prompting_modes", ['zero_shot']) # Default

            for llm_model_name in llm_models_to_run:
                print(f"\n   --- Evaluating LLM: {llm_model_name} ---")

                for mode in prompting_modes:
                    print(f"      Mode: {mode}")
                    few_shot_examples_df = None
                    prompt_template = ""
                    example_format = ""

                    # Get configuration based on mode
                    if mode == 'zero_shot':
                        prompt_template = llm_config.get('zero_shot_prompt_template')
                        if not prompt_template:
                             print("      ⚠️ Zero-shot prompt template not found in config. Skipping.")
                             continue
                    elif mode == 'few_shot':
                        fs_config = llm_config.get('few_shot', {})
                        num_examples = fs_config.get('num_examples')
                        strategy = fs_config.get('example_selection_strategy')
                        example_format = fs_config.get('example_format')
                        prompt_template = fs_config.get('prompt_template')

                        if not all([num_examples, strategy, example_format, prompt_template]):
                             print("      ⚠️ Few-shot configuration incomplete in config. Skipping.")
                             continue

                        # Select few-shot examples from the *original* TRAINING data
                        print("         Selecting few-shot examples from training data...")
                        few_shot_examples_df = llm_prediction.select_few_shot_examples(
                            train_df=train_df, # Use the full training split
                            num_examples=num_examples,
                            strategy=strategy,
                            seed=seed # Use run seed for reproducibility
                        )
                        if few_shot_examples_df is None or few_shot_examples_df.empty:
                             print("      ⚠️ Failed to select valid few-shot examples. Skipping few-shot mode.")
                             continue
                    else:
                        print(f"      ⚠️ Unknown prompting mode: {mode}. Skipping.")
                        continue

                    # --- Get Predictions ---
                    print(f"         Getting LLM predictions for {len(test_texts_llm)} test samples...")
                    try:
                         predictions_llm, failed_count = llm_prediction.get_llm_predictions(
                            client=client,
                            texts_to_classify=test_texts_llm,
                            model_name=llm_model_name,
                            mode=mode,
                            prompt_template=prompt_template,
                            request_timeout=llm_config.get("request_timeout", 30),
                            max_retries=llm_config.get("max_retries", 3),
                            retry_delay=llm_config.get("retry_delay", 5),
                            few_shot_examples_df=few_shot_examples_df, # None if zero_shot
                            few_shot_example_format=example_format # None if zero_shot
                         )

                         # --- Handle Failed Predictions & Evaluate ---
                         # Strategy: Evaluate only on successfully predicted samples
                         valid_indices = [i for i, p in enumerate(predictions_llm) if p is not None]
                         if len(valid_indices) == 0:
                             print(f"      ⚠️ No valid predictions returned by LLM for {mode} mode. Cannot evaluate.")
                             continue # Skip evaluation if all failed

                         if failed_count > 0:
                              print(f"      Evaluating on {len(valid_indices)} successfully parsed predictions (out of {len(test_texts_llm)}).")

                         # Align true labels and predictions based on successful calls
                         y_true_eval = test_labels_llm[valid_indices]
                         y_pred_eval = [predictions_llm[i] for i in valid_indices]

                         # --- Evaluate ---
                         print("         Calculating evaluation metrics...")
                         eval_model_name = f"{category}_{llm_model_name}_{mode}_seed{seed}"
                         eval_results = evaluation.evaluate_model(
                            y_true=y_true_eval,
                            y_pred=y_pred_eval,
                            y_proba=None, # LLMs don't typically provide usable probabilities here
                            model_name=eval_model_name,
                            output_dir=seed_output_dir
                         )
                         all_results_this_seed.append(eval_results)

                    except Exception as e:
                         print(f"❌ Error during LLM prediction/evaluation for {llm_model_name} ({mode}): {e}")
                         import traceback; traceback.print_exc()
        else:
            print("   No LLM models specified or runnable.")


    # --- End of Category Loop ---
    print(f"\n✅ Finished processing category: {category} | Seed: {seed}")
    return all_results_this_seed


def main():
    """Main function to orchestrate the experiment runs."""
    start_time = datetime.now()
    run_timestamp = start_time.strftime('%Y%m%d_%H%M%S')

    # --- Load Configuration ---
    try:
        # Assuming load_config uses a default path like 'configs/experiment_config.yaml'
        config = load_config()
        print("✅ Configuration loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error loading config: {e}. Please ensure 'configs/experiment_config.yaml' exists or provide correct path.")
        return
    except Exception as e:
        print(f"❌ Error parsing config file: {e}")
        return

    # --- Create Run Directory & Save Config ---
    base_output_dir = config.get("output_dir", "results/")
    main_run_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"\n======== Starting Run: {run_timestamp} ========")
    print(f"Output will be saved in: {main_run_dir}")
    config_save_path = os.path.join(main_run_dir, 'config_used.yaml')
    try:
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        print(f"   Configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save config file: {e}")

    # --- Run Pipeline for each Seed ---
    seeds_to_run = config.get("random_seeds", [42])
    if not isinstance(seeds_to_run, list) or not seeds_to_run:
        print("⚠️ Invalid 'random_seeds' format in config. Using default [42].")
        seeds_to_run = [42]

    all_run_results = [] # Collect results across all seeds
    print(f"\n--- Starting runs for seeds: {seeds_to_run} ---")
    for seed in seeds_to_run:
        try:
            current_seed = int(seed) # Ensure seed is integer
            print(f"\n------- Processing Seed: {current_seed} -------")
            # Call the main pipeline function for the current seed
            seed_results = run_pipeline_for_seed(config, current_seed, main_run_dir)
            if seed_results: # Add results if the run for the seed produced any
                 all_run_results.extend(seed_results)
        except (ValueError, TypeError):
            print(f"⚠️ Invalid seed value '{seed}' encountered. Skipping.")
            continue
        except KeyboardInterrupt:
             print("\n🚫 Run interrupted by user (KeyboardInterrupt). Exiting.")
             break # Allow graceful exit if user interrupts
        except Exception as e:
            # Catch any unexpected errors during a seed run
            print(f"❌❌❌ Unhandled Error during pipeline for seed {seed}: {e}")
            import traceback; traceback.print_exc()

    # --- Final Summary ---
    print("\n--- Final Summary Generation ---")
    if not all_run_results:
         print("⚠️ No results were generated across any seeds. Skipping summary.")
    else:
        print(f"📊 Summarizing evaluation results for this run (all seeds) from: {main_run_dir}")
        try:
            # Ensure the summary function uses the correct directory containing all seed results for this run
            summary = evaluation.summarize_evaluations(result_dir=main_run_dir)
            print(f"\n📋 Final Summary (Run ID: {run_timestamp}):")
            if isinstance(summary, dict) and summary:
                # Sort summary dictionary by model name key for consistent output
                sorted_summary = dict(sorted(summary.items()))
                for model, metrics in sorted_summary.items():
                    print(f"  Model: {model}")
                    # Sort metrics within each model
                    sorted_metrics = dict(sorted(metrics.items()))
                    for metric, value in sorted_metrics.items():
                         if metric == 'run_count': print(f"    {metric}: {value}")
                         # Format numbers, handle None gracefully (summary func should return N/A or None)
                         elif isinstance(value, (int, float, np.number)): print(f"    {metric}: {value:.4f}")
                         else: print(f"    {metric}: {value}") # Print N/A or other strings as is
            elif not summary:
                 print("  No valid result files found in the run directory to summarize.")
            else:
                 # Handle case where summary function might return an error message as a string
                 print(f"  Summary generation returned: {summary}")
        except Exception as e:
            print(f"❌ Error during final result summarization: {e}")
            import traceback; traceback.print_exc()

    # --- Run Completion ---
    end_time = datetime.now()
    print(f"\n======== Run {run_timestamp} Finished ========")
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    # Optional: Suppress specific warnings globally if needed
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # warnings.filterwarnings("ignore", category=UserWarning, module='llm_prediction') # Example

    main()