# main.py
import os
import numpy as np
import pandas as pd
from datetime import datetime # <-- Import datetime
from scripts.utils import load_config
# Import specific functions needed
from scripts.data_loader import load_reviews, identify_label_candidates, create_balanced_temporal_splits
# Updated feature engineering imports
from scripts.feature_engineering import fit_feature_extractor, transform_features, save_featurizer
from scripts import preprocessing, model_training, evaluation # evaluation.py should have recursive summary already

import random
import tensorflow as tf # <-- Keep TF import here

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception as e:
        print(f"Warning: Could not set TensorFlow seed: {e}")
    os.environ['PYTHONHASHSEED'] = str(seed)

# MODEL_DISPATCH points to the correct functions
MODEL_DISPATCH = {
    "random_forest": model_training.train_random_forest,
    "svm": model_training.train_svm,
    "gradient_boosting": model_training.train_gbm,
    "cnn": model_training.train_cnn,
    "rcnn": model_training.train_rcnn
}

# Modified function to accept the main run directory path
def run_pipeline_for_seed(config, seed, main_run_dir): # <-- Added main_run_dir argument
    set_seed(seed)

    all_results_this_seed = [] # Store results for this seed run

    # --- Create seed-specific output directory INSIDE main_run_dir ---
    seed_output_dir = os.path.join(main_run_dir, f"seed_{seed}") # <-- Use main_run_dir
    os.makedirs(seed_output_dir, exist_ok=True)
    print(f"   Output directory for this seed: {seed_output_dir}")

    categories_to_run = config.get("categories", [])
    if not categories_to_run:
        print("⚠️ No categories specified in config. Skipping seed.")
        return all_results_this_seed # Return empty list for this seed

    for category in categories_to_run:
        print(f"\n📦 Processing category: {category} | Seed: {seed}")
        path = config.get("data_path_template", "data/{category}.jsonl").format(category=category)

        # --- Step 1: Load ---
        df_raw = load_reviews(
            filepath=path,
            year_range=config.get("year_range"),
            max_rows=config.get("max_reviews"),
            seed=seed
        )
        if df_raw.empty: print(f"⚠️ No data loaded for {category}. Skipping category."); continue

        # --- Step 2: Preprocess ---
        try:
            df_processed = preprocessing.preprocess_reviews(df_raw)
        except ValueError as e:
            print(f"❌ Error during preprocessing for {category}: {e}. Skipping category.")
            continue
        if df_processed.empty: print(f"⚠️ Data empty after preprocessing for {category}. Skipping category."); continue

        # --- Step 3: Identify Label Candidates ---
        try:
            df_labeled_pool = identify_label_candidates(df_processed, config)
        except ValueError as e:
             print(f"❌ Error during label identification for {category}: {e}. Skipping category.")
             continue
        if df_labeled_pool.empty: print(f"⚠️ No labeled candidates found for {category}. Skipping category."); continue

        # --- Step 4: Create Splits ---
        try:
            train_df, val_df, test_df = create_balanced_temporal_splits(df_labeled_pool, config, seed)
        except ValueError as e:
            print(f"❌ Error during split creation for {category}: {e}. Skipping category.")
            continue

        # --- Validation: Check splits ---
        if train_df.empty or test_df.empty:
            print(f"⚠️ Train or Test split is empty for {category}. Skipping models for this category.")
            continue
        if train_df['label'].nunique() < 2 or test_df['label'].nunique() < 2:
            print(f"⚠️ Train ({train_df['label'].nunique()} labels) or Test ({test_df['label'].nunique()} labels) lacks both classes. Skipping models.")
            print("   Train labels:\n", train_df['label'].value_counts())
            print("   Test labels:\n", test_df['label'].value_counts())
            continue

        print("\n✅ Data splits created successfully:")
        print(f"   Train shape: {train_df.shape}, Labels: {dict(train_df['label'].value_counts())}")
        print(f"   Val shape: {val_df.shape}, Labels: {dict(val_df['label'].value_counts()) if not val_df.empty else 'Empty'}")
        print(f"   Test shape: {test_df.shape}, Labels: {dict(test_df['label'].value_counts())}")

        # --- Step 5: Feature Engineering (for ML models) ---
        models_config = config.get("models_to_run", {})
        ml_models_to_run = models_config.get("ml", [])
        dl_models_to_run = models_config.get("dl", [])

        featurizer = None
        feature_names = []
        X_train, y_train, X_val, y_val, X_test, y_test = [None] * 6 # Initialize all

        if ml_models_to_run:
            try:
                print("\n🛠️ Applying Feature Engineering for ML models...")
                featurizer, feature_names = fit_feature_extractor(
                    train_df,
                    feature_set=config.get("feature_set", "hybrid"),
                    text_max_features=config.get("text_max_features", 1000)
                )
                # --- Save featurizer to the specific seed output directory ---
                featurizer_path = os.path.join(seed_output_dir, f"{category}_featurizer.joblib") # <-- Use seed_output_dir
                save_featurizer(featurizer, featurizer_path)
                # --- ---

                X_train, y_train = transform_features(train_df, featurizer)
                if not val_df.empty:
                    X_val, y_val = transform_features(val_df, featurizer)
                else:
                     num_features = X_train.shape[1] if X_train is not None else 0
                     X_val, y_val = np.empty((0, num_features)), np.empty((0,))
                X_test, y_test = transform_features(test_df, featurizer)

                print("\n✅ Feature Engineering completed:")
                # (print shapes...)
                if X_train.shape[0] == 0 or X_test.shape[0] == 0 or X_train.shape[1] == 0:
                     print("⚠️ Empty arrays or zero features after FE. Skipping ML models.")
                     ml_models_to_run = []
            except Exception as e:
                print(f"❌ Error during feature engineering for {category}: {e}")
                import traceback; traceback.print_exc()
                ml_models_to_run = []

        # --- Step 6: Model Training & Evaluation ---

        # --- ML Models ---
        if ml_models_to_run and X_train is not None and X_test is not None:
            for model_name in ml_models_to_run:
                print(f"\n🚀 Training ML model: {model_name}")
                if model_name not in MODEL_DISPATCH: print(f"⚠️ Model '{model_name}' not found."); continue
                try:
                    model_fn = MODEL_DISPATCH[model_name]
                    class_weight_config = 'balanced' if not config.get('balanced_sampling',{}).get('use_strict_balancing', False) else None
                    _, model = model_fn(X_train, y_train) # Add class_weight=class_weight_config if functions updated

                    print(f"   Fitting {model_name} completed.")
                    y_pred = model.predict(X_test)
                    y_proba = None
                    if hasattr(model, "predict_proba"):
                        # (predict_proba logic...)
                         try:
                            probas = model.predict_proba(X_test)
                            if probas.ndim == 2 and probas.shape[1] == 2: y_proba = probas[:, 1]
                            elif probas.ndim == 1: y_proba = probas
                            elif probas.ndim == 2 and probas.shape[1] == 1: y_proba = probas[:, 0]
                            else: print(f"   ⚠️ Unexpected predict_proba shape {probas.shape}")
                         except Exception as e: print(f"   ⚠️ Could not get probabilities: {e}")

                    # --- Pass seed_output_dir for saving evaluation results ---
                    eval_results = evaluation.evaluate_model(
                        y_true=y_test, y_pred=y_pred, y_proba=y_proba,
                        model_name=f"{category}_{model_name}_seed{seed}",
                        output_dir=seed_output_dir # <-- Use seed_output_dir
                    )
                    all_results_this_seed.append(eval_results)
                    # --- ---
                except Exception as e:
                    print(f"❌ Error during ML model {model_name}: {e}")
                    import traceback; traceback.print_exc()

        # --- DL Models ---
        if dl_models_to_run:
            print("\n🧠 Preparing data for DL models...")
            try:
                texts_train = train_df["full_text"].tolist()
                labels_train = train_df["label"].astype(int).values
                texts_val = val_df["full_text"].tolist() if not val_df.empty else []
                labels_val = val_df["label"].astype(int).values if not val_df.empty else np.array([])
                texts_test = test_df["full_text"].tolist()
                labels_test = test_df["label"].astype(int).values
                if not texts_train or not texts_test: print("⚠️ Empty text lists. Skipping DL."); continue
            except KeyError as e: print(f"❌ Missing column for DL: {e}. Skipping DL."); continue

            dl_params = { # (Get DL params from config...)
                "max_words": config.get("dl_max_words", 10000),
                "max_len": config.get("dl_max_len", 300),
                "embedding_dim": config.get("dl_embedding_dim", 64),
                "epochs": config.get("dl_epochs", 5),
                "batch_size": config.get("dl_batch_size", 64),
                "seed": seed
            }
            dl_params["class_weight"] = None # (Calculate class weights if needed...)
            if not config.get('balanced_sampling',{}).get('use_strict_balancing', False):
                 from sklearn.utils.class_weight import compute_class_weight
                 try:
                      weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
                      dl_params["class_weight"] = dict(enumerate(weights))
                      print(f"   Calculated class weights for DL: {dl_params['class_weight']}")
                 except Exception as e: print(f"   ⚠️ Could not compute class weights for DL: {e}")


            for model_name in dl_models_to_run:
                print(f"\n🧠 Training DL model: {model_name}")
                if model_name not in MODEL_DISPATCH: print(f"⚠️ Model '{model_name}' not found."); continue
                try:
                    model_fn = MODEL_DISPATCH[model_name]
                    model, tokenizer = model_fn( # Get model and tokenizer back
                        texts_train=texts_train, labels_train=labels_train,
                        texts_val=texts_val, labels_val=labels_val,
                        **dl_params
                    )
                    print(f"   Fitting {model_name} completed.")

                    # --- Correct DL Evaluation ---
                    print("   Preprocessing test data using fitted tokenizer...")
                    sequences_test = tokenizer.texts_to_sequences(texts_test)
                    padded_test = tf.keras.preprocessing.sequence.pad_sequences(
                        sequences_test, maxlen=dl_params["max_len"]
                    )
                    print(f"   Test data shape after padding: {padded_test.shape}")
                    print("   Predicting on test data...")
                    probas_test = model.predict(padded_test)
                    preds_test = (probas_test > 0.5).astype("int32").flatten()
                    proba_test_for_eval = probas_test.flatten()

                    # --- Pass seed_output_dir for saving evaluation results ---
                    eval_results = evaluation.evaluate_model(
                        y_true=labels_test, y_pred=preds_test, y_proba=proba_test_for_eval,
                        model_name=f"{category}_{model_name}_seed{seed}",
                        output_dir=seed_output_dir # <-- Use seed_output_dir
                    )
                    all_results_this_seed.append(eval_results)
                    # --- ---
                except Exception as e:
                    print(f"❌ Error during DL model {model_name}: {e}")
                    import traceback; traceback.print_exc()

    # --- End of Category Loop ---
    return all_results_this_seed # Return results collected for this seed


def main():
    start_time = datetime.now()
    run_timestamp = start_time.strftime('%Y%m%d_%H%M%S') # Generate timestamp for the run

    try:
        config = load_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}"); return

    # --- Create Main Run Directory ---
    base_output_dir = config.get("output_dir", "results/")
    main_run_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"======== Starting Run: {run_timestamp} ========")
    print(f"Output will be saved in: {main_run_dir}")
    # --- ---

    # Optional: Weave Init (Consider initializing weave per run?)
    # ...

    seeds_to_run = config.get("random_seeds", [42])
    if not isinstance(seeds_to_run, list) or not seeds_to_run:
        print("⚠️ Invalid 'random_seeds'. Using default [42]."); seeds_to_run = [42]

    all_run_results = [] # Accumulate results across all seeds within this run
    for seed in seeds_to_run:
        try:
            current_seed = int(seed)
            print(f"\n------- Processing Seed: {current_seed} -------")
            # Pass main_run_dir to the seed function
            seed_results = run_pipeline_for_seed(config, current_seed, main_run_dir)
            all_run_results.extend(seed_results) # Add results from this seed
        except (ValueError, TypeError):
            print(f"⚠️ Invalid seed value '{seed}'. Skipping."); continue
        except Exception as e:
             print(f"❌❌❌ Unhandled Error during pipeline run for seed {seed}! Error: {e} ❌❌❌")
             import traceback; traceback.print_exc()

    # --- Final Summary (Summarize results for THIS run only) ---
    print("\n📊 Summarizing evaluation results for this run...")
    try:
        # Pass the specific main_run_dir to the summarization function
        summary = evaluation.summarize_evaluations(result_dir=main_run_dir) # <-- Use main_run_dir
        print(f"\n📋 Final Summary (Run: {run_timestamp}):")
        if isinstance(summary, dict) and summary:
            # (Pretty print summary...)
            sorted_summary = dict(sorted(summary.items()))
            for model, metrics in sorted_summary.items():
                print(f"  Model: {model}")
                sorted_metrics = dict(sorted(metrics.items()))
                for metric, value in sorted_metrics.items():
                     if metric == 'run_count': print(f"    {metric}: {value}") # Don't format count
                     elif isinstance(value, (int, float, np.number)): print(f"    {metric}: {value:.4f}")
                     else: print(f"    {metric}: {value}")
        elif not summary: print("  No results found in this run to summarize.")
        else: print(summary)
    except Exception as e:
        print(f"❌ Error during result summarization: {e}")

    end_time = datetime.now()
    print(f"\n======== Run {run_timestamp} Finished ========")
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()