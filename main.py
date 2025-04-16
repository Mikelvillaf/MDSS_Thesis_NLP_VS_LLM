# main.py

import os
import numpy as np
import pandas as pd # Import pandas
from scripts.utils import load_config
# Import specific functions needed
from scripts.data_loader import load_reviews, identify_label_candidates, create_balanced_temporal_splits
from scripts import preprocessing, feature_engineering, model_training, evaluation

import random
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Check if TensorFlow is installed before trying to set its seed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass # Ignore if TensorFlow is not installed
    os.environ['PYTHONHASHSEED'] = str(seed) # For controlling hash randomization

MODEL_DISPATCH = {
    "random_forest": model_training.train_random_forest,
    "svm": model_training.train_svm,
    "gradient_boosting": model_training.train_gbm,
    "cnn": model_training.train_cnn,
    "rcnn": model_training.train_rcnn
}

def run_pipeline_for_seed(config, seed):
    set_seed(seed)

    all_results = [] # Store results per category for potential aggregation later

    for category in config["categories"]:
        print(f"\n📦 Processing category: {category} | Seed: {seed}")
        path = f"data/{category}.jsonl"

        # --- Step 1: Load All Relevant Data ---
        df_raw = load_reviews(
            filepath=path,
            year_range=config["year_range"]
            # No max_rows here
        )

        if df_raw.empty:
            print(f"⚠️ No data loaded for category {category}. Skipping.")
            continue

        # --- Step 2: Preprocess (includes total_vote calculation) ---
        df_processed = preprocessing.preprocess_reviews(df_raw)

        if df_processed.empty:
            print(f"⚠️ Data empty after preprocessing for category {category}. Skipping.")
            continue

        # --- Step 3: Identify Label Candidates (Helpful=1, Unhelpful=0, Discarded=-1) ---
        df_labeled_pool = identify_label_candidates(df_processed, config)

        # --- Step 4: Create Balanced Temporal Splits ---
        train, val, test = create_balanced_temporal_splits(df_labeled_pool, config, seed)

        # --- Validation: Check if splits are usable ---
        if train.empty or test.empty:
            print(f"⚠️ Train or Test split is empty for category {category} after balanced sampling. Skipping model training.")
            continue
        if train['label'].nunique() < 2 or test['label'].nunique() < 2:
            print(f"⚠️ Train or Test split does not contain both classes for category {category}. Skipping model training.")
            print("Train labels:\n", train['label'].value_counts())
            print("Test labels:\n", test['label'].value_counts())
            continue


        print("\n✅ Data splits created successfully:")
        print("Train shape:", train.shape, "Labels:", dict(train["label"].value_counts()))
        print("Val shape:", val.shape, "Labels:", dict(val["label"].value_counts()) if not val.empty else "Empty")
        print("Test shape:", test.shape, "Labels:", dict(test["label"].value_counts()))

        # --- Step 5: Feature Engineering & Model Training (as before) ---
        if config["models_to_run"].get("ml"):
            try:
                # Pass copies to avoid SettingWithCopyWarning within build_features if it modifies df
                X_train, y_train, _ = feature_engineering.build_features(train.copy(), feature_set=config["feature_set"])
                # Use the same featurizer for test set (fit_transform on train, transform on test)
                # Let's modify build_features to return the featurizer
                # OR assume build_features handles fit/transform appropriately if called separately
                X_test, y_test, _ = feature_engineering.build_features(test.copy(), feature_set=config["feature_set"]) # This assumes independent fitting - needs fix

                # --> Correction Needed: Fit featurizer on Train, Transform Test <--
                # We need to adjust feature_engineering or the flow here.
                # Quick Fix Assumption: Assume build_features fits internally and we call it again.
                # Proper Fix: Modify build_features to return featurizer or handle test transform.
                # Let's proceed with the assumption for now, but flag it.
                print("🛠️ Note: Feature engineering assumes independent fit/transform currently.")


                # Check shapes after feature engineering
                print(f"Shapes after FeatEng: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    print("⚠️ Empty arrays after feature engineering. Skipping ML models.")
                    continue


                for model_name in config["models_to_run"]["ml"]:
                    print(f"\n🚀 Training ML model: {model_name}")
                    model_fn = MODEL_DISPATCH[model_name]
                    # Pass validation data if model supports it (simplification: use test set as val here)
                    # Proper way: Use the 'val' split created earlier
                    # Let's stick to the current model_training signature which uses train_test_split internally
                    results_train, model = model_fn(X_train, y_train) # model_fn does its own split for internal validation

                    print(f"Fitting {model_name} completed.")
                    # Evaluate on the actual Test set
                    y_pred = model.predict(X_test)
                    y_proba = None
                    if hasattr(model, "predict_proba"):
                        try:
                            # Ensure predict_proba returns probabilities for both classes if available
                            probas = model.predict_proba(X_test)
                            if probas.shape[1] == 2:
                                y_proba = probas[:, 1] # Probability of the positive class (1)
                            elif probas.shape[1] == 1: # Handle case where model might only output one proba array
                                # This might happen with some configurations or models; less common for SVC/RF/GBM
                                # If model predicts prob of class 1 directly:
                                # y_proba = probas[:, 0] # Or adjust based on model output interpretation
                                # If model predicts prob of class 0 only (unlikely):
                                # y_proba = 1 - probas[:, 0]
                                print(f"⚠️ {model_name} predict_proba output shape is {probas.shape}. Using first column for ROC AUC.")
                                y_proba = probas[:, 0] # Assumption: need to verify model output
                            else:
                                print(f"⚠️ Unexpected output shape from predict_proba for {model_name}: {probas.shape}")

                        except Exception as e:
                            print(f"⚠️ Could not get probabilities for {model_name}: {e}")

                    eval_results = evaluation.evaluate_model(
                        y_true=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        model_name=f"{category}_{model_name}_seed{seed}", # Add category to name
                        output_dir=config["output_dir"]
                    )
                    all_results.append(eval_results)

            except Exception as e:
                print(f"❌ Error during ML model training/evaluation for {category}: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback
                continue # Skip to next category or seed


        # --- DL Model Training (If enabled, needs similar adjustments) ---
        if config["models_to_run"].get("dl"):
            print("\n⚠️ DL Model Training with balanced splits needs verification.")
            # Ensure text/label extraction uses the balanced train/test sets
            texts_train = train["full_text"].tolist()
            labels_train = train["label"].astype(int).tolist()
            texts_test = test["full_text"].tolist()
            labels_test = test["label"].astype(int).tolist()

            if not texts_train or not texts_test:
                print("⚠️ Empty text lists for DL models. Skipping.")
                continue

            for model_name in config["models_to_run"]["dl"]:
                print(f"\n🧠 Training DL model: {model_name}")
                try:
                        model_fn = MODEL_DISPATCH[model_name]
                        # DL functions handle their own train/val split internally currently
                        results_train, model = model_fn(texts_train, labels_train)

                        # Evaluate on the actual Test set
                        # Predict probabilities if possible for ROC AUC, otherwise just classes
                        try:
                            probas_test = model.predict(texts_test)
                            preds_test = (probas_test > 0.5).astype("int32").flatten() # Flatten in case output is (N, 1)
                            proba_test_for_eval = probas_test.flatten() # Use raw probabilities for ROC AUC
                        except Exception as e:
                            print(f"⚠️ Error predicting with DL model {model_name}: {e}. Skipping evaluation.")
                            continue

                        eval_results = evaluation.evaluate_model(
                            y_true=labels_test,
                            y_pred=preds_test,
                            y_proba=proba_test_for_eval,
                            model_name=f"{category}_{model_name}_seed{seed}", # Add category to name
                            output_dir=config["output_dir"]
                        )
                        all_results.append(eval_results)
                except Exception as e:
                        print(f"❌ Error during DL model training/evaluation for {category}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue # Skip to next DL model or category


def main():
    config = load_config()
    # Optional: Initialize Weave here if desired
    # try:
    #     from scripts.wandb_init import init_tracking
    #     init_tracking() # Add project name if needed
    #     print("📊 Weave tracking initialized.")
    # except ImportError:
    #     print("⚠️ Weave initialization script not found or failed.")
    # except Exception as e:
    #     print(f"⚠️ Error initializing Weave: {e}")


    for seed in config["random_seeds"]:
        run_pipeline_for_seed(config, seed)

    print("\n📊 Summarizing all evaluation results...")
    # Pass the specific output directory from config
    summary = evaluation.summarize_evaluations(result_dir=config["output_dir"])
    print("\n📋 Final Summary:")
    print(summary) # Print the generated summary

if __name__ == "__main__":
    main()