# tune_hyperparameters.py

import os
import yaml
import numpy as np
import pandas as pd
import random
import time
import warnings
import traceback
from typing import Dict, Any, Optional, List, Union 

# Optuna for hyperparameter tuning
import optuna
from optuna.trial import TrialState 

# Project script imports
from scripts.utils import load_config
from scripts.data_loader import load_reviews, identify_label_candidates, create_balanced_temporal_splits, load_and_clean_metadata
from scripts import preprocessing
from scripts.feature_engineering import fit_feature_extractor, transform_features 

# ML Model imports 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# DL Model imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, concatenate)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    print("TensorFlow imported successfully.")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. Skipping DL model tuning.")
    TF_AVAILABLE = False
    Tokenizer = object; EarlyStopping = object # Dummy classes

# --- Configuration ---
CATEGORY_TO_TUNE = 'CDs_and_Vinyl' 
MODELS_TO_TUNE = ['svm', 'random_forest', 'gradient_boosting']
if TF_AVAILABLE:
     MODELS_TO_TUNE.extend(['cnn', 'rcnn']) 
     
# --- Adjusted Trial Counts ---
N_TRIALS_ML = 30       # Trials for SVM, RF, GB
N_TRIALS_CNN = 15      # Trials for CNN
N_TRIALS_RCNN = 15     # Trials for RCNN (already reduced)

SEED = 42      
OUTPUT_HP_FILE = 'configs/hyperparameters.yaml' 
CONFIG_FILE = 'configs/experiment_config.yaml' 

# --- Set Seed ---
def set_seed_tune(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE: tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed); print(f"Tuning Seed set to {seed}")

# --- Data Preparation & Feature Engineering (No FastText) ---
def prepare_data_and_features(config: Dict[str, Any], category: str, seed: int) -> Optional[Dict[str, Any]]:
    """Loads, preprocesses, splits data, and engineers features for tuning."""
    print(f"\n--- Preparing Data & Features for Category: {category} ---")
    try:
        # Steps 1-4 (Load, Preprocess, Label, Split)
        data_path = config.get("data_path_template", "data/{category}.jsonl").format(category=category)
        metadata_path = config.get("metadata_path_template", "data/meta_{category}.jsonl").format(category=category)
        year_range = config.get("year_range"); df_raw = load_reviews(filepath=data_path, year_range=year_range, seed=seed)
        if df_raw.empty: print(f"⚠️ No data loaded."); return None
        meta_df = load_and_clean_metadata(metadata_path); print(f"Metadata loaded: {'Yes' if not meta_df.empty else 'No'}")
        df_processed = preprocessing.preprocess_reviews(df_raw, metadata_df=meta_df)
        if df_processed.empty: print(f"⚠️ Data empty after preprocessing."); return None
        df_labeled_pool = identify_label_candidates(df_processed, config)
        if df_labeled_pool.empty: print(f"⚠️ No labeled candidates found."); return None
        train_df, val_df, test_df = create_balanced_temporal_splits(df_labeled_pool, config, seed)
        if train_df.empty or val_df.empty: print(f"❌ Train ({len(train_df)}) or Val ({len(val_df)}) split empty."); return None
        if train_df['label'].nunique() < 2 or val_df['label'].nunique() < 2: print(f"❌ Train/Val split lacks both classes."); return None
        print(f"   Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Step 5: Feature Engineering (ML)
        print("\n   Engineering ML Features..."); ml_feature_set = config.get("feature_set", "hybrid")
        text_max_features = config.get("text_max_features", 1000)
        featurizer, _ = fit_feature_extractor(train_df, feature_set=ml_feature_set, text_max_features=text_max_features)
        X_train_ml, y_train_ml = transform_features(train_df, featurizer); X_val_ml, y_val_ml = transform_features(val_df, featurizer)
        print(f"   ML Feature Shapes: Train={X_train_ml.shape}, Val={X_val_ml.shape}")

        # Step 6: Feature Engineering (DL Prep)
        print("\n   Preparing DL Data..."); texts_train = train_df["full_text"].tolist(); labels_train = train_df["label"].astype(int).values
        texts_val = val_df["full_text"].tolist(); labels_val = val_df["label"].astype(int).values
        tokenizer = None; actual_vocab_size = 0; X_train_structured_dl = None; X_val_structured_dl = None
        
        if TF_AVAILABLE: 
            dl_max_words = config.get("dl_max_words", 10000) 
            tokenizer = Tokenizer(num_words=dl_max_words, oov_token="<OOV>")
            tokenizer.fit_on_texts(texts_train)
            actual_vocab_size = len(tokenizer.word_index) + 1 
            print(f"   DL Tokenizer fitted (Max words: {dl_max_words}, Actual indexed: {actual_vocab_size -1})")
            
            dl_feature_set = config.get("dl_feature_set", "hybrid"); dl_num_structured = config.get("dl_num_structured_features", 5)
            if dl_feature_set == 'hybrid':
                if ml_feature_set == 'nlp': print(f"   ⚠️ Cannot extract structured feats for Hybrid DL.")
                elif X_train_ml is None or X_train_ml.shape[1] < dl_num_structured: print(f"   ⚠️ Not enough ML feats ({X_train_ml.shape[1] if X_train_ml is not None else 0}) for Hybrid DL.")
                else:
                    print(f"   Extracting first {dl_num_structured} columns for Hybrid DL...")
                    X_train_structured_dl = X_train_ml[:, :dl_num_structured]
                    X_val_structured_dl = X_val_ml[:, :dl_num_structured] if X_val_ml.shape[0] > 0 else np.empty((0, dl_num_structured))

        print("--- Data & Features Ready ---")
        return {"X_train_ml": X_train_ml, "y_train_ml": y_train_ml, "X_val_ml": X_val_ml, "y_val_ml": y_val_ml,
                "texts_train": texts_train, "labels_train": labels_train, "texts_val": texts_val, "labels_val": labels_val,
                "tokenizer": tokenizer, "actual_vocab_size": actual_vocab_size, 
                "X_train_structured_dl": X_train_structured_dl, "X_val_structured_dl": X_val_structured_dl,
                "config": config }
    except Exception as e: print(f"❌ Error during data prep/FE: {e}"); traceback.print_exc(); return None

# --- Optuna Objective Function (Updated Search Spaces) ---
def objective(trial: optuna.trial.Trial, model_name: str, data: Dict[str, Any]) -> float:
    """Optuna objective function to train and evaluate a model."""
    print(f"\n--- Trial {trial.number} for {model_name} ---")
    X_train_ml, y_train_ml, X_val_ml, y_val_ml = data["X_train_ml"], data["y_train_ml"], data["X_val_ml"], data["y_val_ml"]
    texts_train, labels_train, texts_val, labels_val = data["texts_train"], data["labels_train"], data["texts_val"], data["labels_val"]
    tokenizer, actual_vocab_size = data["tokenizer"], data["actual_vocab_size"] 
    X_train_structured_dl, X_val_structured_dl = data["X_train_structured_dl"], data["X_val_structured_dl"]
    config = data["config"]
    class_weight_dict = None; unique_classes = np.unique(y_train_ml); 
    if len(unique_classes) > 1: weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_ml); class_weight_dict = dict(enumerate(weights))
    val_accuracy = 0.0 

    try:
        # === ML Models (narrow Search Spaces for speed) ===
        if model_name == 'svm':
            hp_c_pow = trial.suggest_int("C_pow", -3, 1) # C from 0.001 to 10
            hp_c = 10**hp_c_pow 
            hp_loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
            hp_class_weight_svm = trial.suggest_categorical("class_weight", [None, "balanced"]) 
            hp_max_iter = trial.suggest_categorical("max_iter", [1000, 2000, 3000]) # Reduced max
            print(f"   Params: C={hp_c:.5f}, loss={hp_loss}, cw={hp_class_weight_svm}, iter={hp_max_iter}")
            model = LinearSVC(C=hp_c, loss=hp_loss, class_weight=hp_class_weight_svm, max_iter=hp_max_iter, random_state=SEED, dual='auto')
            model.fit(X_train_ml, y_train_ml); y_pred_val = model.predict(X_val_ml); val_accuracy = accuracy_score(y_val_ml, y_pred_val)
        
        elif model_name == 'random_forest':
            hp_n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50) # Narrowed range
            hp_max_depth = trial.suggest_int("max_depth", 10, 30, step=5)       # Narrowed range
            hp_min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=2) 
            hp_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, step=1)    
            hp_class_weight_rf = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]) 
            print(f"   Params: n_est={hp_n_estimators}, depth={hp_max_depth}, split={hp_min_samples_split}, leaf={hp_min_samples_leaf}, cw={hp_class_weight_rf}")
            model = RandomForestClassifier(n_estimators=hp_n_estimators, max_depth=hp_max_depth, min_samples_split=hp_min_samples_split, min_samples_leaf=hp_min_samples_leaf, class_weight=hp_class_weight_rf, random_state=SEED, n_jobs=-1)
            model.fit(X_train_ml, y_train_ml); y_pred_val = model.predict(X_val_ml); val_accuracy = accuracy_score(y_val_ml, y_pred_val)
            
        elif model_name == 'gradient_boosting':
            hp_n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50) # Narrowed range
            hp_learning_rate_pow = trial.suggest_int("learning_rate_pow", -3, -1); hp_learning_rate = 10**hp_learning_rate_pow # LR=0.001 to 0.1
            hp_max_depth = trial.suggest_int("max_depth", 2, 5, step=1)        # Narrowed range
            hp_subsample = trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]) 
            print(f"   Params: n_est={hp_n_estimators}, lr={hp_learning_rate:.4f}, depth={hp_max_depth}, subsample={hp_subsample}")
            model = GradientBoostingClassifier(n_estimators=hp_n_estimators, learning_rate=hp_learning_rate, max_depth=hp_max_depth, subsample=hp_subsample, random_state=SEED)
            model.fit(X_train_ml, y_train_ml); y_pred_val = model.predict(X_val_ml); val_accuracy = accuracy_score(y_val_ml, y_pred_val)
        
        # === DL Models (Focusing DL search based on previous good performance) ===
        elif model_name in ['cnn', 'rcnn'] and TF_AVAILABLE:
            # --- Suggest Hyperparameters ---
            # Focus embedding_dim based on previous good CNN trials (around 128, 256)
            hp_embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256]) 
            # Focus conv_filters based on previous good CNN trials (around 128, 256)
            hp_conv1d_filters = trial.suggest_categorical("conv1d_filters", [64, 128, 256]) 
            hp_lstm_units = 0 
            if model_name == 'rcnn': 
                hp_lstm_units = trial.suggest_categorical("lstm_units", [64, 128]) # Keeping this range for now
            # Focus dense_units based on previous good CNN trials (around 32, 64, 128)
            hp_dense_units = trial.suggest_categorical("dense_units", [32, 64, 128]) 
            hp_dropout = trial.suggest_categorical("dropout_cat", [0.3, 0.4, 0.5]) # Focus mid-range
            # Focus learning_rate based on previous good CNN trials (around 1e-4, 1e-3)
            hp_learning_rate_pow = trial.suggest_int("learning_rate_pow", -4, -3); hp_learning_rate = 10**hp_learning_rate_pow
            
            print(f"   Params: emb={hp_embedding_dim}, conv={hp_conv1d_filters}, lstm={hp_lstm_units if model_name=='rcnn' else 'N/A'}, dense={hp_dense_units}, drop={hp_dropout}, lr={hp_learning_rate:.5f}")

            # --- Prepare Data ---
            dl_max_len = config.get("dl_max_len", 300) 
            sequences_train = tokenizer.texts_to_sequences(texts_train); padded_train = pad_sequences(sequences_train, maxlen=dl_max_len)
            sequences_val = tokenizer.texts_to_sequences(texts_val); padded_val = pad_sequences(sequences_val, maxlen=dl_max_len)
            keras_train_inputs = [padded_train]; keras_val_inputs = [padded_val]
            dl_feature_set = config.get("dl_feature_set", "hybrid"); dl_num_structured = config.get("dl_num_structured_features", 5)
            is_hybrid = (dl_feature_set == 'hybrid' and X_train_structured_dl is not None and X_val_structured_dl is not None and 
                        X_train_structured_dl.shape[1] == dl_num_structured and X_val_structured_dl.shape[1] == dl_num_structured)
            if is_hybrid: print("   (Running DL in Hybrid mode)"); keras_train_inputs.append(X_train_structured_dl); keras_val_inputs.append(X_val_structured_dl)
            else: print("   (Running DL in Text-Only mode)")

            # --- Define Model ---
            tf.keras.backend.clear_session() 
            text_input = Input(shape=(dl_max_len,), name='text_input'); model_inputs = [text_input]
            emb_layer = Embedding( input_dim=actual_vocab_size, output_dim=hp_embedding_dim, trainable=True, name='Embedding_Layer')
            emb = emb_layer(text_input)
            if model_name == 'cnn':
                text_features = Conv1D(hp_conv1d_filters, 5, activation='relu', name='Conv1D_Layer')(emb)
                text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
            elif model_name == 'rcnn':
                lstm_out = LSTM(hp_lstm_units, return_sequences=True, name='LSTM_Layer')(emb)
                text_features = Conv1D(hp_conv1d_filters, 5, activation='relu', name='Conv1D_Layer')(lstm_out)
                text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
            final_features = text_features
            if is_hybrid:
                structured_input = Input(shape=(dl_num_structured,), name='structured_input'); model_inputs.append(structured_input)
                structured_processed = Dense(16, activation='relu', name='Structured_Dense')(structured_input) 
                final_features = concatenate([text_features, structured_processed], name='concatenate_features')
            hidden_dense = Dense(hp_dense_units, activation='relu', name='Dense_Hidden_Layer')(final_features)
            dropout_layer = Dropout(hp_dropout, name='Dropout_Layer')(hidden_dense)
            output = Dense(1, activation='sigmoid', name='Dense_Output_Layer')(dropout_layer)
            model = Model(inputs=model_inputs, outputs=output)
            
            # --- Compile & Train ---
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            epochs = config.get("dl_epochs", 5) 
            batch_size = config.get("dl_batch_size", 64)
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True) 
            history = model.fit(keras_train_inputs, labels_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(keras_val_inputs, labels_val), class_weight=class_weight_dict, 
                                callbacks=[early_stopping], verbose=0) 
            # --- Evaluate ---
            loss, val_accuracy = model.evaluate(keras_val_inputs, labels_val, verbose=0)
            print(f"   Trial DL Eval: Val Loss={loss:.4f}, Val Accuracy={val_accuracy:.4f}")

        else:
            print(f"   Model type '{model_name}' not recognized or TF not available.")
            raise optuna.exceptions.TrialPruned(f"Model type {model_name} not supported.")

        print(f"   Trial {trial.number} result: Validation Accuracy = {val_accuracy:.4f}")
        if np.isnan(val_accuracy) or np.isinf(val_accuracy): print("   Invalid accuracy, returning -1.0"); return -1.0
        return val_accuracy 

    except Exception as e:
        print(f"❌ Trial {trial.number} FAILED for {model_name}: {type(e).__name__} - {e}")
        traceback.print_exc(); return -1.0 

# --- Main Tuning Execution (Adjusted Trial Counts) ---
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn') 
    warnings.filterwarnings('ignore', category=RuntimeWarning) 
    
    set_seed_tune(SEED)
    
    print("Loading main configuration...")
    try: config = load_config(CONFIG_FILE)
    except Exception as e: print(f"❌ Failed to load config {CONFIG_FILE}: {e}"); exit()

    print(f"Preparing data for tuning category: {CATEGORY_TO_TUNE}")
    prepared_data = prepare_data_and_features(config, CATEGORY_TO_TUNE, SEED)
    if prepared_data is None: print(f"❌ Failed to prepare data. Exiting."); exit()
        
    best_hyperparameters = {}

    for model_to_tune in MODELS_TO_TUNE:
        print(f"\n===== Tuning Model: {model_to_tune.upper()} =====")
        
        # --- Use specific trial counts ---
        if model_to_tune in ['svm', 'random_forest', 'gradient_boosting']:
            n_trials_current = N_TRIALS_ML
        elif model_to_tune == 'cnn':
            n_trials_current = N_TRIALS_CNN
        elif model_to_tune == 'rcnn':
            n_trials_current = N_TRIALS_RCNN
        else: 
            n_trials_current = 10 # Fallback small number for any unexpected model

        print(f"   Running {n_trials_current} trials...")
        
        objective_with_data = lambda trial: objective(trial, model_to_tune, prepared_data)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED)) 
        
        try:
            study.optimize(objective_with_data, n_trials=n_trials_current, timeout=None) 
            print(f"\n--- Best Trial for {model_to_tune} ---")
            print(f"  Value (Validation Accuracy): {study.best_value:.4f}")
            print(f"  Params: ")
            best_params_safe = {}
            for key, value in study.best_params.items():
                if isinstance(value, np.integer): best_params_safe[key] = int(value)
                elif isinstance(value, np.floating): best_params_safe[key] = float(value)
                else: best_params_safe[key] = value
                print(f"    {key}: {best_params_safe[key]}") 
            best_hyperparameters[model_to_tune] = best_params_safe 
        except Exception as e:
            print(f"❌ Optuna study FAILED for model {model_to_tune}: {e}")
            traceback.print_exc(); best_hyperparameters[model_to_tune] = {"error": f"Tuning failed: {e}"}

    # --- Save Best Hyperparameters ---
    print("\n--- Saving Best Hyperparameters ---")
    try:
        output_dir = os.path.dirname(OUTPUT_HP_FILE)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(OUTPUT_HP_FILE, 'w') as f:
            yaml.dump(best_hyperparameters, f, default_flow_style=False, sort_keys=False) 
        print(f"✅ Best hyperparameters saved to: {OUTPUT_HP_FILE}")
    except Exception as e: print(f"❌ Failed to save hyperparameters to {OUTPUT_HP_FILE}: {e}")

    print("\n--- Hyperparameter Tuning Finished ---")