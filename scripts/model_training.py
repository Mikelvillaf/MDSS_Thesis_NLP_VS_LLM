# model_training.py

import numpy as np
import pandas as pd
import weave
from typing import Tuple, List, Optional, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
# Use LinearSVC for potentially faster SVM training on larger datasets
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Ensure necessary TF imports are here
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM, Input # Added Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- ML Models ---
# (train_random_forest and train_gbm remain unchanged from previous correct version)

@weave.op()
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    """Trains RF, uses internal split for simplicity now. Returns report on internal val, model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight, n_jobs=-1) # Use n_jobs=-1 for parallelism
    # Internal split for hyperparameter tuning/early stopping simulation if needed, not ideal
    # Stratify ensures proportion is maintained in split
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training RF on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    # Note: report is on internal validation set, not the main val set from main.py
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0)) # Print formatted report too
    return report, model

@weave.op()
def train_svm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    """
    Trains SVM (using LinearSVC for efficiency), uses internal split.
    Returns report on internal val, model.
    Note: LinearSVC doesn't have predict_proba by default. For ROC AUC, SVC(probability=True) is needed, but slower.
    """
    # Using LinearSVC for potentially better performance on larger datasets
    # Note: `dual='auto'` is default and usually fine. `dual=False` might be needed if n_samples > n_features.
    # Increased max_iter as LinearSVC sometimes needs more iterations to converge
    model = LinearSVC(class_weight=class_weight, random_state=42, dual='auto', max_iter=2000, C=0.1) # Added C=0.1 for regularization, adjust if needed
    # SVC(kernel='linear', probability=True) is needed for predict_proba / ROC AUC
    # If ROC AUC is critical, switch back but expect longer training:
    # model = SVC(kernel="linear", probability=True, class_weight=class_weight, random_state=42, C=0.1)

    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training LinearSVC on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0))
    # If using SVC with probability=True, calculate ROC AUC on internal val set here if desired
    # if hasattr(model, "predict_proba"): try: roc_auc = roc_auc_score(y_v, model.predict_proba(X_v)[:, 1]); print(f"Internal Val ROC AUC: {roc_auc:.4f}") except: pass
    return report, model

@weave.op()
def train_gbm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    """Trains GBM, uses internal split. Returns report on internal val, model."""
    if class_weight:
        print("Warning: GradientBoostingClassifier doesn't directly use class_weight. Ignoring.")

    # Consider adding learning_rate or n_estimators tuning later
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3)
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training GBM on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0))
    # GBM also has predict_proba
    # if hasattr(model, "predict_proba"): try: roc_auc = roc_auc_score(y_v, model.predict_proba(X_v)[:, 1]); print(f"Internal Val ROC AUC: {roc_auc:.4f}") except: pass
    return report, model


# --- DL Models (Refactored) ---

@weave.op()
def train_dl_model(
    model_type: str, # 'cnn' or 'rcnn'
    texts_train: List[str],
    labels_train: np.ndarray,
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
) -> Tuple[tf.keras.Model, Tokenizer]:
    """
    Trains a CNN or RCNN model, handling tokenization and returning the fitted model and tokenizer.
    """
    print(f"   Training DL model: {model_type.upper()}")
    tf.random.set_seed(seed)

    # 1. Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    print(f"   Tokenizer fitted on {len(texts_train)} training texts.")

    # 2. Sequencing and Padding
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    padded_train = pad_sequences(sequences_train, maxlen=max_len)
    print(f"   Training data shape after padding: {padded_train.shape}")

    validation_data_keras = None
    if texts_val and labels_val is not None and len(texts_val) == len(labels_val):
        sequences_val = tokenizer.texts_to_sequences(texts_val)
        padded_val = pad_sequences(sequences_val, maxlen=max_len)
        validation_data_keras = (padded_val, labels_val)
        print(f"   Validation data shape after padding: {padded_val.shape}")
    else:
        print("   No validation data provided for Keras training.")

    # 3. Model Definition (Corrected Layer Names)
    model = Sequential(name=f"{model_type.upper()}_Model")
    model.add(Input(shape=(max_len,), name='Input_Layer')) # Explicit Input layer is good practice
    model.add(Embedding(max_words, embedding_dim, name='Embedding_Layer')) # Removed input_length

    if model_type == 'cnn':
        model.add(Conv1D(64, 5, activation='relu', name='Conv1D_Layer'))
        model.add(GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')) # <-- Renamed
    elif model_type == 'rcnn':
        model.add(LSTM(64, return_sequences=True, name='LSTM_Layer'))
        model.add(Conv1D(64, 5, activation='relu', name='Conv1D_Layer'))
        model.add(GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')) # <-- Renamed
    else:
        raise ValueError("model_type must be 'cnn' or 'rcnn'")

    model.add(Dense(32, activation='relu', name='Dense_Hidden_Layer')) # <-- Renamed (optional, consistency)
    model.add(Dropout(0.5, name='Dropout_Layer')) # <-- Renamed (optional, consistency)
    model.add(Dense(1, activation='sigmoid', name='Dense_Output_Layer')) # <-- Renamed (optional, consistency)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary(line_length=100) # Print model summary with adjusted width

    # 4. Training
    print(f"   Starting model.fit (Epochs: {epochs}, Batch: {batch_size})...")
    # Use validation data for early stopping if available
    monitor_metric = 'val_loss' if validation_data_keras else 'loss'
    history = model.fit(
        padded_train,
        labels_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_keras,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor=monitor_metric, patience=2, verbose=1, restore_best_weights=True)],
        verbose=1 # Show progress per epoch
    )
    print("   model.fit completed.")

    # Log final validation metrics if available
    if validation_data_keras:
        # Evaluate using the restored best weights
        loss, acc = model.evaluate(validation_data_keras[0], validation_data_keras[1], verbose=0, batch_size=batch_size*2) # Use larger batch for eval
        print(f"   Restored best weights. Final Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")
    else:
        # Log final training loss/acc if no validation
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        print(f"   Final Training Loss: {final_train_loss:.4f}, Training Accuracy: {final_train_acc:.4f}")


    return model, tokenizer

# --- Wrapper functions to match MODEL_DISPATCH ---

@weave.op()
def train_cnn(
    texts_train: List[str],
    labels_train: np.ndarray,
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
    ) -> Tuple[tf.keras.Model, Tokenizer]: # Return model and tokenizer
    return train_dl_model(
        model_type='cnn',
        texts_train=texts_train, labels_train=labels_train,
        texts_val=texts_val, labels_val=labels_val,
        max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight, seed=seed
        )

@weave.op()
def train_rcnn(
    texts_train: List[str],
    labels_train: np.ndarray,
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
    ) -> Tuple[tf.keras.Model, Tokenizer]: # Return model and tokenizer
    return train_dl_model(
        model_type='rcnn',
        texts_train=texts_train, labels_train=labels_train,
        texts_val=texts_val, labels_val=labels_val,
        max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight, seed=seed
        ) 