# model_training.py (Modified for Optional Hybrid DL Input)

import numpy as np
import pandas as pd
import weave
from typing import Tuple, List, Optional, Dict, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# --- TensorFlow / Keras Imports ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM,
    Dense, Dropout, concatenate, BatchNormalization # Added BatchNormalization as an example
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- ML Models (Unchanged) ---
@weave.op()
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    # (Function unchanged)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight, n_jobs=-1)
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training RF on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0))
    return report, model

@weave.op()
def train_svm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    # (Function unchanged)
    model = LinearSVC(class_weight=class_weight, random_state=42, dual='auto', max_iter=2000, C=0.1) # dual='auto' recommended
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training LinearSVC on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0))
    return report, model

@weave.op()
def train_gbm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None) -> Tuple[dict, object]:
    # (Function unchanged)
    if class_weight: print("Warning: GradientBoostingClassifier doesn't use class_weight. Ignoring.")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3)
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    print(f"   Training GBM on {X_t.shape[0]} samples, validating on {X_v.shape[0]}.")
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(f"   Internal validation report (on {len(y_v)} samples):")
    report = classification_report(y_v, preds, output_dict=True, zero_division=0)
    print(classification_report(y_v, preds, zero_division=0))
    return report, model


# --- Core DL Model Training Function (Handles Both Text-Only and Hybrid) ---

@weave.op()
def train_dl_model_internal(
    model_type: str, # 'cnn' or 'rcnn'
    # Text Data (Required)
    texts_train: List[str],
    labels_train: np.ndarray,
    # Structured Data (Optional - Provide for Hybrid)
    structured_train: Optional[np.ndarray] = None,
    num_structured_features: int = 0, # Default to 0 if not hybrid
    # Validation Data (Optional)
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    structured_val: Optional[np.ndarray] = None,
    # Model Hyperparameters
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    # Training Hyperparameters
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
) -> Tuple[tf.keras.Model, Tokenizer]:
    """
    Internal function to train a CNN or RCNN model.
    Handles both text-only and hybrid (text + structured) inputs based on
    whether structured_train is provided.
    """
    is_hybrid = structured_train is not None and num_structured_features > 0
    input_mode = "Hybrid" if is_hybrid else "Text-Only"
    print(f"   Training Internal DL model: {model_type.upper()} ({input_mode})")
    tf.random.set_seed(seed)

    # 1. Tokenization (Always based on text)
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    print(f"   Tokenizer fitted on {len(texts_train)} training texts.")

    # 2. Sequencing and Padding (Text Data)
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    padded_train = pad_sequences(sequences_train, maxlen=max_len)
    print(f"   Text training data shape after padding: {padded_train.shape}")

    # Prepare training inputs list/dict for model.fit
    keras_train_inputs = [padded_train] # Text input always present

    # 3. Validate and Prepare Structured Data (if Hybrid)
    if is_hybrid:
        if structured_train.shape[0] != padded_train.shape[0]:
            raise ValueError(f"Mismatch between text ({padded_train.shape[0]}) and structured ({structured_train.shape[0]}) train data sample counts!")
        if structured_train.shape[1] != num_structured_features:
            raise ValueError(f"Structured train data has {structured_train.shape[1]} features, but expected {num_structured_features}")
        print(f"   Structured training data shape: {structured_train.shape}")
        keras_train_inputs.append(structured_train) # Add structured input

    # 4. Prepare Validation Data
    keras_val_data = None
    use_val = False
    if texts_val and labels_val is not None:
        sequences_val = tokenizer.texts_to_sequences(texts_val)
        padded_val = pad_sequences(sequences_val, maxlen=max_len)
        keras_val_inputs = [padded_val] # Start with text

        if is_hybrid:
            # For hybrid, validation structured data MUST also be provided and valid
            if structured_val is not None and structured_val.shape[0] == padded_val.shape[0] and structured_val.shape[1] == num_structured_features:
                print(f"   Text validation data shape after padding: {padded_val.shape}")
                print(f"   Structured validation data shape: {structured_val.shape}")
                keras_val_inputs.append(structured_val)
                keras_val_data = (keras_val_inputs, labels_val)
                use_val = True
            else:
                print("   ⚠️ Hybrid mode: Validation structured data missing, invalid shape, or count mismatch. Not using Keras validation.")
        else:
            # For text-only, we only need text and labels
            if len(texts_val) == len(labels_val):
                 print(f"   Text validation data shape after padding: {padded_val.shape}")
                 keras_val_data = (keras_val_inputs, labels_val) # Only text input needed
                 use_val = True
            else:
                 print("   ⚠️ Text-only mode: Validation text/label count mismatch. Not using Keras validation.")

    if not use_val:
        print("   No complete validation data provided/usable for Keras training.")

    # 5. Model Definition (Functional API - Conditional Structure)
    text_input = Input(shape=(max_len,), name='text_input')
    model_inputs = [text_input] # Start with text input

    # Text processing branch (always present)
    emb = Embedding(max_words, embedding_dim, name='Embedding_Layer')(text_input)
    if model_type == 'cnn':
        text_features = Conv1D(64, 5, activation='relu', name='Conv1D_Layer')(emb)
        text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
    elif model_type == 'rcnn':
        lstm_out = LSTM(64, return_sequences=True, name='LSTM_Layer')(emb)
        text_features = Conv1D(64, 5, activation='relu', name='Conv1D_Layer')(lstm_out)
        text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
    else:
        raise ValueError("model_type must be 'cnn' or 'rcnn'")

    # --- Conditional Hybrid Branch ---
    if is_hybrid:
        structured_input = Input(shape=(num_structured_features,), name='structured_input')
        model_inputs.append(structured_input) # Add to model's input list

        # Process structured features (example: simple dense layer)
        structured_processed = Dense(16, activation='relu', name='Structured_Dense')(structured_input)
        # structured_processed = BatchNormalization(name='Structured_BN')(structured_processed) # Optional BN

        # Concatenate features
        combined_features = concatenate([text_features, structured_processed], name='concatenate_features')
        final_features = combined_features
    else:
        # Text-only: the features from the text branch are the final features
        final_features = text_features
    # --- End Conditional Branch ---

    # Common final layers
    hidden_dense = Dense(32, activation='relu', name='Dense_Hidden_Layer')(final_features)
    dropout_layer = Dropout(0.5, name='Dropout_Layer')(hidden_dense)
    output = Dense(1, activation='sigmoid', name='Dense_Output_Layer')(dropout_layer)

    # Create the model with the correct inputs
    model = Model(inputs=model_inputs, outputs=output, name=f"{input_mode}_{model_type.upper()}_Model")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary(line_length=120)

    # 6. Training
    print(f"   Starting model.fit (Epochs: {epochs}, Batch: {batch_size})...")
    monitor_metric = 'val_loss' if use_val else 'loss'
    history = model.fit(
        keras_train_inputs, # List of inputs [text] or [text, structured]
        labels_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=keras_val_data, # Will be None if use_val is False
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor=monitor_metric, patience=2, verbose=1, restore_best_weights=True)],
        verbose=1
    )
    print("   model.fit completed.")

    # Report final metrics based on restored best weights if validation was used
    if use_val:
        loss, acc = model.evaluate(keras_val_data[0], keras_val_data[1], verbose=0, batch_size=batch_size*2)
        print(f"   Restored best weights. Final Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")
    else:
        # Get last recorded training loss/acc if no validation
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        print(f"   Final Training Loss: {final_train_loss:.4f}, Training Accuracy: {final_train_acc:.4f}")

    return model, tokenizer


# --- Wrapper functions calling the internal trainer ---
# These wrappers maintain a consistent interface for main.py

@weave.op()
def train_cnn(
    # Required text args first
    texts_train: List[str],
    labels_train: np.ndarray,
    # Optional args (structured, val, hypers)
    structured_train: Optional[np.ndarray] = None,
    num_structured_features: int = 0,
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    structured_val: Optional[np.ndarray] = None,
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
    ) -> Tuple[tf.keras.Model, Tokenizer]:
    """Wrapper for training a CNN model (Text-Only or Hybrid)."""
    return train_dl_model_internal(
        model_type='cnn',
        texts_train=texts_train, labels_train=labels_train,
        structured_train=structured_train, num_structured_features=num_structured_features,
        texts_val=texts_val, labels_val=labels_val, structured_val=structured_val,
        max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight, seed=seed
    )

@weave.op()
def train_rcnn(
     # Required text args first
    texts_train: List[str],
    labels_train: np.ndarray,
    # Optional args (structured, val, hypers)
    structured_train: Optional[np.ndarray] = None,
    num_structured_features: int = 0,
    texts_val: Optional[List[str]] = None,
    labels_val: Optional[np.ndarray] = None,
    structured_val: Optional[np.ndarray] = None,
    max_words: int = 10000,
    max_len: int = 300,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42
    ) -> Tuple[tf.keras.Model, Tokenizer]:
    """Wrapper for training an RCNN model (Text-Only or Hybrid)."""
    return train_dl_model_internal(
        model_type='rcnn',
        texts_train=texts_train, labels_train=labels_train,
        structured_train=structured_train, num_structured_features=num_structured_features,
        texts_val=texts_val, labels_val=labels_val, structured_val=structured_val,
        max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight, seed=seed
    )