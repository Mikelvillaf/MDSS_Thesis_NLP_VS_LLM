# model_training.py

import numpy as np
import pandas as pd
import weave # Ensure weave is imported
from typing import Tuple, List, Optional, Dict, Union, Any 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

# --- TensorFlow / Keras Imports ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM,
    Dense, Dropout, concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- ML Models ---

@weave.op()
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, class_weight=None, seed=42, **kwargs) -> RandomForestClassifier:
    print(f"   Training RandomForestClassifier on {X_train.shape[0]} samples...")
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
    rf_class_weight = kwargs.get('class_weight', class_weight) 
    
    print(f"   RF Params: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, class_weight='{rf_class_weight}'")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=seed, 
        class_weight=rf_class_weight, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete.")
    return model

@weave.op()
def train_svm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None, seed=42, **kwargs) -> LinearSVC:
    print(f"   Training LinearSVC on {X_train.shape[0]} samples...")
    C_pow = kwargs.get('C_pow', -1) 
    C_value = 10**float(C_pow)
    loss = kwargs.get('loss', 'squared_hinge') 
    svm_max_iter = kwargs.get('max_iter', 2000) 
    svm_class_weight = kwargs.get('class_weight', class_weight)
    
    print(f"   SVM Params: C={C_value:.5f}, loss='{loss}', max_iter={svm_max_iter}, class_weight='{svm_class_weight}'")

    model = LinearSVC(
        C=C_value,
        loss=loss,
        class_weight=svm_class_weight, 
        random_state=seed, 
        dual='auto', 
        max_iter=svm_max_iter
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete.")
    return model

@weave.op()
def train_gbm(X_train: np.ndarray, y_train: np.ndarray, class_weight=None, seed=42, **kwargs) -> GradientBoostingClassifier:
    if class_weight or kwargs.get('class_weight'): 
        print("   ⚠️ Warning: GradientBoostingClassifier doesn't directly use 'class_weight'. Training proceeds.")
        
    print(f"   Training GradientBoostingClassifier on {X_train.shape[0]} samples...")
    n_estimators = kwargs.get('n_estimators', 100)
    learning_rate_pow = kwargs.get('learning_rate_pow', -1) 
    learning_rate_value = 10**float(learning_rate_pow)
    max_depth = kwargs.get('max_depth', 3)
    subsample = kwargs.get('subsample', 1.0)

    print(f"   GBM Params: n_estimators={n_estimators}, learning_rate={learning_rate_value:.4f}, max_depth={max_depth}, subsample={subsample}")
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate_value,
        max_depth=max_depth,
        subsample=subsample,
        random_state=seed
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete.")
    return model


# --- Core DL Model Training Function (Handles Text-Only and Hybrid) ---
@weave.op()
def train_dl_model_internal(
    model_type: str, 
    texts_train: List[str],
    labels_train: np.ndarray,
    structured_train: Optional[np.ndarray], 
    num_structured_features: int, 
    class_weight: Optional[Dict[int, float]], 
    seed: int,
    validation_data: Optional[Tuple[Union[List[np.ndarray], np.ndarray], np.ndarray]] = None,
    **dl_hyperparams_from_main 
) -> Tuple[tf.keras.Model, Tokenizer]:

    is_hybrid = structured_train is not None and num_structured_features > 0
    input_mode = "Hybrid" if is_hybrid else "Text-Only"
    print(f"   Training Internal DL model: {model_type.upper()} ({input_mode})")
    tf.random.set_seed(seed) 

    max_words_val = int(dl_hyperparams_from_main.get('max_words', 10000))
    max_len_val = int(dl_hyperparams_from_main.get('max_len', 300))
    embedding_dim_val = int(dl_hyperparams_from_main.get('embedding_dim', 64))
    epochs_val = int(dl_hyperparams_from_main.get('epochs', 5))
    batch_size_val = int(dl_hyperparams_from_main.get('batch_size', 64))
    
    conv1d_filters_val = int(dl_hyperparams_from_main.get('conv1d_filters', 64))
    lstm_units_val = int(dl_hyperparams_from_main.get('lstm_units', 64)) 
    dense_units_val = int(dl_hyperparams_from_main.get('dense_units', 32))
    dropout_rate_val = float(dl_hyperparams_from_main.get('dropout_cat', 0.5)) 
    
    learning_rate_pow_val = float(dl_hyperparams_from_main.get('learning_rate_pow', -3)) 
    actual_learning_rate = 10**learning_rate_pow_val

    print(f"   DL Effective Params: max_words={max_words_val}, max_len={max_len_val}, embed_dim={embedding_dim_val}, "
          f"epochs={epochs_val}, batch={batch_size_val}, conv_filters={conv1d_filters_val}, "
          f"lstm_units={lstm_units_val if model_type == 'rcnn' else 'N/A'}, dense_units={dense_units_val}, "
          f"dropout={dropout_rate_val}, lr={actual_learning_rate:.6f}")

    tokenizer = Tokenizer(num_words=max_words_val, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    padded_train = pad_sequences(sequences_train, maxlen=max_len_val)

    keras_train_inputs: Union[np.ndarray, List[np.ndarray]]
    if is_hybrid:
        if structured_train is None or structured_train.shape[1] != num_structured_features or structured_train.shape[0] != len(labels_train):
             raise ValueError(f"Hybrid mode error: structured_train invalid (shape {structured_train.shape if structured_train is not None else 'None'}), expected ({len(labels_train)}, {num_structured_features})")
        keras_train_inputs = [padded_train, structured_train]
    else:
        keras_train_inputs = padded_train

    keras_val_data = validation_data 
    if keras_val_data:
        if not (isinstance(keras_val_data, tuple) and len(keras_val_data) == 2):
             print("   ⚠️ Warning: Invalid format received for validation_data. Disabling Keras validation.")
             keras_val_data = None

    tf.keras.backend.clear_session() 
    text_input = Input(shape=(max_len_val,), name='text_input')
    model_inputs = [text_input]

    emb = Embedding(input_dim=max_words_val, output_dim=embedding_dim_val, name='Embedding_Layer')(text_input) 
    
    if model_type == 'cnn':
        text_features = Conv1D(conv1d_filters_val, 5, activation='relu', name='Conv1D_Layer')(emb)
        text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
    elif model_type == 'rcnn':
        lstm_out = LSTM(lstm_units_val, return_sequences=True, name='LSTM_Layer')(emb)
        text_features = Conv1D(conv1d_filters_val, 5, activation='relu', name='Conv1D_Layer')(lstm_out)
        text_features = GlobalMaxPooling1D(name='GlobalMaxPooling_Layer')(text_features)
    else:
        raise ValueError(f"Invalid internal model_type: {model_type}")

    final_features = text_features
    if is_hybrid:
        structured_input = Input(shape=(num_structured_features,), name='structured_input')
        model_inputs.append(structured_input)
        structured_processed = Dense(16, activation='relu', name='Structured_Dense')(structured_input) 
        final_features = concatenate([text_features, structured_processed], name='concatenate_features')
    
    hidden_dense = Dense(dense_units_val, activation='relu', name='Dense_Hidden_Layer')(final_features)
    dropout_layer = Dropout(dropout_rate_val, name='Dropout_Layer')(hidden_dense)
    output = Dense(1, activation='sigmoid', name='Dense_Output_Layer')(dropout_layer)

    model = Model(inputs=model_inputs, outputs=output, name=f"{input_mode}_{model_type.upper()}_Model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=actual_learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(f"   Starting model.fit (Epochs: {epochs_val}, Batch: {batch_size_val})...")
    monitor_metric = 'val_accuracy' if keras_val_data else 'accuracy' 
    # Ensure restore_best_weights=True so Keras handles it.
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=3, verbose=1, mode='max', restore_best_weights=True)

    history = model.fit(
        keras_train_inputs, 
        labels_train,
        epochs=epochs_val,
        batch_size=batch_size_val,
        validation_data=keras_val_data,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1 
    )
    print("   model.fit completed.")

    # Corrected logic for checking early stopping and evaluating
    if keras_val_data:
        # Check if early stopping was actually triggered and if best_epoch is available
        stopped_epoch = early_stopping.stopped_epoch
        best_epoch_info = ""
        if hasattr(early_stopping, 'best_epoch') and early_stopping.best_epoch is not None : # Check if best_epoch is available
            # In Keras, best_epoch is 0-indexed, so add 1 for human-readable epoch number
            best_epoch_info = f" (best recorded at epoch {early_stopping.best_epoch + 1})" 

        if stopped_epoch > 0 : # 0 means training completed all epochs without stopping early
             print(f"   Early stopping triggered{best_epoch_info}. Model has weights from the best epoch if restore_best_weights=True.")
        else:
             print(f"   Training finished all epochs{best_epoch_info}. Model has weights from the best epoch if restore_best_weights=True and improvement was found, otherwise final epoch.")
        
        # Evaluate the model. If restore_best_weights=True, it already has the best weights.
        val_loss, val_acc = model.evaluate(keras_val_data[0], keras_val_data[1], verbose=0, batch_size=batch_size_val*2)
        print(f"   Final Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    else:
        # No validation data, report final training metrics
        final_train_loss = history.history['loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        print(f"   Final Training Loss: {final_train_loss:.4f}, Acc: {final_train_acc:.4f} (no validation data used)")

    return model, tokenizer


# --- Wrapper functions calling the internal trainer ---
@weave.op()
def train_cnn(
    texts_train: List[str],
    labels_train: np.ndarray,
    structured_train: Optional[np.ndarray],
    num_structured_features: int,
    class_weight: Optional[Dict[int, float]],
    seed: int,
    validation_data: Optional[Tuple[Union[List[np.ndarray], np.ndarray], np.ndarray]] = None,
    **dl_model_params 
    ) -> Tuple[tf.keras.Model, Tokenizer]:
    """Trains CNN model via internal function, passing all specific DL parameters."""
    return train_dl_model_internal(
        model_type='cnn',
        texts_train=texts_train, labels_train=labels_train,
        structured_train=structured_train,
        num_structured_features=num_structured_features,
        class_weight=class_weight, seed=seed,
        validation_data=validation_data,
        **dl_model_params 
    )

@weave.op()
def train_rcnn(
    texts_train: List[str],
    labels_train: np.ndarray,
    structured_train: Optional[np.ndarray],
    num_structured_features: int,
    class_weight: Optional[Dict[int, float]],
    seed: int,
    validation_data: Optional[Tuple[Union[List[np.ndarray], np.ndarray], np.ndarray]] = None,
    **dl_model_params 
    ) -> Tuple[tf.keras.Model, Tokenizer]:
    """Trains RCNN model via internal function, passing all specific DL parameters."""
    return train_dl_model_internal(
        model_type='rcnn',
        texts_train=texts_train, labels_train=labels_train,
        structured_train=structured_train,
        num_structured_features=num_structured_features,
        class_weight=class_weight, seed=seed,
        validation_data=validation_data,
        **dl_model_params 
    )