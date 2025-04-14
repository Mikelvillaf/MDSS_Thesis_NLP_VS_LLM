# model_training.py

import numpy as np
import pandas as pd
import weave
from typing import Tuple, Literal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

@weave.op()
def train_random_forest(X: np.ndarray, y: np.ndarray) -> Tuple[dict, object]:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True), model

@weave.op()
def train_svm(X: np.ndarray, y: np.ndarray) -> Tuple[dict, object]:
    model = SVC(kernel="linear", probability=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True), model

@weave.op()
def train_gbm(X: np.ndarray, y: np.ndarray) -> Tuple[dict, object]:
    model = GradientBoostingClassifier(n_estimators=100)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True), model

@weave.op()
def train_cnn(texts: list, labels: list, max_words=10000, max_len=300) -> Tuple[dict, object]:
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, stratify=labels)

    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=max_len))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=2)])

    preds = (model.predict(X_val) > 0.5).astype("int32")
    return classification_report(y_val, preds, output_dict=True), model

@weave.op()
def train_rcnn(texts: list, labels: list, max_words=10000, max_len=300) -> Tuple[dict, object]:
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)

    X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, stratify=labels)

    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=max_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=2)])

    preds = (model.predict(X_val) > 0.5).astype("int32")
    return classification_report(y_val, preds, output_dict=True), model