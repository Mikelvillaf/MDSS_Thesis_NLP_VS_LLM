# main.py

import os
import numpy as np
from scripts.utils import load_config
from scripts import data_loader, label_generation, preprocessing, feature_engineering, model_training, evaluation

import random
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

MODEL_DISPATCH = {
    "random_forest": model_training.train_random_forest,
    "svm": model_training.train_svm,
    "gradient_boosting": model_training.train_gbm,
    "cnn": model_training.train_cnn,
    "rcnn": model_training.train_rcnn
}

def run_pipeline_for_seed(config, seed):
    set_seed(seed)

    for category in config["categories"]:
        print(f"\n📦 Running category: {category} | Seed: {seed}")
        path = f"data/{category}.jsonl.gz"

        df = data_loader.load_reviews(
            filepath=path,
            year_range=config["year_range"],
            max_rows=config["max_reviews"],
            seed=seed
        )

        df = preprocessing.preprocess_reviews(df)  # timestamp → year
        train, val, test = data_loader.temporal_split(
            df,
            train_years=config["temporal_split"]["train_years"],
            val_year=config["temporal_split"]["val_year"],
            test_year=config["temporal_split"]["test_year"]
        )

        # 💡 Label each split independently
        top_p = config["labeling"]["top_percentile"]
        bottom_p = config["labeling"]["bottom_percentile"]
        train = label_generation.generate_labels(train, top_percentile=top_p, bottom_percentile=bottom_p)
        val = label_generation.generate_labels(val, top_percentile=top_p, bottom_percentile=bottom_p)
        test = label_generation.generate_labels(test, top_percentile=top_p, bottom_percentile=bottom_p)

        print("✅ Before building features:")
        print("Train shape:", train.shape)
        print("Test shape:", test.shape)
        print("Train labels:", train["label"].value_counts(dropna=False))
        print("Contains NaNs in label:", train["label"].isna().sum())

        if config["models_to_run"].get("ml"):
            X_train, y_train, _ = feature_engineering.build_features(train, feature_set=config["feature_set"])
            X_test, y_test, _ = feature_engineering.build_features(test, feature_set=config["feature_set"])

            for model_name in config["models_to_run"]["ml"]:
                print(f"\n🚀 Training ML model: {model_name}")
                model_fn = MODEL_DISPATCH[model_name]
                results, model = model_fn(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and model.predict_proba(X_test).shape[1] > 1 else None
                evaluation.evaluate_model(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    model_name=f"{model_name}_seed{seed}"
                )

        if config["models_to_run"].get("dl"):
            texts_train = train["full_text"].tolist()
            labels_train = train["label"].astype(int).tolist()
            texts_test = test["full_text"].tolist()
            labels_test = test["label"].astype(int).tolist()

            for model_name in config["models_to_run"]["dl"]:
                print(f"\n🧠 Training DL model: {model_name}")
                model_fn = MODEL_DISPATCH[model_name]
                results, model = model_fn(texts_train, labels_train)
                preds = (model.predict(texts_test) > 0.5).astype("int32")
                evaluation.evaluate_model(
                    y_true=labels_test,
                    y_pred=preds,
                    y_proba=None,
                    model_name=f"{model_name}_seed{seed}"
                )

def main():
    config = load_config()
    for seed in config["random_seeds"]:
        run_pipeline_for_seed(config, seed)

    print("\n📊 Summarizing all evaluation results...")
    summary = evaluation.summarize_evaluations(result_dir=config["output_dir"])
    print(summary)

if __name__ == "__main__":
    main()