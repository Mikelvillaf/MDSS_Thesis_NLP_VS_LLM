# Master pipeline script for experiment
# Example flow
from scripts import (
    data_loader, label_generation, preprocessing,
    feature_engineering, model_training, evaluation
)
from tracking.wandb_init import init_tracking

def main():
    init_tracking("amazon-helpfulness")

    df = data_loader.load_data(...)
    df = label_generation.generate_labels(df)
    df = preprocessing.clean(df)
    X_train, y_train, X_test, y_test = feature_engineering.build_features(df)
    model_training.train_models(X_train, y_train)
    evaluation.evaluate_models(X_test, y_test)
    # Optional:
    # llm_prediction.predict_with_llm(df)

if __name__ == '__main__':
    main()