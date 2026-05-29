import time
import pandas as pd

from src.models import build_ann, get_sklearn_models
from src.evaluate import evaluate_predictions


def train_ann(X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
    """
    Train and evaluate ANN.
    """
    model = build_ann(input_dim=X_train.shape[1])

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    end_time = time.time()

    results = evaluate_predictions(y_test, y_pred)
    results["model"] = "ann"
    results["time_ms"] = (end_time - start_time) * 1000

    return model, results


def train_sklearn_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate SVM, Logistic Regression, and Naive Bayes.
    """
    all_results = []
    trained_models = {}

    models = get_sklearn_models()

    for model_name, model in models.items():
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()

        results = evaluate_predictions(y_test, y_pred)
        results["model"] = model_name
        results["time_ms"] = (end_time - start_time) * 1000

        trained_models[model_name] = model
        all_results.append(results)

    return trained_models, all_results