import time
import copy
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.models import build_ann, get_sklearn_models
from src.preprocessing import scale_features
from src.evaluate import evaluate_predictions


def run_cross_validation(data, n_splits=5, random_state=42, ann_epochs=50, ann_batch_size=32):
    """
    Run stratified k-fold cross-validation for ANN and classical ML models.
    """

    X = data.drop(columns=["label"])
    y = data["label"]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold}/{n_splits} ==========")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train, X_test, scaler = scale_features(X_train, X_test)

        # ANN
        print("Training ann...")

        ann_model = build_ann(input_dim=X_train.shape[1])

        start_time = time.time()

        ann_model.fit(
            X_train,
            y_train,
            epochs=ann_epochs,
            batch_size=ann_batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred_prob = ann_model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

        end_time = time.time()

        metrics = evaluate_predictions(y_test, y_pred)

        cv_results.append({
            "model": "ann",
            "fold": fold,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "time_ms": (end_time - start_time) * 1000,
        })

        print(f"ANN Fold {fold}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

        # Classical ML models
        sklearn_models = get_sklearn_models()

        for model_name, model in sklearn_models.items():
            print(f"Training {model_name}...")

            model = copy.deepcopy(model)

            start_time = time.time()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            end_time = time.time()

            metrics = evaluate_predictions(y_test, y_pred)

            cv_results.append({
                "model": model_name,
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "time_ms": (end_time - start_time) * 1000,
            })

            print(
                f"{model_name} Fold {fold}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_score']:.4f}"
            )

    cv_df = pd.DataFrame(cv_results)

    summary_df = cv_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        time_ms_mean=("time_ms", "mean"),
        time_ms_std=("time_ms", "std"),
    ).reset_index()

    return cv_df, summary_df