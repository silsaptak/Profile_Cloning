import os
import pandas as pd
import joblib

from src.dataloader import load_datasets
from src.preprocessing import clean_data, split_data, scale_features
from src.train import train_ann, train_sklearn_models
from src.utils import plot_model_comparison
from src.cross_validation import run_cross_validation


def main():
    fake_path = "data/fusers.csv"
    genuine_path = "data/users.csv"

    os.makedirs("results", exist_ok=True)

    print("Loading datasets...")
    data = load_datasets(fake_path, genuine_path)

    print("Cleaning and preprocessing data...")
    data = clean_data(data)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)

    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Save scaler and feature column order for future inference
    joblib.dump(scaler, "results/scaler.pkl")
    joblib.dump(list(X_train.columns), "results/feature_columns.pkl")

    print("Saved scaler to results/scaler.pkl")
    print("Saved feature columns to results/feature_columns.pkl")

    final_results = []

    print("Training ANN...")
    ann_model, ann_results = train_ann(X_train, X_test, y_train, y_test)

    # Save trained ANN model for inference
    ann_model.save("results/best_ann_model.keras")
    print("Saved ANN model to results/best_ann_model.keras")

    final_results.append(ann_results)

    print("Training classical ML models...")
    sklearn_models, sklearn_results = train_sklearn_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    final_results.extend(sklearn_results)

    # Save clean metrics
    metrics_rows = []

    for result in final_results:
        metrics_rows.append({
            "model": result["model"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "time_ms": result["time_ms"],
        })

        print("\nModel:", result["model"])
        print("Accuracy:", result["accuracy"])
        print("Precision:", result["precision"])
        print("Recall:", result["recall"])
        print("F1-score:", result["f1_score"])
        print("Time:", result["time_ms"], "ms")
        print("Confusion Matrix:\n", result["confusion_matrix"])
        print("Classification Report:\n", result["classification_report"])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("results/metrics.csv", index=False)

    plot_model_comparison(metrics_df, "results/model_comparison.png")

    print("\nSaved metrics to results/metrics.csv")
    print("Saved model comparison plot to results/model_comparison.png")

    print("\nRunning cross-validation...")
    cv_df, cv_summary_df = run_cross_validation(
        data,
        n_splits=5,
        ann_epochs=50,
        ann_batch_size=32
    )

    cv_df.to_csv("results/cv_results.csv", index=False)
    cv_summary_df.to_csv("results/cv_summary.csv", index=False)

    print("Saved cross-validation results to results/cv_results.csv")
    print("Saved cross-validation summary to results/cv_summary.csv")


if __name__ == "__main__":
    main()