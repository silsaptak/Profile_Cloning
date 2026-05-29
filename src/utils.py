import matplotlib.pyplot as plt


def plot_model_comparison(metrics_df, save_path="results/model_comparison.png"):
    """
    Plot model accuracy, F1-score, and execution time.
    """

    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["model"], metrics_df["accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()