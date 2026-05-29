from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def build_ann(input_dim: int):
    """
    Build ANN model for binary classification.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_sklearn_models():
    """
    Return classical machine learning models.
    """
    models = {
        "svm": SVC(kernel="linear", random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": GaussianNB(),
    }

    return models