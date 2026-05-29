import joblib
import pandas as pd
from tensorflow.keras.models import load_model

from src.preprocessing import clean_data


def predict_single_profile(profile_dict, model_path, scaler_path, feature_columns_path):
    """
    Predict whether a single user profile is fake or genuine.

    Returns:
    1 -> Fake user
    0 -> Genuine user
    """

    # Convert dictionary to dataframe
    data = pd.DataFrame([profile_dict])

    # Dummy label is needed because clean_data expects label column structure
    data["label"] = 0

    # Apply same preprocessing
    data = clean_data(data)

    # Separate features
    X = data.drop(columns=["label"])

    # Load saved feature columns
    feature_columns = joblib.load(feature_columns_path)

    # Ensure all required columns exist
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Keep same column order as training
    X = X[feature_columns]

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Scale features
    X_scaled = X.copy()
    scale_cols = scaler.feature_names_in_
    X_scaled[scale_cols] = scaler.transform(X_scaled[scale_cols])

    # Load model
    model = load_model(model_path)

    # Predict
    probability = model.predict(X_scaled, verbose=0)[0][0]
    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability