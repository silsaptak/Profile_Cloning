from src.predict import predict_single_profile


def main():
    # demo profile -- 
    sample_profile = {
        "statuses_count": 2500,
        "followers_count": 1200,
        "friends_count": 800,
        "favourites_count": 500,
        "listed_count": 20,
        "description": "Research student interested in machine learning and data science.",
        "created_at": "2015-01-01",
        "default_profile": 0,
        "profile_background_tile": 0,
        "geo_enabled": 1,
        "profile_use_background_image": 1,
        "time_zone": "Kolkata",
    }

    prediction, probability = predict_single_profile(
        sample_profile,
        model_path="results/best_ann_model.keras",
        scaler_path="results/scaler.pkl",
        feature_columns_path="results/feature_columns.pkl",
    )

    if prediction == 1:
        print(f"Prediction: Fake user | Probability: {probability:.4f}")
    else:
        print(f"Prediction: Genuine user | Probability: {probability:.4f}")


if __name__ == "__main__":
    main()