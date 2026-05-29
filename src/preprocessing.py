import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DROP_COLUMNS = [
    "name",
    "screen_name",
    "id",
    "url",
    "location",
    "default_profile_image",
    "profile_image_url",
    "profile_banner_url",
    "profile_background_image_url_https",
    "profile_text_color",
    "profile_image_url_https",
    "protected",
    "verified",
    "dataset",
    "utc_offset",
    "profile_link_color",
    "profile_background_color",
    "profile_background_image_url",
    "profile_sidebar_border_color",
    "profile_sidebar_fill_color",
    "lang",
]


NUMERIC_FEATURES_TO_SCALE = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "account_age_days",
]


def clean_data(data: pd.DataFrame, reference_date: str = "2025-01-01") -> pd.DataFrame:
    """
    Clean raw user profile data and create final ML-ready features.
    """

    data = data.copy()

    # Drop unnecessary columns if present
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns], errors="ignore")

    # Fill boolean-like columns
    bool_columns = [
        "default_profile",
        "profile_background_tile",
        "geo_enabled",
        "profile_use_background_image",
    ]

    for col in bool_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0).astype(int)

    # Convert text fields to binary indicators
    if "description" in data.columns:
        data["description"] = data["description"].apply(
            lambda x: 1 if pd.notnull(x) and str(x).strip() != "" else 0
        )

    if "time_zone" in data.columns:
        data["time_zone"] = data["time_zone"].apply(
            lambda x: 1 if pd.notnull(x) and str(x).strip() != "" else 0
        )

    # Date feature engineering
    if "created_at" in data.columns:
        data["created_at"] = pd.to_datetime(
                data["created_at"],
                errors="coerce",
                utc=True
            ).dt.tz_localize(None)
        ref_time = pd.Timestamp(reference_date)
        data["account_age_days"] = (ref_time - data["created_at"]).dt.days
        data = data.drop(columns=["created_at"])

    if "updated" in data.columns:
        data = data.drop(columns=["updated"])

    # Remove duplicates
    data = data.drop_duplicates()

    # Fill remaining missing values
    data = data.fillna(0)

    return data


def split_data(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Split data into train and test sets.
    """
    X = data.drop(columns=["label"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Fit scaler only on training data to avoid data leakage.
    """
    scaler = MinMaxScaler()

    scale_cols = [col for col in NUMERIC_FEATURES_TO_SCALE if col in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, scaler