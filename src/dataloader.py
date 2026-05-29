import pandas as pd


def load_datasets(fake_path: str, genuine_path: str) -> pd.DataFrame:
    """
    Load fake and genuine user datasets and combine them.

    Label convention:
    1 -> Fake user
    0 -> Genuine user
    """
    fake_users = pd.read_csv(fake_path)
    genuine_users = pd.read_csv(genuine_path)

    fake_users["label"] = 1
    genuine_users["label"] = 0

    data = pd.concat([fake_users, genuine_users], ignore_index=True)
    return data