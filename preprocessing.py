"""
Preprocessing for the ridings dataset.

IGNORE THIS FILE -- IT WAS JUST FOR DATA EXPLORATION
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "ridings.csv"
TARGET_COLUMN = "Political Affiliation"
BINARY_TARGET_COLUMN = "is_liberal"
CONSTITUENCY_COLUMN = "Constituency"

# reduced feature set chosen during eda (please see notebooks/eda.ipynb)
# male/female splits are dropped because of multicollinearity (r > 0.9)
# except for income: male and female have opposite correlation with the target
FEATURE_COLUMNS = [
    "Ave_Age_All",
    "bike_rate",
    "pop",
    "NoPostsecondary_all",
    "french",
    "HouseNeed",
    "income_all",
    "income_male",
    "income_female",
    "unemploy_all",
]
def load_ridings_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw ridings CSV
    """
    return pd.read_csv(data_path, encoding="cp1252")


def clean_ridings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the ridings dataset for analysis or modeling.

    - Drop `Constituency` because it duplicates the riding name in a different format.
    - Remove rows with missing political affiliation.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=[CONSTITUENCY_COLUMN], errors="ignore")
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # binary target
    cleaned[BINARY_TARGET_COLUMN] = (cleaned[TARGET_COLUMN] == "Liberal").astype(int)

    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a cleaned dataframe into features and target.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def load_features_target(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset, clean it, and return features plus target.
    """
    cleaned = clean_ridings_data(load_ridings_data(data_path))
    return split_features_target(cleaned)

def load_modeling_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset for SVM
    :param  data_path
    :return: features set and the binary target (1=Liberal, 0=Non-Liberal)
    """
    cleaned = clean_ridings_data(load_ridings_data(data_path))
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[BINARY_TARGET_COLUMN]
    return X, y

if __name__ == "__main__":
    cleaned_df = clean_ridings_data(load_ridings_data())
    X, y = split_features_target(cleaned_df)

    print(f"Cleaned rows: {cleaned_df.shape[0]}")
    print(f"Feature columns: {X.shape[1]}")
    print("\nTarget distribution:")
    print(y.value_counts())

    X_model, y_model = load_modeling_data()
    print(f"\nModeling features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"\nBinary target distribution:")
    print(y_model.value_counts().rename({1: "Liberal (1)", 0: "Non-Liberal (0)"}))
