"""
helpers.py — shared utilities
Import what you need in any notebook: e.g., from helpers import load_data

## TO-DO
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath="../data/ridings.csv", encoding="latin1"):
    """
    Load dataset from csv file.
    """
    df = pd.read_csv(filepath, encoding=encoding)
    return df


def clean_target(df):
    """
    Drop rows with missing Political Affiliation
    and create binary target:
    Liberal = 1, non-Liberal = 0
    """
    df = df.dropna(subset=["Political Affiliation"]).copy()
    df["target"] = (df["Political Affiliation"] == "Liberal").astype(int)
    return df


def drop_unused_columns(df):
    """
    Drop columns that should not be used for modeling.
    """
    drop_cols = [
        "Unnamed: 0",
        "riding",
        "Constituency",
        "First Name",
        "Last Name",
        "Political Affiliation"
    ]
    return df.drop(columns=drop_cols, errors="ignore")


PROVINCE_TO_REGION = {
    "Newfoundland and Labrador": "Atlantic",
    "Prince Edward Island":      "Atlantic",
    "Nova Scotia":               "Atlantic",
    "New Brunswick":             "Atlantic",
    "Quebec":                    "Quebec",
    "Ontario":                   "Ontario",
    "Manitoba":                  "Prairies",
    "Saskatchewan":              "Prairies",
    "Alberta":                   "Prairies",
    "British Columbia":          "British Columbia",
    "Yukon":                     "Territories",
    "Northwest Territories":     "Territories",
    "Nunavut":                   "Territories",
}


def get_feature_lists():
    """
    Return numeric and categorical feature lists.
    """
    numeric_features = [
        "Ave_Age_All",
        "bike_rate",
        "pop",
        "NoPostsecondary_all",
        "french",
        "HouseNeed",
        "income_all",
        "unemploy_all"
    ]

    categorical_features = [
        "Province / Territory"
    ]

    return numeric_features, categorical_features


def get_feature_lists_region():
    """
    Return numeric and categorical feature lists for the region-based model.
    Uses 'region' instead of 'Province / Territory'.
    """
    numeric_features = [
        "Ave_Age_All",
        "bike_rate",
        "pop",
        "NoPostsecondary_all",
        "french",
        "HouseNeed",
        "income_all",
        "unemploy_all"
    ]

    categorical_features = [
        "region"
    ]

    return numeric_features, categorical_features


def prepare_X_y(df):
    """
    Prepare X and y for modeling.
    """
    numeric_features, categorical_features = get_feature_lists()
    X = df[numeric_features + categorical_features]
    y = df["target"]
    return X, y


def prepare_X_y_region(df):
    """
    Prepare X and y for the region-based model.
    Expects df to already contain the 'region' column.
    """
    numeric_features, categorical_features = get_feature_lists_region()
    X = df[numeric_features + categorical_features]
    y = df["target"]
    return X, y


def build_preprocessor():
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    numeric_features, categorical_features = get_feature_lists()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def build_preprocessor_region():
    """
    Build preprocessing pipeline identical to build_preprocessor() but using
    'region' as the categorical feature instead of 'Province / Territory'.
    """
    numeric_features, categorical_features = get_feature_lists_region()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor