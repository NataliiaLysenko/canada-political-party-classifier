"""
Refactored Alex's CV_tuning notebook for reuse
cv_tuning.py — data preparation, CV configuration

Import what you need in any notebook: e.g., from cv_tuning import prepare_train_test_data, CV_FOLDS, tune_sklearn_svm
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helpers import (
    load_data, clean_target, drop_unused_columns,
    prepare_X_y, build_preprocessor,
)

CV_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def prepare_train_test_data(filepath="../data/ridings.csv", test_size=0.2,
                            random_state=42):
    """
    Full pipeline is here: load -> clean -> split -> preprocess -> convert to SVM labels.

    Returns
    -------
    data : dict with keys
        X_train, X_test       : Raw data
        y_train, y_test       : Raw data
        X_train_proc, X_test_proc : Processed data (scaled + one-hot encoded)
        y_train_svm, y_test_svm   : SVM labels
        preprocessor          : Preprocessor object
    """
    df = load_data(filepath)
    df = clean_target(df)
    df = drop_unused_columns(df)
    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Preprocessing -- scaling & encoding
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # checks if X_train_proc & X_test_proc is a sparse matrix after transforming, and if so converts to an array
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    # convert SVM labels: 1->+1 and 0 -> -1 for subsequent math operations
    y_train_svm = np.where(y_train.values == 1, 1, -1)
    y_test_svm = np.where(y_test.values == 1, 1, -1)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train_svm": y_train_svm,
        "y_test_svm": y_test_svm,
        "preprocessor": preprocessor,
    }

def run_baseline_lr(X_train, y_train, X_test, y_test, cv=CV_FOLDS):
    """
    Logistic Regression baseline: cross-validate, fit, and report test accuracy.

    Returns
    -------
    dict with cv_scores, test_acc, y_pred, pipeline
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring="accuracy")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "cv_scores": cv_scores,
        "mean_cv": cv_scores.mean(),
        "std_cv": cv_scores.std(),
        "test_acc": test_acc,
        "y_pred": y_pred,
        "pipeline": pipeline,
    }

def tune_lr(X_train, y_train, X_test, y_test, cv=CV_FOLDS):
    """
    GridSearchCV over Logistic Regression hyperparameters.

    Returns
    -------
    dict with grid (fitted GridSearchCV), best_params, best_cv_score,
    test_acc, y_pred
    """
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(
        Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "grid": grid,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_acc": test_acc,
        "y_pred": y_pred,
    }


def tune_sklearn_svm(X_train, y_train, X_test, y_test, cv=CV_FOLDS):
    """
    GridSearchCV over sklearn SVC hyperparameters.

    Returns
    -------
    dict with grid, best_params, best_cv_score, test_acc, y_pred
    """
    param_grid = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto"],
    }

    grid = GridSearchCV(
        Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", SVC()),
        ]),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "grid": grid,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_acc": test_acc,
        "y_pred": y_pred,
    }


def print_eval_report(y_true, y_pred, model_name="Model"):
    """Print classification report and confusion matrix."""
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()


def build_results_summary(baseline_result, lr_result, svm_result):
    return pd.DataFrame({
        "Model": [
            "Baseline Logistic Regression",
            "Tuned Logistic Regression",
            "Tuned SVM (sklearn)",
        ],
        "CV Score": [
            baseline_result["mean_cv"],
            lr_result["best_cv_score"],
            svm_result["best_cv_score"],
        ],
        "Test Score": [
            baseline_result["test_acc"],
            lr_result["test_acc"],
            svm_result["test_acc"],
        ],
    })
