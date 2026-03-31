"""
optimizers.py — optimization algorithms, and cross-validation framework .

Import what you need in any notebook: e.g., from optimizers import Adam, SGD, grid_search_cv, final_evaluate
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cv_tuning import prepare_train_test_data, CV_FOLDS

# Compare different MLP optimizers using grid search cross-validation.
# Tries sgd, adam, and lbfgs with several network sizes and learning settings.
# Returns the best model, best parameters, CV score, and test predictions.

def tune_mlp_optimizers(X_train_proc, y_train, X_test_proc, y_test, cv=CV_FOLDS):
    """
    Compare different optimization algorithms in MLPClassifier.
    Main optimizers:
        - sgd
        - adam
        - lbfgs
    """
    param_grid = [
        {
            "solver": ["sgd"],
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "max_iter": [2000],
            "random_state": [42],
        },
        {
            "solver": ["adam"],
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
            "max_iter": [2000],
            "random_state": [42],
        },
        {
            "solver": ["lbfgs"],
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "alpha": [0.0001, 0.001, 0.01],
            "max_iter": [2000],
            "random_state": [42],
        },
    ]

    grid = GridSearchCV(
        estimator=MLPClassifier(),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )

    grid.fit(X_train_proc, y_train)
    y_pred = grid.best_estimator_.predict(X_test_proc)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "grid": grid,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_acc": test_acc,
        "y_pred": y_pred,
    }


# Print a standard evaluation report for the selected optimizer model.
# Includes test accuracy, classification report, and confusion matrix.

def print_optimizer_eval_report(y_true, y_pred, model_name="Optimizer Model"):
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()

# Build a compact one-row summary table for the optimizer search result.
# Stores the best CV score, test score, and corresponding best parameters.

def build_optimizer_results_summary(result):
    return pd.DataFrame({
        "Best CV Score": [result["best_cv_score"]],
        "Test Score": [result["test_acc"]],
        "Best Params": [result["best_params"]],
    })

# End-to-end entry point for optimizer experiments.
# Loads the shared preprocessed train/test split and runs the optimizer search.
# Returns the prepared data, full result object, and summary table.

def run_optimizer_experiments(filepath="../data/ridings.csv"):
    """
    End-to-end optimizer experiment entry point.
    Reuses the same preprocessed train/test data from cv_tuning.py.
    """
    data = prepare_train_test_data(filepath=filepath)

    result = tune_mlp_optimizers(
        X_train_proc=data["X_train_proc"],
        y_train=data["y_train"],
        X_test_proc=data["X_test_proc"],
        y_test=data["y_test"],
    )

    return {
        "data": data,
        "result": result,
        "summary_df": build_optimizer_results_summary(result),
    }
