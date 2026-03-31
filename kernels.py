"""
kernels.py — kernel functions, kernelized SVM (TODO), and cross-validation framework (TODO).

Import what you need in any notebook: e.g., from kernels import compute_kernel_matrix, KernelSVM, grid_search_cv, final_evaluate
"""


import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cv_tuning import prepare_train_test_data, CV_FOLDS


# Kernel functions
# Convention: kernel_fn(X1, X2, **params) -> K  where K[i,j] = K(X1[i], X2[j])

def linear_kernel(X1, X2, **kwargs):
    """K(x, x') = x · x'"""
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=1.0, **kwargs):
    """K(x, x') = exp(-gamma * ||x - x'||^2)"""
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(dist_sq, 0))


def polynomial_kernel(X1, X2, gamma=1.0, coef0=1.0, degree=3, **kwargs):
    """K(x, x') = (gamma * x · x' + coef0)^degree"""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def sigmoid_kernel(X1, X2, gamma=1.0, coef0=0.0, **kwargs):
    """K(x, x') = tanh(gamma * x · x' + coef0)"""
    return np.tanh(gamma * (X1 @ X2.T) + coef0)



# Kernel registry
# Maps name -> function, default hyperparams, CV search grid

KERNEL_REGISTRY = {
    "linear": {
        "fn": linear_kernel,
        "defaults": {},
        "param_grid": [{}],
    },
    "rbf": {
        "fn": rbf_kernel,
        "defaults": {"gamma": 1.0},
        "param_grid": [{"gamma": g} for g in [0.001, 0.01, 0.1, 1.0, 10.0]],
    },
    "poly": {
        "fn": polynomial_kernel,
        "defaults": {"gamma": 1.0, "coef0": 1.0, "degree": 3},
        "param_grid": [
            {"degree": d, "gamma": g, "coef0": c}
            for d in [2, 3, 4]
            for g in [0.01, 0.1, 1.0]
            for c in [0.0, 1.0]
        ],
    },
    "sigmoid": {
        "fn": sigmoid_kernel,
        "defaults": {"gamma": 0.01, "coef0": 0.0},
        "param_grid": [
            {"gamma": g, "coef0": c}
            for g in [0.001, 0.01, 0.1]
            for c in [0.0, 1.0]
        ],
    },
}

# Compute the Gram matrix for the selected kernel.
# Uses the kernel registry to fetch default parameters and the kernel function.

def compute_kernel_matrix(X1, X2, kernel_name, **kernel_params):
    """Return Gram matrix using the selected kernel."""
    entry = KERNEL_REGISTRY[kernel_name]
    params = {**entry["defaults"], **kernel_params}
    return entry["fn"](X1, X2, **params)


# Build an sklearn SVC that matches the selected kernel setting.
# This keeps model fitting stable while we control the kernel search manually.

def _build_svc_from_kernel(kernel_name, C=1.0, **kernel_params):
    """
    Build an sklearn SVC configured to match our kernel registry.
    We use sklearn SVC here for stable fitting, while controlling
    the search logic ourselves.
    """
    if kernel_name == "linear":
        return SVC(kernel="linear", C=C)

    if kernel_name == "rbf":
        return SVC(
            kernel="rbf",
            C=C,
            gamma=kernel_params.get("gamma", 1.0),
        )

    if kernel_name == "poly":
        return SVC(
            kernel="poly",
            C=C,
            gamma=kernel_params.get("gamma", 1.0),
            degree=kernel_params.get("degree", 3),
            coef0=kernel_params.get("coef0", 1.0),
        )

    if kernel_name == "sigmoid":
        return SVC(
            kernel="sigmoid",
            C=C,
            gamma=kernel_params.get("gamma", 0.01),
            coef0=kernel_params.get("coef0", 0.0),
        )

    raise ValueError(f"Unknown kernel: {kernel_name}")

# Train and evaluate one kernel SVM configuration on the test set.
# Returns the test accuracy, predictions, and fitted model.

def evaluate_single_kernel(
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
    kernel_name,
    C=1.0,
    **kernel_params,
):
    """
    Fit one kernel SVM setting and evaluate it on the test set.
    """
    model = _build_svc_from_kernel(kernel_name, C=C, **kernel_params)
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)

    return {
        "kernel": kernel_name,
        "C": C,
        "kernel_params": kernel_params,
        "test_acc": accuracy_score(y_test, y_pred),
        "y_pred": y_pred,
        "model": model,
    }

# Run manual cross-validation for one kernel and parameter setting.
# Trains on each fold split and records validation accuracy across folds.

def cross_validate_single_kernel(X, y, kernel_name, C=1.0, cv=CV_FOLDS, **kernel_params):
    """
    Manual CV loop for one kernel setting.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr = X[train_idx]
        X_val = X[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        model = _build_svc_from_kernel(kernel_name, C=C, **kernel_params)
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)

        fold_scores.append(accuracy_score(y_val, y_val_pred))

    fold_scores = np.array(fold_scores)

    return {
        "kernel": kernel_name,
        "C": C,
        "kernel_params": kernel_params,
        "cv_scores": fold_scores,
        "mean_cv": fold_scores.mean(),
        "std_cv": fold_scores.std(),
    }

# Search over all registered kernels, parameter grids, and C values.
# Compares settings using cross-validation and also records test accuracy.
# Returns the full results table and the best configuration found.

def search_kernels(
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
    c_values=(0.1, 1, 10),
    cv=CV_FOLDS,
):
    """
    Search over all registered kernels and their parameter grids.
    Returns:
        all_results_df
        best_cv_result
        best_test_result
    """
    all_results = []
    best_cv_result = None

    for kernel_name, entry in KERNEL_REGISTRY.items():
        for params in entry["param_grid"]:
            for C in c_values:
                cv_result = cross_validate_single_kernel(
                    X_train_proc, y_train,
                    kernel_name=kernel_name,
                    C=C,
                    cv=cv,
                    **params,
                )

                test_result = evaluate_single_kernel(
                    X_train_proc, y_train,
                    X_test_proc, y_test,
                    kernel_name=kernel_name,
                    C=C,
                    **params,
                )

                row = {
                    "kernel": kernel_name,
                    "C": C,
                    "params": params,
                    "mean_cv": cv_result["mean_cv"],
                    "std_cv": cv_result["std_cv"],
                    "test_acc": test_result["test_acc"],
                }
                all_results.append(row)

                if (best_cv_result is None) or (cv_result["mean_cv"] > best_cv_result["mean_cv"]):
                    best_cv_result = {
                        **cv_result,
                        "test_acc": test_result["test_acc"],
                        "y_pred": test_result["y_pred"],
                        "model": test_result["model"],
                    }

    results_df = pd.DataFrame(all_results).sort_values(
        by=["mean_cv", "test_acc"], ascending=False
    ).reset_index(drop=True)

    return {
        "results_df": results_df,
        "best_result": best_cv_result,
    }

# Print a standard evaluation report for the chosen kernel model.
# Includes test accuracy, classification report, and confusion matrix.

def print_kernel_eval_report(y_true, y_pred, model_name="Kernel SVM"):
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()

# End-to-end entry point for kernel experiments.
# Reuses the shared preprocessing pipeline and runs the full kernel search.
# Returns the prepared data, all search results, and the best model result.

def run_kernel_experiments(filepath="../data/ridings.csv"):
    """
    End-to-end kernel experiment entry point.
    Uses the same shared data prep structure defined in cv_tuning.py.
    """
    data = prepare_train_test_data(filepath=filepath)

    result = search_kernels(
        X_train_proc=data["X_train_proc"],
        y_train=np.asarray(data["y_train"]),
        X_test_proc=data["X_test_proc"],
        y_test=np.asarray(data["y_test"]),
    )

    return {
        "data": data,
        "results_df": result["results_df"],
        "best_result": result["best_result"],
    }