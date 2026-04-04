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
            for c in [0.0, 1.0, 10]
        ],
    },
    "sigmoid": {
        "fn": sigmoid_kernel,
        "defaults": {"gamma": 0.01, "coef0": 0.0},
        "param_grid": [
            {"gamma": g, "coef0": c}
            for g in [0.001, 0.01, 0.1]
            for c in [0.0, 1.0, 10]
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

def _build_svc(kernel_name, C=1.0, **kernel_params):
    """Build an sklearn SVC matching a registry kernel configuration."""
    p = {**KERNEL_REGISTRY[kernel_name]["defaults"], **kernel_params}
    if kernel_name == "linear":
        return SVC(kernel="linear", C=C)
    if kernel_name == "rbf":
        return SVC(kernel="rbf",     C=C, gamma=p["gamma"])
    if kernel_name == "poly":
        return SVC(kernel="poly",    C=C, gamma=p["gamma"], degree=p["degree"], coef0=p["coef0"])
    if kernel_name == "sigmoid":
        return SVC(kernel="sigmoid", C=C, gamma=p["gamma"], coef0=p["coef0"])
    raise ValueError(f"Unknown kernel: {kernel_name}")


def search_kernels(X_train, y_train, X_test, y_test, c_values=(0.1, 1, 10), cv=CV_FOLDS):
    """
    Grid search over all registered kernels, their param grids, and C values.
    Model selection is by mean CV accuracy; test accuracy is also recorded.

    Returns dict: results_df, best_result.
    """
    X_tr, y_tr = np.asarray(X_train), np.asarray(y_train)
    X_te, y_te = np.asarray(X_test), np.asarray(y_test)

    rows, best = [], None

    for kernel_name, entry in KERNEL_REGISTRY.items():
        for params in entry["param_grid"]:
            for C in c_values:
                fold_scores = []
                for tr_idx, val_idx in cv.split(X_tr, y_tr):
                    clf = _build_svc(kernel_name, C=C, **params)
                    clf.fit(X_tr[tr_idx], y_tr[tr_idx])
                    fold_scores.append(accuracy_score(y_tr[val_idx], clf.predict(X_tr[val_idx])))
                cv_scores = np.array(fold_scores)

                clf = _build_svc(kernel_name, C=C, **params)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                test_acc = accuracy_score(y_te, y_pred)

                row = {
                    "kernel": kernel_name, "C": C, "params": params,
                    "mean_cv": cv_scores.mean(), "std_cv": cv_scores.std(),
                    "test_acc": test_acc,
                }
                rows.append(row)
                if best is None or cv_scores.mean() > best["mean_cv"]:
                    best = {**row, "y_pred": y_pred, "model": clf}

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["mean_cv", "test_acc"], ascending=False)
        .reset_index(drop=True)
    )
    return {"results_df": results_df, "best_result": best}


def print_kernel_eval_report(y_true, y_pred, model_name="Kernel SVM"):
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))