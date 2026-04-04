"""
optimizers.py — optimization algorithms and cross-validation framework.

Import what you need in any notebook:
    from optimizers import robbins_monro_svm, predict_rm, tune_rm_joint_cv, tune_mlp_optimizers
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cv_tuning import prepare_train_test_data, CV_FOLDS
from kernels import compute_kernel_matrix, KERNEL_REGISTRY


# ---------------------------------------------------------------------------
# Robbins-Monro SVM  —  dual representation, any registered kernel
# ---------------------------------------------------------------------------

def robbins_monro_svm(
        X_train, y_train_svm, X_test, y_test_svm,
        kernel_name="linear", kernel_params=None,
        lambda_reg=0.01, n_epochs=50, eta0=0.1, decay=0.01,
        patience=None, random_state=42,
):
    """
    Robbins-Monro SGD for soft-margin SVM via the representer
    theorem.

    Maintains alpha in R^n so that w = sum_i alpha_i * phi(x_i).
    Decision function: f(x) = K(X_train, x)^T @ alpha - b.

    Primal loss minimised (per Pegasos / Shalev-Shwartz et al. 2007):
        L = (1/n) sum_j max(0, 1 - y_j * f(x_j)) + lambda * ||w||^2

    Per-step updates (sample j, learning rate eta_t = eta0 / (1 + decay * t)):
        alpha   *= (1 - 2 * eta * lambda_reg)      # weight-decay from L2 term
        alpha[j] += eta * y_j   }  only when        # hinge subgradient
        b        -= eta * y_j   }  margin < 1

    Parameters
    ----------
    y_train_svm, y_test_svm : array-like, labels in {-1, +1}
    kernel_params : dict, passed to compute_kernel_matrix (e.g. {"gamma": 0.1})
    patience      : int or None.  If set, stop early when test accuracy has not
                    improved for this many consecutive epochs.

    Returns
    -------
    dict with keys: alpha, b, X_train, kernel_name, kernel_params,
                    train_acc_history, test_acc_history, fit_time_s
    """
    kp = kernel_params or {}
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train_svm, dtype=float)
    y_te = np.asarray(y_test_svm, dtype=float)

    K_tr = compute_kernel_matrix(X_tr, X_tr, kernel_name, **kp)   # (n, n)
    K_te = compute_kernel_matrix(X_tr, X_te, kernel_name, **kp)   # (n, n_test)

    n = len(y_tr)
    alpha = np.zeros(n)
    b = 0.0
    rng = np.random.default_rng(random_state)

    train_hist, test_hist = [], []
    best_test_acc = -1.0
    best_alpha, best_b = alpha.copy(), b
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        eta = eta0 / (1.0 + decay * epoch)
        for j in rng.permutation(n):
            margin = y_tr[j] * (K_tr[j] @ alpha - b)
            alpha *= (1.0 - 2.0 * eta * lambda_reg)
            if margin < 1.0:
                alpha[j] += eta * y_tr[j]
                b -= eta * y_tr[j]

        tr_acc = accuracy_score(y_tr > 0, (K_tr @ alpha - b) > 0)
        te_acc = accuracy_score(y_te > 0, (K_te.T @ alpha - b) > 0)
        train_hist.append(tr_acc)
        test_hist.append(te_acc)

        # ---- early stopping ----
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_alpha, best_b = alpha.copy(), b
            wait = 0
        else:
            wait += 1
        if patience is not None and wait >= patience:
            break

    # restore best checkpoint when using early stopping
    if patience is not None:
        alpha, b = best_alpha, best_b

    return {
        "alpha": alpha,
        "b": b,
        "X_train": X_tr,
        "kernel_name": kernel_name,
        "kernel_params": kp,
        "train_acc_history": train_hist,
        "test_acc_history": test_hist,
        "fit_time_s": time.perf_counter() - t0,
    }


def predict_rm(result, X_new):
    """Return binary (0/1) predictions for X_new from a fitted robbins_monro_svm result."""
    K = compute_kernel_matrix(
        result["X_train"], np.asarray(X_new, dtype=float),
        result["kernel_name"], **result["kernel_params"],
    )
    return (K.T @ result["alpha"] - result["b"] > 0).astype(int)


# ---------------------------------------------------------------------------
# Cross-validation tuning for Robbins-Monro
# ---------------------------------------------------------------------------

_DEFAULT_RM_GRID = [
    {"lambda_reg": lr, "eta0": e, "decay": d, "n_epochs": ne}
    for lr in [0.001, 0.01, 0.1]
    for e in [0.01, 0.1]
    for d in [0.01, 0.1]
    for ne in [50, 100]
]


def tune_robbins_monro_cv(
        X_train, y_train,
        kernel_name="linear", kernel_params=None,
        param_grid=None, cv=CV_FOLDS, random_state=42,
):
    """
    k-fold CV grid search over RM hyperparameters for a FIXED kernel setting.

    y_train      : 0/1 labels (converted to {-1,+1} per fold internally).
    kernel_params: fixed kernel params for this CV run (e.g. {"gamma": 0.1}).
    param_grid   : list of dicts with keys lambda_reg, eta0, decay, n_epochs.

    Returns dict: results_df, best_params, best_cv_score, fit_time_s.
    """
    kp = kernel_params or {}
    param_grid = param_grid or _DEFAULT_RM_GRID
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)

    rows = []
    t0 = time.perf_counter()

    for params in param_grid:
        fold_scores = []
        for tr_idx, val_idx in cv.split(X, y):
            y_tr_svm = np.where(y[tr_idx] == 1, 1, -1).astype(float)
            y_val_svm = np.where(y[val_idx] == 1, 1, -1).astype(float)
            res = robbins_monro_svm(
                X[tr_idx], y_tr_svm, X[val_idx], y_val_svm,
                kernel_name=kernel_name, kernel_params=kp,
                random_state=random_state, **params,
            )
            fold_scores.append(res["test_acc_history"][-1])

        s = np.array(fold_scores)
        rows.append({**params, "kernel": kernel_name, "mean_cv": s.mean(), "std_cv": s.std()})

    results_df = (
        pd.DataFrame(rows)
        .sort_values("mean_cv", ascending=False)
        .reset_index(drop=True)
    )
    best = results_df.iloc[0]
    return {
        "results_df": results_df,
        "best_params": {k: float(best[k]) for k in ("lambda_reg", "eta0", "decay", "n_epochs")},
        "best_cv_score": float(best["mean_cv"]),
        "fit_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# JOINT kernel + RM hyperparameter tuning  (the main improvement)
# ---------------------------------------------------------------------------

_DEFAULT_RM_GRID_COMPACT = [
    {"lambda_reg": lr, "eta0": e, "decay": d, "n_epochs": ne}
    for lr in [0.001, 0.01, 0.1]
    for e in [0.01, 0.1, 0.5]
    for d in [0.01, 0.1]
    for ne in [50, 100]
]


def tune_rm_joint_cv(
        X_train, y_train,
        kernel_names=None,
        rm_param_grid=None,
        c_values=None,
        cv=CV_FOLDS,
        random_state=42,
        verbose=True,
):
    """
    Joint grid search over kernels (including their hyperparameters) AND
    Robbins-Monro optimizer settings.

    This is the key function that was missing — the original code fixed kernel
    params to param_grid[0] and only tuned RM params, which crippled non-linear
    kernels.

    For each (kernel_name, kernel_params, rm_params) triple, runs k-fold CV
    and selects the combination with the highest mean CV accuracy.

    Parameters
    ----------
    kernel_names   : list of str, subset of KERNEL_REGISTRY keys (default: all)
    rm_param_grid  : list of dicts with lambda_reg, eta0, decay, n_epochs
    c_values       : ignored (kept for API compatibility with search_kernels)
    cv             : cross-validation splitter
    verbose        : print progress per kernel

    Returns
    -------
    dict with results_df (all combos), best_result (dict with best settings)
    """
    kernel_names = kernel_names or list(KERNEL_REGISTRY.keys())
    rm_param_grid = rm_param_grid or _DEFAULT_RM_GRID_COMPACT
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)

    rows = []
    best = None
    t0 = time.perf_counter()

    for kernel_name in kernel_names:
        entry = KERNEL_REGISTRY[kernel_name]
        kernel_param_list = entry["param_grid"]         # full kernel param grid

        for kp in kernel_param_list:
            for rm_params in rm_param_grid:
                fold_scores = []
                for tr_idx, val_idx in cv.split(X, y):
                    y_tr_svm = np.where(y[tr_idx] == 1, 1, -1).astype(float)
                    y_val_svm = np.where(y[val_idx] == 1, 1, -1).astype(float)
                    res = robbins_monro_svm(
                        X[tr_idx], y_tr_svm, X[val_idx], y_val_svm,
                        kernel_name=kernel_name, kernel_params=kp,
                        random_state=random_state, **rm_params,
                    )
                    fold_scores.append(res["test_acc_history"][-1])

                s = np.array(fold_scores)
                row = {
                    "kernel": kernel_name,
                    "kernel_params": kp,
                    **rm_params,
                    "mean_cv": s.mean(),
                    "std_cv": s.std(),
                }
                rows.append(row)

                if best is None or s.mean() > best["mean_cv"]:
                    best = row.copy()

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"  {kernel_name:8s}  combos tested: {len(rows):>4d}  "
                  f"best_cv so far: {best['mean_cv']:.4f}  [{elapsed:.0f}s]")

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["mean_cv"], ascending=False)
        .reset_index(drop=True)
    )

    return {
        "results_df": results_df,
        "best_result": best,
        "fit_time_s": time.perf_counter() - t0,
    }


def fit_best_rm(X_train, y_train_svm, X_test, y_test_svm, best_result,
                n_epochs_override=None):
    """
    Convenience: refit the RM SVM on the full training set using the best
    settings found by tune_rm_joint_cv.

    Returns the robbins_monro_svm result dict.
    """
    rm_keys = {"lambda_reg", "eta0", "decay", "n_epochs"}
    rm_params = {k: float(v) for k, v in best_result.items() if k in rm_keys}
    if n_epochs_override is not None:
        rm_params["n_epochs"] = int(n_epochs_override)
    return robbins_monro_svm(
        X_train, y_train_svm, X_test, y_test_svm,
        kernel_name=best_result["kernel"],
        kernel_params=best_result["kernel_params"],
        **rm_params,
    )


# ---------------------------------------------------------------------------
# MLP optimizer comparison
# ---------------------------------------------------------------------------

def tune_mlp_optimizers(X_train, y_train, X_test, y_test, cv=CV_FOLDS):
    """
    Compare different optimization algorithms in MLPClassifier.
    Main optimizers: sgd, adam, lbfgs.
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

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    return {
        "grid": grid,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_acc": accuracy_score(y_test, y_pred),
        "y_pred": y_pred,
        "fit_time_s": time.perf_counter() - t0,
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rm_eval_report(y_true, y_pred, model_name="RM SVM"):
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

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
        X_train=data["X_train_proc"],
        y_train=data["y_train"],
        X_test=data["X_test_proc"],
        y_test=data["y_test"],
    )

    return {
        "data": data,
        "result": result,
        "summary_df": build_optimizer_results_summary(result),
    }
