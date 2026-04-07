"""
optimizers.py — SVM optimization algorithms and a generic cross-validation
tuning framework that works with any of the three optimizers.

Public API
----------
Algorithms:
    robbins_monro_svm, adagrad_svm, adam_svm
    predict_rm

Default hyperparameter grids:
    RM_PARAM_GRID, ADAGRAD_PARAM_GRID, ADAM_PARAM_GRID

Generic tuning helpers (work with any optimizer_fn):
    tune_optimizer_joint_cv
    fit_best
    run_svm_experiments

MLP comparison:
    tune_mlp_optimizers, run_optimizer_experiments

Reporting:
    print_rm_eval_report, build_optimizer_results_summary
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
# Adam SVM — dual representation with Adaptive Moment Estimation
# ---------------------------------------------------------------------------

def adam_svm(
    X_train, y_train_svm, X_test, y_test_svm,
    kernel_name="linear", kernel_params=None,
    lambda_reg=0.01, n_epochs=50,
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
    random_state=42,
):
    """
    SVM optimization using the Adam algorithm.
    Maintains moving averages of the gradient (m) and its square (v).
    """
    kp = kernel_params or {}
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train_svm, dtype=float)
    y_te = np.asarray(y_test_svm, dtype=float)

    K_tr = compute_kernel_matrix(X_tr, X_tr, kernel_name, **kp)
    K_te = compute_kernel_matrix(X_tr, X_te, kernel_name, **kp)

    n = len(y_tr)
    alpha = np.zeros(n)
    b = 0.0

    # Adam initializations
    m_alpha, v_alpha = np.zeros(n), np.zeros(n)
    m_b, v_b = 0.0, 0.0
    t = 0  # timestep for bias correction

    rng = np.random.default_rng(random_state)
    train_hist, test_hist = [], []
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        for j in rng.permutation(n):
            t += 1

            margin = y_tr[j] * (K_tr[j] @ alpha - b)

            grad_alpha = 2 * lambda_reg * alpha
            grad_b = 0.0

            if margin < 1.0:
                grad_alpha[j] -= y_tr[j]
                grad_b += y_tr[j]

            m_alpha = beta1 * m_alpha + (1 - beta1) * grad_alpha
            v_alpha = beta2 * v_alpha + (1 - beta2) * (grad_alpha ** 2)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

            m_hat_alpha = m_alpha / (1 - beta1 ** t)
            v_hat_alpha = v_alpha / (1 - beta2 ** t)
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)

            alpha -= learning_rate * m_hat_alpha / (np.sqrt(v_hat_alpha) + epsilon)
            b -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

        train_hist.append(accuracy_score(y_tr > 0, (K_tr @ alpha - b) > 0))
        test_hist.append(accuracy_score(y_te > 0, (K_te.T @ alpha - b) > 0))

    return {
        "alpha": alpha, "b": b, "X_train": X_tr,
        "kernel_name": kernel_name, "kernel_params": kp,
        "train_acc_history": train_hist, "test_acc_history": test_hist,
        "fit_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Robbins-Monro SVM — dual representation, any registered kernel
# ---------------------------------------------------------------------------

def robbins_monro_svm(
        X_train, y_train_svm, X_test, y_test_svm,
        kernel_name="linear", kernel_params=None,
        lambda_reg=0.01, n_epochs=50, eta0=0.1, decay=0.01,
        random_state=42,
):
    """
    Stochastic subgradient descent for soft-margin SVM via the representer
    theorem, using a Robbins-Monro learning rate schedule.

    Primal loss minimised:
        L = (1/n) sum_j max(0, 1 - y_j * f(x_j)) + lambda * ||w||^2

    Per-step updates (sample j, learning rate eta_t = eta0 / (1 + decay * t)):
        alpha   *= (1 - 2 * eta * lambda_reg)
        alpha[j] += eta * y_j   }  only when margin < 1
        b        -= eta * y_j   }
    """
    kp = kernel_params or {}
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train_svm, dtype=float)
    y_te = np.asarray(y_test_svm, dtype=float)

    K_tr = compute_kernel_matrix(X_tr, X_tr, kernel_name, **kp)
    K_te = compute_kernel_matrix(X_tr, X_te, kernel_name, **kp)

    n = len(y_tr)
    alpha = np.zeros(n)
    b = 0.0
    rng = np.random.default_rng(random_state)

    train_hist, test_hist = [], []
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        eta = eta0 / (1.0 + decay * epoch)
        for j in rng.permutation(n):
            margin = y_tr[j] * (K_tr[j] @ alpha - b)
            alpha *= (1.0 - 2.0 * eta * lambda_reg)
            if margin < 1.0:
                alpha[j] += eta * y_tr[j]
                b -= eta * y_tr[j]

        train_hist.append(accuracy_score(y_tr > 0, (K_tr @ alpha - b) > 0))
        test_hist.append(accuracy_score(y_te > 0, (K_te.T @ alpha - b) > 0))

    return {
        "alpha": alpha, "b": b, "X_train": X_tr,
        "kernel_name": kernel_name, "kernel_params": kp,
        "train_acc_history": train_hist, "test_acc_history": test_hist,
        "fit_time_s": time.perf_counter() - t0,
    }


def predict_rm(result, X_new):
    """Return binary (0/1) predictions for X_new from a fitted SVM result dict."""
    K = compute_kernel_matrix(
        result["X_train"], np.asarray(X_new, dtype=float),
        result["kernel_name"], **result["kernel_params"],
    )
    return (K.T @ result["alpha"] - result["b"] > 0).astype(int)


# ---------------------------------------------------------------------------
# AdaGrad SVM — dual representation, any registered kernel
# ---------------------------------------------------------------------------

def adagrad_svm(
        X_train, y_train_svm, X_test, y_test_svm,
        kernel_name="linear", kernel_params=None,
        lambda_reg=0.01, n_epochs=50, eta0=0.1, eps=1e-8,
        patience=None, random_state=42,
):
    """
    AdaGrad for soft-margin SVM via the representer theorem.

    Uses coordinate-wise adaptive learning rates with sparse per-sample
    gradient updates (only component j is updated each inner step):
        G_t[j] += g_j^2
        alpha[j] -= eta0 / sqrt(G_t[j] + eps) * g_j
    """
    kp = kernel_params or {}
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train_svm, dtype=float)
    y_te = np.asarray(y_test_svm, dtype=float)

    K_tr = compute_kernel_matrix(X_tr, X_tr, kernel_name, **kp)
    K_te = compute_kernel_matrix(X_tr, X_te, kernel_name, **kp)

    n = len(y_tr)
    alpha = np.zeros(n)
    b = 0.0

    G_alpha = np.zeros(n)
    G_b = 0.0

    rng = np.random.default_rng(random_state)
    train_hist, test_hist = [], []
    best_test_acc = -1.0
    best_alpha, best_b = alpha.copy(), b
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        for j in rng.permutation(n):
            margin = y_tr[j] * (K_tr[j] @ alpha - b)

            # sparse gradient: only component j
            grad_j = 2.0 * lambda_reg * alpha[j]
            grad_b = 0.0

            if margin < 1.0:
                grad_j -= y_tr[j]
                grad_b = y_tr[j]

            G_alpha[j] += grad_j ** 2
            G_b += grad_b ** 2

            alpha[j] -= (eta0 / np.sqrt(G_alpha[j] + eps)) * grad_j

            if grad_b != 0.0:
                b -= (eta0 / np.sqrt(G_b + eps)) * grad_b

        tr_acc = accuracy_score(y_tr > 0, (K_tr @ alpha - b) > 0)
        te_acc = accuracy_score(y_te > 0, (K_te.T @ alpha - b) > 0)
        train_hist.append(tr_acc)
        test_hist.append(te_acc)

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_alpha, best_b = alpha.copy(), b
            wait = 0
        else:
            wait += 1

        if patience is not None and wait >= patience:
            break

    if patience is not None:
        alpha, b = best_alpha, best_b

    return {
        "alpha": alpha, "b": b, "X_train": X_tr,
        "kernel_name": kernel_name, "kernel_params": kp,
        "train_acc_history": train_hist, "test_acc_history": test_hist,
        "fit_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Default hyperparameter grids
# ---------------------------------------------------------------------------

RM_PARAM_GRID = [
    {"lambda_reg": lr, "eta0": e, "decay": d, "n_epochs": ne}
    for lr in [0.001, 0.01, 0.1]
    for e  in [0.01, 0.1, 0.5]
    for d  in [0.01, 0.1]
    for ne in [50, 100]
]

ADAGRAD_PARAM_GRID = [
    {"lambda_reg": lr, "eta0": e, "eps": eps, "n_epochs": ne}
    for lr  in [0.001, 0.01, 0.1]
    for e   in [0.01, 0.05, 0.1]
    for eps in [1e-8, 1e-6]
    for ne  in [50, 100]
]

ADAM_PARAM_GRID = [
    {"lambda_reg": lr, "learning_rate": lr2, "beta1": b1,
     "beta2": 0.999, "epsilon": 1e-8, "n_epochs": ne}
    for lr  in [0.001, 0.01, 0.1]
    for lr2 in [0.0001, 0.001, 0.01]
    for b1  in [0.9, 0.5]
    for ne  in [50, 100]
]

# Keys in best_result that are metadata, not optimizer hyperparameters
_META_KEYS = {"kernel", "kernel_params", "mean_cv", "std_cv"}


# ---------------------------------------------------------------------------
# Generic tuning helpers — work with any SVM optimizer function
# ---------------------------------------------------------------------------

def tune_optimizer_joint_cv(
        optimizer_fn, X_train, y_train,
        param_grid,
        kernel_names=None,
        cv=CV_FOLDS,
        random_state=42,
        verbose=True,
):
    """
    Joint grid search over kernels and optimizer hyperparameters.

    Parameters
    ----------
    optimizer_fn : callable — one of robbins_monro_svm, adagrad_svm, adam_svm
    param_grid   : list of dicts with the optimizer's hyperparameters
                   (use RM_PARAM_GRID / ADAGRAD_PARAM_GRID / ADAM_PARAM_GRID)
    kernel_names : subset of KERNEL_REGISTRY keys (default: all)

    Returns
    -------
    dict with results_df, best_result, fit_time_s
    """
    kernel_names = kernel_names or list(KERNEL_REGISTRY.keys())
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)

    rows = []
    best = None
    t0   = time.perf_counter()

    for kernel_name in kernel_names:
        for kp in KERNEL_REGISTRY[kernel_name]["param_grid"]:
            for opt_params in param_grid:
                fold_scores = []
                for tr_idx, val_idx in cv.split(X, y):
                    y_tr  = np.where(y[tr_idx]  == 1,  1, -1).astype(float)
                    y_val = np.where(y[val_idx] == 1,  1, -1).astype(float)
                    res = optimizer_fn(
                        X[tr_idx], y_tr, X[val_idx], y_val,
                        kernel_name=kernel_name, kernel_params=kp,
                        random_state=random_state, **opt_params,
                    )
                    fold_scores.append(res["test_acc_history"][-1])

                s   = np.array(fold_scores)
                row = {"kernel": kernel_name, "kernel_params": kp,
                       **opt_params, "mean_cv": s.mean(), "std_cv": s.std()}
                rows.append(row)
                if best is None or s.mean() > best["mean_cv"]:
                    best = row.copy()

        if verbose:
            print(f"  {kernel_name:8s}  combos tested: {len(rows):>4d}  "
                  f"best_cv so far: {best['mean_cv']:.4f}  "
                  f"[{time.perf_counter() - t0:.0f}s]")

    results_df = (
        pd.DataFrame(rows)
        .sort_values("mean_cv", ascending=False)
        .reset_index(drop=True)
    )
    return {"results_df": results_df, "best_result": best,
            "fit_time_s": time.perf_counter() - t0}


def fit_best(
        optimizer_fn, X_train, y_train_svm, X_test, y_test_svm,
        best_result, n_epochs_override=None,
):
    """
    Refit optimizer_fn on the full training set using the best configuration
    found by tune_optimizer_joint_cv.
    """
    opt_params = {k: v for k, v in best_result.items() if k not in _META_KEYS}
    if n_epochs_override is not None:
        opt_params["n_epochs"] = int(n_epochs_override)
    return optimizer_fn(
        X_train, y_train_svm, X_test, y_test_svm,
        kernel_name=best_result["kernel"],
        kernel_params=best_result["kernel_params"],
        **opt_params,
    )


def run_svm_experiments(
        optimizer_fn, param_grid,
        filepath="../data/ridings.csv",
        n_epochs_override=None,
        verbose=True,
):
    """
    End-to-end pipeline: load data → joint CV search → refit on full train set.

    Parameters
    ----------
    optimizer_fn : robbins_monro_svm | adagrad_svm | adam_svm
    param_grid   : RM_PARAM_GRID | ADAGRAD_PARAM_GRID | ADAM_PARAM_GRID

    Returns
    -------
    dict with data, search, fit, y_pred, test_acc
    """
    data = prepare_train_test_data(filepath=filepath)

    search = tune_optimizer_joint_cv(
        optimizer_fn, data["X_train_proc"], data["y_train"],
        param_grid=param_grid, verbose=verbose,
    )
    fit = fit_best(
        optimizer_fn,
        data["X_train_proc"], data["y_train_svm"],
        data["X_test_proc"],  data["y_test_svm"],
        best_result=search["best_result"],
        n_epochs_override=n_epochs_override,
    )
    y_pred   = predict_rm(fit, data["X_test_proc"])
    test_acc = accuracy_score(data["y_test"], y_pred)

    return {"data": data, "search": search, "fit": fit,
            "y_pred": y_pred, "test_acc": test_acc}


# ---------------------------------------------------------------------------
# MLP optimizer comparison
# ---------------------------------------------------------------------------

def tune_mlp_optimizers(X_train, y_train, X_test, y_test, cv=CV_FOLDS):
    """
    Compare SGD, Adam, and L-BFGS solvers in MLPClassifier via grid search.
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


def run_optimizer_experiments(filepath="../data/ridings.csv"):
    """End-to-end MLP optimizer experiment entry point."""
    data   = prepare_train_test_data(filepath=filepath)
    result = tune_mlp_optimizers(
        X_train=data["X_train_proc"], y_train=data["y_train"],
        X_test=data["X_test_proc"],  y_test=data["y_test"],
    )
    return {
        "data": data,
        "result": result,
        "summary_df": build_optimizer_results_summary(result),
    }
# ---------------------------------------------------------------------------
# Repeated split evaluation
# ---------------------------------------------------------------------------

def repeated_eval_comparison(
        experiments,
        filepath="../data/ridings.csv",
        n_repetitions=10,
        seeds=None,
):
    """
    Evaluate pre-tuned SVM configurations across repeated stratified train/test
    splits to get robust accuracy estimates.

    Hyperparameters are fixed (from a prior CV search); only the train/test
    split randomness changes across repetitions.

    Parameters
    ----------
    experiments : list of dicts, each with keys:
        "name"              — str label for the optimizer
        "optimizer_fn"      — callable (robbins_monro_svm | adagrad_svm | adam_svm)
        "best_result"       — dict from tune_optimizer_joint_cv
        "n_epochs_override" — int or None
    filepath : str
    n_repetitions : int
        Number of repeated splits (default 10).
    seeds : list[int] or None
        Explicit random seeds. If None, uses ``range(n_repetitions)``.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per optimizer — Mean Accuracy, Std, Min, Max.
    raw_df : pd.DataFrame
        One row per (optimizer × seed) — Optimizer, Seed, Accuracy.
    """
    seeds = list(seeds) if seeds is not None else list(range(n_repetitions))

    raw_rows = []
    for seed in seeds:
        data = prepare_train_test_data(filepath=filepath, random_state=seed)
        for exp in experiments:
            fit = fit_best(
                exp["optimizer_fn"],
                data["X_train_proc"], data["y_train_svm"],
                data["X_test_proc"],  data["y_test_svm"],
                best_result=exp["best_result"],
                n_epochs_override=exp.get("n_epochs_override"),
            )
            y_pred = predict_rm(fit, data["X_test_proc"])
            acc = accuracy_score(data["y_test"], y_pred)
            raw_rows.append({"Optimizer": exp["name"], "Seed": seed, "Accuracy": acc})

    raw_df = pd.DataFrame(raw_rows)

    summary_df = (
        raw_df.groupby("Optimizer")["Accuracy"]
        .agg(
            Mean="mean",
            Std="std",
            Min="min",
            Max="max",
        )
        .rename(columns={"Mean": "Mean Accuracy", "Std": "Std Accuracy"})
        .reset_index()
        .sort_values("Mean Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.index = summary_df.index + 1
    summary_df.index.name = "#"
    summary_df["Mean Accuracy"] = summary_df["Mean Accuracy"].map("{:.4f}".format)
    summary_df["Std Accuracy"]  = summary_df["Std Accuracy"].map("±{:.4f}".format)
    summary_df["Min"]           = summary_df["Min"].map("{:.4f}".format)
    summary_df["Max"]           = summary_df["Max"].map("{:.4f}".format)

    return summary_df, raw_df

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rm_eval_report(y_true, y_pred, model_name="RM SVM"):
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def build_optimizer_results_summary(result):
    return pd.DataFrame({
        "Best CV Score": [result["best_cv_score"]],
        "Test Score":    [result["test_acc"]],
        "Best Params":   [result["best_params"]],
    })
