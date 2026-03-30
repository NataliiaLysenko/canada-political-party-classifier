"""
kernels.py — kernel functions, kernelized SVM (TODO), and cross-validation framework (TODO).

Import what you need in any notebook: e.g., from kernels import compute_kernel_matrix, KernelSVM, grid_search_cv, final_evaluate
"""


import numpy as np

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
            {"degree": d, "gamma": g, "coef0": 1.0}
            for d in [2, 3, 4]
            for g in [0.01, 0.1, 1.0]
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


def compute_kernel_matrix(X1, X2, kernel_name, **kernel_params):
    """Look up kernel by name, merge defaults with overrides, return Gram matrix."""
    entry = KERNEL_REGISTRY[kernel_name]
    params = {**entry["defaults"], **kernel_params}
    return entry["fn"](X1, X2, **params)


