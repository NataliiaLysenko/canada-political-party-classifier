import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cv_tuning import prepare_train_test_data
from optimizers import run_optimizer_experiments


def sigmoid(z):
    """
    Numerically stable sigmoid function.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias_column(X):
    """
    Add a column of 1s to the front of X for the intercept term.
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])


def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    """
    Compute average binary cross-entropy loss.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )


def robbins_monro_logistic(
    X_train,
    y_train,
    epochs=30,
    eta0=0.1,
    decay=0.01,
    random_state=42,
):
    """
    Robbins-Monro style stochastic approximation for logistic regression.

    Update rule:
        theta_{t+1} = theta_t - eta_t * g_t
    where g_t is the stochastic gradient from one randomly chosen sample.

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_features)
        Preprocessed training features.
    y_train : ndarray, shape (n_samples,)
        Binary labels in {0,1}.
    epochs : int
        Number of passes through the training data.
    eta0 : float
        Initial learning rate.
    decay : float
        Step-size decay parameter.
    random_state : int
        Random seed.

    Returns
    -------
    result : dict
        Contains learned weights, loss history, and configuration.
    """
    rng = np.random.default_rng(random_state)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # add intercept
    X_train_b = add_bias_column(X_train)

    n_samples, n_features = X_train_b.shape
    theta = np.zeros(n_features)

    loss_history = []
    step_count = 0

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)

        for idx in indices:
            x_i = X_train_b[idx]          # shape: (n_features,)
            y_i = y_train[idx]            # scalar

            # predicted probability
            p_i = sigmoid(x_i @ theta)

            # stochastic gradient for logistic loss
            grad_i = (p_i - y_i) * x_i

            # Robbins-Monro step size
            eta_t = eta0 / (1.0 + decay * step_count)

            # parameter update
            theta = theta - eta_t * grad_i

            step_count += 1

        # track full training loss after each epoch
        train_probs = sigmoid(X_train_b @ theta)
        epoch_loss = binary_cross_entropy(y_train, train_probs)
        loss_history.append(epoch_loss)

    return {
        "theta": theta,
        "loss_history": loss_history,
        "epochs": epochs,
        "eta0": eta0,
        "decay": decay,
    }


def predict_rm_logistic(X, theta, threshold=0.5):
    """
    Predict class labels from learned RM logistic weights.
    """
    X = np.asarray(X, dtype=float)
    X_b = add_bias_column(X)
    probs = sigmoid(X_b @ theta)
    preds = (probs >= threshold).astype(int)
    return preds, probs


def evaluate_rm_baseline(
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
    epochs=30,
    eta0=0.1,
    decay=0.01,
    random_state=42,
):
    """
    Train and evaluate Robbins-Monro logistic baseline.
    """
    rm_result = robbins_monro_logistic(
        X_train=X_train_proc,
        y_train=y_train,
        epochs=epochs,
        eta0=eta0,
        decay=decay,
        random_state=random_state,
    )

    y_pred, y_prob = predict_rm_logistic(X_test_proc, rm_result["theta"])
    test_acc = accuracy_score(y_test, y_pred)

    return {
        "theta": rm_result["theta"],
        "loss_history": rm_result["loss_history"],
        "epochs": epochs,
        "eta0": eta0,
        "decay": decay,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "test_acc": test_acc,
    }


def print_algorithm_eval_report(y_true, y_pred, model_name="Algorithm Model"):
    """
    Print standard classification metrics.
    """
    print(f"=== {model_name} ===")
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()


def compare_rm_with_optimizer_sgd(filepath="data/ridings.csv"):
    """
    Compare a Robbins-Monro logistic baseline against the best optimizer-based model.

    Returns
    -------
    dict containing:
        data
        rm_result
        optimizer_result
        comparison_df
    """
    # shared prepared data
    data = prepare_train_test_data(filepath)

    X_train_proc = data["X_train_proc"]
    X_test_proc = data["X_test_proc"]
    y_train = np.asarray(data["y_train"])
    y_test = np.asarray(data["y_test"])

    # Robbins-Monro baseline
    rm_result = evaluate_rm_baseline(
        X_train_proc=X_train_proc,
        y_train=y_train,
        X_test_proc=X_test_proc,
        y_test=y_test,
        epochs=30,
        eta0=0.1,
        decay=0.01,
        random_state=42,
    )

    # Best optimizer experiment (contains SGD/Adam/LBFGS search in MLP)
    optimizer_out = run_optimizer_experiments(filepath)
    optimizer_result = optimizer_out["result"]

    comparison_df = pd.DataFrame([
        {
            "Model": "Robbins-Monro Logistic Baseline",
            "CV Score": np.nan,   # no CV loop implemented here
            "Test Accuracy": rm_result["test_acc"],
            "Best Params / Notes": (
                f"epochs={rm_result['epochs']}, "
                f"eta0={rm_result['eta0']}, "
                f"decay={rm_result['decay']}"
            ),
        },
        {
            "Model": "Best Optimizer Model (MLP)",
            "CV Score": optimizer_result["best_cv_score"],
            "Test Accuracy": optimizer_result["test_acc"],
            "Best Params / Notes": str(optimizer_result["best_params"]),
        },
    ])

    return {
        "data": data,
        "rm_result": rm_result,
        "optimizer_result": optimizer_result,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    out = compare_rm_with_optimizer_sgd("data/ridings.csv")

    print("=== RM vs Optimizer Comparison ===")
    print(out["comparison_df"])
    print()

    print_algorithm_eval_report(
        out["data"]["y_test"],
        out["rm_result"]["y_pred"],
        model_name="Robbins-Monro Logistic Baseline"
    )

    print_algorithm_eval_report(
        out["data"]["y_test"],
        out["optimizer_result"]["y_pred"],
        model_name="Best Optimizer Model (MLP)"
    )