"""Evaluation metrics (RMSE, MAE, R², KGE, Wasserstein, …) for imputation model comparison."""

import numpy as np
from scipy import stats
from permetrics import RegressionMetric

import config


def kge(y_true, y_pred):
    """
    Modified Kling-Gupta efficiency (Gupta et al. 2009).

    Same formula as used in ``permetrics``, but the correlation is set to 0 for a constant prediction.
    ``permetrics`` returns NaN there (0/0) if numerical rounding yields exactly 0 variance and -0.41 otherwise.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N,)
        Observed and predicted values without NaNs.

    Returns
    -------
    float
        KGE.
    """
    # Means and sum of squared deviations
    m1, m2 = np.mean(y_true), np.mean(y_pred)
    ss_true = np.sum((y_true - m1) ** 2)
    ss_pred = np.sum((y_pred - m2) ** 2)

    tol = 1e-20  # tolerance catching float noise
    if ss_pred <= tol * len(y_pred):
        # Constant prediction: Undefined correlation, treated as 0
        r = 0.0
        std_pred = 0.0
    else:
        # Not-zero variance
        r = np.sum((y_true - m1) * (y_pred - m2)) / np.sqrt(ss_true * ss_pred)
        std_pred = np.std(y_pred)

    beta = m2 / m1
    gamma = (std_pred / m2) / (np.std(y_true) / m1)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))


def compute_metrics(y_true, y_pred, var_names=None):
    """
    Compute global and per-variable regression metrics.

    Parameters
    ----------
    y_true : ndarray, shape (N, F)
        Ground-truth values; ``NaN`` at missing positions.
    y_pred : ndarray, shape (N, F)
        Predicted values; ``NaN`` at missing positions.
    var_names : list[str] or None, default=None
        Variable names; defaults to ``['var_0', 'var_1', …]``.

    Returns
    -------
    dict
        ``{'Global': {metric: value}, 'PerVariable': {name: {metric: value}}}``.
        Global metrics are computed by aggregating over all valid entries across variables.
    """
    # Variables
    n_vars = y_true.shape[1]
    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    # Initialize containers
    var_metrics = {}
    global_valid_true = []
    global_valid_pred = []

    # Per-variable  metrics
    for i, name in enumerate(var_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # Check number of valid entries (min. 2 required for metrics)
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) <= 1:
            var_metrics[name] = {m: np.nan for m in config.EVAL_METRICS}
            continue

        yt = yt[mask]
        yp = yp[mask]

        # Compute metrics
        evaluator = RegressionMetric(yt, yp)
        var_metrics[name] = evaluator.get_metrics_by_list_names(config.EVAL_METRICS)
        if "KGE" in var_metrics[name]:
            var_metrics[name]["KGE"] = kge(yt, yp)  # robust to constant predictions (e.g. mean imputation)

        wasserstein = stats.wasserstein_distance(yt, yp)
        var_metrics[name]["wasserstein"] = wasserstein

        # Store in dict
        global_valid_true.append(yt)
        global_valid_pred.append(yp)

    # Global metrics
    if len(global_valid_true) == 0:
        global_metrics = {m: np.nan for m in config.EVAL_METRICS}
    else:
        # Combine for global metrics
        global_valid_true = np.concatenate(global_valid_true)
        global_valid_pred = np.concatenate(global_valid_pred)

        # Compute metrics
        evaluator = RegressionMetric(global_valid_true, global_valid_pred)
        global_metrics = evaluator.get_metrics_by_list_names(config.EVAL_METRICS)

    metrics = {
        "Global": global_metrics,
        "PerVariable": var_metrics
    }
    return metrics
