"""Evaluation metrics (RMSE, MAE, R², KGE, Wasserstein, …) for imputation model comparison."""

import numpy as np
from scipy import stats
from permetrics import RegressionMetric

import config


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
