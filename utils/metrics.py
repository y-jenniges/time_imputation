import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy import stats
from permetrics import RegressionMetric

import config


def compute_metrics(y_true, y_pred, var_names=None):
    """
    Compute global and per-variable metrics for model evaluation.

    :param y_true: True values, numpy array of shape (n_samples, n_variables)
    :param y_pred: Predicted values, numpy array of shape (n_samples, n_variables)
    :param var_names: List of variable names (optional)
    :return: Dict with global metrics and per-variable metrics
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

        # Check number of valid entries
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) == 0:
            var_metrics[name] = {m: np.nan for m in config.EVAL_METRICS}
            continue

        yt = yt[mask]
        yp = yp[mask]

        # Compute metrics
        evaluator = RegressionMetric(yt, yp)
        var_metrics[name] = evaluator.get_metrics_by_list_names(config.EVAL_METRICS)

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
