import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy import stats


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
            var_metrics[name] = {"RMSE": np.nan, "NSE": np.nan, "Pearson": np.nan, "Pearson_p": np.nan}
            continue

        yt = yt[mask]
        yp = yp[mask]

        # Compute metrics
        mae = mean_absolute_error(yt, yp)
        rmse = root_mean_squared_error(yt, yp)
        r, p = stats.pearsonr(yt, yp)
        r2 = r2_score(yt, yp)

        # Store in dict
        var_metrics[name] = {"MAE": mae, "RMSE": rmse, "Pearson": r, "Pearson_p": p, "R2": r2}
        global_valid_true.append(yt)
        global_valid_pred.append(yp)

    # Global metrics
    if len(global_valid_true) == 0:
        mae_global, rmse_global, nse_global, r_global, p_global, mse_global, r2_global = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        # Combine for global metrics
        global_valid_true = np.concatenate(global_valid_true)
        global_valid_pred = np.concatenate(global_valid_pred)

        # Compute metrics
        mae_global = mean_absolute_error(global_valid_true, global_valid_pred)
        rmse_global = root_mean_squared_error(global_valid_true, global_valid_pred)
        r_global, p_global = stats.pearsonr(global_valid_true, global_valid_pred)
        r2_global = r2_score(global_valid_true, global_valid_pred)

    metrics = {
        "Global": {"MAE": mae_global, "RMSE": rmse_global, "Pearson": r_global, "Pearson_p": p_global, "R2": r2_global},
        "PerVariable": var_metrics
    }
    return metrics
