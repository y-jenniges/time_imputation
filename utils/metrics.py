import numpy as np
from scipy import stats


def taylor_skill(y_true, y_pred, coords=None, per_variable=True):
    """
    Compute Taylor Skill Score for oceanographic variables.

    Args:
        y_true: (N,D) array of observed values
        y_pred: (N,D) array of predicted values
        coords: (N,4) array of [lat, lon, depth, time], optional
        per_variable: if True, return score per variable; else global

    Returns:
        dict of Taylor Skill scores (per variable or global)
    """
    assert y_true.shape == y_pred.shape
    n_vars = y_true.shape[1]

    ts_scores = {}

    for i in range(n_vars):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) == 0:
            ts_scores[i] = np.nan
            continue

        yt = yt[mask]
        yp = yp[mask]

        # Standard deviations
        std_true = np.std(yt)
        std_pred = np.std(yp)

        # Correlation
        r = np.corrcoef(yt, yp)[0,1]

        # Taylor Skill (original formulation)
        ts = (4 * (1 + r)**2) / ((std_pred/std_true + 1)**2)

        ts_scores[i] = ts

    if not per_variable:
        # Weighted global score (mean over variables)
        ts_scores = {"Global": np.nanmean(list(ts_scores.values()))}

    return ts_scores


def compute_metrics(y_true, y_pred, var_names=None, coords=None):
    """
    Compute global and per-variable metrics for model evaluation.
    Args:
        y_true: numpy array of shape (N, D)
        y_pred: numpy array of shape (N, D)
        var_names: list of variable names (optional)
    Returns:
        metrics: dict with global metrics and dicts of per-variable metrics
    """
    # Ensure same shape
    # assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"

    n_vars = y_true.shape[1]
    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    # Initialize containers
    var_metrics = {}
    global_valid_true = []
    global_valid_pred = []

    for i, name in enumerate(var_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) == 0:
            var_metrics[name] = {"RMSE": np.nan, "NSE": np.nan, "Pearson": np.nan}
            continue

        yt = yt[mask]
        yp = yp[mask]

        mse = np.mean((yt - yp) ** 2)
        rmse = np.sqrt(mse)
        nse = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
        r, p = stats.pearsonr(yt, yp)
        #  ts = taylor_skill(yt, yp, coords=coords) if coords is not None else np.nan

        var_metrics[name] = {"RMSE": rmse, "NSE": nse, "Pearson": r, "Pearson_p": p}  # , "Taylor_skill": ts}
        global_valid_true.append(yt)
        global_valid_pred.append(yp)

    # Combine for global metrics
    global_valid_true = np.concatenate(global_valid_true)
    global_valid_pred = np.concatenate(global_valid_pred)

    mse_global = np.mean((global_valid_true - global_valid_pred) ** 2)
    rmse_global = np.sqrt(mse_global)
    nse_global = 1 - np.sum((global_valid_true - global_valid_pred) ** 2) / np.sum(
        (global_valid_true - np.mean(global_valid_true)) ** 2
    )
    r_global, p_global = stats.pearsonr(global_valid_true, global_valid_pred)

    metrics = {
        "Global": {"RMSE": rmse_global, "NSE": nse_global, "Pearson": r_global, "Pearson_p": p_global},
        "PerVariable": var_metrics
    }
    return metrics
