"""Dataset loading and feature preprocessing for the COMFORT ocean dataset."""

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

import config


def load_dataset(data_path=None) -> DataFrame:
    """
    Load the CSV dataset (based on COMFORT), convert timestamps to years, and drop unused columns.

    Parameters
    ----------
    data_path : str or Path or None, default=None
        Path to the CSV file. Uses ``config.data_path`` when None.

    Returns
    -------
    DataFrame
        Dataset with DATEANDTIME converted to integer years and 'idx'/'water' dropped.
    """
    if data_path is None:
        data_path = config.data_path

    df = pd.read_csv(data_path)
    df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year

    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    if "water" in df.columns:
        df = df.drop(columns=["water"])
    return df


def preprocess(coords, values, coord_names, parameter_names, cyclic_time: bool = False,
               scaler_dict: dict = None):
    """
    Scale and encode coordinates and feature values.

    Latitude and longitude are mapped to Cartesian (x, y, z) on the unit sphere.
    Depth and time use MinMax scaling (or cyclic sin/cos encoding for time).
    Features are independently MinMax-scaled on observed values only.

    Parameters
    ----------
    coords : Tensor, shape (N, 4)
        Raw coordinates: [lat, lon, depth, time].
    values : Tensor, shape (N, F)
        Raw feature values; ``NaN`` marks missing positions.
    coord_names : list[str]
        Names of the coordinate columns (e.g. ``config.coordinates``).
    parameter_names : list[str]
        Names of the feature columns (e.g. ``config.parameters``).
    cyclic_time : bool, default=False
        Encode time as (cos, sin) pair for periodic monthly data.
    scaler_dict : dict or None, default=None
        Pre-fitted scalers to apply; new scalers are created and added when None.

    Returns
    -------
    x_scaled : Tensor, shape (N, coord_dim)
        Preprocessed coordinate matrix (coord_dim=5 for linear, 6 for cyclic time).
    y_scaled : Tensor, shape (N, F)
        Preprocessed feature matrix; ``NaN`` preserved at missing positions.
    scaler_dict : dict
        Fitted scalers keyed by coordinate/feature name.
    """
    if scaler_dict is None:
        scaler_dict = {}

    coord_dict = {name: coords[:, i] for i, name in enumerate(coord_names)}

    lat_radians = np.pi / 180 * coord_dict["LATITUDE"]
    lon_radians = np.pi / 180 * coord_dict["LONGITUDE"]

    x = (np.cos(lat_radians) * np.cos(lon_radians) + 1) / 2
    y = (np.cos(lat_radians) * np.sin(lon_radians) + 1) / 2
    z = (np.sin(lat_radians) + 1) / 2

    if "LEV_M" in scaler_dict:
        depth_scaled = scaler_dict["LEV_M"].transform(coord_dict["LEV_M"].reshape(-1, 1))
    else:
        scaler_depth = MinMaxScaler()
        depth_scaled = scaler_depth.fit_transform(coord_dict["LEV_M"].reshape(-1, 1))
        scaler_dict["LEV_M"] = scaler_depth

    if cyclic_time:
        ctime = (np.cos(2 * np.pi * coord_dict["DATEANDTIME"] / 12) + 1) / 2
        stime = (np.sin(2 * np.pi * coord_dict["DATEANDTIME"] / 12) + 1) / 2
        x_scaled = np.column_stack([x, y, z, depth_scaled, ctime, stime])
    else:
        if "DATEANDTIME" in scaler_dict:
            scaled_time = scaler_dict["DATEANDTIME"].transform(coord_dict["DATEANDTIME"].reshape(-1, 1))
        else:
            scaler_time = MinMaxScaler()
            scaled_time = scaler_time.fit_transform(coord_dict["DATEANDTIME"].reshape(-1, 1))
            scaler_dict["DATEANDTIME"] = scaler_time
        x_scaled = np.column_stack([x, y, z, depth_scaled, scaled_time])

    observed_mask = ~torch.isnan(values).cpu().numpy()

    y_scaled = np.full(values.shape, np.nan)
    for i, pn in enumerate(parameter_names):
        y_col = values[:, i].cpu().numpy()
        obs = observed_mask[:, i]
        y_obs = y_col[obs].reshape(-1, 1)

        if len(y_obs) == 0:
            continue

        if pn in scaler_dict:
            y_scaled[obs, i] = scaler_dict[pn].transform(y_obs).flatten()
        else:
            scaler_feat = MinMaxScaler()
            scaler_feat.fit(y_obs)
            y_scaled[obs, i] = scaler_feat.transform(y_obs).flatten()
            scaler_dict[pn] = scaler_feat

    x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
    y_scaled = torch.tensor(y_scaled, dtype=torch.float32)

    return x_scaled, y_scaled, scaler_dict


def get_availability_per_time(coords, values):
    """Return the fraction of observed feature values per time step."""
    time = coords[:, -1].long()

    valid = ~torch.isnan(values)
    n_features = values.shape[1]

    total = torch.bincount(time).float() * n_features

    available = torch.zeros_like(total).scatter_add(
        0,
        time.repeat_interleave(n_features),
        valid.flatten().float()
    )

    return available / total.clamp(min=1)
