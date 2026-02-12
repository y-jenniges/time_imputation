from typing import Tuple
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


class OceanMAEDataset(Dataset):
    """ Dataset for OceanMAE.
    Returns:
        coords: [N, coord_dim] tensor
        values: [N, value_dim] tensor
        mask_indices: [N, value_dim] boolean tensor, additional masking
        n_neighbors: int, number of neighbours
    """

    def __init__(self,
                 coords: torch.Tensor,
                 values: torch.Tensor,
                 neighbour_indices: torch.Tensor,
                 mask_indices: np.ndarray | None = None,
                 ):
        self.coords = coords
        self.values = values
        self.n_samples, self.n_features = values.shape
        self.neighbour_indices = neighbour_indices

        # Initial mask: True = observed (False = NaN)
        self.feature_mask = ~torch.isnan(values)

        # Mask indices
        if mask_indices is not None:
            mask_indices = torch.as_tensor(mask_indices, dtype=torch.long)
            self.feature_mask[mask_indices, :] = False

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Query point
        q_feat = self.values[idx]
        q_mask = self.feature_mask[idx]
        q_coord = self.coords[idx]

        # Neighbours
        n_idx = self.neighbour_indices[idx]
        n_feat = self.values[n_idx]
        n_mask = self.feature_mask[n_idx]
        n_coord = self.coords[n_idx]

        # Relative positions
        rel_positions = n_coord - q_coord

        sample = {
            "query_features": q_feat,
            "query_mask": q_mask,
            "query_coords": q_coord,
            "neighbour_features": n_feat,
            "neighbour_mask": n_mask,
            "rel_positions": rel_positions
        }

        return sample


def prepare_mae_loaders(coords: torch.Tensor,
                        values: torch.Tensor,
                        train_idx: np.ndarray,
                        val_idx: np.ndarray,
                        test_idx: np.ndarray,
                        batch_size: int,
                        generator: torch.Generator,
                        cyclic_time: bool = False,
                        n_neighbours: int = 24,
                        ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict, int, int]:
    # Split raw data
    coords_train_raw = coords[train_idx]
    values_train_raw = values[train_idx]

    # Preprocess training data
    coords_train, values_train, train_scaler_dict = preprocess(coords=coords_train_raw,
                                                               values=values_train_raw,
                                                               coord_names=config.coordinates,
                                                               parameter_names=config.parameters,
                                                               cyclic_time=cyclic_time,
                                                               scaler_dict=None)

    # Preprocess full dataset using training scalers
    coords_full, values_full, _ = preprocess(
        coords=coords,
        values=values,
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=train_scaler_dict
    )

    # Build graph for neighbour search
    n_samples = values.shape[0]
    neighbours = NearestNeighbors(n_neighbors=min(n_neighbours, n_samples), algorithm="auto").fit(coords.cpu().numpy())
    neighbour_indices = neighbours.kneighbors(coords.cpu().numpy(), return_distance=False)[:, 1:]  # Exclude self
    neighbour_indices = torch.as_tensor(neighbour_indices, dtype=torch.long, device="cpu")

    # Define datasets
    train_dataset = OceanMAEDataset(coords=coords_full, values=values_full, mask_indices=train_idx, neighbour_indices=neighbour_indices)
    val_dataset = OceanMAEDataset(coords=coords_full, values=values_full, mask_indices=val_idx, neighbour_indices=neighbour_indices)
    test_dataset = OceanMAEDataset(coords=coords_full, values=values_full, mask_indices=test_idx, neighbour_indices=neighbour_indices)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loader for complete dataset (for complete reconstruction)
    full_dataset = OceanMAEDataset(coords=coords_full, values=values_full, mask_indices=None, neighbour_indices=neighbour_indices)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(1), values_train.size(1)


def prepare_sklearn_data(df, train_idx, val_idx, test_idx):
    values_raw = df[config.parameters].to_numpy().astype(float)
    coords_raw = df[config.coordinates].to_numpy().astype(float)

    # Scale training data
    coords_train_scaled, values_train_scaled, scaler_dict = preprocess(
        coords=torch.tensor(coords_raw[train_idx]),
        values=torch.tensor(values_raw[train_idx]),
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=False,
        scaler_dict=None
    )
    values_train_scaled = values_train_scaled.cpu().numpy()
    coords_train_scaled = coords_train_scaled.cpu().numpy()

    # Scale test and validation data with scalers from training
    coords_val_scaled, values_val_scaled, _ = preprocess(
        coords=torch.tensor(coords_raw[val_idx]),
        values=torch.tensor(values_raw[val_idx]),
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=False,
        scaler_dict=scaler_dict
    )
    values_val_scaled = values_val_scaled.cpu().numpy()
    coords_val_scaled = coords_val_scaled.cpu().numpy()

    coords_test_scaled, values_test_scaled, _ = preprocess(
        coords=torch.tensor(coords_raw[test_idx]),
        values=torch.tensor(values_raw[test_idx]),
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=False,
        scaler_dict=scaler_dict
    )
    values_test_scaled = values_test_scaled.cpu().numpy()
    coords_test_scaled = coords_test_scaled.cpu().numpy()

    # Prepare training input
    coord_dim = coords_train_scaled.shape[1]
    values_dim = values_train_scaled.shape[1]
    n_samples = values_raw.shape[0]
    x_train = np.full((n_samples, coord_dim + values_dim), np.nan)
    x_train[:, coord_dim:][train_idx] = values_train_scaled
    x_train[:, coord_dim:][val_idx] = np.nan
    x_train[:, coord_dim:][test_idx] = np.nan

    x_train[:, :coord_dim][train_idx] = coords_train_scaled
    x_train[:, :coord_dim][val_idx] = coords_val_scaled
    x_train[:, :coord_dim][test_idx] = coords_test_scaled

    # Generate y_true (scaled)
    y_true_scaled = x_train[:, coord_dim:].copy()
    y_true_scaled[val_idx, :] = values_val_scaled
    y_true_scaled[test_idx, :] = values_test_scaled

    return x_train, y_true_scaled, scaler_dict, coord_dim, values_dim


def random_feature_mask(batch_size: int,
                        feature_dim: int,
                        mask_ratio: float = 0.5,
                        n_neighbours: int = 24,
                        device="cpu",
                        mask_query: bool = True,
                        mask_neighbours: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Query mask
    if mask_query:
        # Init mask with "False"
        query_mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)

        # Get number of features to mask
        n_query_mask = max(1, int(feature_dim * mask_ratio))

        # Per sample random ordering of features
        rand_idx_q = torch.rand(batch_size, feature_dim, device=device).argsort(dim=1)

        # Scatter into bool mask (each row then has n_mask features set to True)
        query_mask.scatter_(1, rand_idx_q[:, :n_query_mask], True)
    else:
        query_mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)

    # Neighbour mask
    if mask_neighbours:
        # Init mask with "False"
        neighbour_mask = torch.zeros(batch_size, n_neighbours, feature_dim, dtype=torch.bool, device=device)

        # Get number of features to mask
        n_neighbour_mask = max(1, int(feature_dim * mask_ratio))

        # Per sample random ordering of features
        rand_idx_n = torch.rand(batch_size, n_neighbours, feature_dim, device=device).argsort(dim=2)

        # Scatter into bool mask (each row then has n_mask features set to True)
        neighbour_mask.scatter_(2, rand_idx_n[:, :, :n_neighbour_mask], True)
    else:
        neighbour_mask = torch.zeros(batch_size, n_neighbours, feature_dim, dtype=torch.bool, device=device)

    return query_mask, neighbour_mask


def preprocess(coords, values, coord_names, parameter_names, cyclic_time=False, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}

    # Convert to dict for easier column referencing
    coord_dict = {name: coords[:, i] for i, name in enumerate(coord_names)}

    # Transform latitude and longitude to cartesian coords on unit sphere (and scale to range [0, 1])
    lat_radians = np.pi / 180 * coord_dict["LATITUDE"]
    lon_radians = np.pi / 180 * coord_dict["LONGITUDE"]

    x = (np.cos(lat_radians) * np.cos(lon_radians) + 1 ) / 2
    y = (np.cos(lat_radians) * np.sin(lon_radians) + 1 ) / 2
    z = (np.sin(lat_radians) + 1 ) / 2

    # Scale depth
    if "LEV_M" in scaler_dict.keys():
        depth_scaled = scaler_dict["LEV_M"].transform(coord_dict["LEV_M"].reshape(-1, 1))
    else:
        scaler_depth = MinMaxScaler()
        depth_scaled = scaler_depth.fit_transform(coord_dict["LEV_M"].reshape(-1, 1))
        scaler_dict["LEV_M"] = scaler_depth

    # Cyclic encoding of months / minmax scaling for years
    if cyclic_time:
        ctime = (np.cos(2 * np.pi * coord_dict["DATEANDTIME"] / 12) + 1) / 2
        stime = (np.sin(2 * np.pi * coord_dict["DATEANDTIME"] / 12) + 1) / 2

        # Combine coordinates
        x_scaled = np.column_stack([x, y, z, depth_scaled, ctime, stime])
    else:
        if "DATEANDTIME" in scaler_dict.keys():
            scaled_time = scaler_dict["DATEANDTIME"].transform(coord_dict["DATEANDTIME"].reshape(-1, 1))
        else:
            scaler_time = MinMaxScaler()
            scaled_time = scaler_time.fit_transform(coord_dict["DATEANDTIME"].reshape(-1, 1))
            scaler_dict["DATEANDTIME"] = scaler_time

        # Combine coordinates
        x_scaled = np.column_stack([x, y, z, depth_scaled, scaled_time])

    # Generate mask of observed values
    observed_mask = ~torch.isnan(values).cpu().numpy()

    # Scale features individually, ignoring masked values
    y_scaled = np.full(values.shape, np.nan)
    for i, pn in enumerate(parameter_names):
        y_col = values[:, i].cpu().numpy()
        obs = observed_mask[:, i]
        y_obs = y_col[obs].reshape(-1, 1)

        # Transform observed values
        if pn in scaler_dict.keys():
            y_scaled[obs, i] = scaler_dict[pn].transform(y_obs).flatten()
        else:
            scaler_feat = MinMaxScaler()
            scaler_feat.fit(y_obs)
            y_scaled[obs, i] = scaler_feat.transform(y_obs).flatten()
            scaler_dict[pn] = scaler_feat

    # Convert to torch tensors
    x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
    y_scaled = torch.tensor(y_scaled, dtype=torch.float32)

    return x_scaled, y_scaled, scaler_dict


def load_dataset():
    # Load data
    df = pd.read_csv(config.data_path)
    df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year

    # Drop not-needed columns
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    if "water" in df.columns:
        df = df.drop(columns=["water"])
    return df
