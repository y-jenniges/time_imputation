from typing import Tuple, Any
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from nn_utils.graph import GraphProvider


class LearnedNeighbourDataset(Dataset):
    def __init__(self, coords: torch.Tensor, values: torch.Tensor, graph_provider: GraphProvider, query_indices: np.ndarray | None = None):
        self.coords = coords
        self.values = values
        self.query_indices = query_indices

        # KNN graph
        self.graph_provider = graph_provider

        # Initial mask: True = observed (False = NaN)
        self.feature_mask = ~torch.isnan(values)

        # Store original indices of query points
        if query_indices is None:
            self.query_indices = torch.arange(values.shape[0], dtype=torch.long)
        else:
            self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)

    def __len__(self):
        return self.query_indices.shape[0]

    def __getitem__(self, idx):
        # Map local query point idx to global idx
        q_idx = self.query_indices[idx]

        # Query point
        q_feat = self.values[q_idx]
        q_mask = self.feature_mask[q_idx]
        q_coord = self.coords[q_idx]

        # Neighbours (global indices)
        n_idx = self.graph_provider.neighbour_indices[q_idx]

        n_feat = self.values[n_idx]
        n_mask = self.feature_mask[n_idx]
        n_coord = self.coords[n_idx]

        # Relative positions
        rel_positions = n_coord - q_coord

        return {
            "query_features": q_feat,
            "query_mask": q_mask,
            "query_coords": q_coord,
            "neighbour_features": n_feat,
            "neighbour_mask": n_mask,
            "neighbour_coords": n_coord,
            "rel_positions": rel_positions
        }



class NeighbourDataset(Dataset):
    """ Dataset returning query point data and neighbourhood data. """

    def __init__(self,
                 coords: torch.Tensor,
                 values: torch.Tensor,
                 neighbour_indices: torch.Tensor,
                 query_indices: np.ndarray | None = None
                 ):
        self.coords = coords
        self.values = values

        # self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)
        self.neighbour_indices = torch.as_tensor(neighbour_indices, dtype=torch.long)

        # Store original indices of query points
        if query_indices is None:
            self.query_indices = torch.arange(values.shape[0], dtype=torch.long)
        else:
            self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)

        # Initial mask: True = observed (False = NaN)
        self.feature_mask = ~torch.isnan(values)

    def __len__(self):
        return self.query_indices.shape[0]

    def __getitem__(self, idx):
        # Map local query point idx to global idx
        q_idx = self.query_indices[idx]

        # Query point
        q_feat = self.values[q_idx]
        q_mask = self.feature_mask[q_idx]
        q_coord = self.coords[q_idx]

        # Neighbours (global indices)
        n_idx = self.neighbour_indices[idx]

        n_feat = self.values[n_idx]
        n_mask = self.feature_mask[n_idx]
        n_coord = self.coords[n_idx]

        # Relative positions
        rel_positions = n_coord - q_coord

        return {
            "query_features": q_feat,
            "query_mask": q_mask,
            "query_coords": q_coord,
            "neighbour_features": n_feat,
            "neighbour_mask": n_mask,
            "neighbour_coords": n_coord,
            "rel_positions": rel_positions
        }


class PointwiseDataset(Dataset):
    """ Dataset returning pointwise data. """

    def __init__(self, coords: torch.Tensor, values: torch.Tensor):
        self.coords = coords
        self.values = values
        self.feature_mask = ~torch.isnan(values)  # Mask indices (True = observed)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return {"features": self.values[idx], "coords": self.coords[idx], "mask": self.feature_mask[idx]}


def prepare_learned_neighbourhood_loaders(coords: torch.Tensor,
                                          values: torch.Tensor,
                                          train_idx: np.ndarray,
                                          val_idx: np.ndarray,
                                          test_idx: np.ndarray,
                                          batch_size: int,
                                          generator: torch.Generator,
                                          graph_provider: GraphProvider,
                                          cyclic_time: bool = False,
                                          n_neighbours: int = 24,
                                          ) -> tuple[
    DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any], dict | dict[
        Any, Any], int, int, None, Tensor, Tensor, Tensor]:
    """
    Preprocess data (use training scalers to scale validation and test sets) and create respective data loaders that
    return query and spatio-temporally neighbouring tokens.

    :param coords: Latitude, longitude, depth and time data
    :param values: Variable data (torch.tensor)
    :param train_idx: Rows meant for training
    :param val_idx: Rows meant for validation
    :param test_idx: Rows meant for testing
    :param batch_size: Batch size
    :param generator: Random number generator
    :param graph_provider: GraphProvider object
    :param cyclic_time: Whether to encode time cyclically (e.g. for monthly data) or not (e.g. for yearly data)
    :param n_neighbours: Number of neighbours to use
    :return:
        - full_loader
        - train_loader
        - val_loader
        - test_loader
        - train_scaler_dict
        - n_coords
        - n_values
        - dists
    """
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
    mask_full = ~torch.isnan(values_full)

    # Build initial graph for neighbour search
    init_encoder = lambda coords, values, mask, t_fourier=None: coords_full
    graph_provider.update(encoder=init_encoder, coords=coords_full, values=values_full, mask=mask_full)

    # Define datasets
    train_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=train_idx, graph_provider=graph_provider)
    val_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=val_idx, graph_provider=graph_provider)
    test_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=test_idx, graph_provider=graph_provider)
    full_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=None, graph_provider=graph_provider)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    dists_train = None
    # print(f"Median distance of training data: {np.nanmedian(dists_train)}")

    return full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(1), values_train.size(1), dists_train, coords_full, values_full, mask_full


def prepare_neighbourhood_loaders(coords: torch.Tensor,
                                  values: torch.Tensor,
                                  train_idx: np.ndarray,
                                  val_idx: np.ndarray,
                                  test_idx: np.ndarray,
                                  batch_size: int,
                                  generator: torch.Generator,
                                  cyclic_time: bool = False,
                                  n_neighbours: int = 24,
                                  ) -> Tuple[
    DataLoader, DataLoader, DataLoader, DataLoader, dict, int, int, np.ndarray]:
    """
    Preprocess data (use training scalers to scale validation and test sets) and create respective data loaders that
    return query and spatio-temporally neighbouring tokens.

    :param coords: Latitude, longitude, depth and time data
    :param values: Variable data (torch.tensor)
    :param train_idx: Rows meant for training
    :param val_idx: Rows meant for validation
    :param test_idx: Rows meant for testing
    :param batch_size: Batch size
    :param generator: Random number generator
    :param cyclic_time: Whether to encode time cyclically (e.g. for monthly data) or not (e.g. for yearly data)
    :param n_neighbours: Number of neighbours to use
    :return:
        - full_loader
        - train_loader
        - val_loader
        - test_loader
        - train_scaler_dict
        - n_coords
        - n_values
        - dists
    """
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
    # Compute neighbours only on non-empty training data points
    train_valid_mask = ~torch.isnan(values_train_raw).all(dim=1)
    train_valid_idx = train_idx[train_valid_mask]
    coords_valid = coords_full[train_valid_idx]

    n_samples = values.shape[0]
    # neighbours_train = NearestNeighbors(n_neighbors=min(n_neighbours+1, n_samples), algorithm="auto").fit(coords_full[train_idx].cpu().numpy())
    neighbours_train = NearestNeighbors(n_neighbors=min(n_neighbours + 1, n_samples), algorithm="auto").fit(
        coords_valid.cpu().numpy())
    dists_train, neighbour_indices_train = neighbours_train.kneighbors(coords_full[train_idx].cpu().numpy(),
                                                                       return_distance=True)
    neighbour_indices_train = torch.as_tensor(neighbour_indices_train[:, 1:], dtype=torch.long,
                                              device="cpu")  # Exclude self and convert to tensor

    dists_val, neighbour_indices_val = neighbours_train.kneighbors(coords_full[val_idx].cpu().numpy(),
                                                                   return_distance=True)
    neighbour_indices_val = torch.as_tensor(neighbour_indices_val[:, :n_neighbours], dtype=torch.long, device="cpu")

    dists_test, neighbour_indices_test = neighbours_train.kneighbors(coords_full[test_idx].cpu().numpy(),
                                                                     return_distance=True)
    neighbour_indices_test = torch.as_tensor(neighbour_indices_test[:, :n_neighbours], dtype=torch.long, device="cpu")

    neighbour_indices_train = train_valid_idx[neighbour_indices_train]
    neighbour_indices_val = train_valid_idx[neighbour_indices_val]
    neighbour_indices_test = train_valid_idx[neighbour_indices_test]

    # Define datasets
    train_dataset = NeighbourDataset(coords=coords_full, values=values_full, query_indices=train_idx,
                                     neighbour_indices=neighbour_indices_train)
    val_dataset = NeighbourDataset(coords=coords_full, values=values_full, query_indices=val_idx,
                                   neighbour_indices=neighbour_indices_val)
    test_dataset = NeighbourDataset(coords=coords_full, values=values_full, query_indices=test_idx,
                                    neighbour_indices=neighbour_indices_test)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loader for complete dataset (for complete reconstruction)
    dists, neighbours_local = neighbours_train.kneighbors(coords_full.cpu().numpy(), return_distance=True)

    neighbour_local = neighbours_local[:, :n_neighbours]
    neighbour_indices = train_valid_idx[neighbour_local]
    neighbour_indices = torch.as_tensor(neighbour_indices, dtype=torch.long, device="cpu")

    # neighbour_indices = torch.as_tensor(neighbour_indices[:, :n_neighbours], dtype=torch.long, device="cpu")
    # neighbour_indices = train_idx[neighbour_indices]
    full_dataset = NeighbourDataset(coords=coords_full, values=values_full, query_indices=None,
                                    neighbour_indices=neighbour_indices)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    print(f"Median distance of training data: {np.median(dists_train)}")

    return (full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(1),
            values_train.size(1), dists_train)


def prepare_pointwise_loaders(coords: torch.Tensor,
                              values: torch.Tensor,
                              train_idx: np.ndarray,
                              val_idx: np.ndarray,
                              test_idx: np.ndarray,
                              batch_size: int,
                              generator: torch.Generator,
                              cyclic_time: bool = False):
    """
    Preprocess data (use training scalers to scale validation and test sets) and create respective data loaders that
    return data pointwise.

    :param coords: Latitude, longitude, depth and time data
    :param values: Variable data (torch.tensor)
    :param train_idx: Rows meant for training
    :param val_idx: Rows meant for validation
    :param test_idx: Rows meant for testing
    :param batch_size: Batch size
    :param generator: Random number generator
    :param cyclic_time: Whether to encode time cyclically (e.g. for monthly data) or not (e.g. for yearly data)
    :return:
        - full_loader
        - train_loader
        - val_loader
        - test_loader
        - train_scaler_dict
        - n_coords
        - n_values
        - None (for compatibility)
    """
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

    # Full dataset and loader (no splitting yet, for reconstruction)
    full_dataset = PointwiseDataset(coords=coords_full, values=values_full)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Define datasets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(
        1), values_train.size(1), None, None, None, None


def prepare_sklearn_data(df, train_idx, val_idx, test_idx):
    """
    Prepare data for sklearn model training.

    :param df: Input pandas.DataFrame
    :param train_idx: Rows meant for training
    :param val_idx: Rows meant for validation
    :param test_idx: Rows meant for testing
    :return:
        - x_train
        - y_true_scaled
        - scaler_dict
        - coord_dim
        - values_dim
    """
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
    """
    Generates masks that hide a given proportion of the data randomly per feature.

    :param batch_size: Batch size
    :param feature_dim: Number of features
    :param mask_ratio: Proportion to hide
    :param n_neighbours: Number of neighbours
    :param device: Device to use
    :param mask_query: Whether to mask the query points
    :param mask_neighbours: Whether to mask the neighbour points
    :return:
        - query_mask
        - neighbour_mask
    """
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


def preprocess(coords, values, coord_names, parameter_names, cyclic_time: bool = False, scaler_dict: dict = None):
    """
    Preprocess coordinates and values of the inputa data by scaling and/or encoding them. Most variables are
    transformed using MinMax scaling. Latitude and longitude are mapped to Cartesian coordinates on a unit sphere.

    :param coords: Input coordinates
    :param values: Input values
    :param coord_names: Names of the coordinates
    :param parameter_names: Names of the parameters
    :param cyclic_time: Whether to encode time cyclically. Default is False.
    :param scaler_dict: Optional dict of scalers to use instead of creating new ones. Default is None.
    :return:
        - x_scaled - Scaled coordinates
        - y_scaled - Scaled values
        - scaler_dict - Dictionary of used scalers
    """
    if scaler_dict is None:
        scaler_dict = {}

    # Convert to dict for easier column referencing
    coord_dict = {name: coords[:, i] for i, name in enumerate(coord_names)}

    # Transform latitude and longitude to cartesian coords on unit sphere (and scale to range [0, 1])
    lat_radians = np.pi / 180 * coord_dict["LATITUDE"]
    lon_radians = np.pi / 180 * coord_dict["LONGITUDE"]

    x = (np.cos(lat_radians) * np.cos(lon_radians) + 1) / 2
    y = (np.cos(lat_radians) * np.sin(lon_radians) + 1) / 2
    z = (np.sin(lat_radians) + 1) / 2

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

        if len(y_obs) == 0:
            continue

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


def load_dataset(data_path=None):
    """
    Load the dataset. It converts the DATEANDTIME column to years and drops 'idx' and 'water' columns.

    :param data_path: Optional path to the dataset. If None, config.data_path is used.
    :return: Dataset as pandas.DataFrame
    """
    if data_path is None:
        data_path = config.data_path

    # Load data
    df = pd.read_csv(data_path)
    df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year

    # Drop not-needed columns
    if "idx" in df.columns:
        df = df.drop(columns=["idx"])
    if "water" in df.columns:
        df = df.drop(columns=["water"])
    return df
