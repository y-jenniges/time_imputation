from typing import Tuple, Any, Dict
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


import config
from nn_utils.graph import GraphProvider
from utils.preprocessing import get_scopes


class LearnedNeighbourDataset(Dataset):
    def __init__(self, coords: torch.Tensor, values: torch.Tensor, graph_provider: GraphProvider, query_indices: np.ndarray | None = None,
                 scopes=None):
        self.scopes = scopes if scopes is not None else ["default"]

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

    def __getitem__(self, idx) -> Dict[str, Any]:
        # Map local query point idx to global idx
        q_idx = self.query_indices[idx]

        # Query point
        q_feat = self.values[q_idx]
        q_mask = self.feature_mask[q_idx]
        q_coord = self.coords[q_idx]

        out_dict: Dict[str, Any] = {
            "query_features": q_feat,
            "query_mask": q_mask,
            "query_coords": q_coord
        }

        # Neighbours (global indices)
        for scope in self.scopes:
            n_batch = self.graph_provider.get_neighbour_batch(idx=q_idx, coords=self.coords, values=self.values, feature_mask=self.feature_mask, scope=scope)

            # Relative positions
            rel_positions = n_batch["coords"] - q_coord

            out_dict[scope] = {
                "features": n_batch["values"],
                "mask": n_batch["mask"],
                "coords": n_batch["coords"],
                "rel_positions": rel_positions
            }

        return out_dict


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


class TimeSequenceDataset(Dataset):
    def __init__(self, coords: torch.Tensor, values: torch.Tensor, query_indices: np.ndarray | None = None,
                 sequence_length: int = None,
                 scopes=None):
        # self.scopes = scopes if scopes is not None else ["time"]

        if sequence_length is None:
            sequence_length = len(torch.unique(coords[:, -1]))
        assert sequence_length % 2 == 1, "sequence_length must be odd"

        self.coords = coords
        self.values = values
        self.query_indices = query_indices

        # Sequence info
        self.seq_len = sequence_length
        self.half = sequence_length // 2

        # Initial mask: True = observed (False = NaN)
        self.feature_mask = ~torch.isnan(values)

        # Sort by time
        self.sorted_idx = torch.argsort(coords[:, -1])
        self.coords_sorted = coords[self.sorted_idx]
        self.values_sorted = values[self.sorted_idx]
        self.mask_sorted = self.feature_mask[self.sorted_idx]

        # Store original indices of query points
        if query_indices is None:
            self.query_indices = torch.arange(len(coords), dtype=torch.long)
        else:
            self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)

        # Map query indices to sorted indices
        self.query_indices = torch.searchsorted(self.sorted_idx, self.query_indices)

        print("first 10 sorted times:", coords[self.sorted_idx[:10], -1])
        print("query_indices sample:", self.query_indices[:10])

    def __len__(self):
        return self.query_indices.shape[0]

    def __getitem__(self, idx) -> Dict[str, Any]:
        # Map local query point idx to global idx
        q_idx = self.query_indices[idx]

        # Sequence
        seq_start = q_idx - self.half
        seq_end = q_idx + self.half + 1

        # Sequence boundaries
        seq_start = max(int(seq_start), 0)
        seq_end = min(int(seq_end), len(self.coords_sorted))

        if idx < 5:
            print("q_idx:", q_idx.item())
            print("seq_start/end:", seq_start, seq_end)

        seq_coords = self.coords_sorted[seq_start:seq_end]
        seq_values = self.values_sorted[seq_start:seq_end]
        seq_mask = self.mask_sorted[seq_start:seq_end]

        # Query point is the center
        q_local = min(self.half, seq_coords.shape[0] - 1)

        # Pad (if sequence shorter than seq length)
        if seq_coords.shape[0] < self.seq_len:
            pad_size = self.seq_len - seq_coords.shape[0]

            seq_coords = torch.cat([seq_coords, seq_coords[-1:].repeat(pad_size, 1)], dim=0)
            seq_values = torch.cat([seq_values, seq_values[-1:].repeat(pad_size, 1)], dim=0)
            seq_mask = torch.cat([seq_mask, seq_mask[-1:].repeat(pad_size, 1)], dim=0)

        if idx < 5:
            print("q_local:", q_local)
            print("seq_len actual:", seq_coords.shape[0])
            print("query_features:", seq_values[q_local])
            print("query_mask:", seq_mask[q_local])

        return {
            "query_features": seq_values[q_local],
            "query_mask": seq_mask[q_local],
            "query_coords": seq_coords[q_local],
            "time": {
                "features": seq_values,
                "mask": seq_mask,
                "coords": seq_coords,
                "rel_positions": seq_coords - seq_coords[q_local]
            }
        }



def prepare_time_sequence_loaders(coords: torch.Tensor,
                                  values: torch.Tensor,
                                  train_idx: np.ndarray,
                                  val_idx: np.ndarray,
                                  test_idx: np.ndarray,
                                  batch_size: int,
                                  sequence_length: int,
                                  generator: torch.Generator,
                                  cyclic_time: bool = False,
                                  cfg=None,
                                  graph_provider=None,
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
    :param cfg: Configuration object
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

    # Define datasets
    train_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=train_idx, sequence_length=sequence_length)
    val_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=val_idx, sequence_length=sequence_length)
    test_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=test_idx, sequence_length=sequence_length)
    full_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=None, sequence_length=sequence_length)

    print(f"Lenghts of train, val, test datasets: {len(train_dataset), len(val_dataset), len(test_dataset)}")
    print("NaNs values_full:", torch.isnan(values_full).sum())

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    dists_train = None
    # print(f"Median distance of training data: {np.nanmedian(dists_train)}")

    return full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(1), values_train.size(1), dists_train, coords_full, values_full, mask_full


def prepare_learned_neighbourhood_loaders(coords: torch.Tensor,
                                          values: torch.Tensor,
                                          train_idx: np.ndarray,
                                          val_idx: np.ndarray,
                                          test_idx: np.ndarray,
                                          batch_size: int,
                                          generator: torch.Generator,
                                          graph_provider: GraphProvider,
                                          cyclic_time: bool = False,
                                          cfg=None
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
    :param cfg: Configuration object
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
    init_encoder = lambda coords, times, values, mask: coords_full
    #init_encoder = lambda coords, times: coords_full
    graph_provider.update(encoder=init_encoder, coords=coords_full, values=values_full, mask=mask_full)

    print("dataset graph provider updated")

    # Define datasets
    scopes = get_scopes(cfg=cfg)
    train_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=train_idx, graph_provider=graph_provider, scopes=scopes)
    val_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=val_idx, graph_provider=graph_provider, scopes=scopes)
    test_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=test_idx, graph_provider=graph_provider, scopes=scopes)
    full_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=None, graph_provider=graph_provider, scopes=scopes)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    dists_train = None
    # print(f"Median distance of training data: {np.nanmedian(dists_train)}")

    return full_loader, train_loader, val_loader, test_loader, train_scaler_dict, coords_train.size(1), values_train.size(1), dists_train, coords_full, values_full, mask_full


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
    mask_full = ~torch.isnan(values_full)

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
        1), values_train.size(1), None, coords_full, values_full, mask_full


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


def random_per_feature_mask(batch_size,
                            feature_dim: int,
                            mask_ratio: float = 0.5,
                            n_neighbours: int = 24,
                            device="cpu",
                            mask_query: bool = True,
                            mask_neighbours: bool = False
                            ):
    """
    Masks a proportion of samples per feature (column-wise masking).
    Each feature has mask_ratio * batch_size samples masked.
    Note: With gridded data, it can happen that only missing entries are masked... @todo
    Returns:
        mask: [B, F] boolean tensor
    """
    # Init mask
    mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)

    # Number of samples to mask per feature
    n_mask = int(batch_size * mask_ratio)

    if n_mask == 0:
        return mask

    # For each feature, randomly select samples to mask
    rand_idx = torch.rand(feature_dim, batch_size, device=device).argsort(dim=1)

    # Take first n_mask indices per feature
    selected = rand_idx[:, :n_mask]  # [F, n_mask]

    # Scatter into mask
    for f in range(feature_dim):
        mask[selected[f], f] = True

    return mask


def random_feature_mask(batch_size: int,
                        feature_dim: int,
                        mask_ratio: float = 0.5,
                        n_neighbours: int = 24,
                        device="cpu",
                        mask_query: bool = True,
                        mask_neighbours: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates masks that hide a given proportion of each sample randomly.

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


def spherical_feature_mask(batch, feature_dim, size=0.1, p=0.3, device="cpu"):
    """ Creates single spherical mask. """
    lonlats = batch["query_coords"][:, :3]  # [B, 3]
    B = lonlats.shape[0]

    # Apply block mask in p % of batches
    if torch.rand(1) > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    # Random center
    # # center = torch.rand(3, device=device)
    # center = torch.randn(3, device=device)
    # center = center / center.norm()

    idx = torch.randint(0, B, (1,), device=device)
    center = lonlats[idx]

    # Distance to center
    dist = torch.norm(lonlats - center, dim=1)

    # Spherical block
    mask_block = dist < size  # [B]

    return mask_block.unsqueeze(1).expand(B, feature_dim)


def transect_feature_mask(batch, feature_dim, width=0.05, p=0.3, device="cpu", orientation=0.0):
    """ Creates single transect mask. Orientation: 0 random orientation and 1 horizontal or vertical"""
    lonlats = batch["query_coords"][:, :3]  # [B, 3]
    B = lonlats.shape[0]

    # Only apply with probability p
    if torch.rand(1, device=device) > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    # Normalize lons and lats
    lonlats = lonlats / torch.norm(lonlats, dim=1, keepdim=True)

    # Orientation probability
    use_structured = torch.rand(1, device=device) < orientation
    if use_structured:
        if torch.rand(1, device=device) < 0.5:
            # Latitude transect (parallel to equator, constant z)
            z0 = torch.empty(1, device=device).uniform_(-1, 1)
            dist = torch.abs(lonlats[:, 2] - z0)

        else:
            # Longitude transect (meridian, great circle through poles)
            # Normal vector constrained to equatorial plane
            n = torch.randn(3, device=device)
            n[2] = 0.0  # Force vertical plane
            n = n / torch.norm(n)
            dist = torch.abs(lonlats @ n)

    else:
        # Fully random orientation
        # Point on sphere
        # p0 = torch.randn(3, device=device)
        # p0 = p0 / torch.norm(p0)

        idx = torch.randint(0, B, (1,), device=device)
        p0 = lonlats[idx][0]  # Anchor point on data

        # Random direction
        d = torch.randn(3, device=device)
        d = d / torch.norm(d)

        n = torch.cross(p0, d)
        n_norm = torch.norm(n)
        if n_norm < 1e-6:
            return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)
        n = n / n_norm

        # Great circle distance
        dist = torch.abs(lonlats @ n)

    # Points close to line
    mask_line = dist < width  # [B]

    # Expand to features
    mask = mask_line.unsqueeze(1).expand(B, feature_dim)

    return mask

class WeightedMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), use_sqrt=True, weights=None):
        self.feature_range = feature_range
        self.use_sqrt = use_sqrt
        self.weights = weights
        self.scaler_ = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64)
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        sample_weight = self.weights[X.long()]

        check_is_fitted(self, "scaler_")
        X = check_array(X, dtype=np.float64)

        X_scaled = self.scaler_.transform(X)

        if sample_weight is None:
            return X_scaled

        sample_weight = np.asarray(sample_weight)

        if self.use_sqrt:
            sample_weight = np.sqrt(sample_weight)

        # Reshape for broadcasting if needed
        if sample_weight.ndim == 1:
            sample_weight = sample_weight[:, None]

        return X_scaled * sample_weight

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X).transform(X)


def get_availability_per_time(coords, values):
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


def preprocess(coords, values, coord_names, parameter_names, cyclic_time: bool = False, scaler_dict: dict = None, time_scaler_weights=None):
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
            # scaler_time =  MinMaxScaler(feature_range=(0, 0.5))
            # time_scaler_weights = get_availability_per_time(coords, values)
            # scaler_time = WeightedMinMaxScaler(weights=time_scaler_weights)
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
