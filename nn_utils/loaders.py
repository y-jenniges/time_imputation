"""DataLoader factory functions for MaSTNeT training."""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

import config
from nn_utils.datasets import LearnedNeighbourDataset, PointwiseDataset, TimeSequenceDataset
from nn_utils.graph import GraphProvider
from utils.data import preprocess
from utils.preprocessing import get_scopes


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
                                  graph_provider=None) -> tuple[
    DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any],
    dict, int, int, None, Tensor, Tensor, Tensor]:
    """
    Preprocess data and build DataLoaders returning centred temporal windows.

    Scalers are fit on training rows only and then applied to the full dataset.

    Parameters
    ----------
    coords : Tensor
        Raw coordinate matrix (lat, lon, depth, time).
    values : Tensor
        Raw feature matrix; ``NaN`` marks missing values.
    train_idx : ndarray
        Row indices for the training split.
    val_idx : ndarray
        Row indices for the validation split.
    test_idx : ndarray
        Row indices for the test split.
    batch_size : int
    sequence_length : int
        Odd temporal window size passed to :class:`~nn_utils.datasets.TimeSequenceDataset`.
    generator : torch.Generator
        Seeded RNG for reproducible shuffling.
    cyclic_time : bool, default=False
        Encode time cyclically (sine/cosine) for monthly data.
    cfg : ModelConfig or None, default=None
    graph_provider : ignored
        Accepted for interface compatibility.

    Returns
    -------
    full_loader, train_loader, val_loader, test_loader : DataLoader
    train_scaler_dict : dict
    n_coords : int
    n_values : int
    dists : None
    coords_full : Tensor
    values_full : Tensor
    mask_full : Tensor
    """
    coords_train, values_train, train_scaler_dict = preprocess(
        coords=coords[train_idx],
        values=values[train_idx],
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=None
    )

    coords_full, values_full, _ = preprocess(
        coords=coords,
        values=values,
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=train_scaler_dict
    )
    mask_full = ~torch.isnan(values_full)

    train_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=train_idx,
                                        sequence_length=sequence_length)
    val_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=val_idx,
                                      sequence_length=sequence_length)
    test_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=test_idx,
                                       sequence_length=sequence_length)
    full_dataset = TimeSequenceDataset(coords=coords_full, values=values_full, query_indices=None,
                                       sequence_length=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return (full_loader, train_loader, val_loader, test_loader, train_scaler_dict,
            coords_train.size(1), values_train.size(1), None, coords_full, values_full, mask_full)


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
    DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any],
    dict, int, int, None, Tensor, Tensor, Tensor]:
    """
    Preprocess data and build DataLoaders returning query + k-NN neighbour tokens.

    Scalers are fit on training rows only and then applied to the full dataset.
    The graph is initialised once from preprocessed coordinate space.

    Parameters
    ----------
    coords : Tensor
        Raw coordinate matrix (lat, lon, depth, time).
    values : Tensor
        Raw feature matrix; ``NaN`` marks missing values.
    train_idx : ndarray
        Row indices for the training split.
    val_idx : ndarray
        Row indices for the validation split.
    test_idx : ndarray
        Row indices for the test split.
    batch_size : int
    generator : torch.Generator
        Seeded RNG for reproducible shuffling.
    graph_provider : GraphProvider
        Manages k-NN neighbour look-ups; initialised inside this function.
    cyclic_time : bool, default=False
        Encode time cyclically (sine/cosine) for monthly data.
    cfg : ModelConfig or None, default=None

    Returns
    -------
    full_loader, train_loader, val_loader, test_loader : DataLoader
    train_scaler_dict : dict
    n_coords : int
    n_values : int
    dists : None
    coords_full : Tensor
    values_full : Tensor
    mask_full : Tensor
    """
    coords_train, values_train, train_scaler_dict = preprocess(
        coords=coords[train_idx],
        values=values[train_idx],
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=None
    )

    coords_full, values_full, _ = preprocess(
        coords=coords,
        values=values,
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=train_scaler_dict
    )
    mask_full = ~torch.isnan(values_full)

    init_encoder = lambda coords, times, values, mask: coords_full
    graph_provider.update(encoder=init_encoder, coords=coords_full, values=values_full, mask=mask_full)

    scopes = get_scopes(cfg=cfg)
    train_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=train_idx,
                                            graph_provider=graph_provider, scopes=scopes)
    val_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=val_idx,
                                          graph_provider=graph_provider, scopes=scopes)
    test_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=test_idx,
                                           graph_provider=graph_provider, scopes=scopes)
    full_dataset = LearnedNeighbourDataset(coords=coords_full, values=values_full, query_indices=None,
                                           graph_provider=graph_provider, scopes=scopes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return (full_loader, train_loader, val_loader, test_loader, train_scaler_dict,
            coords_train.size(1), values_train.size(1), None, coords_full, values_full, mask_full)


def prepare_pointwise_loaders(coords: torch.Tensor,
                              values: torch.Tensor,
                              train_idx: np.ndarray,
                              val_idx: np.ndarray,
                              test_idx: np.ndarray,
                              batch_size: int,
                              generator: torch.Generator,
                              cyclic_time: bool = False,
                              cfg=None,
                              graph_provider=None):
    """
    Preprocess data and build DataLoaders returning individual points (for MLP / pointwise models).

    Scalers are fit on training rows only and then applied to the full dataset.

    Parameters
    ----------
    coords : Tensor
        Raw coordinate matrix (lat, lon, depth, time).
    values : Tensor
        Raw feature matrix; ``NaN`` marks missing values.
    train_idx : ndarray
    val_idx : ndarray
    test_idx : ndarray
    batch_size : int
    generator : torch.Generator
    cyclic_time : bool, default=False
    cfg : ignored
    graph_provider : ignored

    Returns
    -------
    full_loader, train_loader, val_loader, test_loader : DataLoader
    train_scaler_dict : dict
    n_coords : int
    n_values : int
    None
        Placeholder for compatibility with other prepare functions.
    coords_full : Tensor
    values_full : Tensor
    mask_full : Tensor
    """
    coords_train, values_train, train_scaler_dict = preprocess(
        coords=coords[train_idx],
        values=values[train_idx],
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=None
    )

    coords_full, values_full, _ = preprocess(
        coords=coords,
        values=values,
        coord_names=config.coordinates,
        parameter_names=config.parameters,
        cyclic_time=cyclic_time,
        scaler_dict=train_scaler_dict
    )
    mask_full = ~torch.isnan(values_full)

    full_dataset = PointwiseDataset(coords=coords_full, values=values_full)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False)

    return (full_loader, train_loader, val_loader, test_loader, train_scaler_dict,
            coords_train.size(1), values_train.size(1), None, coords_full, values_full, mask_full)


def prepare_sklearn_data(df, train_idx, val_idx, test_idx):
    """
    Prepare scaled arrays for sklearn-based imputation models.

    Parameters
    ----------
    df : DataFrame
        Full dataset with coordinate and feature columns as defined in ``config``.
    train_idx : ndarray
    val_idx : ndarray
    test_idx : ndarray

    Returns
    -------
    x_train : ndarray, shape (N, coord_dim + n_features)
        Coordinate + feature matrix; features for val/test rows are set to NaN.
    y_true_scaled : ndarray, shape (N, n_features)
        Scaled ground-truth features for all rows (used for evaluation).
    scaler_dict : dict
    coord_dim : int
    values_dim : int
    """
    import numpy as np

    values_raw = df[config.parameters].to_numpy().astype(float)
    coords_raw = df[config.coordinates].to_numpy().astype(float)

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

    y_true_scaled = x_train[:, coord_dim:].copy()
    y_true_scaled[val_idx, :] = values_val_scaled
    y_true_scaled[test_idx, :] = values_test_scaled

    return x_train, y_true_scaled, scaler_dict, coord_dim, values_dim
