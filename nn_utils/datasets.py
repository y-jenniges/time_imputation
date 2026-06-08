"""PyTorch Dataset classes for MaSTNeT training."""

from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from nn_utils.graph import GraphProvider


class LearnedNeighbourDataset(Dataset):
    """
    PyTorch Dataset returning a query point and its k-NN neighbours per scope.

    Used by MaSTNeT's graph-based attention modes. Neighbours are retrieved
    from a :class:`GraphProvider` at each ``__getitem__`` call.

    Parameters
    ----------
    coords : Tensor, shape (N, coord_dim)
        Preprocessed (scaled) coordinate matrix for the full dataset.
    values : Tensor, shape (N, n_features)
        Preprocessed feature matrix; ``NaN`` marks missing values.
    graph_provider : GraphProvider
        Provides neighbour index look-ups for each scope.
    query_indices : ndarray or None, default=None
        Subset of rows to use as queries; None uses all rows.
    scopes : list[str] or None, default=None
        Neighbourhood scopes to include (e.g. ``['space', 'time']``).
        Defaults to ``['default']``.
    """

    def __init__(self, coords: torch.Tensor,
                 values: torch.Tensor,
                 graph_provider: GraphProvider,
                 query_indices: np.ndarray | None = None,
                 scopes=None):
        self.scopes = scopes if scopes is not None else ["default"]

        self.coords = coords
        self.values = values
        self.query_indices = query_indices

        self.graph_provider = graph_provider
        self.feature_mask = ~torch.isnan(values)

        if query_indices is None:
            self.query_indices = torch.arange(values.shape[0], dtype=torch.long)
        else:
            self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)

    def __len__(self):
        return self.query_indices.shape[0]

    def __getitem__(self, idx) -> Dict[str, Any]:
        q_idx = self.query_indices[idx]

        q_feat = self.values[q_idx]
        q_mask = self.feature_mask[q_idx]
        q_coord = self.coords[q_idx]

        out_dict: Dict[str, Any] = {
            "query_features": q_feat,
            "query_mask": q_mask,
            "query_coords": q_coord
        }

        for scope in self.scopes:
            n_batch = self.graph_provider.get_neighbour_batch(idx=q_idx, coords=self.coords, values=self.values,
                                                              feature_mask=self.feature_mask, scope=scope)
            rel_positions = n_batch["coords"] - q_coord

            out_dict[scope] = {
                "features": n_batch["values"],
                "mask": n_batch["mask"],
                "coords": n_batch["coords"],
                "rel_positions": rel_positions
            }

        return out_dict


class PointwiseDataset(Dataset):
    """PyTorch dataset returning individual points with coordinates, features, and an observation mask."""

    def __init__(self, coords: torch.Tensor, values: torch.Tensor):
        self.coords = coords
        self.values = values
        self.feature_mask = ~torch.isnan(values)  # True = observed

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return {"features": self.values[idx], "coords": self.coords[idx], "mask": self.feature_mask[idx]}


class TimeSequenceDataset(Dataset):
    """
    PyTorch Dataset returning a centred temporal window around each query point.

    Data are sorted by time; each sample contains a fixed-length sequence centred
    on the query point, padded at boundaries by repeating the last row.

    Parameters
    ----------
    coords : Tensor, shape (N, coord_dim)
        Preprocessed coordinate matrix; last column is the (scaled) time dimension.
    values : Tensor, shape (N, n_features)
        Preprocessed feature matrix; ``NaN`` marks missing values.
    query_indices : ndarray or None, default=None
        Subset of rows to use as query centres; None uses all rows.
    sequence_length : int or None, default=None
        Odd integer defining the temporal window size; defaults to the number of
        unique time steps in ``coords``.
    scopes : ignored
        For API compatibility.
    """

    def __init__(self, coords: torch.Tensor, values: torch.Tensor, query_indices: np.ndarray | None = None,
                 sequence_length: int = None,
                 scopes=None):
        if sequence_length is None:
            sequence_length = len(torch.unique(coords[:, -1]))
        assert sequence_length % 2 == 1, "sequence_length must be odd"

        self.coords = coords
        self.values = values
        self.query_indices = query_indices

        self.seq_len = sequence_length
        self.half = sequence_length // 2

        self.feature_mask = ~torch.isnan(values)

        self.sorted_idx = torch.argsort(coords[:, -1])
        self.coords_sorted = coords[self.sorted_idx]
        self.values_sorted = values[self.sorted_idx]
        self.mask_sorted = self.feature_mask[self.sorted_idx]

        if query_indices is None:
            self.query_indices = torch.arange(len(coords), dtype=torch.long)
        else:
            self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)

        self.query_indices = self.sorted_idx[self.query_indices]

    def __len__(self):
        return self.query_indices.shape[0]

    def __getitem__(self, idx) -> Dict[str, Any]:
        q_idx = self.query_indices[idx]

        seq_start = max(int(q_idx - self.half), 0)
        seq_end = min(int(q_idx + self.half + 1), len(self.coords_sorted))

        seq_coords = self.coords_sorted[seq_start:seq_end]
        seq_values = self.values_sorted[seq_start:seq_end]
        seq_mask = self.mask_sorted[seq_start:seq_end]

        q_local = min(self.half, seq_coords.shape[0] - 1)

        if seq_coords.shape[0] < self.seq_len:
            pad_size = self.seq_len - seq_coords.shape[0]
            seq_coords = torch.cat([seq_coords, seq_coords[-1:].repeat(pad_size, 1)], dim=0)
            seq_values = torch.cat([seq_values, seq_values[-1:].repeat(pad_size, 1)], dim=0)
            seq_mask = torch.cat([seq_mask, seq_mask[-1:].repeat(pad_size, 1)], dim=0)

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
