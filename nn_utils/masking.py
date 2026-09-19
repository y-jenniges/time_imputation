"""Masking utilities for MaSTNeT training: random, spherical, transect and copy masks."""

from typing import Tuple
import torch


def random_per_feature_mask(batch_size,
                            feature_dim: int,
                            mask_ratio: float = 0.5,
                            n_neighbours: int = 24,
                            device="cpu",
                            mask_query: bool = True,
                            mask_neighbours: bool = False,
                            generator: torch.Generator | None = None
                            ):
    """
    Column-wise masking: each feature has ``mask_ratio * batch_size`` samples masked.

    Parameters
    ----------
    generator : torch.Generator, optional
        Generator for reproducible masks across validation epochs.

    Returns
    -------
    mask : Tensor, shape (B, F), dtype bool
    """
    mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)
    n_mask = int(batch_size * mask_ratio)

    if n_mask == 0:
        return mask

    # Define values to hide
    rand_idx = torch.rand(feature_dim, batch_size, device=device, generator=generator).argsort(dim=1)
    selected = rand_idx[:, :n_mask]  # [F, n_mask]

    # Hide values
    for f in range(feature_dim):
        mask[selected[f], f] = True

    return mask


def random_token_mask(batch_size, feature_dim, mask_ratio, device, generator: torch.Generator | None = None):
    """Return an element-wise random bool mask of shape (B, F) with approximately ``mask_ratio`` True entries."""
    return torch.rand(batch_size, feature_dim, device=device, generator=generator) < mask_ratio


def random_feature_mask(batch_size: int,
                        feature_dim: int,
                        mask_ratio: float = 0.5,
                        n_neighbours: int = 24,
                        device="cpu",
                        mask_query: bool = True,
                        mask_neighbours: bool = False,
                        generator: torch.Generator | None = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate row-wise masks that hide a fixed proportion of features per sample.

    Parameters
    ----------
    batch_size : int
    feature_dim : int
    mask_ratio : float, default=0.5
        Fraction of features to hide per sample.
    n_neighbours : int, default=24
        Used only when ``mask_neighbours=True``.
    device : str, default='cpu'
    mask_query : bool, default=True
        Apply masking to query points.
    mask_neighbours : bool, default=False
        Apply independent masking to neighbour tokens.
    generator : torch.Generator, optional
        Generator for reproducible masks across validation epochs.

    Returns
    -------
    query_mask : Tensor, shape (B, F), dtype bool
    neighbour_mask : Tensor, shape (B, K, F), dtype bool
    """
    if mask_query:
        query_mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)
        n_query_mask = max(1, int(feature_dim * mask_ratio))
        rand_idx_q = torch.rand(batch_size, feature_dim, device=device, generator=generator).argsort(dim=1)
        query_mask.scatter_(1, rand_idx_q[:, :n_query_mask], True)
    else:
        query_mask = torch.zeros(batch_size, feature_dim, dtype=torch.bool, device=device)

    if mask_neighbours:
        neighbour_mask = torch.zeros(batch_size, n_neighbours, feature_dim, dtype=torch.bool, device=device)
        n_neighbour_mask = max(1, int(feature_dim * mask_ratio))
        rand_idx_n = torch.rand(batch_size, n_neighbours, feature_dim, device=device, generator=generator).argsort(dim=2)
        neighbour_mask.scatter_(2, rand_idx_n[:, :, :n_neighbour_mask], True)
    else:
        neighbour_mask = torch.zeros(batch_size, n_neighbours, feature_dim, dtype=torch.bool, device=device)

    return query_mask, neighbour_mask


def spherical_feature_mask(batch, feature_dim, size=0.1, p=0.3, device="cpu",
                           generator: torch.Generator | None = None):
    """Create a spherical block mask centred on a random query point, applied with probability ``p``."""
    lonlats = batch["query_coords"][:, :3]  # [B, 3]
    B = lonlats.shape[0]

    # Skip mask with probability 1-p
    if torch.rand(1, device=device, generator=generator) > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    # Define spherical block mask
    idx = torch.randint(0, B, (1,), device=device, generator=generator)
    center = lonlats[idx]
    dist = torch.norm(lonlats - center, dim=1)
    mask_block = dist < size  # [B]

    return mask_block.unsqueeze(1).expand(B, feature_dim)


def transect_feature_mask(batch, feature_dim, width=0.05, p=0.3, device="cpu",
                          orientation=0.0, generator: torch.Generator | None = None):
    """Create a great-circle transect mask; applied with probability ``p``. ``orientation=0`` is fully random, ``1``
    forces axis-aligned transects."""
    lonlats = batch["query_coords"][:, :3]  # [B, 3]
    B = lonlats.shape[0]

    # Skip mask with probability 1-p
    if torch.rand(1, device=device, generator=generator) > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    lonlats = lonlats / torch.norm(lonlats, dim=1, keepdim=True)

    # Use a mask parallel to lat/lon
    use_structured = torch.rand(1, device=device, generator=generator) < orientation
    if use_structured:
        if torch.rand(1, device=device, generator=generator) < 0.5:
            z0 = torch.empty(1, device=device).uniform_(-1, 1, generator=generator)
            dist = torch.abs(lonlats[:, 2] - z0)
        else:
            n = torch.randn(3, device=device, generator=generator)
            n[2] = 0.0
            n = n / torch.norm(n)
            dist = torch.abs(lonlats @ n)
    else:
        # Randomly oriented transect mask
        idx = torch.randint(0, B, (1,), device=device, generator=generator)
        p0 = lonlats[idx][0]
        d = torch.randn(3, device=device, generator=generator)
        d = d / torch.norm(d)
        n = torch.cross(p0, d)
        n_norm = torch.norm(n)
        if n_norm < 1e-6:
            return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)
        n = n / n_norm
        dist = torch.abs(lonlats @ n)

    mask_line = dist < width  # [B]
    return mask_line.unsqueeze(1).expand(B, feature_dim)


def copy_feature_mask(query_mask: torch.Tensor,
                      p: float = 0.3,
                      min_observed: int = 1,
                      donor_pool_mask: torch.Tensor | None = None,
                      device="cpu",
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Copy masking: For each sample, additionally hide the features that are
    missing in a different, randomly drawn donor sample.

    Adapted from the "copy masking" strategy in Gorla, Wang, Liu, An &
    Sankararaman, "CACTI: Leveraging Copy Masking and Contextual Information to
    Improve Tabular Data Imputation", PMLR 267:20187-20225, 2025,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC13060770/ with reference implementation
    at github.com/sriramlab/CACTI, ``src/loaders/copyloader.py``.

    Parameters
    ----------
    query_mask : Tensor, shape (B, F), dtype bool
        True where the query feature is observed.
    p : float, default=0.3
        Probability of applying a copy mask to each sample.
    min_observed : int, default=1
        Minimum number of features that must remain observed after masking.
    donor_pool_mask : Tensor, shape (N_pool, F), dtype bool, optional
        Observation mask to draw donors from. If ``None``, donors are drawn
        from ``query_mask`` (i.e. from the current batch only). Pass the
        full training-set mask here to draw donors from the whole population,
        as in the reference implementation.
    device : str, default='cpu'
    generator : torch.Generator, optional
        Generator for reproducible masks across validation epochs.

    Returns
    -------
    mask : Tensor, shape (B, F), dtype bool
        True where a feature should be additionally hidden.
    """
    batch_size = query_mask.shape[0]
    pool = donor_pool_mask if donor_pool_mask is not None else query_mask
    apply_mask = torch.rand(batch_size, device=device, generator=generator) < p
    donor_idx = torch.randint(0, pool.shape[0], (batch_size,), device=device, generator=generator)
    donor_missing = ~pool[donor_idx]  # features missing in the donor sample

    # Mask replicates the missingness pattern of the donor sample
    mask = donor_missing & apply_mask.unsqueeze(1)

    # Add the minimum-observed condition
    if min_observed > 0:
        would_drop_below_min = (query_mask & ~mask).sum(dim=1) < min_observed
        mask[would_drop_below_min] = False

    return mask
