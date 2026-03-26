import torch
from typing import Union, Dict
from sklearn.neighbors import NearestNeighbors

from utils.preprocessing import fill_feature_tensor


def knn_feature_variance(values, mask, knn_idx, scope="space"):
    if isinstance(knn_idx, dict):
        knn_idx = knn_idx[scope]

    # How similar are feature values in neighbourhood? Low variance --> Similar neighbours
    v_i = values.unsqueeze(1)  # [N, 1, F]
    v_j = values[knn_idx]  # [N, K, F]

    # Prepare passed mask
    m_i = mask.unsqueeze(1)            # [N, 1, F]
    m_j = mask[knn_idx]                # [N, K, F]

    # NaN mask
    nan_mask = ~torch.isnan(values)    # [N, F]
    n_i = nan_mask.unsqueeze(1)        # [N, 1, F]
    n_j = nan_mask[knn_idx]            # [N, K, F]

    # Combine masks
    joint_mask = m_i & m_j & n_i & n_j   # [N, K, F]

    # Compute squared differences (avoid NaN propagation)
    diff2 = (torch.nan_to_num(v_i) - torch.nan_to_num(v_j)) ** 2

    # Apply mask and compute variance
    diff2 = torch.where(joint_mask, diff2, 0.0)
    var = diff2.sum() / joint_mask.sum().clamp(min=1)

    return var.item()


def knn_time_difference(coords, knn_idx, scope="space"):
    # How far apart are neighbours in time? (Lower is better)
    if isinstance(knn_idx, dict):
        knn_idx = knn_idx[scope]

    t = coords[:, -1]
    t_i = t.unsqueeze(1)
    t_j = t[knn_idx]
    return torch.mean(torch.abs(t_i - t_j)).item()


def knn_overlap(prev_idx, new_idx):
    # How many neighbours stayed the same after graph update? (1 = identical neighbours)
    prev = prev_idx.unsqueeze(2)  # [N, K, 1]
    new  = new_idx.unsqueeze(1)  # [N, 1, K]

    matches = (prev.eq(new)).any(dim=-1).float()  # [N, K]
    return matches.mean().item()


class GraphProvider:
    def __init__(self,
                 n_neighbours=20,
                 update_every: int = 5,
                 cfg = None,
                 test_idx: torch.Tensor = None,
                 val_idx: torch.Tensor = None,
                 device: torch.device = torch.device("cpu")
                 ):
        self.n_neighbours = n_neighbours
        self.update_every = update_every
        self.cfg = cfg

        self.test_idx = test_idx
        self.val_idx = val_idx

        self.device = device

        # Graph properties and analytics
        self.neighbour_indices = None
        self.prev_neighbour_indices = None
        self.history = {"feat_variance": [], "time_difference": [], "overlap": []}

    @torch.no_grad()
    def build_graph(self, encoded, coords, values, mask):
        if self.cfg.attention_type == "space_time_attention":
            # Time neighbours
            time_indices = compute_candidates(coords[:, -1:], k=self.n_neighbours+1)
            time_indices = time_indices[:, 1:]  # Remove self

            # Space neighbours
            space_indices = compute_candidates(coords[:, :4], k=self.n_neighbours+1)
            space_indices = space_indices[:, 1:]  # Remove self

            self.neighbour_indices = {"space": space_indices, "time": time_indices}

        elif self.cfg.graph_metric == "isotropic":
            indices = compute_candidates(encoded, k=self.n_neighbours+1)
            self.neighbour_indices = indices[:, 1:]  # Remove self

        elif self.cfg.graph_metric == "anisotropic":
            candidates = compute_candidates(coords, k=200)
            self.neighbour_indices = compute_anisotropic_knn(coords, values, mask, k=self.n_neighbours,
                                                             candidate_idx=candidates,
                                                             lambda_=1.0, weights=None, batch_size=2048,
                                                             device=self.device)

        else:
            raise ValueError(f"Unknown graph metric: {self.cfg.graph_metric}")

    def encode_input(self, encoder, coords, values, mask, mean_values=None):
        if self.cfg.graph_space == "encoded":
            # Fill masked values and nan values (test/val idxs)
            values_filled = fill_feature_tensor(features=values, mask=mask, fill_strategy=self.cfg.fill_strategy, mean_values=mean_values)
            values_filled = fill_feature_tensor(features=values_filled, mask=None, fill_strategy=self.cfg.fill_strategy, mean_values=mean_values)
            encoded = encoder(coords=coords[:, :4], values=values_filled, mask=mask.float(), times=coords[:, -1:].float())
            return encoded.detach().cpu().numpy()

        elif self.cfg.graph_space == "raw":
            return coords
        else:
            raise ValueError(f"Unknown graph space: {self.cfg.graph_space}")

    def update_eval_metrics(self, values, coords, mask):
        feat_variance = knn_feature_variance(values, mask, self.neighbour_indices, scope="space")
        time_difference = knn_time_difference(coords, self.neighbour_indices, scope="space")

        if self.cfg.attention_type == "space_time_attention":
            feat_variance_time = knn_feature_variance(values, mask, self.neighbour_indices, scope="time")
            time_difference_time = knn_time_difference(coords, self.neighbour_indices, scope="time")

            overlap, overlap_time = None, None
            if self.prev_neighbour_indices is not None:
                overlap = knn_overlap(prev_idx=self.prev_neighbour_indices["space"], new_idx=self.neighbour_indices["space"])
                overlap_time = knn_overlap(prev_idx=self.prev_neighbour_indices["time"], new_idx=self.neighbour_indices["time"])

            print(f"Space graph: ")
            print(f"feat_variance: {feat_variance:.6f}, time_difference: {time_difference:.6f}, overlap: {overlap}")

            print("Time graph: ")
            print(f"feat_variance: {feat_variance_time:.6f}, time_difference: {time_difference_time:.6f}, overlap: {overlap_time}")

            self.history["feat_variance"].append(feat_variance)
            self.history["time_difference"].append(time_difference)
            self.history["overlap"].append(overlap)

            self.history["feat_variance_time"].append(feat_variance_time)
            self.history["time_difference_time"].append(time_difference_time)
            self.history["overlap_time"].append(overlap_time)

            self.prev_neighbour_indices = {
                k: v.clone() for k, v in self.neighbour_indices.items()
            }

        else:
            overlap = None
            if self.prev_neighbour_indices is not None:
                overlap = knn_overlap(prev_idx=self.prev_neighbour_indices, new_idx=self.neighbour_indices)

            print(f"feat_variance: {feat_variance:.6f}, time_difference: {time_difference:.6f}, overlap: {overlap}")

            self.history["feat_variance"].append(feat_variance)
            self.history["time_difference"].append(time_difference)
            self.history["overlap"].append(overlap)

            self.prev_neighbour_indices = self.neighbour_indices.clone()

    def update(self, encoder, coords, values, mask, mean_values=None):
        """ Recompute graph from latent space. """
        values = values.clone()
        mask = mask.clone()

        #coords = coords.to(self.device)
        #values = values.to(self.device)
        #mask = mask.to(self.device)

        if self.test_idx is not None:
            values[self.test_idx] = torch.nan

        if self.val_idx is not None:
            values[self.val_idx] = torch.nan

        # Encoding
        encoded = self.encode_input(encoder=encoder, coords=coords, values=values, mask=mask, mean_values=mean_values)  # second visit here somehow leaves nan

        # Graph building
        self.build_graph(encoded=encoded, coords=coords, values=values, mask=mask)

        # KNN eval metrics
        self.update_eval_metrics(values=values, coords=coords, mask=mask)

    def set(self, indices):
        self.neighbour_indices = indices

    def get(self, idx):
        if isinstance(self.neighbour_indices, dict):
            return {
                "space": self.neighbour_indices["space"][idx],
                "time": self.neighbour_indices["time"][idx]
            }
        return self.neighbour_indices[idx]


def compute_candidates(coords, k=200):
    """
    coords: [N, D] (CPU tensor or numpy)
    returns: [N, k] candidate indices
    """
    if torch.is_tensor(coords):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = coords

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs.fit(coords_np)

    indices = nbrs.kneighbors(return_distance=False)
    return torch.tensor(indices, dtype=torch.long)


def compute_anisotropic_knn(
    coords,
    values,
    mask,
    candidate_idx,
    k=20,
    lambda_=1.0,
    weights=None,
    batch_size=2048,
    device=torch.device("cpu")
):
    if weights is None:
        weights = [1] * coords.shape[-1]

    coords = coords.to(device)
    values = values.to(device)
    mask   = mask.to(device)
    candidate_idx = candidate_idx.to(device)

    N, D = coords.shape
    F = values.shape[-1]

    w = torch.tensor(weights, device=device, dtype=coords.dtype)

    neighbors = []

    for i in range(0, N, batch_size):
        q_coords = coords[i:i+batch_size]
        q_values = values[i:i+batch_size]
        q_mask   = mask[i:i+batch_size]

        cand = candidate_idx[i:i+batch_size]  # [B, Kc]

        coords_ref = coords[cand]   # [B, Kc, D]
        values_ref = values[cand]   # [B, Kc, F]
        mask_ref   = mask[cand]     # [B, Kc, F]

        B, Kc = cand.shape

        # Spatio-temporal distances
        d_geo = 0
        for d in range(D):
            a = q_coords[:, d].unsqueeze(1)
            b = coords_ref[:, :, d]
            d_geo += w[d] * (a - b) ** 2

        # Feature distances
        d_feat = torch.zeros((B, Kc), device=device, dtype=coords.dtype)
        denom  = torch.zeros((B, Kc), device=device, dtype=coords.dtype)

        for f in range(F):
            qi = q_values[:, f].unsqueeze(1)
            qj = values_ref[:, :, f]

            mi = q_mask[:, f].unsqueeze(1).float()
            mj = mask_ref[:, :, f].float()

            m = mi * mj

            diff2 = (qi - qj) ** 2

            d_feat += diff2 * m
            denom  += m

        d_feat = torch.where(
            denom > 0,
            d_feat / denom,
            torch.zeros_like(d_feat)
        )

        # Combine geo and feature distances
        dist = d_geo * (1.0 + lambda_ * d_feat)

        # Remove self
        rows = torch.arange(B, device=device)
        self_idx = i + rows
        mask_self = cand == self_idx.unsqueeze(1)
        dist[mask_self] = float("inf")

        # KNN
        knn_local = torch.topk(dist, k=k, largest=False).indices
        knn_idx = torch.gather(cand, 1, knn_local)

        neighbors.append(knn_idx)

    return torch.cat(neighbors, dim=0)
