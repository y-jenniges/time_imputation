import torch
from sklearn.neighbors import NearestNeighbors


class GraphProvider:
    def __init__(self, n_neighbours=20, update_every: int = 5, test_idx: torch.Tensor = None, val_idx: torch.Tensor = None, device: torch.device = torch.device("cpu")):
        self.n_neighbours = n_neighbours
        self.update_every = update_every
        self.neighbour_indices = None

        self.test_idx = test_idx
        self.val_idx = val_idx

        self.device = device

    def update(self, encoder, coords, values, mask):
        """ Recompute graph from latent space. """
        values = values.clone()
        mask = mask.clone()

        coords = coords.to(self.device)
        values = values.to(self.device)
        mask = mask.to(self.device)

        if self.test_idx is not None:
            values[self.test_idx] = torch.nan

        if self.val_idx is not None:
            values[self.val_idx] = torch.nan

        with torch.no_grad():
            # --- Encoder-KNN -----------------------------------------------
            # # Encode
            # encoded = encoder(coords=coords[:, :4], values=torch.nan_to_num(values, nan=0.0), mask=mask.float(), times=coords[:, -1:].float())
            # encoded_np = encoded.detach().cpu().numpy()
            #
            # # Construct KNN graph
            # knn = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
            # knn.fit(encoded_np)

            # # Get neighbour indices
            # indices = torch.from_numpy(knn.kneighbors(encoded_np, return_distance=False)).long()
            # self.neighbour_indices = indices[:, 1:]  # Remove self

            # # --- KNN only on coordinates -----------------------------------------------
            # # Construct KNN graph
            # knn = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
            # knn.fit(coords)
            #
            # # Get neighbour indices
            # indices = torch.from_numpy(knn.kneighbors(coords, return_distance=False)).long()
            # self.neighbour_indices = indices[:, 1:]  # Remove self

            # --- Anisotropic KNN -----------------------------------------------
            self.neighbour_indices = compute_anisotropic_knn(coords, values, mask, k=self.n_neighbours,
                                                             lambda_=1, weights=None, batch_size=5000,
                                                             device=self.device)

    def get(self, idx):
        return self.neighbour_indices[idx]


def compute_anisotropic_knn(coords, values, mask, lambda_=1, weights=None, k=20, batch_size=5000, device=torch.device("cpu")):
    if weights is None:
        weights = [1, 1, 1, 1, 1]

    coords = coords.to(device)
    values = values.to(device)
    mask   = mask.to(device)

    w = torch.tensor(weights, device=device, dtype=coords.dtype)

    neighbors = []
    for i in range(0, coords.shape[0], batch_size):
        q_coords = coords[i:i+batch_size]         # [B, 5]
        q_values = values[i:i+batch_size]         # [B, F]
        q_mask   = mask[i:i+batch_size]           # [B, F]

        # Spatio-temporal distances
        # diff = q_coords.unsqueeze(1) - coords.unsqueeze(0)  # [B, N, 5]
        # d_geo = (diff ** 2 * w).sum(-1)  # B, N
        d_geo = 0
        for d in range(coords.shape[-1]):
            diff_d = q_coords[:, d].unsqueeze(1) - coords[:, d].unsqueeze(0)
            d_geo = d_geo + w[d] * diff_d ** 2

        # Feature distances
        v_i = q_values.unsqueeze(1)              # [B, 1, F]
        v_j = values.unsqueeze(0)                # [1, N, F]

        m_i = q_mask.unsqueeze(1).float()
        m_j = mask.unsqueeze(0).float()

        joint_mask = m_i * m_j                   # [B, N, F]

        diff2 = (v_i - v_j) ** 2

        denom = joint_mask.sum(-1)  # [B, N]
        valid = denom > 0
        d_feat = torch.where(
            valid,
            (diff2 * joint_mask).sum(-1) / denom,
            torch.zeros_like(denom)
        )
        # Combine
        dist_combined = d_geo * (1.0 + lambda_ * d_feat)

        # Remove self (only if in batch)
        for b in range(q_coords.shape[0]):
            global_idx = i + b
            if global_idx < coords.shape[0]:
                dist_combined[b, global_idx] = float("inf")

        # KNN
        knn_idx = torch.topk(dist_combined, k=k, largest=False).indices  # [B, k]
        neighbors.append(knn_idx)

    return torch.cat(neighbors, dim=0)
