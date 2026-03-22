import torch
from sklearn.neighbors import NearestNeighbors


class GraphProvider:
    def __init__(self, n_neighbours=20, update_every: int = 5, test_idx: torch.Tensor = None, val_idx: torch.Tensor = None):
        self.n_neighbours = n_neighbours
        self.update_every = update_every
        self.neighbour_indices = None

        self.test_idx = test_idx
        self.val_idx = val_idx

    def update(self, encoder, coords, values, mask):
        """ Recompute graph from latent space. """
        values = values.clone()
        mask = mask.clone()

        if self.test_idx is not None:
            values[self.test_idx] = torch.nan

        if self.val_idx is not None:
            values[self.val_idx] = torch.nan

        with torch.no_grad():
            # Encode
            encoded = encoder(coords=coords[:, :4], values=torch.nan_to_num(values, nan=0.0), mask=mask.float(), times=coords[:, -1:].float())
            encoded_np = encoded.detach().cpu().numpy()

            # Construct KNN graph
            knn = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
            knn.fit(encoded_np)

            # Get neighbour indices
            indices = torch.from_numpy(knn.kneighbors(encoded_np, return_distance=False)).long()
            self.neighbour_indices = indices[:, 1:]  # Remove self

    def get(self, idx):
        return self.neighbour_indices[idx]
