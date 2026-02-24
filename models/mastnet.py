import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from nn_utils.dataset import NeighbourDataset
from nn_utils.trainer import NeighbourAdapter


class MaSTNeT(nn.Module):
    def __init__(self, coord_dim=5, value_dim=6, d_model=64, nhead=4, nlayers=2, dropout=0.1, dim_feedforward=128):
        super().__init__()

        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.input_dim = self.coord_dim + self.value_dim * 2  # coords, features, feature masks

        # Input encoder
        self.encoder_input = nn.Linear(self.input_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        # Decoder: Mean head (for reconstruction)
        self.mean_decoder = nn.Sequential(
            nn.Linear(d_model, value_dim),
            nn.Sigmoid()
        )

        # Decoder: Variance head (for heteroscedastic uncertainty)
        self.var_decoder = nn.Sequential(
            nn.Linear(d_model, value_dim),
            nn.Softplus()
        )

    def forward(self, query_features, query_mask, query_coords, neighbour_features, neighbour_mask, rel_positions):
        """
            query_features: [batch_size, n_features]
            query_mask: [batch_size, n_features]
            neighbour_features: [batch_size, n_neighbours, n_features]
            neighbour_mask: [batch_size, n_neighbours, n_features]
            rel_positions: [batch_size, n_neighbours, coord_dim]
        """
        batch_size, n_features = query_features.shape

        # Prepare query token
        query_mask_float = query_mask.float().unsqueeze(1)  # [batch_size, 1, n_features]
        query_coords_float = query_coords.float().unsqueeze(1)
        query_feat_filled = torch.where(query_mask, query_features, torch.zeros_like(query_features)).unsqueeze(1)  # Fill missing features with 0
        query_token = torch.cat([query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]

        # Prepare neighbour token
        neighbour_mask_float = neighbour_mask.float()
        neighbour_feat_filled = torch.where(neighbour_mask, neighbour_features, torch.zeros_like(neighbour_features))  # Fill missing features with 0
        neighbour_tokens = torch.cat([rel_positions, neighbour_feat_filled, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]

        # Concatenate query and neighbour tokens (along sequence dimension)
        sequence = torch.cat([query_token, neighbour_tokens], dim=1)  # [batch_size, 1+n_neighbours, input_dim]

        # Encoding input
        x = self.encoder_input(sequence)  # [batch_size, 1+n_neighbours, d_model]
        encoded = self.encoder(x)

        # Take the query token output only for reconstruction (not neighbours)
        query_encoded = encoded[:, 0, :]  # [batch_size, d_model]

        # Predict mean and variance
        pmean = self.mean_decoder(query_encoded)  # [batch_size, n_features]
        pvar = self.var_decoder(query_encoded)  # [batch_size, n_features]

        return pmean, pvar

    def predict(self, x, y, n_neighbours, batch_size, device: torch.device = torch.device("cpu")):
        """ Impute missing data given scaled x (coordinates) and y (values). """
        # Compute neighbours
        n_samples = y.shape[0]
        neighbours = NearestNeighbors(n_neighbors=min(n_neighbours, n_samples), algorithm="auto").fit(x.cpu().numpy())
        neighbour_indices = neighbours.kneighbors(x.cpu().numpy(), return_distance=False)
        neighbour_indices = torch.as_tensor(neighbour_indices[:, 1:], dtype=torch.long, device="cpu")  # Exclude self and convert to tensor

        # Define dataset and loader (to predict everything)
        dataset = NeighbourDataset(coords=x, values=y, query_indices=None, neighbour_indices=neighbour_indices)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        adapter = NeighbourAdapter()

        # Predict batch-wise
        y_hat = []
        y_var = []
        with torch.no_grad():
            for batch in loader:
                batch = adapter.prepare_batch(batch, device=device)
                masks = adapter.make_masks(batch=batch, mask_ratio=0.0, mode="reconstruct", device=device)
                pmean, pvar = adapter.forward(self, batch=batch, masks=masks)

                y_hat.append(pmean.cpu())
                y_var.append(pvar.cpu())

        return torch.cat(y_hat, dim=0).cpu().numpy(), torch.cat(y_var, dim=0).cpu().numpy()
