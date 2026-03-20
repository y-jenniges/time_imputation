import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from nn_utils.dataset import NeighbourDataset
from nn_utils.trainer import NeighbourAdapter
from nn_utils.embed import PositionalEncoder, CoordEncoder


class MaSTNeT(nn.Module):
    def __init__(self, coord_dim=5, value_dim=6, pos_hidden_dim=64, coord_encoder_hidden_dim=32, d_model=64, nhead=4, nlayers=2, dropout=0.1, dim_feedforward=128):
        super().__init__()

        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.input_dim = self.coord_dim * 2 + self.value_dim * 2  # coords and relative coords, features and feature masks
        # self.input_dim = self.coord_dim + d_model + self.value_dim * 2  # coords and relative coords embedding, features and feature masks
        # self.input_dim = self.coord_dim + self.value_dim * 2  # coords and relative coords embedding, features and feature masks

        # Positional embeddings
        # self.pos_encoder = PositionalEncoder(input_dim=self.coord_dim, hidden_dim=pos_hidden_dim, output_dim=d_model)
        # self.pos_bias = nn.Linear(d_model, nhead)

        self.coord_encoder = CoordEncoder(hidden_dim=coord_encoder_hidden_dim, coord_dim=coord_dim-1, value_dim=value_dim, time_dim=1)

        # Input encoder
        self.encoder_input = nn.Linear(self.input_dim, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Decoder: Mean head (for reconstruction)
        self.mean_decoder = nn.Linear(d_model, value_dim)

        # Decoder: Variance head (for heteroscedastic uncertainty)
        self.var_decoder = nn.Linear(d_model, value_dim)

        self.feature_mixer = nn.Sequential(
            nn.Linear(self.value_dim * 2, self.value_dim),
            nn.ReLU(),
            nn.Linear(self.value_dim, self.value_dim),
        )

    def forward(self, query_features, query_mask, query_coords, neighbour_features, neighbour_coords, neighbour_mask,
                rel_positions):
        """
            query_features: [batch_size, n_features]
            query_mask: [batch_size, n_features]
            neighbour_features: [batch_size, n_neighbours, n_features]
            neighbour_mask: [batch_size, n_neighbours, n_features]
            rel_positions: [batch_size, n_neighbours, coord_dim]
        """
        batch_size, n_features = query_features.shape

        # Prepare neighbour token
        # rel_pos_embed = self.pos_encoder(rel_positions)  # Embed relative positions

        # Prepare query token
        query_mask_float = query_mask.float().unsqueeze(1)  # [batch_size, 1, n_features]
        query_coords_float = query_coords.float().unsqueeze(1)
        query_feat_filled = torch.where(query_mask, query_features, torch.zeros_like(query_features)).unsqueeze(1)  # Fill missing features with 0
        query_feat_filled = self.feature_mixer(torch.cat([query_feat_filled, query_mask_float], dim=-1))
        #query_feat_filled = self.feature_mixer(query_feat_filled)

        # global_pred = self.global_mlp(torch.cat([query_feat_filled, query_mask_float], dim=-1).squeeze(1))
        # print(global_pred.shape)

        # rel_positions_dummy = torch.zeros(query_coords.shape[0], 1, self.coord_dim, device=query_coords.device)
        # rel_pos_query_embed = self.pos_encoder(rel_positions_dummy)
        ## query_token = torch.cat([rel_pos_query_embed, query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]
        # query_token = torch.cat([query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]

        rel_positions_dummy = torch.zeros_like(query_coords_float)
        query_token = torch.cat([rel_positions_dummy, query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]

        neighbour_mask_float = neighbour_mask.float()
        neighbour_feat_filled = torch.where(neighbour_mask, neighbour_features, torch.zeros_like(neighbour_features))  # Fill missing features with 0
        neighbour_feat_filled = self.feature_mixer(torch.cat([neighbour_feat_filled, neighbour_mask_float], dim=-1))
        #neighbour_feat_filled = self.feature_mixer(neighbour_feat_filled)
        neighbour_tokens = torch.cat([rel_positions, neighbour_coords, neighbour_feat_filled, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]
        # neighbour_tokens = torch.cat([rel_pos_embed, neighbour_coords, neighbour_feat_filled, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]
        # neighbour_tokens = torch.cat([neighbour_coords, neighbour_feat_filled, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]

        # Weight for relative positions
        # weights = torch.softmax(-torch.norm(rel_positions, dim=-1), dim=-1)
        # weighted_neighbours = weights.unsqueeze(-1) * neighbour_feat_filled
        # neighbour_tokens = torch.cat([rel_positions, neighbour_coords, weighted_neighbours, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]

        # Encode tokens and perform cross attention (query attends to neighbours)
        q = self.encoder_input(query_token)
        n = self.encoder_input(neighbour_tokens) # + rel_pos_embed
        attn_out, _ = self.attn(q, n, n, attn_mask=None, key_padding_mask=None)  # query attends to neighbours
        q = self.norm1(q + attn_out)
        mlp_out = self.mlp(q)
        encoded = self.norm2(q + mlp_out)

        # Take the query token output only for reconstruction (not neighbours)
        query_encoded = encoded[:, 0, :]  # [batch_size, d_model]

        # Predict mean and variance
        pmean = self.mean_decoder(query_encoded)
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

        return torch.cat(y_hat, dim=0).cpu().numpy(), torch.exp(torch.cat(y_var, dim=0)).cpu().numpy()
