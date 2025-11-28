import torch
import torch.nn as nn


class OceanMAE(nn.Module):
    def __init__(self, coord_dim=5, value_dim=6, d_model=64, nhead=4, nlayers=2, dropout=0.1, dim_feedforward=128):
        super().__init__()

        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.input_dim = self.coord_dim + self.value_dim * 2  # coords, features, feature masks
        self.dropout = nn.Dropout(dropout)

        # Input encoder
        self.encoder_input = nn.Linear(self.input_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True
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

    def forward(self, coords, values, feature_mask, mc_dropout=False):
        # Fill nans with 0 (in features)
        values_filled = torch.where(torch.isnan(values), torch.zeros_like(values), values)
        x = torch.cat([coords, values_filled, feature_mask.float(), ], dim=-1)

        # Encoding input and dropout
        x = self.encoder_input(x)
        if mc_dropout:
            x = self.dropout(x)

        encoded = self.encoder(x.unsqueeze(1)).squeeze(1)

        # Predict mean and variance
        pmean = self.mean_decoder(encoded)
        pvar = torch.exp(self.var_decoder(encoded))

        return pmean, pvar
