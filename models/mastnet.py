import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

#from nn_utils.dataset import NeighbourDataset
#from nn_utils.trainer import NeighbourAdapter
from nn_utils.embed import PositionalEncoder, CoordEncoder
from tuning_studies.modelconfig import ModelConfig
from utils.preprocessing import fill_feature_tensor

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Multi-head attention (only query attends to neighbours)
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

    def forward(self, q, n):
        attn_out, _ = self.attn(q, n, n)
        q = self.norm1(q + attn_out)

        mlp_out = self.mlp(q)
        q = self.norm2(q + mlp_out)

        return q

class MaSTNeT(nn.Module):
    def __init__(self,
                 cfg: ModelConfig,
                 coord_dim=5, value_dim=6, pos_hidden_dim=64,
                 d_model=64, nhead=4,
                 nlayers=3, dropout=0.1, dim_feedforward=128, global_means=None):
        super().__init__()
        self.cfg = cfg
        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.input_dim = self.compute_input_dim()

        # Define global means (and automatically make it move to correct device)
        if global_means is None:
            self.global_means = None
        else:
            self.register_buffer("global_means", torch.tensor(global_means, dtype=torch.float32))

        # Optional coordinate encoder
        if self.cfg.encoder_scope != "none":
            self.coord_encoder = CoordEncoder(
                cfg=self.cfg,
                hidden_dim=cfg.encoder_hidden_dim,
                output_dim=cfg.encoder_output_dim,
                coord_dim=coord_dim-1,
                time_dim=1,
                value_dim=value_dim,
            )
        else:
            self.coord_encoder = None

        # Optional feature encoder
        if self.cfg.feature_mixer:
            if self.cfg.feature_mixer_input == "feat":
                self.feature_mixer = nn.Sequential(
                    nn.Linear(self.value_dim, self.value_dim),
                    nn.ReLU(),
                    nn.Linear(self.value_dim, self.value_dim),
                )
            elif self.cfg.feature_mixer_input == "feat_mask":
                self.feature_mixer = nn.Sequential(
                    nn.Linear(self.value_dim * 2, self.value_dim),
                    nn.ReLU(),
                    nn.Linear(self.value_dim, self.value_dim),
                )
            else:
                raise ValueError(f"MaSTNeT: Unknown feature_mixer_input {self.cfg.feature_mixer_input}")
        else:
            self.feature_mixer = None

        # Input projector
        self.input_projector = nn.Linear(self.input_dim, d_model)

        if self.cfg.attention_type == "mha":
            self.mha_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

        elif self.cfg.attention_type == "transformer_encoder_layer":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
                dropout=dropout
            )
            self.encoder_transformer = nn.TransformerEncoder(encoder_layer, nlayers)

        elif self.cfg.attention_type == "encoder_decoder":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
                dropout=dropout
            )
            self.encoder_transformer = nn.TransformerEncoder(encoder_layer, nlayers)

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
                dropout=dropout
            )
            self.decoder_transformer = nn.TransformerDecoder(decoder_layer, nlayers)

        elif self.cfg.attention_type == "space_time_attention":
            self.time_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

            self.space_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

        else:
            raise ValueError(f"MaSTNeT: Unknown attention_type {self.cfg.attention_type}")

        # Decoding
        self.mean_decoder = nn.Linear(d_model, value_dim)  # Mean head (for reconstruction)
        self.var_decoder = nn.Linear(d_model, value_dim)  # Variance head (for heteroscedastic uncertainty)

    def compute_input_dim(self):
        dim = 0

        # Coordinate encoding
        if self.cfg.encoder_scope not in ["none", "graph"]:
            dim += self.cfg.encoder_output_dim
        else:
            dim += self.coord_dim

        # Feature encoding
        dim += self.value_dim

        # Relative positions
        if self.cfg.use_rel_pos:
            dim += self.coord_dim

        # Masks
        if self.cfg.use_masks:
            dim += self.value_dim

        return dim

    def construct_tokens(self, query_coords, encoded_query_coords, encoded_query_features, query_mask,
                         rel_positions, encoded_neighbour_coords, encoded_neighbour_features, neighbour_mask):
        query_token_input = []
        neighbour_token_input = []

        if self.cfg.use_rel_pos:
            rel_positions_dummy = torch.zeros_like(query_coords)
            query_token_input.append(rel_positions_dummy)
            neighbour_token_input.append(rel_positions)

        query_token_input.append(encoded_query_coords)
        query_token_input.append(encoded_query_features)
        neighbour_token_input.append(encoded_neighbour_coords)
        neighbour_token_input.append(encoded_neighbour_features)

        if self.cfg.use_masks:
            query_token_input.append(query_mask.float())
            neighbour_token_input.append(neighbour_mask.float())

        query_token = torch.cat(query_token_input, dim=-1).unsqueeze(1)  # [B, 1, input_dim]
        neighbour_tokens = torch.cat(neighbour_token_input, dim=-1)  # [B, K, input_dim]

        return query_token, neighbour_tokens

    def forward_space_time(self, query_features, query_mask, query_coords, neighbour_features, neighbour_coords,
                           neighbour_mask, rel_positions):
        # Get space/time info
        nf_space = neighbour_features["space"]
        nf_time = neighbour_features["time"]

        nc_space = neighbour_coords["space"]
        nc_time = neighbour_coords["time"]

        nm_space = neighbour_mask["space"]
        nm_time = neighbour_mask["time"]

        rp_space = rel_positions["space"]
        rp_time = rel_positions["time"]

        # Fill features
        query_features_filled = fill_feature_tensor(features=query_features, mask=query_mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)
        n_space_features_filled = fill_feature_tensor(features=nf_space, mask=nm_space, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)
        n_time_features_filled = fill_feature_tensor(features=nf_time, mask=nm_time, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)

        # Optional feature encoding
        if self.feature_mixer is not None and self.cfg.feature_mixer_input in ["feat", "feat_mask"]:
            if self.cfg.feature_mixer_input == "feat":
                encoded_query_features = self.feature_mixer(query_features_filled)
                encoded_n_space_features = self.feature_mixer(n_space_features_filled)
                encoded_n_time_features = self.feature_mixer(n_time_features_filled)
            elif self.cfg.feature_mixer_input == "feat_mask":
                encoded_query_features = self.feature_mixer(torch.cat([query_features_filled, query_mask], dim=-1))
                encoded_n_space_features = self.feature_mixer(torch.cat([n_space_features_filled, nm_space], dim=-1))
                encoded_n_time_features = self.feature_mixer(torch.cat([n_time_features_filled, nm_time], dim=-1))
        else:
            encoded_query_features = query_features_filled
            encoded_n_space_features = n_space_features_filled
            encoded_n_time_features = n_time_features_filled

        # Optional coordinate encoding
        if self.coord_encoder is not None and self.cfg.encoder_scope in ["model", "both"]:
            encoded_query_coords = self.coord_encoder(coords=query_coords[:, :self.coord_dim - 1],
                                                      times=query_coords[:, -1:], values=query_features_filled,
                                                      mask=query_mask.float())
            encoded_n_space_coords = self.coord_encoder(coords=nc_space[:, :, :self.coord_dim - 1],
                                                          times=nc_space[:, :, -1:], values=n_space_features_filled,
                                                          mask=nm_space.float())
            encoded_n_time_coords = self.coord_encoder(coords=nc_time[:, :, :self.coord_dim - 1],
                                                          times=nc_time[:, :, -1:], values=n_time_features_filled,
                                                          mask=nm_time.float())
        else:
            encoded_query_coords = query_coords
            encoded_n_space_coords = nc_space
            encoded_n_time_coords = nc_time

        # Construct tokens
        query_token, n_space_tokens = self.construct_tokens(
            query_coords=query_coords, encoded_query_coords=encoded_query_coords,
            encoded_query_features=encoded_query_features, query_mask=query_mask,
            rel_positions=rp_space, encoded_neighbour_coords=encoded_n_space_coords,
            encoded_neighbour_features=encoded_n_space_features, neighbour_mask=nm_space)
        _, n_time_tokens = self.construct_tokens(
            query_coords=query_coords, encoded_query_coords=encoded_query_coords,
            encoded_query_features=encoded_query_features, query_mask=query_mask,
            rel_positions=rp_time, encoded_neighbour_coords=encoded_n_time_coords,
            encoded_neighbour_features=encoded_n_time_features, neighbour_mask=nm_time)

        # Attention (time, then space)
        q = self.input_projector(query_token)
        ns = self.input_projector(n_space_tokens)
        nt = self.input_projector(n_time_tokens)

        for layer in self.time_layers:
            q = layer(q, nt)

        for layer in self.space_layers:
            q = layer(q, ns)

        # Take the query token output only for reconstruction (not neighbours)
        query_encoded = q[:, 0, :]  # [B, d_model]

        # Decode
        pmean = self.mean_decoder(query_encoded)
        pvar = self.var_decoder(query_encoded)  # [B, F]

        return pmean, pvar

    def forward(self, query_features, query_mask, query_coords, neighbour_features, neighbour_coords, neighbour_mask,
                rel_positions):
        """
            query_features: [batch_size, n_features]
            query_mask: [batch_size, n_features]
            neighbour_features: [batch_size, n_neighbours, n_features]
            neighbour_mask: [batch_size, n_neighbours, n_features]
            rel_positions: [batch_size, n_neighbours, coord_dim]
        """
        if self.cfg.attention_type == "space_time_attention":
            return self.forward_space_time(query_features, query_mask, query_coords, neighbour_features,
                                           neighbour_coords, neighbour_mask, rel_positions)

        # Feature filling
        query_features_filled = fill_feature_tensor(features=query_features, mask=query_mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)
        neighbour_features_filled = fill_feature_tensor(features=neighbour_features, mask=neighbour_mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)

        # Optional feature encoding
        if self.feature_mixer is not None and self.cfg.feature_mixer_input in ["feat", "feat_mask"]:
            if self.cfg.feature_mixer_input == "feat":
                encoded_query_features = self.feature_mixer(query_features_filled)
                encoded_neighbour_features = self.feature_mixer(neighbour_features_filled)
            elif self.cfg.feature_mixer_input == "feat_mask":
                encoded_query_features = self.feature_mixer(torch.cat([query_features_filled, query_mask], dim=-1))
                encoded_neighbour_features = self.feature_mixer(torch.cat([neighbour_features_filled, neighbour_mask], dim=-1))
        else:
            encoded_query_features = query_features_filled
            encoded_neighbour_features = neighbour_features_filled

        # Optional coordinate encoding
        if self.coord_encoder is not None and self.cfg.encoder_scope in ["model", "both"]:
            encoded_query_coords = self.coord_encoder(coords=query_coords[:, :self.coord_dim - 1],
                                                      times=query_coords[:, -1:], values=query_features_filled,
                                                      mask=query_mask.float())
            encoded_neighbour_coords = self.coord_encoder(coords=neighbour_coords[:, :, :self.coord_dim - 1],
                                                          times=neighbour_coords[:, :, -1:], values=neighbour_features_filled,
                                                          mask=neighbour_mask.float())
        else:
            encoded_query_coords = query_coords
            encoded_neighbour_coords = neighbour_coords

        # Construct tokens
        query_token, neighbour_tokens = self.construct_tokens(
            query_coords=query_coords, encoded_query_coords=encoded_query_coords,
            encoded_query_features=encoded_query_features, query_mask=query_mask,
            rel_positions=rel_positions, encoded_neighbour_coords=encoded_neighbour_coords,
            encoded_neighbour_features=encoded_neighbour_features, neighbour_mask=neighbour_mask)

        # Attention
        if self.cfg.attention_type == "mha":
            q = self.input_projector(query_token)
            n = self.input_projector(neighbour_tokens)  # + rel_pos_embed

            for layer in self.mha_layers:
                q = layer(q, n)
            encoded = q

        elif self.cfg.attention_type == "transformer_encoder_layer":
            sequence = torch.cat([query_token, neighbour_tokens], dim=1)
            x = self.input_projector(sequence)  # [B, K, d_model]
            encoded = self.encoder_transformer(x)

        elif self.cfg.attention_type == "encoder_decoder":
            # Encode neighbours
            n = self.input_projector(neighbour_tokens)  # [B, K, d_model]
            n = self.encoder_transformer(n)

            # Encode query
            q = self.input_projector(query_token)  # [B, 1, d_model]

            # Decode query conditioned on memory (neighbours)
            encoded = self.decoder_transformer(tgt=q, memory=n)

        else:
            raise ValueError(f"MaSTNeT: Unknown attention type: {self.cfg.attention_type}")

        # Take the query token output only for reconstruction (not neighbours)
        query_encoded = encoded[:, 0, :]  # [B, d_model]

        # Decode
        pmean = self.mean_decoder(query_encoded)
        pvar = self.var_decoder(query_encoded)  # [B, F]

        return pmean, pvar



        batch_size, n_features = query_features.shape

        # Prepare neighbour token
        # rel_pos_embed = self.pos_encoder(rel_positions)  # Embed relative positions

        # Encode query coordinates
        q_time = query_coords[:, -1:].float()
        q_space = query_coords[:, :4].float()

        q_feat_for_encoder = fill_feature_tensor(features=query_features, mask=query_mask, fill_strategy="mean", mean_values=self.global_means)
        z_query = self.coord_encoder(coords=q_space, times=q_time, values=q_feat_for_encoder, mask=query_mask.float()).unsqueeze(1)  # [B, 1, d_enc]

        # Encode neighbour coordinates
        n_time = neighbour_coords[:, :, -1:].float()
        n_space = neighbour_coords[:, :, :4].float()

        n_feat_for_encoder = fill_feature_tensor(features=neighbour_features, mask=neighbour_mask, fill_strategy="mean", mean_values=self.global_means)
        z_neighbours = self.coord_encoder(coords=n_space, times=n_time, values=n_feat_for_encoder, mask=neighbour_mask.float()).unsqueeze(1)  # [B, 1, d_enc]

        # Prepare query token
        query_mask_float = query_mask.float().unsqueeze(1)  # [batch_size, 1, n_features]
        query_coords_float = query_coords.float().unsqueeze(1)
        query_feat_filled = fill_feature_tensor(features=query_features, mask=query_mask, fill_strategy="mean", mean_values=self.global_means).unsqueeze(1)
        query_feat_filled = self.feature_mixer(torch.cat([query_feat_filled, query_mask_float], dim=-1))
        #query_feat_filled = self.feature_mixer(query_feat_filled)

        # global_pred = self.global_mlp(torch.cat([query_feat_filled, query_mask_float], dim=-1).squeeze(1))
        # print(global_pred.shape)

        # rel_positions_dummy = torch.zeros(query_coords.shape[0], 1, self.coord_dim, device=query_coords.device)
        # rel_pos_query_embed = self.pos_encoder(rel_positions_dummy)
        ## query_token = torch.cat([rel_pos_query_embed, query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]
        # query_token = torch.cat([query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]

        rel_positions_dummy = torch.zeros_like(query_coords_float)
        # query_token = torch.cat([rel_positions_dummy, query_coords_float, query_feat_filled, query_mask_float], dim=-1)  # [batch_size, 1, input_dim]
        query_token = torch.cat([rel_positions_dummy, z_query, query_feat_filled], dim=-1)  # [batch_size, 1, input_dim]

        neighbour_mask_float = neighbour_mask.float()
        neighbour_feat_filled = torch.where(neighbour_mask, neighbour_features, torch.zeros_like(neighbour_features) if self.global_means is None else self.global_means.unsqueeze(0))  # Fill missing features with 0
        neighbour_feat_filled = self.feature_mixer(torch.cat([neighbour_feat_filled, neighbour_mask_float], dim=-1))
        #neighbour_feat_filled = self.feature_mixer(neighbour_feat_filled)
        # neighbour_tokens = torch.cat([rel_positions, neighbour_coords, neighbour_feat_filled, neighbour_mask_float], dim=-1)  # [batch_size, neighbours, input_dim]
        neighbour_tokens = torch.cat([rel_positions, z_neighbours, neighbour_feat_filled], dim=-1)  # [batch_size, neighbours, input_dim]
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
    #
    # def predict(self, x, y, n_neighbours, batch_size, device: torch.device = torch.device("cpu")):
    #     """ Impute missing data given scaled x (coordinates) and y (values). """
    #     # Compute neighbours
    #     n_samples = y.shape[0]
    #     neighbours = NearestNeighbors(n_neighbors=min(n_neighbours, n_samples), algorithm="auto").fit(x.cpu().numpy())
    #     neighbour_indices = neighbours.kneighbors(x.cpu().numpy(), return_distance=False)
    #     neighbour_indices = torch.as_tensor(neighbour_indices[:, 1:], dtype=torch.long, device="cpu")  # Exclude self and convert to tensor
    #
    #     # Define dataset and loader (to predict everything)
    #     dataset = NeighbourDataset(coords=x, values=y, query_indices=None, neighbour_indices=neighbour_indices)
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #     adapter = NeighbourAdapter()
    #
    #     # Predict batch-wise
    #     y_hat = []
    #     y_var = []
    #     with torch.no_grad():
    #         for batch in loader:
    #             batch = adapter.prepare_batch(batch, device=device)
    #             masks = adapter.make_masks(batch=batch, mask_ratio=0.0, mode="reconstruct", device=device)
    #             pmean, pvar = adapter.forward(self, batch=batch, masks=masks)
    #
    #             y_hat.append(pmean.cpu())
    #             y_var.append(pvar.cpu())
    #
    #     return torch.cat(y_hat, dim=0).cpu().numpy(), torch.exp(torch.cat(y_var, dim=0)).cpu().numpy()
    #

        #
        #
        # # Define global means (and automatically make it move to correct device)
        # if global_means is None:
        #     self.global_means = None
        # else:
        #     self.register_buffer("global_means", torch.tensor(global_means, dtype=torch.float32))
        #
        # self.coord_dim = coord_dim
        # self.value_dim = value_dim
        # self.coord_encoder_hidden_dim = coord_encoder_hidden_dim
        # self.coord_encoder_output_dim = coord_encoder_output_dim
        #
        # self.input_dim = self.coord_dim * 2 + self.value_dim  # coords and relative coords, features
        # self.input_dim = self.coord_dim + self.value_dim + self.coord_encoder_output_dim # relative coords, features, encoded coords
        # # self.input_dim = self.coord_dim * 2 + self.value_dim * 2  # coords and relative coords, features and feature masks
        # # self.input_dim = self.coord_dim + d_model + self.value_dim * 2  # coords and relative coords embedding, features and feature masks
        # # self.input_dim = self.coord_dim + self.value_dim * 2  # coords and relative coords embedding, features and feature masks
        #
        # # Positional embeddings
        # # self.pos_encoder = PositionalEncoder(input_dim=self.coord_dim, hidden_dim=pos_hidden_dim, output_dim=d_model)
        # # self.pos_bias = nn.Linear(d_model, nhead)
        #
        # self.coord_encoder = CoordEncoder(hidden_dim=coord_encoder_hidden_dim,
        #                                   coord_dim=coord_dim-1, value_dim=value_dim, time_dim=1,
        #                                   output_dim=3
        #                                   )
        #
        # # Input encoder
        # self.encoder_input = nn.Linear(self.input_dim, d_model)
        #
        # # Multi-head attention
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=nhead,
        #     batch_first=True,
        #     dropout=dropout
        # )
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        #
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, d_model),
        # )
        #
        # # Decoder: Mean head (for reconstruction)
        # self.mean_decoder = nn.Linear(d_model, value_dim)
        #
        # # Decoder: Variance head (for heteroscedastic uncertainty)
        # self.var_decoder = nn.Linear(d_model, value_dim)
        #
        # self.feature_mixer = nn.Sequential(
        #     nn.Linear(self.value_dim * 2, self.value_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.value_dim, self.value_dim),
        # )
