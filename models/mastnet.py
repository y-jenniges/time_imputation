import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

#from nn_utils.dataset import NeighbourDataset
#from nn_utils.trainer import NeighbourAdapter
from nn_utils.embed import PositionalEncoder, CoordEncoder
from tuning_studies.modelconfig import ModelConfig
from utils.preprocessing import fill_feature_tensor, get_scopes


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
        attn_out, attn_weights = self.attn(q, n, n)
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
        self.d_model = d_model
        self.input_dim = self.compute_input_dim()

        # Define global means (and automatically make it move to correct device)
        if global_means is None:
            self.global_means = None
        else:
            self.register_buffer("global_means", torch.tensor(global_means, dtype=torch.float32))

        # Optional learnable anisotropic graph weights
        if self.cfg.learn_anisotropic_weights:
            self.anisotropic_weights = nn.Parameter(torch.ones(coord_dim))
            print("Weights in mastnet: ", self.anisotropic_weights)
        else:
            self.anisotropic_weights = None

        # Optional positional encoding
        if self.cfg.positional_encoding:
            pos_input_dim = self.coord_dim if not self.cfg.positional_encoding_time_only else 1
            self.pos_encoder = PositionalEncoder(input_dim=pos_input_dim, hidden_dim=pos_hidden_dim, output_dim=d_model)
            # self.pos_bias = nn.Linear(d_model, nhead)

        # ...
        if self.cfg.graph_mode == "single_feature":
            feat_input_dim = 1 + (1 if self.cfg.use_masks else 0)  # [value, mask_bit]
            self.feat_projector = nn.Linear(feat_input_dim, d_model)
            self.coord_projector = nn.Linear(1, d_model)  # One scalar per coord token
            self.coord_embedding = nn.Embedding(coord_dim, d_model)

            self.feature_embedding = nn.Embedding(value_dim, d_model)

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

        # Optional global context
        if self.cfg.global_context:
            self.global_context = nn.Sequential(
                nn.Linear(d_model, d_model),
            )

            self.gate_mlp = nn.Sequential(
                # nn.Linear(d_model, d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
        else:
            self.global_context = None
            self.gate_mlp = None

        # Input projector
        self.input_projector = nn.Linear(self.input_dim, d_model)

        if self.cfg.attention_type == "mha":
            self.mha_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

        elif self.cfg.attention_type == "transformer_encoder":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
                dropout=dropout
            )
            self.encoder_transformer = nn.TransformerEncoder(encoder_layer, nlayers)

        elif self.cfg.attention_type == "autoencoder":
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

        elif self.cfg.attention_type == "mha_decoder":
            # Encoder
            self. mha_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", batch_first=True,
                dropout=dropout
            )
            self.decoder_transformer = nn.TransformerDecoder(decoder_layer, nlayers)

        elif self.cfg.attention_type == "space_time_attention" or cfg.attention_type == "time_space_attention":
            self.time_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(self.cfg.n_time_layers)
            ])

            self.space_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

        elif self.cfg.attention_type == "space_time_depth_attention":
            self.time_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(self.cfg.n_time_layers)
            ])

            self.space_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

            self.depth_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])
        elif self.cfg.attention_type == "weighted_space_time_depth_attention":
            self.time_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(self.cfg.n_time_layers)
            ])

            self.space_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])

            self.depth_layers = nn.ModuleList([
                CrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(nlayers)
            ])
            self.gate = nn.Linear(d_model, 3)

        else:
            raise ValueError(f"MaSTNeT: Unknown attention_type {self.cfg.attention_type}")

        # Decoding
        output_dim = value_dim if self.cfg.graph_mode != "single_feature" else 1
        self.mean_decoder = nn.Linear(d_model, output_dim)  # Mean head (for reconstruction)
        self.var_decoder = nn.Linear(d_model, output_dim)  # Variance head (for heteroscedastic uncertainty)

    def compute_input_dim(self):
        if self.cfg.graph_mode == "single_feature":
            dim = self.coord_dim
            dim += 1  # single feature value instead of value_dim
            if self.cfg.use_masks:
                dim += 1  # single mask bit instead of mask vector
            return dim

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
            if self.cfg.positional_encoding:
                dim += self.d_model
            else:
                if self.cfg.positional_encoding:
                    dim += 1
                else:
                    dim += self.coord_dim

        # Masks
        if self.cfg.use_masks:
            dim += self.value_dim

        return dim

    def construct_token(self, encoded_coords, encoded_features, mask, rel_positions):
        token_input = []

        if self.cfg.use_rel_pos:
            token_input.append(rel_positions)

        token_input.append(encoded_coords)
        token_input.append(encoded_features)

        if self.cfg.use_masks:
            token_input.append(mask.float())

        return torch.cat(token_input, dim=-1)

    def encode_features(self, features, mask):
        if self.feature_mixer is not None and self.cfg.feature_mixer_input in ["feat", "feat_mask"]:
            if self.cfg.feature_mixer_input == "feat":
                return self.feature_mixer(features)
            elif self.cfg.feature_mixer_input == "feat_mask":
                return self.feature_mixer(torch.cat([features, mask], dim=-1))

        return features

    def encode_coordinates(self, coords, features, mask):
        if self.coord_encoder is not None and self.cfg.encoder_scope in ["model", "both"]:
            if coords.dim() == 2:
                # Query coords
                return self.coord_encoder(coords=coords[:, :self.coord_dim - 1], times=coords[:, -1:], values=features, mask=mask)
            else:
                # Neighbour coords
                return self.coord_encoder(coords=coords[:, :, :self.coord_dim - 1], times=coords[:, :, -1:], values=features, mask=mask)

        return coords

    def positional_encoding(self, rel_positions):
        if self.cfg.positional_encoding:
            if self.cfg.positional_encoding_time_only:
                return self.pos_encoder(rel_positions=rel_positions[:, -1])
            return self.pos_encoder(rel_positions)

        return rel_positions

    def _forward_single_feature(self, batch):
        feat = batch["features"]  # (B, F)
        mask = batch["mask"]  # (B, F) bool
        coords = batch["coords"]  # (B, coord_dim)
        B, F = feat.shape

        # Feature filling
        feat_filled = fill_feature_tensor(features=feat, mask=mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)

        # Optional feature encoding
        encoded_features = self.encode_features(features=feat_filled, mask=mask)

        # Optional coordinate encoding
        encoded_coords = self.encode_coordinates(coords=coords, features=feat_filled, mask=mask)

        # Broadcast coords into each feature token: (B, F, coord_out_dim)
        coords_exp = encoded_coords.unsqueeze(1).expand(-1, F, -1)

        # Build per-feature tokens: [coord, value_i, (mask_i)]
        parts = [coords_exp,  # (B, F, coord_out_dim)
                 encoded_features.unsqueeze(-1)]  # (B, F, 1)
        if self.cfg.use_masks:
            parts.append(mask.unsqueeze(-1).float())  # (B, F, 1)

        tokens = torch.cat(parts, dim=-1)  # (B, F, input_dim)

        # Project to d_model and add feature identity embedding
        x = self.input_projector(tokens)  # (B, F, d_model)
        feat_ids = torch.arange(F, device=feat.device).unsqueeze(0).expand(B, F)
        x = x + self.feature_embedding(feat_ids)  # (B, F, d_model)

        # Attention
        if self.cfg.attention_type == "transformer_encoder":
            encoded = self.encoder_transformer(x)  # (B, F, d_model)

        elif self.cfg.attention_type == "autoencoder":
            # Encoder: Self-attention over all feature tokens
            memory = self.encoder_transformer(x)  # (B, F, d_model)
            # Decoder: feature tokens attend to encoded memory
            # Refine "query" from encoded "context" (mirrors exp11/47)
            encoded = self.decoder_transformer(tgt=x, memory=memory)  # (B, F, d_model)

        else:
            raise ValueError(f"single_feature mode does not support {self.cfg.attention_type}")

        # Decode per-feature token to scalar prediction
        pmean = self.mean_decoder(encoded).squeeze(-1)  # (B, F)
        pvar = self.var_decoder(encoded).squeeze(-1)  # (B, F)

        return pmean, pvar
    #
    # def _forward_single_feature(self, batch):
    #     """ More ReMasker-style """
    #     assert self.cfg.attention_type in ["transformer_encoder", "autoencoder"]
    #
    #     feat = batch["features"]  # (B, F)  IndexError: too many indices for tensor of dimension 3
    #     mask = batch["mask"]
    #     coords = batch["coords"]
    #
    #     B, F = feat.shape
    #     C = coords.shape[1]
    #
    #     # Zero-fill masked positions
    #     feat_filled = fill_feature_tensor(features=feat, mask=mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)
    #
    #     # Coord tokens: one token per coordinate dimension ---
    #     # Each coord scalar gets projected + a learned coord-position embedding
    #     coord_ids = torch.arange(C, device=coords.device).unsqueeze(0).expand(B, C)
    #     coord_tokens = self.coord_projector(coords.unsqueeze(-1))  # (B, C, d_model)
    #     coord_tokens = coord_tokens + self.coord_embedding(coord_ids)  # (B, C, d_model)
    #
    #     # Feature tokens: [value, mask_bit] per feature ---
    #     parts = [feat_filled.unsqueeze(-1)]  # (B, F, 1)
    #     if self.cfg.use_masks:
    #         parts.append(mask.unsqueeze(-1).float())  # (B, F, 1)
    #     feat_tokens_raw = torch.cat(parts, dim=-1)  # (B, F, 1 or 2)
    #
    #     feat_ids = torch.arange(F, device=feat.device).unsqueeze(0).expand(B, F)
    #     feat_tokens = self.feat_projector(feat_tokens_raw)  # (B, F, d_model)
    #     feat_tokens = feat_tokens + self.feature_embedding(feat_ids)
    #
    #     # Concatenate: [coord tokens | feat tokens]
    #     sequence = torch.cat([coord_tokens, feat_tokens], dim=1)  # (B, C+F, d_model)
    #
    #     # Attention
    #     if self.cfg.attention_type == "transformer_encoder":
    #         encoded = self.encoder_transformer(sequence)
    #
    #     elif self.cfg.attention_type == "autoencoder":
    #         # Encoder sees full sequence as context
    #         memory = self.encoder_transformer(sequence)
    #
    #         # Decoder: only feature tokens as tgt, full encoded sequence as memory
    #         feat_tgt = sequence[:, C:, :]  # (B, F, d_model) — feature tokens only
    #         encoded_feats = self.decoder_transformer(tgt=feat_tgt, memory=memory)
    #
    #         # Pad coord positions back so indexing stays consistent
    #         encoded = torch.cat([memory[:, :C, :], encoded_feats], dim=1)
    #
    #     else:
    #         raise ValueError(f"single_feature mode does not support {self.cfg.attention_type}")
    #
    #     # Decode only the feature part
    #     pmean = self.mean_decoder(encoded[:, C:, :]).squeeze(-1)
    #     pvar = self.var_decoder(encoded[:, C:, :]).squeeze(-1)
    #     return pmean, pvar

    def forward(self, batch):
        if self.cfg.graph_mode == "single_feature":
            return self._forward_single_feature(batch)

        # Unpack
        query_features = batch["query_features"]
        query_mask = batch["query_mask"]
        query_coords = batch["query_coords"]

        # Feature filling
        query_features_filled = fill_feature_tensor(features=query_features, mask=query_mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)

        # Optional feature encoding
        encoded_query_features = self.encode_features(features=query_features_filled, mask=query_mask)

        # Optional coordinate encoding
        encoded_query_coords = self.encode_coordinates(coords=query_coords, features=query_features_filled, mask=query_mask)

        # Relative position encoding
        query_rel_pos_embed = self.positional_encoding(torch.zeros_like(query_coords))  # Embed relative positions

        # Build query token
        query_token = self.construct_token(
            encoded_coords=encoded_query_coords,
            encoded_features=encoded_query_features,
            mask=query_mask,
            rel_positions=query_rel_pos_embed,
        ).unsqueeze(1)  # [B, 1, input_dim]

        # Project query token
        q = self.input_projector(query_token)  # [B, 1, d_model]
        q_dict = {}

        # Iterate over scopes
        scopes = get_scopes(cfg=self.cfg)
        for scope in scopes:
            data = batch[scope]
            n_feat = data["features"]
            n_mask = data["mask"]
            n_coords = data["coords"]
            rel_pos = self.positional_encoding(data["rel_positions"])

            # Filling features
            n_features_filled = fill_feature_tensor(features=n_feat, mask=n_mask, fill_strategy=self.cfg.fill_strategy, mean_values=self.global_means)

            # Optional feature encoding
            encoded_n_features = self.encode_features(features=n_features_filled, mask=n_mask)

            # Optional coordinate encoding
            encoded_n_coords = self.encode_coordinates(coords=n_coords, features=n_features_filled, mask=n_mask)

            # Construct token
            n_tokens = self.construct_token(
                encoded_coords=encoded_n_coords,
                encoded_features=encoded_n_features,
                mask=n_mask,
                rel_positions=rel_pos,
            )  # [B, K, input_dim]

            # Attention
            if self.cfg.attention_type == "mha":
                n = self.input_projector(n_tokens)

                for layer in self.mha_layers:
                    q = layer(q, n)
                encoded = q

            elif self.cfg.attention_type == "transformer_encoder":
                sequence = torch.cat([query_token, n_tokens], dim=1)
                sequence = self.input_projector(sequence)  # [B, K, d_model]
                encoded = self.encoder_transformer(sequence)

            elif self.cfg.attention_type == "autoencoder":
                n = self.input_projector(n_tokens)  # [B, K, d_model]

                # Encode query
                n = self.encoder_transformer(n)

                # Decode query conditioned on memory (neighbours)
                encoded = self.decoder_transformer(tgt=q, memory=n)

            elif self.cfg.attention_type == "mha_decoder":
                n = self.input_projector(n_tokens)

                for layer in self.mha_layers:
                    q = layer(q, n)
                encoded = self.decoder_transformer(tgt=q, memory=n)

            elif self.cfg.attention_type == "space_time_attention" or self.cfg.attention_type == "time_space_attention":
                n = self.input_projector(n_tokens)  # [B, K, d_model]

                if scope == "time":
                    layers = self.time_layers
                elif scope == "space":
                    layers = self.space_layers
                else:
                    raise ValueError(f"Unknown attention scope {scope}")

                for layer in layers:
                    q = layer(q, n)
                encoded = q

            elif self.cfg.attention_type == "space_time_depth_attention":
                n = self.input_projector(n_tokens)  # [B, K, d_model]

                if scope == "time":
                    layers = self.time_layers
                elif scope == "space":
                    layers = self.space_layers
                elif scope == "depth":
                    layers = self.depth_layers
                else:
                    raise ValueError(f"Unknown attention scope {scope}")

                for layer in layers:
                    q = layer(q, n)
                encoded = q

            elif self.cfg.attention_type == "weighted_space_time_depth_attention":
                n = self.input_projector(n_tokens)  # [B, K, d_model]

                if scope == "time":
                    layers = self.time_layers
                elif scope == "space":
                    layers = self.space_layers
                elif scope == "depth":
                    layers = self.depth_layers
                else:
                    raise ValueError(f"Unknown attention scope {scope}")

                q_temp = 0.0
                for layer in layers:
                    q_temp = layer(q, n)
                q_dict[scope] = q_temp

            else:
                raise ValueError(f"MaSTNeT: Unknown attention type: {self.cfg.attention_type}")

        if self.cfg.attention_type == "weighted_space_time_depth_attention":
            # print(q_dict["time"].shape, q_dict["space"].shape, q_dict["depth"].shape)
            h_shared = torch.cat([q_dict["time"], q_dict["space"], q_dict["depth"]], dim=1)

            # Pool across branches
            h_pooled = h_shared.mean(dim=1)  # [B, d_model]

            # Get weights
            w = torch.softmax(self.gate(h_pooled), dim=-1)  # [B, 3]
            w = w.unsqueeze(-1)  # [B, 3, 1]

            # Compute encoded q
            encoded = w[:, 0:1] * q_dict["time"] + w[:, 1:2] * q_dict["space"] + w[:, 2:3] * q_dict["depth"]

        # Add global context to each node
        if self.global_context is not None and self.gate_mlp is not None:
            h_global = encoded.mean(dim=1, keepdim=True)
            g = self.global_context(h_global)

            # Gating
            alpha = self.gate_mlp(torch.cat([encoded, g], dim=-1))  # [B,T,1]
            encoded = encoded + alpha * (g - encoded)

            # alpha = self.gate_mlp(encoded)
            # encoded = encoded + alpha * g
            # # encoded = encoded + g

        # Take the query token output only for reconstruction (not neighbours)
        query_encoded = encoded[:, 0, :]  # [B, d_model]

        # Decode
        pmean = self.mean_decoder(query_encoded)
        pvar = self.var_decoder(query_encoded)  # [B, F]

        return pmean, pvar
