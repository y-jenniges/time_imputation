import torch
import torch.nn as nn


class CoordEncoder(nn.Module):
    def __init__(self, cfg, hidden_dim=32, coord_dim=5, value_dim=6, time_dim=1, output_dim=None):
        super().__init__()
        self.cfg = cfg

        if output_dim is None:
            output_dim = hidden_dim

        # Spatio-temporal encoding
        self.spatial = nn.Sequential(
            nn.Linear(coord_dim, 64),  # lat, lon, depth
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

        self.time = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),  # Time
            nn.GELU()
        )

        # Optional additional feature / feature+mask encoding
        if self.cfg.encoder_input == "coords_feat":
            self.feature_state = nn.Sequential(
                nn.Linear(value_dim, 64),
                nn.GELU(),
                nn.Linear(64, hidden_dim)
            )

            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 3, 64),
                nn.GELU(),
                nn.Linear(64, output_dim)
            )
        elif self.cfg.encoder_input == "coords_feat_mask":
            self.feature_state = nn.Sequential(
                nn.Linear(value_dim * 2, 64),  # feature + mask
                nn.GELU(),
                nn.Linear(64, hidden_dim)
            )

            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 3, 64),
                nn.GELU(),
                nn.Linear(64, output_dim)
            )
        else:
            self.feature_state = None

            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, output_dim),
                nn.GELU(),
            )

    def forward(self, coords, times, values, mask):
        h_s = self.spatial(coords)
        h_t = self.time(times)

        if self.cfg.encoder_input == "coords":
            return self.fuse(torch.cat([h_s, h_t], dim=-1))
        elif self.cfg.encoder_input == "coords_feat" and self.feature_state is not None:
            h_v = self.feature_state(values)
            return self.fuse(torch.cat([h_s, h_t, h_v], dim=-1))
        elif self.cfg.encoder_input == "coords_feat_mask" and self.feature_state is not None:
            h_v = self.feature_state(torch.cat([values, mask], dim=-1))
            return self.fuse(torch.cat([h_s, h_t, h_v], dim=-1))
        else:
            if self.feature_state is None:
                raise ValueError(f"CoordEncoder: Feature state is None, but required for {self.cfg.encoder_input}")
            else:
                raise ValueError("Unknown encoder input: %s" % self.cfg.encoder_input)


"""
Reference: H. Zhao, L. Jiang, J. Jia, P. Torr and V. Koltun, "Point Transformer," 2021 IEEE/CVF International 
Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 16239-16248, doi: 10.1109/ICCV48922.2021.01595. 
"""


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=64):
        super().__init__()

        self.pos_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rel_pos):
        return self.pos_embed(rel_pos)



