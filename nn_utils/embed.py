import torch
import torch.nn as nn


class CoordEncoder(nn.Module):
    def __init__(self, hidden_dim=32, coord_dim=5, value_dim=6, time_dim=1):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Linear(coord_dim, 64),  # lat, lon, depth
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

        self.state = nn.Sequential(
            nn.Linear(value_dim * 2, 64),  # feature + mask
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

        self.time = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),  # Time
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, coords, values, mask, times):
        h_s = self.spatial(coords)
        h_v = self.state(torch.cat([values, mask], dim=-1))
        h_t = self.time(times)
        return self.fuse(torch.cat([h_s, h_v, h_t], dim=-1))


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



