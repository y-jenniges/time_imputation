import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = Block(in_channels, base_ch)
        self.enc2 = Block(base_ch, base_ch*2)
        self.enc3 = Block(base_ch*2, base_ch*4)

        # Bottleneck
        self.bottleneck = Block(base_ch*4, base_ch*8)

        # Decoder
        self.dec3 = Block(base_ch*8 + base_ch*4, base_ch*4)
        self.dec2 = Block(base_ch*4 + base_ch*2, base_ch*2)
        self.dec1 = Block(base_ch*2 + base_ch, base_ch)

        # Output layer
        self.out = nn.Linear(base_ch, out_channels)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=-1))
        d2 = self.dec2(torch.cat([d3, e2], dim=-1))
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))

        out = self.out(d1)
        return out


class OceanUNet(nn.Module):
    def __init__(self, coord_dim=5, value_dim=6, include_mask=True):
        """
        Wrapper for point-wise UNet compatible with Trainer.
        """
        super().__init__()
        print("Init OceanPointWiseUNet")
        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.include_mask = include_mask

        in_channels = coord_dim + value_dim + (value_dim if include_mask else 0)
        out_channels = value_dim

        # Define point-wise UNet (MLP blocks)
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

        # Variance head (optional)
        self.var_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Softplus()
        )

    def forward(self, coords, values, feature_mask, mc_dropout=False):
        # Fill missing values
        values_filled = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        # Combine inputs: [B, N, C_total]
        x = torch.cat([coords, values_filled, feature_mask.float()], dim=-1)

        # Forward through point-wise UNet
        pmean = torch.sigmoid(self.unet(x))  # [B, N, value_dim]
        pvar = torch.exp(self.var_head(pmean))  # [B, N, value_dim]

        return pmean, pvar
