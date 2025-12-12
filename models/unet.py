import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, num_blocks=3, dropout=0.1):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        self.encoder_channels = []
        in_ch = in_channels
        for i in range(num_blocks):
            out_ch = base_channels * 2**i
            self.encoders.append(Block(in_ch=in_ch, out_ch=out_ch, dropout=dropout))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = Block(in_ch=in_ch, out_ch=in_ch*2, dropout=dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = in_ch * 2
        for skip_ch in reversed(self.encoder_channels):
            self.decoders.append(Block(in_ch=in_ch + skip_ch, out_ch=skip_ch, dropout=dropout))
            in_ch = skip_ch

        # Output layer
        self.out = nn.Linear(in_ch, out_channels)

    def forward(self, x):
        # Encoder outputs (for skips)
        skips = []

        # Encoder
        h = x
        for eblock in self.encoders:
            h = eblock(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for dblock, skip in zip(self.decoders, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = dblock(h)

        return self.out(h)


class OceanUNet(nn.Module):
    def __init__(self, coord_dim=5, value_dim=6, include_mask=True, base_channels=64, num_blocks=3, dropout=0.0):
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
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels,
                         num_blocks=num_blocks, dropout=dropout)

        # Variance head (optional)
        self.var_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Softplus()
        )

    def forward(self, coords, values, feature_mask, mc_dropout=False):
        # Fill missing values
        values_filled = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        # Construct input
        x = torch.cat([coords, values_filled, feature_mask.float()], dim=-1)

        # Forward through point-wise UNet
        pmean = torch.sigmoid(self.unet(x))  # [B, N, value_dim]
        pvar = torch.exp(self.var_head(pmean))  # [B, N, value_dim]  # @todo exp or not?

        return pmean, pvar
