import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=[128, 64], output_dim=6, dropout=0.1):
        super().__init__()

        layers = []

        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h

        self.shared_nn = nn.Sequential(*layers)

        # Mean head (for reconstruction)
        self.mean_head = nn.Sequential(
            nn.Linear(last_dim, output_dim),
            nn.Sigmoid()
        )

        # Variance head (for heteroscedastic uncertainty)
        self.var_head= nn.Sequential(
            nn.Linear(last_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        h = self.shared_nn(x)
        pmean = self.mean_head(h)
        pvar = self.var_head(h)
        return pmean, pvar
