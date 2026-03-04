import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nn_utils.dataset import PointwiseDataset
from nn_utils.trainer import PointwiseAdapter


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
        self.mean_head = nn.Linear(last_dim, output_dim)

        # Variance head (for heteroscedastic uncertainty)
        self.var_head= nn.Linear(last_dim, output_dim)

    def forward(self, x):
        h = self.shared_nn(x)
        pmean = self.mean_head(h)
        pvar = self.var_head(h)
        return pmean, pvar

    def predict(self, x, y, batch_size, device: torch.device = torch.device("cpu")):
        """ Impute missing data given scaled x (coordinates) and y (values). """
        # Define dataset and loader (to predict everything)
        dataset = PointwiseDataset(coords=x, values=y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        adapter = PointwiseAdapter()

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
