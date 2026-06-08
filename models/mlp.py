"""MLP baseline model for pointwise ocean data imputation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nn_utils.dataset import PointwiseDataset
from nn_utils.model_adapters import PointwiseAdapter


class MLP(nn.Module):
    """
    Masked multi-layer perceptron for pointwise imputation of missing ocean values.

    Outputs a mean and log-variance for each feature, supporting heteroscedastic
    uncertainty quantification.

    Parameters
    ----------
    input_dim : int, default=17
        Dimensionality of the input (coordinate dim + feature dim).
    hidden_dims : list[int] or None, default=None
        Hidden layer sizes; defaults to ``[128, 64]``.
    output_dim : int, default=6
        Number of output features.
    dropout : float, default=0.1
        Dropout rate applied after each hidden layer.
    """

    def __init__(self, input_dim=17, hidden_dims=None, output_dim=6, dropout=0.1):
        super().__init__()

        # Define the network layers
        if hidden_dims is None:
            hidden_dims = [128, 64]

        last_dim = input_dim
        layers = []
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

        # Dummy weights
        self.anisotropic_weights = None

    def forward(self, x):
        h = self.shared_nn(x)
        pmean = self.mean_head(h)
        pvar = self.var_head(h)
        return pmean, pvar

    def predict(self, x, y, batch_size, device: torch.device = torch.device("cpu")):
        """
        Impute the full dataset in batches.

        Parameters
        ----------
        x : Tensor
            Preprocessed coordinate matrix.
        y : Tensor
            Preprocessed feature matrix (NaN at missing positions).
        batch_size : int
        device : torch.device, default=cpu

        Returns
        -------
        y_hat : ndarray, shape (N, F)
            Predicted means.
        y_var : ndarray, shape (N, F)
            Predicted variances (exponentiated from log-variance head).
        """
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
                masks, _ = adapter.make_masks(batch=batch, mask_ratio=0.0, mode="reconstruct", device=device)
                pmean, pvar = adapter.forward(self, batch=batch, masks=masks)

                y_hat.append(pmean.cpu())
                y_var.append(pvar.cpu())

        return torch.cat(y_hat, dim=0).cpu().numpy(), torch.exp(torch.cat(y_var, dim=0)).cpu().numpy()
