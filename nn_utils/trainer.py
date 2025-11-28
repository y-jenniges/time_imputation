import torch
import numpy as np
import copy
from time import time
from scipy import stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_utils.losses import MaskedMSELoss, BaseLoss
from nn_utils.dataset import random_feature_mask

import config


def taylor_skill(y_true, y_pred, coords=None, per_variable=True):
    """
    Compute Taylor Skill Score for oceanographic variables.

    Args:
        y_true: (N,D) array of observed values
        y_pred: (N,D) array of predicted values
        coords: (N,4) array of [lat, lon, depth, time], optional
        per_variable: if True, return score per variable; else global

    Returns:
        dict of Taylor Skill scores (per variable or global)
    """
    assert y_true.shape == y_pred.shape
    n_vars = y_true.shape[1]

    ts_scores = {}

    for i in range(n_vars):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) == 0:
            ts_scores[i] = np.nan
            continue

        yt = yt[mask]
        yp = yp[mask]

        # Standard deviations
        std_true = np.std(yt)
        std_pred = np.std(yp)

        # Correlation
        r = np.corrcoef(yt, yp)[0,1]

        # Taylor Skill (original formulation)
        ts = (4 * (1 + r)**2) / ((std_pred/std_true + 1)**2)

        ts_scores[i] = ts

    if not per_variable:
        # Weighted global score (mean over variables)
        ts_scores = {"Global": np.nanmean(list(ts_scores.values()))}

    return ts_scores


def compute_metrics(y_true, y_pred, var_names=None, coords=None):
    """
    Compute global and per-variable metrics for model evaluation.
    Args:
        y_true: numpy array of shape (N, D)
        y_pred: numpy array of shape (N, D)
        var_names: list of variable names (optional)
    Returns:
        metrics: dict with global metrics and dicts of per-variable metrics
    """
    # Ensure same shape
    # assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"

    n_vars = y_true.shape[1]
    if var_names is None:
        var_names = [f"var_{i}" for i in range(n_vars)]

    # Initialize containers
    var_metrics = {}
    global_valid_true = []
    global_valid_pred = []

    for i, name in enumerate(var_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mask = ~np.isnan(yt) & ~np.isnan(yp)
        if np.sum(mask) == 0:
            var_metrics[name] = {"RMSE": np.nan, "NSE": np.nan, "Pearson": np.nan}
            continue

        yt = yt[mask]
        yp = yp[mask]

        mse = np.mean((yt - yp) ** 2)
        rmse = np.sqrt(mse)
        nse = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
        r, p = stats.pearsonr(yt, yp)
        #  ts = taylor_skill(yt, yp, coords=coords) if coords is not None else np.nan

        var_metrics[name] = {"RMSE": rmse, "NSE": nse, "Pearson": r, "Pearson_p": p}  # , "Taylor_skill": ts}
        global_valid_true.append(yt)
        global_valid_pred.append(yp)

    # Combine for global metrics
    global_valid_true = np.concatenate(global_valid_true)
    global_valid_pred = np.concatenate(global_valid_pred)

    mse_global = np.mean((global_valid_true - global_valid_pred) ** 2)
    rmse_global = np.sqrt(mse_global)
    nse_global = 1 - np.sum((global_valid_true - global_valid_pred) ** 2) / np.sum(
        (global_valid_true - np.mean(global_valid_true)) ** 2
    )
    r_global, p_global = stats.pearsonr(global_valid_true, global_valid_pred)

    metrics = {
        "Global": {"RMSE": rmse_global, "NSE": nse_global, "Pearson": r_global, "Pearson_p": p_global},
        "PerVariable": var_metrics
    }
    return metrics


class Trainer:
    """ Trainer class for OceanMAE, which handles training and evaluation loops."""
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.cuda.device = "cpu",
                 loss_fn: BaseLoss = MaskedMSELoss()):
        """
        Args:
            model: MAE model
            optimizer: Optimizer for training
            device:  'cpu' or 'cuda' device
            loss_fn: Loss function class (default: MaskedMSELoss)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

        # Early stopping
        self.best_model_state = None
        self.best_val_loss = np.inf

        # Init summary writer
        self.writer = SummaryWriter(log_dir=f"/tmp/{self.model.__class__.__name__}_{int(time())}")

    def train_one_epoch(self, loader: DataLoader, mask_ratio: float = 0.5) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for coords, values, feature_mask in tqdm(loader, desc="Train", leave=False):
            n_samples_in_batch = coords.shape[0]

            coords = coords.to(self.device, dtype=torch.float32)
            values = values.to(self.device, dtype=torch.float32)
            feature_mask = feature_mask.to(self.device, dtype=torch.bool)

            # random_feature_mask: True means "mask this feature"
            if mask_ratio > 0:
                rand_mask = random_feature_mask(values.shape[0], values.shape[1], mask_ratio=mask_ratio, device=self.device)

                # Observed inputs after random masking
                input_observed = feature_mask & ~rand_mask

                # Positions to reconstruct: originally observed but randomly hidden
                loss_mask = feature_mask & rand_mask
            else:
                # If no random masking, all observed features are inputs and targets, i.e. reconstruct everything
                input_observed = feature_mask
                loss_mask = feature_mask

            # Mask input values (replace hidden ones with 0)
            values_masked = values.clone()
            values_masked[~input_observed] = float("nan")

            # Predict
            pred_mean, pred_var = self.model(coords=coords, values=values_masked, feature_mask=input_observed, mc_dropout=False)

            # Compute loss
            loss = self.loss_fn(input=pred_mean, target=values, mask=loss_mask, coords=coords, pred_var=pred_var)

            if loss is None:
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * n_samples_in_batch
            n_samples += n_samples_in_batch

        return total_loss / max(1, n_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mc_dropout: bool = False):
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_true, all_pred = [], []

        for coords, values, feature_mask in tqdm(loader, desc="Val", leave=False):
            n_samples_in_batch = coords.shape[0]
            coords = coords.to(self.device, dtype=torch.float32)
            values = values.to(self.device, dtype=torch.float32)
            feature_mask = feature_mask.to(self.device, dtype=torch.bool)

            # No random mask in validation: reconstruct all masked (False) values
            loss_mask = ~feature_mask
            values_filled = torch.where(feature_mask, values, torch.zeros_like(values))
            pred_mean, pred_var = self.model(coords=coords, values=values_filled, feature_mask=feature_mask, mc_dropout=mc_dropout)

            # Compute loss
            loss = self.loss_fn(input=pred_mean, target=values, mask=loss_mask, coords=coords, pred_var=pred_var)
            if loss is None:
                continue

            total_loss += loss.item() * n_samples_in_batch
            n_samples += n_samples_in_batch

            # Collect flattened tensors for metric calculation
            all_true.append(values.detach().cpu().numpy())
            all_pred.append(pred_mean.detach().cpu().numpy())

        # Stack for full array metrics
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

        # Compute and log metrics
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, var_names=config.parameters)
        self.log_metrics("Metrics", metrics)

        return total_loss / max(1, n_samples), metrics

    def log_metrics(self, prefix, d):
        # Recursively log all scalar metrics
        for k, v in d.items():
            if isinstance(v, dict):
                self.log_metrics(f"{prefix}/{k}", v)
            elif np.isscalar(v):
                self.writer.add_scalar(f"{prefix}/{k}", v)

    def update_best_model(self, val_loss: float):
        """ Save model state if current validation loss is the best so far. """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())

    def load_best_model(self):
        """ Restore the best model state. """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    @torch.no_grad()
    def reconstruct_full_dataset(self, loader: DataLoader, mc_dropout: bool = False):
        self.model.eval()
        all_coords = []
        all_preds = []
        all_vars = []

        # Iterate over batches
        for coords, values, feature_mask in loader:
            coords = coords.to(self.device, dtype=torch.float32)
            values = values.to(self.device, dtype=torch.float32)
            feature_mask = feature_mask.to(self.device, dtype=torch.bool)

            # Predict all nan-values (true in feature mask means observed)
            pred_means, pred_vars = self.model(coords=coords, values=values, feature_mask=feature_mask, mc_dropout=mc_dropout)

            all_coords.append(coords.cpu())
            all_preds.append(pred_means.cpu())
            all_vars.append(pred_vars.cpu())

        return torch.cat(all_coords), torch.cat(all_preds), torch.cat(all_vars)


    def fit(self, train_loader:DataLoader, val_loader: DataLoader, max_epochs, early_stopping=None, mc_dropout: bool = False, mask_ratio: float = 0.5, optuna_callback=None):
        history = {"train": {}, "val": {}, "metrics": {}}

        # Iterate over epochs
        for epoch in range(max_epochs):
            # Train and compute losses
            train_loss = self.train_one_epoch(train_loader, mask_ratio=mask_ratio)
            val_loss, val_metrics = self.evaluate(loader=val_loader, mc_dropout=mc_dropout)

            # Update best model
            self.update_best_model(val_loss=val_loss)

            history["train"][epoch] = train_loss
            history["val"][epoch] = val_loss
            history["metrics"][epoch] = val_metrics
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # Optuna logging
            if optuna_callback is not None:
                optuna_callback(epoch=epoch, val_losses=list(history["val"].values()))

            # Early stopping
            if early_stopping and early_stopping(val_loss):
                print(f"Early stopping at epoch {early_stopping.epoch}, best_epoch={early_stopping.best_epoch}")
                break

        # Load best model
        self.load_best_model()

        return history
