import torch
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from oceanmae.losses import MaskedMSELoss, BaseLoss
from oceanmae.dataset import random_feature_mask


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

        return total_loss / max(1, n_samples)

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


    def fit(self, train_loader:DataLoader, val_loader: DataLoader, max_epochs, early_stopping=None, mc_dropout: bool = False, mask_ratio: float = 0.5):
        history = {"train": {}, "val": {}}

        # Iterate over epochs
        for epoch in range(max_epochs):
            # Train and compute losses
            train_loss = self.train_one_epoch(train_loader, mask_ratio=mask_ratio)
            val_loss = self.evaluate(loader=val_loader, mc_dropout=mc_dropout)

            # Update best model
            self.update_best_model(val_loss=val_loss)

            history["train"][epoch] = train_loss
            history["val"][epoch] = val_loss
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

            # Early stopping
            if early_stopping and early_stopping(val_loss):
                print(f"Early stopping at epoch {early_stopping.epoch}, best_epoch={early_stopping.best_epoch}")
                break

        # Load best model
        self.load_best_model()

        return history
