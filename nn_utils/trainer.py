from typing import Union, Callable
import torch
import numpy as np
import copy
from time import time

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_utils.losses import MaskedMSELoss, BaseLoss
from nn_utils.dataset import random_feature_mask
from utils.metrics import compute_metrics

import config


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

        for batch in tqdm(loader, desc="Train", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            q_feat = batch["query_features"]  # .to(self.device)
            q_mask = batch["query_mask"]  #.to(self.device)
            q_coords = batch["query_coords"]  # .to(self.device)
            n_feat = batch["neighbour_features"]  # .to(self.device)
            n_mask = batch["neighbour_mask"]  # .to(self.device)
            rel_pos = batch["rel_positions"]  # .to(self.device)

            batch_size, n_features = q_feat.shape
            n_neighbours = n_feat.shape[1]

            # random_feature_mask: True means "mask this feature"
            if mask_ratio > 0:
                q_random_mask, n_random_mask = (
                    random_feature_mask(batch_size=batch_size, feature_dim=n_features, mask_ratio=mask_ratio,
                                        n_neighbours=n_neighbours, device=self.device, mask_query=True,
                                        mask_neighbours=True))

                # Observed inputs after random masking
                q_input_mask = q_mask & ~q_random_mask
                n_input_mask = n_mask & ~n_random_mask

                # Positions to reconstruct: originally observed but randomly hidden
                q_loss_mask = q_mask & q_random_mask
                n_loss_mask = n_mask & n_random_mask
            else:
                # If no random masking, all observed features are inputs and targets, i.e. reconstruct everything
                q_input_mask, q_loss_mask = q_mask, q_mask
                n_input_mask, n_loss_mask = n_mask, n_mask

            # Predict
            pred_mean, pred_var = self.model(query_features=q_feat, query_mask=q_input_mask, query_coords=q_coords,
                                             neighbour_features=n_feat, neighbour_mask=n_input_mask, rel_positions=rel_pos)

            # Compute loss
            loss = self.loss_fn(input=pred_mean, target=q_feat, mask=q_loss_mask, pred_var=pred_var)
            if loss is None:
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(1, n_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, do_dropout: bool = False):
        self.model.eval()  # Disables dropout
        total_loss = 0.0
        n_samples = 0
        all_true, all_pred = [], []

        for batch in tqdm(loader, desc="Val", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            q_feat = batch["query_features"]#.to(self.device)
            q_mask = batch["query_mask"]#.to(self.device)
            q_coords = batch["query_coords"]#.to(self.device)
            n_feat = batch["neighbour_features"]#.to(self.device)
            n_mask = batch["neighbour_mask"]#.to(self.device)
            rel_pos = batch["rel_positions"]#.to(self.device)

            batch_size, n_features = q_feat.shape

            # Enable dropout
            if do_dropout:
                for module in self.model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()

            # Predict
            pred_mean, pred_var = self.model(query_features=q_feat, query_mask=q_mask, query_coords=q_coords,
                                             neighbour_features=n_feat, neighbour_mask=n_mask, rel_positions=rel_pos)

            # Compute loss (reconstruct originally missing features)
            loss = self.loss_fn(input=pred_mean, target=q_feat, mask=~q_mask, pred_var=pred_var)
            if loss is None:
                continue

            total_loss += loss.item() * batch_size
            n_samples += batch_size

            # Collect flattened tensors for metric calculation
            all_true.append(q_feat.detach().cpu().numpy())
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
    def reconstruct_full_dataset(self, loader: DataLoader, do_dropout: bool = False):
        self.model.eval()
        all_preds = []
        all_vars = []

        # Iterate over batches
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            q_feat = batch["query_features"]#.#to(self.device)
            q_mask = batch["query_mask"]#.to(self.device)
            q_coords = batch["query_coords"]#.to(self.device)
            n_feat = batch["neighbour_features"]#.to(self.device)
            n_mask = batch["neighbour_mask"]#.to(self.device)
            rel_pos = batch["rel_positions"]#.to(self.device)

            batch_size, n_features = q_feat.shape

            # Enable dropout
            if do_dropout:
                for module in self.model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()

            # Predict all nan-values (true in feature mask means observed)
            pred_mean, pred_var = self.model(query_features=q_feat, query_mask=q_mask, query_coords=q_coords,
                                             neighbour_features=n_feat, neighbour_mask=n_mask, rel_positions=rel_pos)

            # Add predictions to the list
            all_preds.append(pred_mean.detach().cpu())
            all_vars.append(pred_var.detach().cpu())

        return torch.cat(all_preds), torch.cat(all_vars)


    def fit(self, train_loader:DataLoader, val_loader: DataLoader, max_epochs, early_stopping=None, do_dropout: bool = False, mask_ratio: Union[float, Callable[[int], float]] = 0.5, optuna_callback=None):
        history = {"train": {}, "val": {}, "metrics": {}}

        # Iterate over epochs
        for epoch in range(max_epochs):
            # Check if mask ratio is a function or a fixed float and determine mask ratio to use
            if callable(mask_ratio):
                current_mask_ratio = mask_ratio(epoch)
            else:
                current_mask_ratio = mask_ratio

            # Train and compute losses
            train_loss = self.train_one_epoch(train_loader, mask_ratio=current_mask_ratio)
            val_loss, val_metrics = self.evaluate(loader=val_loader, do_dropout=do_dropout)

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

        # Close writer
        self.writer.close()

        return history
