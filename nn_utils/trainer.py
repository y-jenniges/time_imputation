import logging
from abc import ABC, abstractmethod
from typing import Union, Callable
import torch
import numpy as np
import copy
from time import time

from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_utils.graph import GraphProvider
from nn_utils.losses import MaskedMSELoss, BaseLoss
from nn_utils.dataset import random_feature_mask
from utils.metrics import compute_metrics

import config


class ModelAdapter(ABC):
    @abstractmethod
    def batch_size(self, batch):
        pass

    @abstractmethod
    def prepare_batch(self, batch, device):
        pass

    @abstractmethod
    def make_masks(self, batch, mask_ratio, mode="train", device:torch.device = torch.device("cpu")):
        pass

    @abstractmethod
    def forward(self, model, batch, masks):
        pass

    @abstractmethod
    def loss_inputs(self, batch, outputs, masks):
        pass

    @abstractmethod
    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        pass


class NeighbourAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["query_features"].shape[0]

    def prepare_batch(self, batch, device):
        return self.to_device(batch, device)

    def to_device(self, x, device):
        if isinstance(x, dict):
            return {k: self.to_device(v, device) for k, v in x.items()}
        return x.to(device)

    def make_masks(self, batch, mask_ratio=0.0, mode="train", device=torch.device("cpu")):
        q_mask = batch["query_mask"]

        scopes = [scope for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]]
        masks = {}

        # Query masking
        if mode in ["train", "eval"] and mask_ratio > 0:
            q_random_mask, _ = random_feature_mask(
                batch_size=q_mask.shape[0],
                feature_dim=q_mask.shape[1],
                mask_ratio=mask_ratio,
                device=device,
                mask_query=True,
                mask_neighbours=False
            )

            masks["q_input_mask"] = q_mask & ~q_random_mask  # Observed inputs after random masking
            masks["q_loss_mask"] = q_mask & q_random_mask  # Positions to reconstruct: Originally observed but randomly hidden
        elif mode == "reconstruct":
            # Reconstruct all missing features, feed all observed
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = torch.zeros_like(q_mask, dtype=torch.bool)
        else:
            masks["q_input_mask"] = q_mask
            masks["q_loss_mask"] = q_mask

        # Neighbour masking (per scope)
        masks["neighbour_input"] = {}
        masks["neighbour_loss"] = {}
        for scope in scopes:
            n_mask = batch[scope]["mask"]

            if mode in ["train", "eval"] and mask_ratio > 0:
                _, n_random_mask = random_feature_mask(
                    batch_size=q_mask.shape[0],
                    feature_dim=q_mask.shape[1],
                    mask_ratio=mask_ratio,
                    n_neighbours=n_mask.shape[1],
                    device=device,
                    mask_query=False,
                    mask_neighbours=True
                )
                masks["neighbour_input"][scope] = n_mask & ~n_random_mask
                masks["neighbour_loss"][scope] = n_mask & n_random_mask

            elif mode == "reconstruct":
                masks["neighbour_input"][scope] = n_mask
                masks["neighbour_loss"][scope] = torch.zeros_like(n_mask, dtype=torch.bool)
            else:
                masks["neighbour_input"][scope] = n_mask
                masks["neighbour_loss"][scope] = n_mask

        return masks

    def forward(self, model, batch, masks):
        batch["query_mask"] = masks["q_input_mask"]

        for scope, m in masks["neighbour_input"].items():
            batch[scope]["mask"] = m

        return model(batch)

    def loss_inputs(self, batch, outputs, masks):
        pred_mean, pred_var = outputs

        return dict(
            input=pred_mean,
            target=batch["query_features"],
            mask=masks["q_loss_mask"],
            pred_var=pred_var,

            # Structured neighbour info
            neighbours={
                scope: {
                    "features": batch[scope]["features"],
                    "coords": batch[scope]["coords"],
                    "mask": masks["neighbour_loss"][scope]
                }
                for scope in batch.keys() if scope not in ["query_features", "query_mask", "query_coords"]
            }
        )

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        pred_mean, _ = outputs

        q_feat = batch["query_features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean


class PointwiseAdapter(ModelAdapter):
    def batch_size(self, batch):
        return batch["features"].shape[0]

    def prepare_batch(self, batch, device):
        return {k: v.to(device) for k, v in batch.items()}

    def make_masks(self, batch, mask_ratio, mode="train", device=torch.device("cpu")):
        feat = batch["features"]
        mask = batch["mask"]
        batch_size, n_features = feat.shape

        if mode in ["train", "eval"] and mask_ratio > 0:
            # random_feature_mask: True means "mask/hide this feature"
            random_mask, _ = (
                random_feature_mask(batch_size=batch_size, feature_dim=n_features, mask_ratio=mask_ratio,
                                    n_neighbours=0, device=device, mask_query=True, mask_neighbours=False))

            # Observed inputs after random masking
            input_mask = mask & ~random_mask

            # Positions to reconstruct: originally observed but randomly hidden
            loss_mask = mask & random_mask

        elif mode == "reconstruct":
            # Reconstruct all missing features
            input_mask = mask
            loss_mask = torch.zeros_like(mask, dtype=torch.bool)

        else:
            input_mask = mask
            loss_mask = mask

        return dict(input_mask=input_mask, loss_mask=loss_mask)

    def forward(self, model, batch, masks):
        feat = batch["features"]
        mask = masks["input_mask"]
        coords = batch["coords"]

        feat_filled = torch.where(mask, feat, torch.zeros_like(feat))
        x = torch.cat([coords, feat_filled, mask.float()], dim=-1)
        return model(x)

    def loss_inputs(self, batch, outputs, masks):
        if len(outputs) == 2:
            pred_mean, pred_var = outputs
        else:
            pred_mean = outputs
            pred_var = None
        return dict(input=pred_mean, target=batch["features"], mask=masks["loss_mask"], pred_var=pred_var)

    def outputs_to_cpu(self, batch, outputs, to_numpy: bool = True):
        if len(outputs) == 2:
            pred_mean, _ = outputs
        else:
            pred_mean = outputs

        q_feat = batch["features"].detach().cpu()
        pred_mean = pred_mean.detach().cpu()

        if to_numpy:
            return q_feat.numpy(), pred_mean.numpy()
        else:
            return q_feat, pred_mean


class Trainer:
    """ Trainer class for OceanMAE, which handles training and evaluation loops."""
    def __init__(self, model: torch.nn.Module, adapter: ModelAdapter, optimizer: torch.optim.Optimizer,
                 full_coords: torch.Tensor, full_values: torch.Tensor, full_mask: torch.Tensor,
                 graph_provider: GraphProvider = None,
                 device: torch.cuda.device = "cpu",
                 loss_fn: BaseLoss = MaskedMSELoss(),
                 global_means: torch.Tensor = None,
                 cfg=None):
        """
        Args:
            model: MAE model
            optimizer: Optimizer for training
            device:  'cpu' or 'cuda' device
            loss_fn: Loss function class (default: MaskedMSELoss)
        """
        self.model = model
        self.adapter = adapter
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

        # Graph logic
        self.graph_provider = graph_provider
        self.freeze_graph = False

        # Early stopping
        self.best_model_state = None
        self.best_val_loss = np.inf

        # Full data
        self.full_coords = full_coords.to(device)
        self.full_values = full_values.to(device)
        self.full_mask = full_mask.to(device)

        self.cfg = cfg
        self.global_means = global_means

        # Init summary writer
        self.writer = SummaryWriter(log_dir=f"/tmp/{self.model.__class__.__name__}_{int(time())}")

    def train_one_epoch(self, loader: DataLoader, mask_ratio: float = 0.5) -> float:
        self.model.train()
        total_loss, n_samples = 0.0, 0

        for batch in tqdm(loader, desc="Train", leave=False):
            batch = self.adapter.prepare_batch(batch=batch, device=self.device)

            masks = self.adapter.make_masks(batch=batch, mask_ratio=mask_ratio, mode="train", device=self.device)

            # Predict
            outputs = self.adapter.forward(self.model, batch=batch, masks=masks)

            # Compute loss
            loss_input = self.adapter.loss_inputs(batch=batch, outputs=outputs, masks=masks)
            loss = self.loss_fn(**loss_input)
            if loss is None:
                continue

            # Optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = self.adapter.batch_size(batch=batch)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(1, n_samples)

    def _enable_dropout(self, enable=True):
        if enable:
            self.model.train()
        else:
            self.model.eval()

        # # Enable dropout
        # for module in self.model.modules():
        #     if isinstance(module, nn.Dropout):
        #         module.train() if enable else module.eval()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mask_ratio: float, metrics_key: str = "Metrics", full_metrics: bool = True):
        self.model.eval()  # Disables dropout
        total_loss, n_samples = 0.0, 0
        all_true, all_pred = [], []

        for batch in tqdm(loader, desc="Val", leave=False):
            batch = self.adapter.prepare_batch(batch, self.device)
            masks = self.adapter.make_masks(batch=batch, mask_ratio=mask_ratio, mode="eval", device=self.device)

            # Predict
            outputs = self.adapter.forward(self.model, batch=batch, masks=masks)

            # Compute loss
            loss_inputs = self.adapter.loss_inputs(batch=batch, outputs=outputs, masks=masks)
            loss = self.loss_fn(**loss_inputs)
            if loss is None:
                continue

            batch_size = self.adapter.batch_size(batch=batch)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            # Collect flattened tensors for metric calculation (only on loss mask)
            if full_metrics:
                y_true, y_pred = self.adapter.outputs_to_cpu(batch, outputs, to_numpy=True)
                loss_mask = loss_inputs["mask"].detach().cpu().numpy()

                y_true_masked = y_true.copy()
                y_pred_masked = y_pred.copy()
                y_true_masked[~loss_mask] = np.nan
                y_pred_masked[~loss_mask] = np.nan

                all_true.append(y_true_masked)
                all_pred.append(y_pred_masked)

        if full_metrics:
            if len(all_true) == 0:
                print("Warning: No valid entries to evaluate!")
                return total_loss / max(1, n_samples), {}

            # Stack for full array metrics
            y_true = np.concatenate(all_true, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)

            # Compute and log metrics
            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, var_names=config.parameters)
            self.log_metrics(metrics_key, metrics)
        else:
            metrics = {}

        return total_loss / max(1, n_samples), metrics

    def update_graph(self):
        if self.graph_provider is None:
            return

        if self.freeze_graph:
            return

        logging.info("Updating graph...")
        st = time()
        self.model.eval()

        self.graph_provider.update(encoder=self.model.coord_encoder, coords=self.full_coords, values=self.full_values, mask=self.full_mask, mean_values=self.global_means)
        logging.info("Updated graph in %.0f minutes." % ((time() - st) / 60 ))

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
            self.best_model_state = { k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_best_model(self):
        """ Restore the best model state. """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    @torch.no_grad()
    def reconstruct_full_dataset(self, loader: DataLoader, do_dropout: bool = False, show_progress: bool = True):
        self.model.eval()
        all_preds = []
        all_vars = []

        self._enable_dropout(enable=do_dropout)

        # Define iterator
        iterator = loader
        if show_progress:
            iterator = tqdm(loader, desc="Val", leave=False)

        # Iterate over batches
        for batch in iterator:
            batch = self.adapter.prepare_batch(batch, self.device)

            # In reconstruction, everything becomes input (truly missing features are predicted)
            masks = self.adapter.make_masks(batch, mask_ratio=0.0, mode="reconstruct", device=self.device)

            # Predict all nan-values (true in feature mask means observed)
            outputs = self.adapter.forward(self.model, batch=batch, masks=masks)

            if len(outputs) == 2:
                pred_mean, pred_var = outputs
                pred_mean = pred_mean.detach().cpu()
                pred_var = pred_var.detach().cpu()

                all_vars.append(pred_var)
            else:
                pred_mean = outputs

            # Add predictions to the list
            all_preds.append(pred_mean)

        return torch.cat(all_preds), torch.cat(all_vars) if all_vars else None


    def fit(self,
            train_loader:DataLoader,
            val_loader: DataLoader,
            max_epochs,
            early_stopping=None,
            mask_ratio: Union[float, Callable[[int], float]] = 0.5,
            optuna_callback=None,
            full_metrics: bool = False):
        history = {"train": {}, "val": {}, "metrics": {}}

        # Iterate over epochs
        for epoch in range(max_epochs):
            # Check if mask ratio is a function or a fixed float and determine mask ratio to use
            if callable(mask_ratio):
                current_mask_ratio = mask_ratio(epoch)
            else:
                current_mask_ratio = mask_ratio

            # Update graph if specified
            if self.graph_provider is not None and self.cfg.graph_mode == "dynamic":
                # Freezing
                if epoch == 20:
                    self.freeze_graph = True  # Freeze graph

                    # Freeze coord_encoder (if scope is limited to graph)
                    if self.cfg.encoder_scope == "graph":
                        for p in self.model.coord_encoder.parameters():
                            p.requires_grad = False
                        logging.info("Coord encoder frozen at epoch 20")

                    logging.info("Graph frozen at epoch 20")
                if(not self.freeze_graph) and (epoch % self.graph_provider.update_every == 0):
                    self.update_graph()

            # Train and compute losses
            train_loss = self.train_one_epoch(train_loader, mask_ratio=current_mask_ratio)
            val_loss, val_metrics = self.evaluate(loader=val_loader, mask_ratio=mask_ratio, metrics_key=f"Epoch_{epoch}", full_metrics=full_metrics)

            # Update best model
            self.update_best_model(val_loss=val_loss)

            history["train"][epoch] = train_loss
            history["val"][epoch] = val_loss
            history["metrics"][epoch] = val_metrics
            tqdm.write(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")

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
