"""Trainer class orchestrating training, evaluation, graph updates, and early stopping for SONATI."""

import logging
from typing import Union, Callable, Tuple
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

import config
from nn_utils.graph import GraphProvider
from nn_utils.losses import MaskedMSELoss, BaseLoss
from nn_utils.model_adapters import ModelAdapter
from utils.metrics import compute_metrics


class Trainer:
    """
    Training loop for SONATI and related models.

    Handles forward/backward passes, graph updates (static / dynamic / frozen),
    early stopping via best-model checkpointing, and TensorBoard metric logging.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train (e.g. :class:`models.sonati.SONATI`).
    adapter : ModelAdapter
        Adapter that prepares batches, generates masks, calls forward, and
        extracts outputs for the specific model/mode combination.
    optimizer : torch.optim.Optimizer
        Optimiser (typically Adam).
    full_coords : Tensor
        Preprocessed coordinates for the entire dataset - used for graph updates
        and full-dataset reconstruction.
    full_values : Tensor
        Preprocessed feature values for the entire dataset.
    full_mask : Tensor
        Boolean observation mask for the entire dataset.
    graph_provider : GraphProvider or None, default=None
        Manages k-NN graph construction and updates. Pass None for models that
        do not use a neighbourhood graph.
    device : str or torch.device, default='cpu'
        Compute device.
    loss_fn : BaseLoss, default=MaskedMSELoss()
        Loss function instance.
    global_means : Tensor or None, default=None
        Per-feature training means; forwarded to the graph provider for
        ``fill_strategy='mean'``.
    cfg : ModelConfig or None, default=None
        Experiment configuration object.
    var_names : list[str] or None, default=None
        Variable names for metric logging; defaults to ``config.parameters``.
    """

    def __init__(self, model: torch.nn.Module, adapter: ModelAdapter, optimizer: torch.optim.Optimizer,
                 full_coords: torch.Tensor, full_values: torch.Tensor, full_mask: torch.Tensor,
                 graph_provider: GraphProvider = None,
                 device: torch.cuda.device = "cpu",
                 loss_fn: BaseLoss = MaskedMSELoss(),
                 global_means: torch.Tensor = None,
                 cfg=None,
                 var_names=None):
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
        self.var_names = var_names if var_names is not None else config.parameters

        # Init summary writer
        self.writer = SummaryWriter(
            log_dir=f"{config.output_dir_tensorboard}{self.model.__class__.__name__}_{int(time())}")

    def train_one_epoch(self, loader: DataLoader, mask_ratio: float = 0.5, verbose: bool = True) -> Tuple[float, float]:
        """
        Run one training epoch over ``loader``.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        mask_ratio : float, default=0.5
            Fraction of observed values randomly hidden per step.

        Returns
        -------
        avg_loss : float
        avg_miss_ratio : float
        """
        self.model.train()
        total_loss, total_miss_ratio, n_samples = 0.0, 0.0, 0

        for batch in tqdm(loader, desc="Train", leave=False, disable=not verbose):
            batch = self.adapter.prepare_batch(batch=batch, device=self.device)

            masks, miss_ratio = self.adapter.make_masks(batch=batch, mask_ratio=mask_ratio, mode="train",
                                                        device=self.device, cfg=self.cfg)

            # Predict
            outputs = self.adapter.forward(self.model, batch=batch, masks=masks)

            # Compute loss
            loss_input = self.adapter.loss_inputs(batch=batch, outputs=outputs, masks=masks,
                                                  anisotropic_weights=self.model.anisotropic_weights)
            loss = self.loss_fn(**loss_input)
            if loss is None:
                continue

            # Optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = self.adapter.batch_size(batch=batch)
            total_loss += loss.item() * batch_size
            total_miss_ratio += miss_ratio.item() * batch_size
            n_samples += batch_size

        return total_loss / max(1, n_samples), total_miss_ratio / max(1, n_samples)

    def _enable_dropout(self, enable=True):
        """Switch the model between train mode (dropout on) and eval mode (dropout off)."""
        if enable:
            self.model.train()
        else:
            self.model.eval()

        # # Enable dropout
        # for module in self.model.modules():
        #     if isinstance(module, nn.Dropout):
        #         module.train() if enable else module.eval()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, mask_ratio: float, metrics_key: str = "Metrics", full_metrics: bool = True,
                verbose: bool = True, generator: torch.Generator | None = None):
        """
        Evaluate the model on ``loader`` without gradient updates.

        Parameters
        ----------
        loader : DataLoader
        mask_ratio : float
            Fraction of observed values randomly hidden (mirrors training masking).
        metrics_key : str, default='Metrics'
            TensorBoard prefix for logged metrics.
        full_metrics : bool, default=True
            When True, compute and return RMSE/MAE/R² metrics; when False, only loss.
        generator : torch.Generator, optional
            Generator for creating the same validation masks across epochs.

        Returns
        -------
        avg_loss : float
        metrics : dict
        avg_miss_ratio : float
        """
        self.model.eval()  # Disables dropout
        total_loss, total_miss_ratio, n_samples = 0.0, 0.0, 0
        all_true, all_pred = [], []

        for batch in tqdm(loader, desc="Val", leave=False, disable=not verbose):
            batch = self.adapter.prepare_batch(batch, self.device)
            masks, miss_ratio = self.adapter.make_masks(batch=batch, mask_ratio=mask_ratio, mode="eval",
                                                        device=self.device, cfg=self.cfg, generator=generator)

            # Predict
            outputs = self.adapter.forward(self.model, batch=batch, masks=masks)

            # Compute loss
            loss_inputs = self.adapter.loss_inputs(batch=batch, outputs=outputs, masks=masks,
                                                   anisotropic_weights=self.model.anisotropic_weights)
            loss = self.loss_fn(**loss_inputs)
            if loss is None:
                continue

            batch_size = self.adapter.batch_size(batch=batch)
            total_loss += loss.item() * batch_size
            total_miss_ratio += miss_ratio.item() * batch_size
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
                logger.warning("No valid entries to evaluate")
                return total_loss / max(1, n_samples), {}, total_miss_ratio / max(1, n_samples)

            # Stack for full array metrics
            y_true = np.concatenate(all_true, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)

            # Compute and log metrics
            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, var_names=self.var_names)
            self.log_metrics(metrics_key, metrics)
        else:
            metrics = {}

        return total_loss / max(1, n_samples), metrics, total_miss_ratio / max(1, n_samples)

    def update_graph(self):
        """Recompute the k-NN graph via the graph provider using the current encoder."""
        if self.graph_provider is None:
            return

        if self.freeze_graph:
            return

        logger.info("Updating graph...")
        st = time()
        self.model.eval()

        self.graph_provider.update(encoder=self.model.coord_encoder, coords=self.full_coords, values=self.full_values,
                                   mask=self.full_mask, mean_values=self.global_means,
                                   anisotropic_weights=self.model.anisotropic_weights)
        logger.info("Updated graph in %.0f minutes." % ((time() - st) / 60))

    def log_metrics(self, prefix, d):
        """Recursively write all scalar entries in ``d`` to TensorBoard under ``prefix``."""
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
            self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_best_model(self):
        """ Restore the best model state. """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    @torch.no_grad()
    def reconstruct_full_dataset(self, loader: DataLoader, do_dropout: bool = False, show_progress: bool = True):
        """
        Run full reconstruction over ``loader`` without artificial masking.

        Parameters
        ----------
        loader : DataLoader
            Typically the full-dataset loader (all splits).
        do_dropout : bool, default=False
            When True, keeps dropout active for MC-Dropout uncertainty estimation.
        show_progress : bool, default=True

        Returns
        -------
        preds : Tensor (n, F)
            Predicted means for every point.
        pred_vars : Tensor (n, F) or None
            Predicted log-variances, or None for models without a variance head.
        avg_miss_ratio : float
            Mean fraction of missing values across the dataset.
        """
        self.model.eval()
        all_preds = []
        all_vars = []
        total_miss_ratio, n_samples = 0.0, 0

        self._enable_dropout(enable=do_dropout)

        # Define iterator
        iterator = loader
        if show_progress:
            iterator = tqdm(loader, desc="Val", leave=False)

        # Iterate over batches
        for batch in iterator:
            batch = self.adapter.prepare_batch(batch, self.device)

            # In reconstruction, everything becomes input (truly missing features are predicted, no artificial masking)
            masks, _ = self.adapter.make_masks(batch, mask_ratio=0.0, mode="reconstruct", device=self.device,
                                               cfg=self.cfg)

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

            # Compute true dataset missingness
            q_mask = self.adapter.get_query_mask(batch=batch)  # True = observed
            total_valid = q_mask.sum()
            total_total = q_mask.numel()
            if total_total > 0:
                miss_ratio = 1.0 - (float(total_valid) / float(total_total))
            else:
                miss_ratio = torch.tensor(0.0, device=q_mask.device)

            batch_size = self.adapter.batch_size(batch=batch)
            total_miss_ratio += miss_ratio * batch_size
            n_samples += batch_size

        avg_miss_ratio = total_miss_ratio / max(1, n_samples)

        return torch.cat(all_preds), torch.cat(all_vars) if all_vars else None, avg_miss_ratio

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs,
            early_stopping=None,
            mask_ratio: Union[float, Callable[[int], float]] = 0.5,
            optuna_callback=None,
            full_metrics: bool = False,
            verbose: bool = True,
            val_seed: int = 0):
        """
        Train the model for up to ``max_epochs`` epochs with optional early stopping.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        max_epochs : int
        early_stopping : callable or None
            Called with the current validation loss; returns True to stop training.
        mask_ratio : float or callable, default=0.5
            Fixed masking fraction or a schedule ``mask_ratio(epoch) -> float``.
        optuna_callback : callable or None
            Optuna pruning callback; called with ``(epoch, val_losses)``.
        full_metrics : bool, default=False
            When True, compute full validation metrics (RMSE, MAE, etc.) each epoch.
        verbose : bool, default=True
            When False, suppress tqdm bars and per-epoch console output (useful for Optuna).
        val_seed : int, default=0
            Generator for creating the same validation masks across epochs.

        Returns
        -------
        history : dict
            Keys: ``'train'``, ``'val'``, ``'metrics'``, ``'train_miss_ratio'``,
            ``'val_miss_ratio'``, each mapping epoch index → value.
        """
        history = {"train": {}, "val": {}, "metrics": {}, "train_miss_ratio": {}, "val_miss_ratio": {}}
        val_mask_generator = torch.Generator(device=self.device)

        # Iterate over epochs
        for epoch in range(max_epochs):
            # Check if mask ratio is a function or a fixed float and determine mask ratio to use
            if callable(mask_ratio):
                current_mask_ratio = mask_ratio(epoch)
            else:
                current_mask_ratio = mask_ratio

            # Update graph if specified
            if self.graph_provider is not None and self.cfg.graph_mode == "dynamic":
                # Warmup phase (no graph update yet)
                if epoch >= self.cfg.graph_warmup:

                    # Freezing (stop updating the graph)
                    if epoch == self.cfg.graph_freeze_epoch:
                        self.freeze_graph = True  # Freeze graph

                        # Freeze coord_encoder (if scope is limited to graph)
                        if self.cfg.encoder_scope == "graph":
                            for p in self.model.coord_encoder.parameters():
                                p.requires_grad = False
                            logger.info(f"Coord encoder frozen at epoch {self.cfg.graph_freeze_epoch}")

                        logger.info(f"Graph frozen at epoch {self.cfg.graph_freeze_epoch}")

                    # Update the graph
                    if (not self.freeze_graph) and (epoch % self.graph_provider.update_every == 0):
                        self.update_graph()

            # Train and compute losses
            train_loss, train_miss_ratio = self.train_one_epoch(train_loader, mask_ratio=current_mask_ratio,
                                                                verbose=verbose)
            val_mask_generator.manual_seed(val_seed)  # same val masks every epoch
            val_loss, val_metrics, val_miss_ratio = self.evaluate(loader=val_loader,
                                                                  mask_ratio=mask_ratio,
                                                                  metrics_key=f"Epoch_{epoch}",
                                                                  full_metrics=full_metrics,
                                                                  verbose=verbose,
                                                                  generator=val_mask_generator)

            # Update best model
            self.update_best_model(val_loss=val_loss)

            history["train"][epoch] = train_loss
            history["val"][epoch] = val_loss
            history["train_miss_ratio"][epoch] = train_miss_ratio
            history["val_miss_ratio"][epoch] = val_miss_ratio
            history["metrics"][epoch] = val_metrics

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # Optuna logging
            if optuna_callback is not None:
                optuna_callback(epoch=epoch, val_losses=list(history["val"].values()))

            # Early stopping (called before print so counter is current)
            should_stop = early_stopping and early_stopping(val_loss)

            if verbose:
                patience_str = ""
                if early_stopping is not None and hasattr(early_stopping, "counter"):
                    patience_str = f" | patience {early_stopping.counter}/{early_stopping.patience}"
                tqdm.write(
                    f"Epoch {epoch:03d}/{max_epochs}"
                    f" | train={train_loss:.6f} | val={val_loss:.6f}"
                    f" | miss={val_miss_ratio:.2f}{patience_str}"
                )

            if should_stop:
                logger.info(f"Early stopping at epoch {early_stopping.epoch}, best_epoch={early_stopping.best_epoch}")
                break

        # Load best model
        self.load_best_model()

        # Close writer
        self.writer.close()

        return history
