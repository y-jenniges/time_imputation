
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from oceanmae.trainer import Trainer
from oceanmae.early_stopping import EarlyStopping


def fit_mae(trainer: Trainer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int = 100,
            mask_ratio: float = 0.5,
            early_stopper: EarlyStopping | None = None):
    """ Training loop for OceanMAE.
     Args:
         trainer (OceanMAE): OceanMAE trainer.
         train_loader (DataLoader): OceanMAE training dataset.
         val_loader (DataLoader): OceanMAE validation dataset.
         n_epochs (int, optional): Number of epochs. Defaults to 100.
         mask_ratio (float, optional): Mask ratio. Defaults to 0.5.
         early_stopper (EarlyStopping, optional): Early stopping callback. Defaults to None.
         device (str, optional): Device. Defaults to "cpu".
     Returns:

    """
    loss_per_epoch = {"train": {}, "val": {}}
    epoch = np.nan

    for epoch in range(n_epochs):
        # Training
        train_loss = trainer.train_one_epoch(loader=train_loader, mask_ratio=mask_ratio)
        loss_per_epoch["train"][epoch] = train_loss

        # Validation
        if val_loader is not None:
            val_loss, _, _, _ = trainer.evaluate(val_loader)
            loss_per_epoch["val"][epoch] = val_loss

            # Track best model
            trainer.update_best_model(val_loss)

            # Early stopping
            if early_stopper is not None and early_stopper(val_loss):
                print(f"[EarlyStop] Stop at epoch {epoch} — best epoch {early_stopper.best_epoch}")
                break

            print(f"Epoch {epoch}: train_loss={train_loss:.10f}, val_loss={val_loss:.10f}")
        else:
            # No validation: best model by train loss
            trainer.update_best_model(train_loss)
            print(f"Epoch {epoch}: train_loss={train_loss:.10}")

    # Restore best model
    trainer.load_best_model()

    # Get stop epoch
    stop_epoch = epoch

    return trainer.model, loss_per_epoch, stop_epoch
