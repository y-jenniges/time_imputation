import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=20, delta_ratio=1e-4, min_delta_abs=1e-8):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            delta_ratio (float): Fraction of current val_loss used to define min_delta adaptively
            min_delta_abs (float): Absolute minimum delta threshold (prevents too small comparisons)
        """
        self.patience = patience
        self.delta_ratio = delta_ratio
        self.min_delta_abs = min_delta_abs

        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.epoch = 0

    def reset(self):
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.epoch = 0

    def __call__(self, val_loss):
        """
        Update the early stopping state with a new validation loss.

        Args:
            val_loss (float or torch.Tensor): current validation loss
        """
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        # Adaptive delta: relative to current loss, but not smaller than absolute min
        min_delta = max(self.min_delta_abs, abs(val_loss) * self.delta_ratio)

        # Check if there is a significant improvement
        if val_loss < self.best_loss - min_delta:
            # Update best values
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        self.epoch += 1

        return self.early_stop
