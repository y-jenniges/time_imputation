"""Early stopping with adaptive improvement threshold."""

import torch
import numpy as np


class EarlyStopping:
    """
    Early stopping with an adaptive minimum-improvement threshold.

    The required improvement is the larger of ``min_delta_abs`` and
    ``delta_ratio * |val_loss|``, so the threshold scales with loss magnitude.

    Parameters
    ----------
    patience : int, default=20
        Epochs to wait without improvement before stopping.
    delta_ratio : float, default=1e-4
        Fraction of the current validation loss used as the minimum improvement.
    min_delta_abs : float, default=1e-8
        Absolute floor for the minimum improvement (prevents near-zero comparisons).
    """

    def __init__(self, patience=20, delta_ratio=1e-4, min_delta_abs=1e-8):
        self.patience = patience
        self.delta_ratio = delta_ratio
        self.min_delta_abs = min_delta_abs

        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.epoch = 0

    def reset(self):
        """Reset all state to allow reuse across multiple training runs."""
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.epoch = 0

    def __call__(self, val_loss):
        """
        Update state with ``val_loss`` and return True if training should stop.

        Parameters
        ----------
        val_loss : float or Tensor
            Current validation loss for the epoch.

        Returns
        -------
        bool
            True when the patience counter has reached ``patience``.
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
