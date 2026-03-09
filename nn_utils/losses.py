import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def build_loss(loss_config):
    """ Instantiate loss class from given configuration. """
    loss_class = loss_config["class"]
    kwargs = loss_config.get("kwargs", {})
    return loss_class(**kwargs)


def name_to_loss_spec(loss_name):
    """
    Map a given name to a loss class and specification.

    :param loss_name: Name of the loss
    :return: Loss specification (dict)
    """
    loss_spec = {}
    if loss_name == "mse":
        loss_spec["class"] = MaskedMSELoss
        loss_spec["kwargs"] = {}
    elif loss_name == "hetero":
        loss_spec["class"] = HeteroscedasticGaussianNLL
        loss_spec["kwargs"] = {}
    elif loss_name == "physics_hetero":
        loss_spec["class"] = PhysicsLoss
        loss_spec["kwargs"] = {}
    else:
        raise ValueError(f"Unknown loss name {loss_name}")
    return loss_spec


class BaseLoss(nn.Module):
    """ Base loss class. """

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        """
        Computes loss.

        :param input: Model prediction (torch.Tensor).
        :param target: Ground truth (torch.Tensor).
        :param mask: Mask of values to consider (torch.Tensor, optional)
        :param coords: Coordinates (torch.Tensor, optional), e.g. needed for PhysicsLoss
        :param pred_var: Predicted variacnce (torch.Tensor, optional)
        :param kwargs: Extra keyword arguments
        :return: Scalar loss (torch.Tensor).
        """
        raise NotImplementedError


class MaskedMSELoss(BaseLoss):
    """ Masked MSE loss. """

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        valid_mask = valid_mask.to(input.device)

        mse = F.mse_loss(input[valid_mask], target[valid_mask], reduction="mean")
        return mse


class HeteroscedasticGaussianNLL(BaseLoss):
    """
    Negative log likelihood loss. Instead of variance, it uses log variance for numeric stability.
    Reference: Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer
    Vision?. In Advances in Neural Information Processing Systems. Curran Associates, Inc.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        sq_error = (input[valid_mask] - target[valid_mask]) ** 2
        loss = 0.5 * (sq_error * (torch.exp(-1 * pred_var[valid_mask])) + pred_var[valid_mask])
        return loss.mean()


class PhysicsLoss(BaseLoss):
    """ Loss combining a base loss with physics-inspired penalties (smoothness and bounds). """

    def __init__(self, base_loss: BaseLoss = MaskedMSELoss(), lambda_smooth: float = 0.1, lambda_bounds: float = 0.0, spacings=None, bounds=None):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
        self.spacings = spacings
        self.bounds = bounds

    def forward(self, input, target, mask=None, coords=None, pred_var=None, **kwargs):
        # Only use not-nan entries (from ground truth)
        if mask is None:
            valid_mask = ~torch.isnan(target)
        else:
            valid_mask = mask & (~torch.isnan(target))

        if valid_mask.sum() == 0:
            return None

        # Compute base loss
        base_loss_kwargs = {"input": input, "target": target, "coords": coords, "pred_var": pred_var, "mask": mask}
        base_loss_val = self.base_loss(**base_loss_kwargs)

        phys_loss_val = 0
        # Spatiotemporal smoothness penalty
        if coords is not None and self.spacings is not None:
            phys_loss_val += self._compute_smoothness_loss(input=input, coords=coords)

        # Enforce physical bounds
        if self.bounds is not None:
            phys_loss_val += self._compute_bounds_loss(input=input, mask=valid_mask)

        return base_loss_val + self.lambda_smooth * phys_loss_val + self.lambda_bounds * self.bounds

    def _compute_smoothness_loss(self, input, coords):
        # Query vs neighbours only
        q = input[:, 0, :]  # [batch_size, 1, n_features]
        n = input[:, 1:, :]  # [batch_size, n_neighbours, n_features]

        q_coord = coords[:, 0, :]
        n_coord = coords[:, 1:, :]

        # Squared distances
        dist2 = ((q_coord - n_coord) ** 2).sum(dim=-1)  # [batch_size, n_neighbours]

        # Gaussian edge weights (Belkin & Niyogi)
        sigma2 = self.sigma ** 2
        w = torch.exp(-dist2 / sigma2).unsqueeze(-1)  # [batch_size, n_neighbours, 1]

        diff2 = (q - n) ** 2  # [batch_size, n_neighbours, n_features]

        loss = (w * diff2).mean()
        return loss

    def _compute_bounds_loss(self, input, mask):
        y_min, y_max = self.bounds

        lower = torch.clamp(y_min - input, min=0) ** 2
        upper = torch.clamp(input - y_max, min=0) ** 2

        # Only consider valid entries
        lower = lower * mask.float()
        upper = upper * mask.float()

        loss = (lower.mean() + upper.mean()) / mask.sum()
        return loss
