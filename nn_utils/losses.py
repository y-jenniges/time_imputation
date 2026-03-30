import torch
import torch.nn as nn
import torch.nn.functional as F


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
        loss_spec["kwargs"] = {"sigma": 1.0, "lambda_smooth": 1e-3}
    elif loss_name == "hetero_smooth":
        loss_spec["class"] = PhysicsLoss
        loss_spec["kwargs"] = {"base_loss": HeteroscedasticGaussianNLL(), "sigma": 1.0, "lambda_smooth": 1e-3}
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

    def __init__(self, base_loss: BaseLoss = MaskedMSELoss(), sigma=1.0, lambda_smooth: float = 0.0, lambda_bounds: float = 0.0, bounds=None):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
        self.bounds = bounds
        self.sigma = sigma

    def forward(self, input, target, mask=None, pred_var=None,
                coords=None, neighbour_features=None, neighbour_coords=None,
                query_coords=None,
                neighbours=None,
                anisotropic_weights=None,
                **kwargs):
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

        # Spatiotemporal smoothness penalty
        smooth_loss_val = 0.0
        if neighbours is not None: #  and neighbour_coords is not None:
            smooth_loss_val += self._compute_smoothness_loss(input=input,
                                                             neighbour_features=neighbours["default"]["features"],  # @todo what to do with multiple scopes? smoothness in every direction?
                                                             neighbour_coords=neighbours["default"]["coords"],
                                                             query_coords=query_coords,
                                                             anisotropic_weights=anisotropic_weights)

        # Enforce physical bounds
        bounds_loss_val = 0.0
        if self.bounds is not None:
            bounds_loss_val += self._compute_bounds_loss(input=input, mask=valid_mask)

        return base_loss_val + self.lambda_smooth * smooth_loss_val + self.lambda_bounds * bounds_loss_val

    def _compute_smoothness_loss(self, input, neighbour_features, neighbour_coords, query_coords, anisotropic_weights=None):
        """ Distance-weighted smoothness penalty. """
        valid = ~torch.isnan(neighbour_features)
        n_feat = torch.nan_to_num(neighbour_features)

        # Squared spatio-temporal distances
        dist2_coords = ((query_coords.unsqueeze(1) - neighbour_coords) ** 2)  # [B, K, C]

        if anisotropic_weights is not None:
            # Ensure positivity
            pos_aniso_weights = torch.nn.functional.softplus(anisotropic_weights)  # [C]

            # Broadcast: [1,1,C]
            dist2_coords = (dist2_coords * pos_aniso_weights.view(1, 1, -1))  # [B, K]

        # RBF kernel edge weight
        w = torch.exp(-dist2_coords.sum(dim=-1) / (2*self.sigma**2) ).unsqueeze(-1)  # [B, K, 1]

        # Squared feature difference
        diff2 = (input.unsqueeze(1) - n_feat) ** 2  # [B, K, F]
        weighted = w * diff2 * valid
        denom = (w * valid).sum()

        if valid.sum() == 0:
            return torch.tensor(0.0, device=input.device)

        return weighted.sum() / denom

    def _compute_bounds_loss(self, input, mask):
        """ Feature bound penalty. """
        y_min, y_max = self.bounds

        lower = torch.clamp(y_min - input, min=0) ** 2
        upper = torch.clamp(input - y_max, min=0) ** 2

        # Only consider valid entries
        lower = lower * mask.float()
        upper = upper * mask.float()

        loss = (lower.mean() + upper.mean()) / mask.sum()
        return loss
